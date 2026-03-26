"""Split dataset into train/val/test with project-disjoint leakage-aware assignment.

Held-out projects are fully excluded from training. No project appears in
more than one split. Produces an assignments CSV and a diagnostics report.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from graph_vul.config import (
    HELDOUT_CWE_IDS,
    SEED,
    SPLIT_ASSIGNMENTS,
    SPLIT_REPORT,
    SPLITS_DIR,
    SUBSET_CSV,
    TEST_SPLIT,
    VAL_SPLIT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Split assignment via StratifiedGroupKFold
# ---------------------------------------------------------------------------

def _assign_splits(
    df: pd.DataFrame,
    seed: int,
    val_frac: float = VAL_SPLIT,
    test_frac: float = TEST_SPLIT,
) -> pd.DataFrame:
    """Assign rows to train/val/test using stratified group k-fold.

    Groups = projects (no project spans splits), stratified on the vul label.
    """
    labels = df["vul"].values
    groups = df["project"].values

    # First split: separate test from (train+val)
    n_folds_test = max(round(1.0 / test_frac), 2)
    sgkf_test = StratifiedGroupKFold(n_splits=n_folds_test, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(sgkf_test.split(df, labels, groups))

    # Second split: separate val from train (within the trainval portion)
    val_frac_of_trainval = val_frac / (1.0 - test_frac)
    n_folds_val = max(round(1.0 / val_frac_of_trainval), 2)
    sgkf_val = StratifiedGroupKFold(n_splits=n_folds_val, shuffle=True, random_state=seed)
    df_trainval = df.iloc[trainval_idx]
    train_rel_idx, val_rel_idx = next(
        sgkf_val.split(df_trainval, labels[trainval_idx], groups[trainval_idx])
    )
    train_idx = trainval_idx[train_rel_idx]
    val_idx = trainval_idx[val_rel_idx]

    assignments = pd.Series("train", index=df.index, name="split")
    assignments.iloc[val_idx] = "val"
    assignments.iloc[test_idx] = "test"

    return pd.DataFrame({"sample_id": df["sample_id"], "split": assignments})


# ---------------------------------------------------------------------------
# Unseen CWE split assignment
# ---------------------------------------------------------------------------

def _assign_unseen_splits(
    df_heldout: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """Split held-out CWE samples into unseen_val and unseen_test (50/50, project-disjoint)."""
    labels = df_heldout["vul"].values
    groups = df_heldout["project"].values

    sgkf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    unseen_val_idx, unseen_test_idx = next(sgkf.split(df_heldout, labels, groups))

    assignments = pd.Series("unseen_val", index=df_heldout.index, name="split")
    assignments.iloc[unseen_test_idx] = "unseen_test"

    return pd.DataFrame({"sample_id": df_heldout["sample_id"], "split": assignments})


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_splits(df: pd.DataFrame, assignments: pd.DataFrame) -> list[str]:
    """Validate split properties. Returns list of warnings."""
    warnings = []

    merged = df.merge(assignments, on="sample_id")

    # Check all rows assigned
    if len(merged) != len(df):
        warnings.append(f"Row count mismatch: {len(df)} input rows, {len(merged)} assigned")

    # Check project disjointness
    split_projects = {}
    for split_name in ["train", "val", "test"]:
        split_df = merged[merged["split"] == split_name]
        split_projects[split_name] = set(split_df["project"].unique())

    for s1, s2 in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = split_projects.get(s1, set()) & split_projects.get(s2, set())
        if overlap:
            warnings.append(f"PROJECT OVERLAP between {s1} and {s2}: {overlap}")

    # Check for function-level code leakage across splits
    if "func_before" in merged.columns:
        for s1, s2 in [("train", "test"), ("train", "val"), ("val", "test")]:
            codes_s1 = set(merged.loc[merged["split"] == s1, "func_before"].dropna())
            codes_s2 = set(merged.loc[merged["split"] == s2, "func_before"].dropna())
            code_overlap = len(codes_s1 & codes_s2)
            if code_overlap > 0:
                warnings.append(
                    f"CODE LEAKAGE: {code_overlap} identical func_before texts in both {s1} and {s2}"
                )

    # Check split sizes
    total = len(merged)
    for split_name in ["train", "val", "test"]:
        split_n = (merged["split"] == split_name).sum()
        if split_n == 0:
            warnings.append(f"Split '{split_name}' is empty")
        elif split_n / total < 0.05:
            warnings.append(
                f"Split '{split_name}' has only {split_n} rows ({split_n/total:.1%} of total)"
            )

        # Check if split has only one class
        split_df = merged[merged["split"] == split_name]
        if len(split_df) > 0 and split_df["vul"].nunique() < 2:
            warnings.append(f"Split '{split_name}' has only one class (vul={split_df['vul'].iloc[0]})")

    return warnings


def _validate_unseen_splits(df_heldout: pd.DataFrame, assignments: pd.DataFrame) -> list[str]:
    """Validate unseen split properties."""
    warnings = []
    if len(df_heldout) == 0:
        return warnings

    merged = df_heldout.merge(assignments, on="sample_id")

    # Check project disjointness between unseen_val and unseen_test
    unseen_val_projects = set(
        merged.loc[merged["split"] == "unseen_val", "project"].unique()
    )
    unseen_test_projects = set(
        merged.loc[merged["split"] == "unseen_test", "project"].unique()
    )
    overlap = unseen_val_projects & unseen_test_projects
    if overlap:
        warnings.append(f"PROJECT OVERLAP between unseen_val and unseen_test: {overlap}")

    # Check no held-out CWE samples leaked into train/val/test
    heldout_ids = set(df_heldout["sample_id"].values)
    for split_name in ["train", "val", "test"]:
        leaked = heldout_ids & set(
            assignments.loc[assignments["split"] == split_name, "sample_id"].values
        )
        if leaked:
            warnings.append(
                f"LEAKAGE: {len(leaked)} held-out CWE samples found in '{split_name}' split"
            )

    # Check unseen splits are non-empty
    for split_name in ["unseen_val", "unseen_test"]:
        n = (merged["split"] == split_name).sum()
        if n == 0:
            warnings.append(f"Split '{split_name}' is empty")

    return warnings


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _generate_split_report(
    df: pd.DataFrame,
    assignments: pd.DataFrame,
    warnings: list[str],
) -> str:
    """Generate a Markdown diagnostics report."""
    merged = df.merge(assignments, on="sample_id")

    lines = [
        "# Split Diagnostics Report",
        "",
        "## Split Sizes",
        "",
        "| Split | Total | Vulnerable | Benign | Vul Ratio |",
        "|-------|-------|------------|--------|-----------|",
    ]

    for split_name in ["train", "val", "test", "unseen_val", "unseen_test"]:
        split_df = merged[merged["split"] == split_name]
        if len(split_df) == 0:
            continue
        total = len(split_df)
        vul = int((split_df["vul"] == 1).sum())
        benign = total - vul
        ratio = f"{vul/max(total,1):.2%}"
        lines.append(f"| {split_name} | {total} | {vul} | {benign} | {ratio} |")

    lines.append(f"| **total** | {len(merged)} | {int((merged['vul']==1).sum())} | "
                 f"{int((merged['vul']==0).sum())} | {(merged['vul']==1).mean():.2%} |")

    # Per-CWE breakdown per split
    if "cwe_id" in merged.columns:
        lines.extend(["", "## Per-CWE Breakdown", ""])
        lines.append("| Split | CWE | Total | Vulnerable | Benign |")
        lines.append("|-------|-----|-------|------------|--------|")
        for split_name in ["train", "val", "test", "unseen_val", "unseen_test"]:
            split_df = merged[merged["split"] == split_name]
            for cwe_id in sorted(split_df["cwe_id"].unique()):
                cwe_rows = split_df[split_df["cwe_id"] == cwe_id]
                vul = int((cwe_rows["vul"] == 1).sum())
                benign = int((cwe_rows["vul"] == 0).sum())
                lines.append(f"| {split_name} | {cwe_id} | {len(cwe_rows)} | {vul} | {benign} |")

    # CVE coverage
    lines.extend(["", "## CVE Coverage", ""])
    lines.append("| Split | Unique CVEs | CVE List |")
    lines.append("|-------|-------------|----------|")
    all_splits = ["train", "val", "test", "unseen_val", "unseen_test"]
    for split_name in all_splits:
        split_df = merged[merged["split"] == split_name]
        if len(split_df) == 0:
            continue
        cves = sorted(split_df["cve_id"].unique())
        lines.append(f"| {split_name} | {len(cves)} | {', '.join(cves)} |")

    # Project coverage
    lines.extend(["", "## Project Coverage", ""])
    lines.append("| Split | Unique Projects | Project List |")
    lines.append("|-------|-----------------|--------------|")
    for split_name in all_splits:
        split_df = merged[merged["split"] == split_name]
        if len(split_df) == 0:
            continue
        projects = sorted(split_df["project"].unique())
        lines.append(f"| {split_name} | {len(projects)} | {', '.join(projects)} |")

    # Overlap verification
    lines.extend(["", "## Overlap Verification", ""])
    split_projects = {}
    for split_name in all_splits:
        split_df = merged[merged["split"] == split_name]
        split_projects[split_name] = set(split_df["project"].unique())

    pairs = [("train", "val"), ("train", "test"), ("val", "test"),
             ("unseen_val", "unseen_test")]
    for s1, s2 in pairs:
        if s1 not in split_projects or s2 not in split_projects:
            continue
        overlap = split_projects[s1] & split_projects[s2]
        status = "NONE" if not overlap else f"**LEAKAGE**: {overlap}"
        lines.append(f"- Project overlap ({s1} ∩ {s2}): {status}")

    # Val/test project details
    for split_name in ["val", "test", "unseen_val", "unseen_test"]:
        split_df = merged[merged["split"] == split_name]
        lines.extend(["", f"## {split_name.title()} Projects", ""])
        for proj in sorted(split_df["project"].unique()):
            proj_df = split_df[split_df["project"] == proj]
            vul = int((proj_df["vul"] == 1).sum())
            benign = int((proj_df["vul"] == 0).sum())
            cves = sorted(proj_df["cve_id"].unique())
            lines.append(f"- **{proj}** ({len(proj_df)} rows, {vul} vul, {benign} benign, CVEs: {', '.join(cves)})")

    # Metadata distributions
    for col in ("code_char_len_bucket", "code_line_count_bucket", "code_token_count_bucket"):
        if col not in merged.columns:
            continue
        lines.extend(["", f"## {col} by Split", ""])
        ct = pd.crosstab(merged["split"], merged[col])
        lines.append("| Split | " + " | ".join(str(c) for c in ct.columns) + " |")
        lines.append("|-------" + "".join("| --- " for _ in ct.columns) + "|")
        for split_name in ["train", "val", "test", "unseen_val", "unseen_test"]:
            if split_name in ct.index:
                row_vals = " | ".join(str(ct.loc[split_name, c]) for c in ct.columns)
                lines.append(f"| {split_name} | {row_vals} |")

    # Warnings
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for w in warnings:
            lines.append(f"- ⚠️ {w}")
    else:
        lines.extend(["", "## Warnings", "", "No warnings — all constraints satisfied."])

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def split_data(
    input_csv: str = str(SUBSET_CSV),
    output_dir: str = str(SPLITS_DIR),
    seed: int = SEED,
) -> None:
    df = pd.read_csv(input_csv)

    # Derive metadata for reporting
    code = df["func_before"].fillna("")
    df["code_char_len"] = code.str.len()
    df["code_line_count"] = code.str.count("\n") + 1
    df["code_token_count"] = code.str.split().str.len().fillna(0).astype(int)
    for col in ("code_char_len", "code_line_count", "code_token_count"):
        try:
            df[col + "_bucket"] = pd.qcut(
                df[col], q=4, labels=["short", "medium", "long", "very_long"],
                duplicates="drop",
            )
        except ValueError:
            df[col + "_bucket"] = "medium"

    # Separate held-out CWE rows for unseen evaluation
    has_cwe = "cwe_id" in df.columns
    df_heldout = pd.DataFrame()
    if has_cwe and HELDOUT_CWE_IDS:
        heldout_mask = df["cwe_id"].isin(HELDOUT_CWE_IDS)
        df_heldout = df[heldout_mask].copy()
        df = df[~heldout_mask].copy()
        logger.info(
            "Separated held-out CWEs (%s): %d rows for unseen splits, %d remaining for train/val/test",
            ", ".join(HELDOUT_CWE_IDS), len(df_heldout), len(df),
        )

    # Assign splits on training CWEs (stratified by label, grouped by project)
    assignments = _assign_splits(df, seed)

    # Split held-out CWE rows into unseen_val / unseen_test
    if len(df_heldout) > 0:
        unseen_assignments = _assign_unseen_splits(df_heldout, seed)
        assignments = pd.concat([assignments, unseen_assignments], ignore_index=True)

    # Validate
    warnings = _validate_splits(df, assignments)
    warnings.extend(_validate_unseen_splits(df_heldout, assignments))
    for w in warnings:
        logger.warning(w)

    # Write outputs
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    assignments_path = out / "assignments.csv"
    assignments.to_csv(assignments_path, index=False)
    logger.info("Wrote split assignments to %s", assignments_path)

    # Combine main + held-out for reporting
    df_all = pd.concat([df, df_heldout], ignore_index=True) if len(df_heldout) > 0 else df
    report = _generate_split_report(df_all, assignments, warnings)
    report_path = out / "split_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info("Wrote split report to %s", report_path)

    # Log summary
    for split_name in ["train", "val", "test", "unseen_val", "unseen_test"]:
        n = (assignments["split"] == split_name).sum()
        if n > 0:
            logger.info("Split '%s': %d rows", split_name, n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data with stratified group-aware assignment")
    parser.add_argument("--input", type=str, default=str(SUBSET_CSV))
    parser.add_argument("--output", type=str, default=str(SPLITS_DIR))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    split_data(input_csv=args.input, output_dir=args.output, seed=args.seed)
