"""Split dataset into train/val/test/unseen_val/unseen_test with project-disjoint assignment.

Two regimes:
- **Seen**: projects that do NOT touch any held-out CWE -> train / val / test
- **Unseen**: projects that contain at least one held-out CWE row -> unseen_val / unseen_test

Full-project holdout: if *any* row in a project has a held-out CWE, the *entire*
project (all CWEs) moves to the unseen pool.  No project appears in more than one
split.

Produces:
- assignments.csv              (sample_id, split)
- train_sampler_weights.csv    (per-row weights for balanced training)
- train_balanced_manifest.csv  (materialized balanced training set)
- split_report.md              (diagnostics)
"""

import argparse
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from graph_vul.config import (
    BALANCE_CWE_SMOOTHING,
    BALANCE_MAX_PROJECT_SHARE,
    BALANCE_MAX_WEIGHT,
    BALANCE_PROJECT_DOWNWEIGHT,
    BALANCE_TARGET_VUL_RATIO,
    HELDOUT_CWE_IDS,
    MIN_HELDOUT_PROJECTS,
    MIN_HELDOUT_TOTAL_ROWS,
    MIN_HELDOUT_VUL_ROWS,
    SEED,
    SPLITS_DIR,
    SUBSET_CSV,
    TEST_SPLIT,
    VAL_SPLIT,
)

logger = logging.getLogger(__name__)

ALL_SPLITS = ["train", "val", "test", "unseen_val", "unseen_test"]


# ---------------------------------------------------------------------------
# Held-out CWE eligibility validation
# ---------------------------------------------------------------------------

def _validate_heldout_eligibility(
    df: pd.DataFrame,
    heldout_cwe_ids: list[str],
    min_total: int = MIN_HELDOUT_TOTAL_ROWS,
    min_vul: int = MIN_HELDOUT_VUL_ROWS,
    min_projects: int = MIN_HELDOUT_PROJECTS,
) -> None:
    """Raise ValueError if any held-out CWE is too small for meaningful evaluation."""
    if "cwe_id" not in df.columns or not heldout_cwe_ids:
        return

    errors = []
    for cwe in heldout_cwe_ids:
        cwe_rows = df[df["cwe_id"] == cwe]
        total = len(cwe_rows)
        vul = int((cwe_rows["vul"] == 1).sum())
        n_projects = cwe_rows["project"].nunique()

        if total < min_total:
            errors.append(
                f"{cwe}: only {total} total rows (minimum {min_total})"
            )
        if vul < min_vul:
            errors.append(
                f"{cwe}: only {vul} vulnerable rows (minimum {min_vul})"
            )
        if n_projects < min_projects:
            errors.append(
                f"{cwe}: only {n_projects} distinct projects (minimum {min_projects})"
            )

    if errors:
        raise ValueError(
            "Held-out CWE eligibility check failed:\n  - " + "\n  - ".join(errors)
        )


# ---------------------------------------------------------------------------
# Full-project holdout pool construction
# ---------------------------------------------------------------------------

def _build_unseen_pool(
    df: pd.DataFrame,
    heldout_cwe_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Partition df into seen and unseen pools based on full-project holdout.

    Returns (df_seen, df_unseen, migration_info) where migration_info maps
    each unseen project to the held-out CWE(s) that caused its migration.
    """
    if "cwe_id" not in df.columns or not heldout_cwe_ids:
        return df.copy(), pd.DataFrame(columns=df.columns), {}

    # Find projects that touch any held-out CWE
    heldout_rows = df[df["cwe_id"].isin(heldout_cwe_ids)]
    migration_info: dict[str, list[str]] = {}
    for project, group in heldout_rows.groupby("project"):
        migration_info[project] = sorted(group["cwe_id"].unique().tolist())

    unseen_projects = set(migration_info.keys())
    unseen_mask = df["project"].isin(unseen_projects)
    df_unseen = df[unseen_mask].copy()
    df_seen = df[~unseen_mask].copy()

    return df_seen, df_unseen, migration_info


# ---------------------------------------------------------------------------
# Split assignment via StratifiedGroupKFold
# ---------------------------------------------------------------------------

def _assign_splits(
    df: pd.DataFrame,
    seed: int,
    val_frac: float = VAL_SPLIT,
    test_frac: float = TEST_SPLIT,
) -> pd.DataFrame:
    """Assign seen-pool rows to train/val/test using stratified group k-fold.

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
# Unseen pool split assignment
# ---------------------------------------------------------------------------

def _assign_unseen_splits(
    df_unseen: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """Split unseen-pool rows into unseen_val and unseen_test (50/50, project-disjoint)."""
    labels = df_unseen["vul"].values
    groups = df_unseen["project"].values

    sgkf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    unseen_val_idx, unseen_test_idx = next(sgkf.split(df_unseen, labels, groups))

    assignments = pd.Series("unseen_val", index=df_unseen.index, name="split")
    assignments.iloc[unseen_test_idx] = "unseen_test"

    return pd.DataFrame({"sample_id": df_unseen["sample_id"], "split": assignments})


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_splits(
    df: pd.DataFrame,
    assignments: pd.DataFrame,
    heldout_cwe_ids: list[str] | None = None,
) -> list[str]:
    """Validate split properties across all five splits. Returns list of warnings."""
    warnings: list[str] = []

    merged = df.merge(assignments, on="sample_id")

    # Check all rows assigned
    if len(merged) != len(df):
        warnings.append(f"Row count mismatch: {len(df)} input rows, {len(merged)} assigned")

    # Collect projects per split
    split_projects: dict[str, set[str]] = {}
    for split_name in ALL_SPLITS:
        split_df = merged[merged["split"] == split_name]
        split_projects[split_name] = set(split_df["project"].unique())

    # Check all 10 split pairs for project overlap
    for s1, s2 in combinations(ALL_SPLITS, 2):
        overlap = split_projects.get(s1, set()) & split_projects.get(s2, set())
        if overlap:
            warnings.append(f"PROJECT OVERLAP between {s1} and {s2}: {overlap}")

    # Check no held-out CWE IDs in seen splits
    if heldout_cwe_ids and "cwe_id" in merged.columns:
        for split_name in ["train", "val", "test"]:
            split_df = merged[merged["split"] == split_name]
            leaked_cwes = set(split_df["cwe_id"].unique()) & set(heldout_cwe_ids)
            if leaked_cwes:
                warnings.append(
                    f"HELD-OUT CWE LEAKAGE: {leaked_cwes} found in '{split_name}' split"
                )

    # Check unseen-pool projects don't appear in seen splits
    unseen_projects = split_projects.get("unseen_val", set()) | split_projects.get("unseen_test", set())
    seen_projects = split_projects.get("train", set()) | split_projects.get("val", set()) | split_projects.get("test", set())
    cross_regime = unseen_projects & seen_projects
    if cross_regime:
        warnings.append(f"CROSS-REGIME PROJECT LEAKAGE: {cross_regime}")

    # Check for function-level code leakage across splits
    if "func_before" in merged.columns:
        for s1, s2 in combinations(ALL_SPLITS, 2):
            codes_s1 = set(merged.loc[merged["split"] == s1, "func_before"].dropna())
            codes_s2 = set(merged.loc[merged["split"] == s2, "func_before"].dropna())
            code_overlap = len(codes_s1 & codes_s2)
            if code_overlap > 0:
                warnings.append(
                    f"CODE LEAKAGE: {code_overlap} identical func_before texts in both {s1} and {s2}"
                )

    # Check split sizes and single-class splits
    total = len(merged)
    for split_name in ALL_SPLITS:
        split_n = (merged["split"] == split_name).sum()
        if split_n == 0:
            warnings.append(f"Split '{split_name}' is empty")
        elif split_n / total < 0.05:
            warnings.append(
                f"Split '{split_name}' has only {split_n} rows ({split_n/total:.1%} of total)"
            )

        split_df = merged[merged["split"] == split_name]
        if len(split_df) > 0 and split_df["vul"].nunique() < 2:
            warnings.append(f"Split '{split_name}' has only one class (vul={split_df['vul'].iloc[0]})")

    return warnings


# ---------------------------------------------------------------------------
# Train balancing
# ---------------------------------------------------------------------------

def _compute_sample_weights(
    df_train: pd.DataFrame,
    target_vul_ratio: float = BALANCE_TARGET_VUL_RATIO,
    cwe_smoothing: float = BALANCE_CWE_SMOOTHING,
    project_downweight: float = BALANCE_PROJECT_DOWNWEIGHT,
    max_weight: float = BALANCE_MAX_WEIGHT,
    max_project_share: float = BALANCE_MAX_PROJECT_SHARE,
) -> pd.DataFrame:
    """Compute per-row sample weights for balanced training.

    Weight strategy:
    - Vulnerable rows: upweighted to target ratio, with CWE inverse-frequency smoothing
    - Benign rows: downweighted, with project inverse-size correction
    - All weights capped and normalized
    """
    n = len(df_train)
    n_vul = int((df_train["vul"] == 1).sum())
    n_benign = n - n_vul

    if n_vul == 0 or n_benign == 0:
        weights = pd.DataFrame({
            "sample_id": df_train["sample_id"],
            "split": "train",
            "project": df_train["project"],
            "cwe_id": df_train.get("cwe_id", ""),
            "vul": df_train["vul"],
            "sample_weight": 1.0,
        })
        return weights

    # Base class weights to achieve target ratio
    # target_vul_ratio = w_vul * n_vul / (w_vul * n_vul + w_benign * n_benign)
    # Solving: w_vul / w_benign = target_vul_ratio * n_benign / ((1 - target_vul_ratio) * n_vul)
    w_vul_base = target_vul_ratio * n_benign / ((1.0 - target_vul_ratio) * n_vul)
    w_benign_base = 1.0

    raw_weights = np.ones(n, dtype=np.float64)

    # Vulnerable rows: CWE inverse-frequency smoothing
    vul_mask = df_train["vul"].values == 1
    if "cwe_id" in df_train.columns and cwe_smoothing > 0:
        vul_cwe_counts = df_train.loc[vul_mask, "cwe_id"].value_counts()
        n_vul_total = vul_mask.sum()
        cwe_factors = {}
        for cwe, count in vul_cwe_counts.items():
            cwe_factors[cwe] = (n_vul_total / count) ** cwe_smoothing
        cwe_weight_arr = df_train.loc[vul_mask, "cwe_id"].map(cwe_factors).values
        # Normalize CWE factors so their mean is 1
        cwe_weight_arr = cwe_weight_arr / cwe_weight_arr.mean()
        raw_weights[vul_mask] = w_vul_base * cwe_weight_arr
    else:
        raw_weights[vul_mask] = w_vul_base

    # Benign rows: project inverse-size downweighting
    benign_mask = ~vul_mask
    if project_downweight > 0:
        project_sizes = df_train.loc[benign_mask, "project"].value_counts()
        median_size = project_sizes.median()
        proj_factors = {}
        for proj, size in project_sizes.items():
            proj_factors[proj] = (median_size / size) ** project_downweight
        proj_weight_arr = df_train.loc[benign_mask, "project"].map(proj_factors).values
        proj_weight_arr = proj_weight_arr / proj_weight_arr.mean()
        raw_weights[benign_mask] = w_benign_base * proj_weight_arr
    else:
        raw_weights[benign_mask] = w_benign_base

    # Cap at max_weight * median
    median_w = np.median(raw_weights)
    if median_w > 0:
        cap = max_weight * median_w
        raw_weights = np.minimum(raw_weights, cap)

    # Max project share capping
    if max_project_share > 0:
        total_weight = raw_weights.sum()
        for proj in df_train["project"].unique():
            proj_mask = df_train["project"].values == proj
            proj_total = raw_weights[proj_mask].sum()
            proj_share = proj_total / total_weight
            if proj_share > max_project_share:
                scale = (max_project_share * total_weight) / proj_total
                raw_weights[proj_mask] *= scale

    # Normalize so weights sum to n
    raw_weights = raw_weights * (n / raw_weights.sum())

    weights = pd.DataFrame({
        "sample_id": df_train["sample_id"].values,
        "split": "train",
        "project": df_train["project"].values,
        "cwe_id": df_train["cwe_id"].values if "cwe_id" in df_train.columns else "",
        "vul": df_train["vul"].values,
        "sample_weight": raw_weights,
    })
    return weights


def _generate_balanced_manifest(
    df_train: pd.DataFrame,
    weights_df: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """Sample from training rows using weights to produce a materialized balanced manifest."""
    rng = np.random.RandomState(seed)
    n = len(df_train)
    probs = weights_df["sample_weight"].values / weights_df["sample_weight"].sum()
    indices = rng.choice(len(df_train), size=n, replace=True, p=probs)

    manifest = pd.DataFrame({
        "sample_id": df_train["sample_id"].values[indices],
        "split": "train",
        "sample_weight": weights_df["sample_weight"].values[indices],
    })
    return manifest


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _generate_split_report(
    df: pd.DataFrame,
    assignments: pd.DataFrame,
    warnings: list[str],
    migration_info: dict[str, list[str]] | None = None,
    weights_df: pd.DataFrame | None = None,
) -> str:
    """Generate a Markdown diagnostics report."""
    merged = df.merge(assignments, on="sample_id")

    lines = [
        "# Split Diagnostics Report",
        "",
        "## Split Sizes",
        "",
        "| Split | Total | Vulnerable | Benign | Vul Ratio | Projects | CWEs |",
        "|-------|-------|------------|--------|-----------|----------|------|",
    ]

    for split_name in ALL_SPLITS:
        split_df = merged[merged["split"] == split_name]
        if len(split_df) == 0:
            continue
        total = len(split_df)
        vul = int((split_df["vul"] == 1).sum())
        benign = total - vul
        ratio = f"{vul/max(total,1):.2%}"
        n_projects = split_df["project"].nunique()
        n_cwes = split_df["cwe_id"].nunique() if "cwe_id" in split_df.columns else 0
        lines.append(f"| {split_name} | {total} | {vul} | {benign} | {ratio} | {n_projects} | {n_cwes} |")

    lines.append(f"| **total** | {len(merged)} | {int((merged['vul']==1).sum())} | "
                 f"{int((merged['vul']==0).sum())} | {(merged['vul']==1).mean():.2%} | "
                 f"{merged['project'].nunique()} | "
                 f"{merged['cwe_id'].nunique() if 'cwe_id' in merged.columns else 0} |")

    # Per-CWE breakdown per split
    if "cwe_id" in merged.columns:
        lines.extend(["", "## Per-CWE Breakdown", ""])
        lines.append("| Split | CWE | Total | Vulnerable | Benign |")
        lines.append("|-------|-----|-------|------------|--------|")
        for split_name in ALL_SPLITS:
            split_df = merged[merged["split"] == split_name]
            for cwe_id in sorted(split_df["cwe_id"].unique()):
                cwe_rows = split_df[split_df["cwe_id"] == cwe_id]
                vul = int((cwe_rows["vul"] == 1).sum())
                benign = int((cwe_rows["vul"] == 0).sum())
                lines.append(f"| {split_name} | {cwe_id} | {len(cwe_rows)} | {vul} | {benign} |")

    # CVE coverage
    if "cve_id" in merged.columns:
        lines.extend(["", "## CVE Coverage", ""])
        lines.append("| Split | Unique CVEs | CVE List |")
        lines.append("|-------|-------------|----------|")
        for split_name in ALL_SPLITS:
            split_df = merged[merged["split"] == split_name]
            if len(split_df) == 0:
                continue
            cves = sorted(split_df["cve_id"].unique())
            lines.append(f"| {split_name} | {len(cves)} | {', '.join(str(c) for c in cves)} |")

    # Project coverage
    lines.extend(["", "## Project Coverage", ""])
    lines.append("| Split | Unique Projects | Project List |")
    lines.append("|-------|-----------------|--------------|")
    for split_name in ALL_SPLITS:
        split_df = merged[merged["split"] == split_name]
        if len(split_df) == 0:
            continue
        projects = sorted(split_df["project"].unique())
        lines.append(f"| {split_name} | {len(projects)} | {', '.join(projects)} |")

    # Overlap verification (all 10 pairs)
    lines.extend(["", "## Overlap Verification", ""])
    split_projects: dict[str, set[str]] = {}
    for split_name in ALL_SPLITS:
        split_df = merged[merged["split"] == split_name]
        split_projects[split_name] = set(split_df["project"].unique())

    for s1, s2 in combinations(ALL_SPLITS, 2):
        p1 = split_projects.get(s1, set())
        p2 = split_projects.get(s2, set())
        if not p1 and not p2:
            continue
        overlap = p1 & p2
        status = "NONE" if not overlap else f"**LEAKAGE**: {overlap}"
        lines.append(f"- Project overlap ({s1} ∩ {s2}): {status}")

    # Project Exclusivity section
    lines.extend(["", "## Project Exclusivity", ""])
    seen_pool = split_projects.get("train", set()) | split_projects.get("val", set()) | split_projects.get("test", set())
    unseen_pool = split_projects.get("unseen_val", set()) | split_projects.get("unseen_test", set())
    lines.append(f"- Seen-pool projects: {len(seen_pool)}")
    lines.append(f"- Unseen-pool projects: {len(unseen_pool)}")
    cross = seen_pool & unseen_pool
    lines.append(f"- Cross-regime overlap: {'NONE' if not cross else f'**LEAKAGE**: {cross}'}")

    if migration_info:
        lines.extend(["", "### Migration Reasons", ""])
        lines.append("| Project | Held-out CWEs |")
        lines.append("|---------|---------------|")
        for proj in sorted(migration_info.keys()):
            cwes = ", ".join(migration_info[proj])
            lines.append(f"| {proj} | {cwes} |")

    # Val/test/unseen project details
    for split_name in ["val", "test", "unseen_val", "unseen_test"]:
        split_df = merged[merged["split"] == split_name]
        if len(split_df) == 0:
            continue
        lines.extend(["", f"## {split_name.replace('_', ' ').title()} Projects", ""])
        for proj in sorted(split_df["project"].unique()):
            proj_df = split_df[split_df["project"] == proj]
            vul = int((proj_df["vul"] == 1).sum())
            benign = int((proj_df["vul"] == 0).sum())
            if "cve_id" in proj_df.columns:
                cves = sorted(proj_df["cve_id"].unique())
                lines.append(f"- **{proj}** ({len(proj_df)} rows, {vul} vul, {benign} benign, CVEs: {', '.join(str(c) for c in cves)})")
            else:
                lines.append(f"- **{proj}** ({len(proj_df)} rows, {vul} vul, {benign} benign)")

    # Metadata distributions
    for col in ("code_char_len_bucket", "code_line_count_bucket", "code_token_count_bucket"):
        if col not in merged.columns:
            continue
        lines.extend(["", f"## {col} by Split", ""])
        ct = pd.crosstab(merged["split"], merged[col])
        lines.append("| Split | " + " | ".join(str(c) for c in ct.columns) + " |")
        lines.append("|-------" + "".join("| --- " for _ in ct.columns) + "|")
        for split_name in ALL_SPLITS:
            if split_name in ct.index:
                row_vals = " | ".join(str(ct.loc[split_name, c]) for c in ct.columns)
                lines.append(f"| {split_name} | {row_vals} |")

    # Train Balancing section
    if weights_df is not None and len(weights_df) > 0:
        lines.extend(["", "## Train Balancing", ""])

        # Raw vs balanced ratio
        raw_vul = int((weights_df["vul"] == 1).sum())
        raw_total = len(weights_df)
        raw_ratio = raw_vul / max(raw_total, 1)

        w = weights_df["sample_weight"].values
        vul_mask = weights_df["vul"].values == 1
        balanced_vul_weight = w[vul_mask].sum()
        balanced_total_weight = w.sum()
        balanced_ratio = balanced_vul_weight / max(balanced_total_weight, 1e-9)

        lines.append(f"- Raw train vul ratio: {raw_ratio:.2%} ({raw_vul}/{raw_total})")
        lines.append(f"- Balanced effective vul ratio: {balanced_ratio:.2%}")
        lines.append(f"- Weight stats: min={w.min():.3f}, median={np.median(w):.3f}, "
                     f"max={w.max():.3f}, std={w.std():.3f}")

        # CWE distribution before/after
        if "cwe_id" in weights_df.columns:
            lines.extend(["", "### Vulnerable CWE Distribution (before/after)", ""])
            lines.append("| CWE | Raw Count | Raw Share | Weighted Share |")
            lines.append("|-----|-----------|-----------|----------------|")
            vul_df = weights_df[weights_df["vul"] == 1]
            if len(vul_df) > 0:
                cwe_raw = vul_df["cwe_id"].value_counts()
                for cwe in cwe_raw.index:
                    raw_count = cwe_raw[cwe]
                    raw_share = raw_count / len(vul_df)
                    cwe_mask = vul_df["cwe_id"] == cwe
                    weighted_share = vul_df.loc[cwe_mask, "sample_weight"].sum() / vul_df["sample_weight"].sum()
                    lines.append(f"| {cwe} | {raw_count} | {raw_share:.2%} | {weighted_share:.2%} |")

        # Top project contribution before/after
        lines.extend(["", "### Top Project Contribution (before/after)", ""])
        lines.append("| Project | Raw Rows | Raw Share | Weighted Share |")
        lines.append("|---------|----------|-----------|----------------|")
        proj_raw = weights_df["project"].value_counts().head(10)
        total_w = weights_df["sample_weight"].sum()
        for proj in proj_raw.index:
            raw_count = proj_raw[proj]
            raw_share = raw_count / raw_total
            proj_mask = weights_df["project"] == proj
            weighted_share = weights_df.loc[proj_mask, "sample_weight"].sum() / total_w
            lines.append(f"| {proj} | {raw_count} | {raw_share:.2%} | {weighted_share:.2%} |")

    # Warnings
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for w in warnings:
            lines.append(f"- {w}")
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

    # Validate held-out CWE eligibility
    has_cwe = "cwe_id" in df.columns
    if has_cwe and HELDOUT_CWE_IDS:
        _validate_heldout_eligibility(df, HELDOUT_CWE_IDS)

    # Build seen/unseen pools using full-project holdout
    migration_info: dict[str, list[str]] = {}
    if has_cwe and HELDOUT_CWE_IDS:
        df_seen, df_unseen, migration_info = _build_unseen_pool(df, HELDOUT_CWE_IDS)
        logger.info(
            "Full-project holdout: %d projects (%d rows) moved to unseen pool, "
            "%d projects (%d rows) in seen pool",
            df_unseen["project"].nunique(), len(df_unseen),
            df_seen["project"].nunique(), len(df_seen),
        )
        if migration_info:
            logger.info(
                "Projects migrated to unseen: %s",
                ", ".join(f"{p} (due to {', '.join(cwes)})" for p, cwes in sorted(migration_info.items())),
            )
    else:
        df_seen = df.copy()
        df_unseen = pd.DataFrame(columns=df.columns)

    # Assign seen splits (stratified by label, grouped by project)
    assignments = _assign_splits(df_seen, seed)

    # Assign unseen splits
    if len(df_unseen) > 0:
        unseen_assignments = _assign_unseen_splits(df_unseen, seed)
        assignments = pd.concat([assignments, unseen_assignments], ignore_index=True)

    # Validate all splits
    df_all = pd.concat([df_seen, df_unseen], ignore_index=True) if len(df_unseen) > 0 else df_seen
    warnings = _validate_splits(df_all, assignments, heldout_cwe_ids=HELDOUT_CWE_IDS if has_cwe else None)
    for w in warnings:
        logger.warning(w)

    # Write outputs
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    assignments_path = out / "assignments.csv"
    assignments.to_csv(assignments_path, index=False)
    logger.info("Wrote split assignments to %s", assignments_path)

    # Train balancing
    weights_df = None
    train_ids = set(assignments.loc[assignments["split"] == "train", "sample_id"].values)
    df_train = df_all[df_all["sample_id"].isin(train_ids)].copy()
    if len(df_train) > 0:
        weights_df = _compute_sample_weights(df_train)
        weights_path = out / "train_sampler_weights.csv"
        weights_df.to_csv(weights_path, index=False)
        logger.info("Wrote train sampler weights to %s", weights_path)

        manifest = _generate_balanced_manifest(df_train, weights_df, seed)
        manifest_path = out / "train_balanced_manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        logger.info("Wrote balanced training manifest to %s", manifest_path)

    # Generate report
    report = _generate_split_report(df_all, assignments, warnings, migration_info, weights_df)
    report_path = out / "split_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info("Wrote split report to %s", report_path)

    # Log summary
    for split_name in ALL_SPLITS:
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
