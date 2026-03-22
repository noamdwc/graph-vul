"""Split dataset into train/val/test with project-disjoint leakage-aware assignment.

Held-out projects are fully excluded from training. No project appears in
more than one split. Produces an assignments CSV and a diagnostics report.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from graph_vul.config import (
    HELD_OUT_PROJECT_COUNT,
    SEED,
    SPLIT_ASSIGNMENTS,
    SPLIT_REPORT,
    SPLITS_DIR,
    SUBSET_CSV,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Held-out project selection
# ---------------------------------------------------------------------------

def _select_held_out_projects(
    df: pd.DataFrame,
    n_held_out: int,
    seed: int,
) -> list[str]:
    """Select held-out projects: prefer smaller/mid-size projects with both classes."""
    rng = np.random.RandomState(seed)

    project_stats = df.groupby("project").agg(
        total=("vul", "count"),
        vul_count=("vul", "sum"),
    ).reset_index()
    project_stats["benign_count"] = project_stats["total"] - project_stats["vul_count"]

    # Must have both classes
    candidates = project_stats[
        (project_stats["vul_count"] > 0) & (project_stats["benign_count"] > 0)
    ].copy()

    if len(candidates) == 0:
        logger.warning("No projects have both vulnerable and benign samples; falling back to largest projects")
        candidates = project_stats.copy()

    if len(candidates) < n_held_out:
        logger.warning(
            "Only %d candidate projects available (need %d); using all",
            len(candidates), n_held_out,
        )
        return candidates["project"].tolist()

    # Score: prefer moderate-size projects (not too small, not too large)
    # Exclude top-2 largest projects from held-out consideration
    top2 = project_stats.nlargest(2, "total")["project"].tolist()
    candidates = candidates[~candidates["project"].isin(top2)]

    if len(candidates) < n_held_out:
        # Fall back to including large projects if necessary
        candidates = project_stats[
            (project_stats["vul_count"] > 0) & (project_stats["benign_count"] > 0)
        ].copy()

    # Score by having enough data for evaluation but not being dominant
    candidates = candidates.copy()
    candidates["score"] = np.sqrt(candidates["vul_count"] * candidates["benign_count"])
    # Penalize very small projects (< 10 rows)
    candidates.loc[candidates["total"] < 10, "score"] *= 0.3

    candidates = candidates.sort_values("score", ascending=False)

    # Pick top candidates with some randomness
    pool = candidates.head(max(n_held_out * 3, 6))
    if len(pool) > n_held_out:
        selected = pool.sample(n=n_held_out, random_state=rng.randint(2**31))
    else:
        selected = pool.head(n_held_out)

    result = selected["project"].tolist()
    logger.info("Selected %d held-out projects: %s", len(result), result)
    return result


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

def _assign_splits(
    df: pd.DataFrame,
    held_out_projects: list[str],
    seed: int,
) -> pd.DataFrame:
    """Assign each row to train/val/test. Project-disjoint."""
    rng = np.random.RandomState(seed)

    assignments = pd.Series("train", index=df.index, name="split")

    # Held-out projects go to val/test
    held_out_mask = df["project"].isin(held_out_projects)

    if held_out_mask.sum() == 0:
        logger.warning("No rows from held-out projects found in subset")
        return pd.DataFrame({"sample_id": df["sample_id"], "split": assignments})

    # Distribute held-out projects across val and test
    # Each project goes entirely to one split
    held_out_projects = list(held_out_projects)  # work on a copy
    rng.shuffle(held_out_projects)

    val_projects = []
    test_projects = []
    val_count = 0
    test_count = 0

    for proj in held_out_projects:
        proj_rows = held_out_mask & (df["project"] == proj)
        proj_n = proj_rows.sum()

        # Assign to whichever split has fewer rows so far
        if val_count <= test_count:
            val_projects.append(proj)
            val_count += proj_n
        else:
            test_projects.append(proj)
            test_count += proj_n

    for proj in val_projects:
        mask = df["project"] == proj
        assignments[mask] = "val"

    for proj in test_projects:
        mask = df["project"] == proj
        assignments[mask] = "test"

    return pd.DataFrame({"sample_id": df["sample_id"], "split": assignments})


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


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _generate_split_report(
    df: pd.DataFrame,
    assignments: pd.DataFrame,
    held_out_projects: list[str],
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

    for split_name in ["train", "val", "test"]:
        split_df = merged[merged["split"] == split_name]
        total = len(split_df)
        vul = int((split_df["vul"] == 1).sum())
        benign = total - vul
        ratio = f"{vul/max(total,1):.2%}"
        lines.append(f"| {split_name} | {total} | {vul} | {benign} | {ratio} |")

    lines.append(f"| **total** | {len(merged)} | {int((merged['vul']==1).sum())} | "
                 f"{int((merged['vul']==0).sum())} | {(merged['vul']==1).mean():.2%} |")

    # CVE coverage
    lines.extend(["", "## CVE Coverage", ""])
    lines.append("| Split | Unique CVEs | CVE List |")
    lines.append("|-------|-------------|----------|")
    for split_name in ["train", "val", "test"]:
        split_df = merged[merged["split"] == split_name]
        cves = sorted(split_df["cve_id"].unique())
        lines.append(f"| {split_name} | {len(cves)} | {', '.join(cves)} |")

    # Project coverage
    lines.extend(["", "## Project Coverage", ""])
    lines.append("| Split | Unique Projects | Project List |")
    lines.append("|-------|-----------------|--------------|")
    for split_name in ["train", "val", "test"]:
        split_df = merged[merged["split"] == split_name]
        projects = sorted(split_df["project"].unique())
        lines.append(f"| {split_name} | {len(projects)} | {', '.join(projects)} |")

    # Overlap verification
    lines.extend(["", "## Overlap Verification", ""])
    split_projects = {}
    for split_name in ["train", "val", "test"]:
        split_df = merged[merged["split"] == split_name]
        split_projects[split_name] = set(split_df["project"].unique())

    for s1, s2 in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = split_projects.get(s1, set()) & split_projects.get(s2, set())
        status = "NONE" if not overlap else f"**LEAKAGE**: {overlap}"
        lines.append(f"- Project overlap ({s1} ∩ {s2}): {status}")

    # Held-out summary
    lines.extend(["", "## Held-Out Projects", ""])
    for proj in held_out_projects:
        proj_df = merged[merged["project"] == proj]
        split = proj_df["split"].iloc[0] if len(proj_df) > 0 else "N/A"
        vul = int((proj_df["vul"] == 1).sum())
        benign = int((proj_df["vul"] == 0).sum())
        cves = sorted(proj_df["cve_id"].unique())
        lines.append(f"- **{proj}** → {split} ({len(proj_df)} rows, {vul} vul, {benign} benign, CVEs: {', '.join(cves)})")

    # Metadata distributions
    for col in ("code_char_len_bucket", "code_line_count_bucket", "code_token_count_bucket"):
        if col not in merged.columns:
            continue
        lines.extend(["", f"## {col} by Split", ""])
        ct = pd.crosstab(merged["split"], merged[col])
        lines.append("| Split | " + " | ".join(str(c) for c in ct.columns) + " |")
        lines.append("|-------" + "".join("| --- " for _ in ct.columns) + "|")
        for split_name in ["train", "val", "test"]:
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
    n_held_out: int = HELD_OUT_PROJECT_COUNT,
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

    # Select held-out projects
    held_out_projects = _select_held_out_projects(df, n_held_out, seed)

    # Assign splits
    assignments = _assign_splits(df, held_out_projects, seed)

    # Validate
    warnings = _validate_splits(df, assignments)
    for w in warnings:
        logger.warning(w)

    # Write outputs
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    assignments_path = out / "assignments.csv"
    assignments.to_csv(assignments_path, index=False)
    logger.info("Wrote split assignments to %s", assignments_path)

    report = _generate_split_report(df, assignments, held_out_projects, warnings)
    report_path = out / "split_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info("Wrote split report to %s", report_path)

    # Log summary
    for split_name in ["train", "val", "test"]:
        n = (assignments["split"] == split_name).sum()
        logger.info("Split '%s': %d rows", split_name, n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data with project-disjoint leakage-aware assignment")
    parser.add_argument("--input", type=str, default=str(SUBSET_CSV))
    parser.add_argument("--output", type=str, default=str(SPLITS_DIR))
    parser.add_argument("--held-out", type=int, default=HELD_OUT_PROJECT_COUNT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    split_data(input_csv=args.input, output_dir=args.output, n_held_out=args.held_out, seed=args.seed)
