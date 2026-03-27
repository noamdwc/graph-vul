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
    MIN_SPLIT_FRAC,
    MIN_SPLIT_ROWS,
    SEED,
    SPLIT_MAJOR_CWE_TOP_K,
    SPLITS_DIR,
    SUBSET_CSV,
    TEST_SPLIT,
    UNSEEN_VAL_FRAC,
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
# Project-level greedy allocation
# ---------------------------------------------------------------------------

def _select_major_cwes(df: pd.DataFrame, top_k: int = SPLIT_MAJOR_CWE_TOP_K) -> list[str]:
    """Return the dominant CWEs to track while preserving mix across splits."""
    if "cwe_id" not in df.columns or top_k <= 0:
        return []

    source = df[df["vul"] == 1]
    if len(source) == 0:
        source = df

    counts = source["cwe_id"].value_counts()
    return sorted(counts.head(top_k).index.tolist())


def _build_project_stats(
    df: pd.DataFrame,
    major_cwes: list[str] | None = None,
    bucket_col: str = "code_char_len_bucket",
) -> pd.DataFrame:
    """Aggregate row-level data to project-level statistics for greedy assignment."""
    agg = (
        df.groupby("project")
        .agg(
            total_rows=("sample_id", "size"),
            vulnerable_rows=("vul", "sum"),
        )
        .reset_index()
    )
    agg["benign_rows"] = agg["total_rows"] - agg["vulnerable_rows"]

    major_cwes = major_cwes or []
    for cwe_id in major_cwes:
        counts = (
            df[df["cwe_id"] == cwe_id]
            .groupby("project")
            .size()
            .reindex(agg["project"], fill_value=0)
            .to_numpy()
        )
        agg[f"cwe::{cwe_id}"] = counts

    if bucket_col in df.columns:
        bucket_values = sorted(df[bucket_col].dropna().astype(str).unique().tolist())
        for bucket in bucket_values:
            counts = (
                df[df[bucket_col].astype(str) == bucket]
                .groupby("project")
                .size()
                .reindex(agg["project"], fill_value=0)
                .to_numpy()
            )
            agg[f"bucket::{bucket}"] = counts

    return agg


def _greedy_project_split(
    project_stats: pd.DataFrame,
    split_fracs: dict[str, float],
    seed: int,
) -> dict[str, str]:
    """Assign projects to splits greedily, optimizing for row targets first."""
    splits = list(split_fracs.keys())
    total_rows = float(project_stats["total_rows"].sum())
    total_vul = float(project_stats["vulnerable_rows"].sum())
    global_vul_ratio = total_vul / total_rows if total_rows else 0.0
    cwe_cols = [col for col in project_stats.columns if col.startswith("cwe::")]
    bucket_cols = [col for col in project_stats.columns if col.startswith("bucket::")]

    tracked_cols = cwe_cols + bucket_cols
    split_targets = {
        split: {
            "rows": total_rows * frac,
            "vul_rows": total_vul * frac,
            **{col: float(project_stats[col].sum()) * frac for col in tracked_cols},
        }
        for split, frac in split_fracs.items()
    }
    base_state = {
        split: {"rows": 0.0, "vul_rows": 0.0, **{col: 0.0 for col in tracked_cols}}
        for split in splits
    }

    def build_state(project_to_split: dict[str, str]) -> dict[str, dict[str, float]]:
        state = {
            split: {"rows": 0.0, "vul_rows": 0.0, **{col: 0.0 for col in tracked_cols}}
            for split in splits
        }
        for _, project in project_stats.iterrows():
            split = project_to_split.get(str(project["project"]))
            if split is None:
                continue
            state[split]["rows"] += float(project["total_rows"])
            state[split]["vul_rows"] += float(project["vulnerable_rows"])
            for col in tracked_cols:
                state[split][col] += float(project[col])
        return state

    def score_state(state: dict[str, dict[str, float]]) -> float:
        total_score = 0.0
        for split in splits:
            new_rows = state[split]["rows"]
            row_target = max(split_targets[split]["rows"], 1.0)
            row_gap = (new_rows - row_target) / row_target
            row_penalty = row_gap ** 2
            overshoot = max(new_rows - row_target, 0.0) / row_target
            undershoot = max(row_target - new_rows, 0.0) / row_target
            row_penalty += 6.0 * (overshoot ** 2)
            row_penalty += 8.0 * (undershoot ** 2)

            new_vul = state[split]["vul_rows"]
            new_vul_ratio = new_vul / new_rows if new_rows else 0.0
            vul_penalty = (new_vul_ratio - global_vul_ratio) ** 2

            cwe_penalty = 0.0
            for col in cwe_cols:
                target = max(split_targets[split][col], 1.0)
                gap = (state[split][col] - target) / target
                cwe_penalty += gap ** 2 + 2.0 * max(gap, 0.0) ** 2

            bucket_penalty = 0.0
            for col in bucket_cols:
                target = max(split_targets[split][col], 1.0)
                gap = (state[split][col] - target) / target
                bucket_penalty += gap ** 2

            total_score += 100.0 * row_penalty + 8.0 * vul_penalty + 3.0 * cwe_penalty + 1.0 * bucket_penalty
        return total_score

    state = {split: values.copy() for split, values in base_state.items()}
    project_to_split: dict[str, str] = {}

    rng = np.random.RandomState(seed)
    ordered_projects = project_stats.copy()
    ordered_projects["_tie"] = rng.random(len(ordered_projects))
    ordered_projects = ordered_projects.sort_values(
        ["total_rows", "_tie", "project"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    split_tie_noise = {
        split: rng.random(len(ordered_projects)) * 1e-9
        for split in splits
    }

    total_projects = len(ordered_projects)
    for idx, project in ordered_projects.iterrows():
        remaining_projects = total_projects - idx - 1
        empty_splits = [split for split in splits if state[split]["rows"] == 0]

        best_split = None
        best_score = None
        project_rows = float(project["total_rows"])
        project_vul = float(project["vulnerable_rows"])

        for split in splits:
            empties_after = len(empty_splits) - (1 if state[split]["rows"] == 0 else 0)
            if remaining_projects < empties_after:
                continue

            new_rows = state[split]["rows"] + project_rows
            row_target = max(split_targets[split]["rows"], 1.0)
            row_gap = (new_rows - row_target) / row_target
            row_penalty = row_gap ** 2
            overshoot = max(new_rows - row_target, 0.0) / row_target
            undershoot = max(row_target - new_rows, 0.0) / row_target
            row_penalty += 6.0 * (overshoot ** 2)
            row_penalty += 8.0 * (undershoot ** 2)

            new_vul = state[split]["vul_rows"] + project_vul
            new_vul_ratio = new_vul / new_rows if new_rows else 0.0
            vul_penalty = (new_vul_ratio - global_vul_ratio) ** 2

            cwe_penalty = 0.0
            for col in cwe_cols:
                target = max(split_targets[split][col], 1.0)
                gap = (state[split][col] + float(project[col]) - target) / target
                cwe_penalty += gap ** 2 + 2.0 * max(gap, 0.0) ** 2

            bucket_penalty = 0.0
            for col in bucket_cols:
                target = max(split_targets[split][col], 1.0)
                gap = (state[split][col] + float(project[col]) - target) / target
                bucket_penalty += gap ** 2

            score = (
                100.0 * row_penalty
                + 8.0 * vul_penalty
                + 3.0 * cwe_penalty
                + 1.0 * bucket_penalty
                + split_tie_noise[split][idx]
            )
            if best_score is None or score < best_score:
                best_score = score
                best_split = split

        if best_split is None:
            raise ValueError("Unable to assign projects while keeping all splits non-empty")

        project_to_split[str(project["project"])] = best_split
        state[best_split]["rows"] += project_rows
        state[best_split]["vul_rows"] += project_vul
        for col in tracked_cols:
            state[best_split][col] += float(project[col])

    # Deterministic local refinement: move whole projects when it improves the global objective.
    project_sizes = project_stats.set_index("project")["total_rows"].to_dict()
    current_score = score_state(build_state(project_to_split))
    for _ in range(len(project_to_split) * max(len(splits), 1)):
        best_move = None
        best_score = current_score
        split_counts = pd.Series(project_to_split).value_counts().to_dict()

        for project, source_split in sorted(project_to_split.items(), key=lambda item: (-project_sizes[item[0]], item[0])):
            if split_counts.get(source_split, 0) <= 1:
                continue
            for target_split in splits:
                if target_split == source_split:
                    continue
                candidate = dict(project_to_split)
                candidate[project] = target_split
                candidate_score = score_state(build_state(candidate))
                if candidate_score + 1e-9 < best_score:
                    best_score = candidate_score
                    best_move = (project, target_split)

        if best_move is None:
            break
        project, target_split = best_move
        project_to_split[project] = target_split
        current_score = best_score

    return project_to_split


def _expand_project_assignments(
    df: pd.DataFrame,
    project_to_split: dict[str, str],
) -> pd.DataFrame:
    """Map project-level assignments back to row-level split assignments."""
    assignments = df[["sample_id", "project"]].copy()
    assignments["split"] = assignments["project"].map(project_to_split)
    if assignments["split"].isna().any():
        missing_projects = sorted(assignments.loc[assignments["split"].isna(), "project"].unique())
        raise ValueError(f"Missing project assignments for: {missing_projects}")
    return assignments[["sample_id", "split"]]


def _assign_splits(
    df: pd.DataFrame,
    seed: int,
    val_frac: float = VAL_SPLIT,
    test_frac: float = TEST_SPLIT,
) -> pd.DataFrame:
    """Assign seen-pool rows to train/val/test with project-level greedy allocation."""
    train_frac = 1.0 - val_frac - test_frac
    if train_frac <= 0:
        raise ValueError("Seen split fractions must leave positive train fraction")

    major_cwes = _select_major_cwes(df)
    project_stats = _build_project_stats(df, major_cwes=major_cwes)
    project_to_split = _greedy_project_split(
        project_stats,
        split_fracs={"train": train_frac, "val": val_frac, "test": test_frac},
        seed=seed,
    )
    return _expand_project_assignments(df, project_to_split)


# ---------------------------------------------------------------------------
# Unseen pool split assignment
# ---------------------------------------------------------------------------

def _assign_unseen_splits(
    df_unseen: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """Split unseen-pool rows into unseen_val and unseen_test using the same allocator."""
    if len(df_unseen) == 0:
        return pd.DataFrame(columns=["sample_id", "split"])

    major_cwes = _select_major_cwes(df_unseen)
    project_stats = _build_project_stats(df_unseen, major_cwes=major_cwes)
    project_to_split = _greedy_project_split(
        project_stats,
        split_fracs={"unseen_val": UNSEEN_VAL_FRAC, "unseen_test": 1.0 - UNSEEN_VAL_FRAC},
        seed=seed,
    )
    return _expand_project_assignments(df_unseen, project_to_split)


def _enforce_min_split_sizes(
    df: pd.DataFrame,
    assignments: pd.DataFrame,
    min_rows: int = MIN_SPLIT_ROWS,
    min_frac: float = MIN_SPLIT_FRAC,
    required_splits: tuple[str, ...] = ("val", "test"),
) -> None:
    """Raise if required evaluation splits are too small to be meaningful."""
    merged = df.merge(assignments, on="sample_id")
    total = len(merged)

    errors: list[str] = []
    for split_name in required_splits:
        split_size = int((merged["split"] == split_name).sum())
        if min_rows > 0 and split_size < min_rows:
            errors.append(
                f"{split_name}: only {split_size} rows (minimum {min_rows})"
            )
        if min_frac > 0 and total > 0 and split_size / total < min_frac:
            errors.append(
                f"{split_name}: only {split_size/total:.1%} of rows (minimum {min_frac:.1%})"
            )

    if errors:
        raise ValueError("Split size validation failed:\n  - " + "\n  - ".join(errors))


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
    _enforce_min_split_sizes(
        df_seen,
        assignments[assignments["split"].isin(["train", "val", "test"])].copy(),
        min_rows=MIN_SPLIT_ROWS,
        min_frac=MIN_SPLIT_FRAC,
    )
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
