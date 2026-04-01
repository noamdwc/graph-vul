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
- unseen_audit.csv             (per-project unseen split statistics)
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
    MIN_SHORT_BENIGN_PER_EVAL_SPLIT,
    MIN_SHORT_VUL_PER_EVAL_SPLIT,
    MIN_SPLIT_FRAC,
    MIN_SPLIT_ROWS,
    SEED,
    SHORT_BUCKET_DEFICIT_WEIGHT,
    SPLIT_MAJOR_CWE_TOP_K,
    SPLITS_DIR,
    SUBSET_CSV,
    TEST_SPLIT,
    TOKEN_BUCKET_EDGES,
    TOKEN_BUCKET_LABELS,
    UNSEEN_VAL_FRAC,
    VAL_SPLIT,
)

logger = logging.getLogger(__name__)

ALL_SPLITS = ["train", "val", "test", "unseen_val", "unseen_test"]
PRIMARY_SPLITS = ["train", "val", "test"]
DERIVED_UNSEEN_ASSIGNMENTS_FILENAME = "unseen_assignments.csv"


def _active_splits(assignments: pd.DataFrame, expected_splits: list[str] | None = None) -> list[str]:
    if expected_splits is not None:
        return expected_splits
    present = set(assignments["split"].unique())
    return [split for split in ALL_SPLITS if split in present]


def _load_derived_unseen_assignments(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["sample_id", "split"])
    return pd.read_csv(path, usecols=["sample_id", "split"])


def _materialize_official_split_assignments(df: pd.DataFrame) -> pd.DataFrame:
    """Build assignments directly from normalized source_split values."""
    if "source_split" not in df.columns:
        raise ValueError("Official split mode requires a source_split column")

    split_map = {"train": "train", "validation": "val", "val": "val", "test": "test"}
    normalized = df[["sample_id", "source_split"]].copy()
    missing = normalized["source_split"].isna().sum()
    if missing:
        logger.warning(
            "Dropping %d rows with missing source_split labels before materializing official assignments",
            int(missing),
        )
        normalized = normalized[normalized["source_split"].notna()].copy()
    normalized["split"] = normalized["source_split"].astype(str).str.lower().map(split_map)
    if normalized["split"].isna().any():
        bad = sorted(normalized.loc[normalized["split"].isna(), "source_split"].astype(str).unique())
        raise ValueError(f"Unsupported official split labels: {bad}")
    return normalized[["sample_id", "split"]]


def _dejoin_official_splits(
    df: pd.DataFrame,
    seed: int,
    val_frac: float = VAL_SPLIT,
    test_frac: float = TEST_SPLIT,
) -> pd.DataFrame:
    """Enforce project disjointness on official source splits.

    1. Anchor single-split projects (projects appearing in exactly one source_split).
    2. Pool multi-split projects for reallocation via the greedy allocator.
    3. Feed deficit-adjusted targets to _greedy_project_split.
    4. Merge anchored + reallocated assignments.
    """
    if "source_split" not in df.columns:
        raise ValueError("Dejoin mode requires a source_split column")

    split_map = {"train": "train", "validation": "val", "val": "val", "test": "test"}
    df = df.copy()
    df["_norm_split"] = df["source_split"].astype(str).str.lower().map(split_map)
    if df["_norm_split"].isna().any():
        bad = sorted(df.loc[df["_norm_split"].isna(), "source_split"].astype(str).unique())
        raise ValueError(f"Unsupported official split labels: {bad}")

    # Classify projects: single-split vs multi-split
    project_splits = df.groupby("project")["_norm_split"].apply(lambda x: set(x.unique()))
    single_split_projects = {p for p, splits in project_splits.items() if len(splits) == 1}
    multi_split_projects = {p for p, splits in project_splits.items() if len(splits) > 1}

    logger.info(
        "Dejoin: %d single-split projects (anchored), %d multi-split projects (to reallocate)",
        len(single_split_projects), len(multi_split_projects),
    )

    # Anchor single-split projects
    anchored_mask = df["project"].isin(single_split_projects)
    anchored_assignments = df.loc[anchored_mask, ["sample_id", "_norm_split"]].rename(
        columns={"_norm_split": "split"}
    )

    # Preserve unseen CWE coverage: anchor smallest carrier per at-risk CWE
    if "cwe_id" in df.columns:
        official_train_cwes = set(
            df.loc[df["_norm_split"] == "train", "cwe_id"].dropna().unique()
        )
        # CWEs that are unseen under official splits (in val/test but not train)
        for eval_split in ["val", "test"]:
            eval_cwes = set(
                df.loc[df["_norm_split"] == eval_split, "cwe_id"].dropna().unique()
            )
            unseen_cwes = eval_cwes - official_train_cwes
            if not unseen_cwes:
                continue

            for cwe in sorted(unseen_cwes):
                # Check if any anchored project already covers this CWE in an eval split
                anchored_eval_ids = set(
                    anchored_assignments.loc[
                        anchored_assignments["split"].isin(["val", "test"]), "sample_id"
                    ]
                )
                anchored_eval_cwes = set(
                    df.loc[df["sample_id"].isin(anchored_eval_ids), "cwe_id"].dropna().unique()
                )
                if cwe in anchored_eval_cwes:
                    continue  # Already covered by a single-split eval project

                # Find multi-split projects carrying this CWE
                cwe_projects = set(
                    df.loc[
                        (df["cwe_id"] == cwe) & df["project"].isin(multi_split_projects),
                        "project",
                    ].unique()
                )
                if not cwe_projects:
                    continue

                # Pick the smallest carrier project
                proj_sizes = (
                    df[df["project"].isin(cwe_projects)]
                    .groupby("project")
                    .size()
                    .sort_values()
                )
                anchor_proj = proj_sizes.index[0]

                # Pre-anchor this project to the eval split
                proj_mask = df["project"] == anchor_proj
                proj_assignments = df.loc[proj_mask, ["sample_id"]].copy()
                proj_assignments["split"] = eval_split
                anchored_assignments = pd.concat(
                    [anchored_assignments, proj_assignments], ignore_index=True
                )
                multi_split_projects.discard(anchor_proj)
                logger.info(
                    "Unseen CWE preservation: anchored project '%s' (%d rows) to '%s' "
                    "to preserve CWE %s",
                    anchor_proj, int(proj_sizes[anchor_proj]), eval_split, cwe,
                )

    # Compute deficit targets for the greedy allocator
    total_rows = len(df)
    train_frac = 1.0 - val_frac - test_frac
    global_targets = {
        "train": total_rows * train_frac,
        "val": total_rows * val_frac,
        "test": total_rows * test_frac,
    }
    anchored_counts = anchored_assignments["split"].value_counts().to_dict()
    deficit = {
        split: max(global_targets[split] - anchored_counts.get(split, 0), 0.0)
        for split in ["train", "val", "test"]
    }
    deficit_total = sum(deficit.values())
    if deficit_total == 0:
        deficit = {"train": train_frac, "val": val_frac, "test": test_frac}
        deficit_total = 1.0
    deficit_fracs = {split: d / deficit_total for split, d in deficit.items()}

    # Reallocate multi-split projects using the greedy allocator
    df_multi = df[df["project"].isin(multi_split_projects)].copy()
    major_cwes = _select_major_cwes(df_multi)
    project_stats = _build_project_stats(df_multi, major_cwes=major_cwes, bucket_col="code_token_count_bucket")
    project_to_split = _greedy_project_split(
        project_stats,
        split_fracs=deficit_fracs,
        seed=seed,
        allow_empty_splits=True,
    )
    reallocated_assignments = _expand_project_assignments(df_multi, project_to_split)

    # Merge anchored + reallocated
    combined = pd.concat([anchored_assignments, reallocated_assignments], ignore_index=True)
    df.drop(columns=["_norm_split"], inplace=True)
    return combined[["sample_id", "split"]]


def _derive_unseen_official_assignments(
    df: pd.DataFrame,
    primary_assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Build derived unseen subsets from official val/test rows with CWEs absent from train."""
    merged = df.merge(primary_assignments, on="sample_id", how="inner")
    merged["cwe_id"] = merged["cwe_id"].fillna("").astype(str).str.strip()
    train_cwes = {
        cwe for cwe in merged.loc[merged["split"] == "train", "cwe_id"].unique()
        if cwe and cwe.lower() != "nan"
    }

    rows = []
    for source_split, unseen_split in (("val", "unseen_val"), ("test", "unseen_test")):
        split_df = merged[merged["split"] == source_split].copy()
        split_df = split_df[split_df["cwe_id"].astype(bool)]
        split_df = split_df[~split_df["cwe_id"].isin(train_cwes)]
        if len(split_df) == 0:
            continue
        rows.append(split_df[["sample_id"]].assign(split=unseen_split))

    if not rows:
        return pd.DataFrame(columns=["sample_id", "split"])
    return pd.concat(rows, ignore_index=True).drop_duplicates(subset=["sample_id", "split"])

# Unseen split balance constants
UNSEEN_PROJECT_COUNT_PENALTY = 20.0
MIN_UNSEEN_PROJECTS = 8
MIN_UNSEEN_VUL_ROWS = 10


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

    # Short-bucket (0-64 token) vulnerable and benign counts per project
    if "code_token_count_bucket" in df.columns and "vul" in df.columns:
        short_mask = df["code_token_count_bucket"].astype(str) == "0-64"
        short_vul = (
            df[short_mask & (df["vul"] == 1)]
            .groupby("project")
            .size()
            .reindex(agg["project"], fill_value=0)
            .to_numpy()
        )
        short_benign = (
            df[short_mask & (df["vul"] == 0)]
            .groupby("project")
            .size()
            .reindex(agg["project"], fill_value=0)
            .to_numpy()
        )
        agg["short_vul"] = short_vul
        agg["short_benign"] = short_benign

    return agg


def _greedy_project_split(
    project_stats: pd.DataFrame,
    split_fracs: dict[str, float],
    seed: int,
    project_count_targets: dict[str, int] | None = None,
    allow_empty_splits: bool = False,
) -> dict[str, str]:
    """Assign projects to splits greedily, optimizing for row targets first."""
    splits = list(split_fracs.keys())
    eval_splits = [s for s in splits if s != "train"]
    has_short = "short_vul" in project_stats.columns
    total_rows = float(project_stats["total_rows"].sum())
    total_vul = float(project_stats["vulnerable_rows"].sum())
    global_vul_ratio = total_vul / total_rows if total_rows else 0.0
    cwe_cols = [col for col in project_stats.columns if col.startswith("cwe::")]
    bucket_cols = [col for col in project_stats.columns if col.startswith("bucket::")]

    tracked_cols = cwe_cols + bucket_cols
    short_keys = ["short_vul", "short_benign"] if has_short else []
    split_targets = {
        split: {
            "rows": total_rows * frac,
            "vul_rows": total_vul * frac,
            **{col: float(project_stats[col].sum()) * frac for col in tracked_cols},
        }
        for split, frac in split_fracs.items()
    }
    base_state = {
        split: {
            "rows": 0.0, "vul_rows": 0.0,
            **{col: 0.0 for col in tracked_cols},
            **{k: 0.0 for k in short_keys},
        }
        for split in splits
    }

    def _short_deficit_penalty(state: dict[str, dict[str, float]]) -> float:
        if not has_short:
            return 0.0
        penalty = 0.0
        for s in eval_splits:
            vul_deficit = max(0.0, MIN_SHORT_VUL_PER_EVAL_SPLIT - state[s]["short_vul"])
            benign_deficit = max(0.0, MIN_SHORT_BENIGN_PER_EVAL_SPLIT - state[s]["short_benign"])
            penalty += (vul_deficit / max(MIN_SHORT_VUL_PER_EVAL_SPLIT, 1)) ** 2
            penalty += (benign_deficit / max(MIN_SHORT_BENIGN_PER_EVAL_SPLIT, 1)) ** 2
        return SHORT_BUCKET_DEFICIT_WEIGHT * penalty

    def build_state(project_to_split: dict[str, str]) -> dict[str, dict[str, float]]:
        state = {
            split: {
                "rows": 0.0, "vul_rows": 0.0,
                **{col: 0.0 for col in tracked_cols},
                **{k: 0.0 for k in short_keys},
            }
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
            for k in short_keys:
                state[split][k] += float(project[k])
        return state

    def _project_count_score(project_to_split: dict[str, str]) -> float:
        if not project_count_targets:
            return 0.0
        counts = {split: 0 for split in splits}
        for split in project_to_split.values():
            counts[split] += 1
        penalty = 0.0
        for split in splits:
            target = max(project_count_targets.get(split, 1), 1)
            penalty += ((counts[split] / target) - 1.0) ** 2
        return UNSEEN_PROJECT_COUNT_PENALTY * penalty

    def score_state(state: dict[str, dict[str, float]], project_to_split: dict[str, str] | None = None) -> float:
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

        total_score += _short_deficit_penalty(state)

        if project_to_split is not None:
            total_score += _project_count_score(project_to_split)

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
            if not allow_empty_splits and remaining_projects < empties_after:
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

            # Short-bucket deficit penalty (eval splits only)
            short_penalty = 0.0
            if has_short and split in eval_splits:
                cand_sv = state[split]["short_vul"] + float(project["short_vul"])
                cand_sb = state[split]["short_benign"] + float(project["short_benign"])
                vul_def = max(0.0, MIN_SHORT_VUL_PER_EVAL_SPLIT - cand_sv)
                ben_def = max(0.0, MIN_SHORT_BENIGN_PER_EVAL_SPLIT - cand_sb)
                short_penalty += (vul_def / max(MIN_SHORT_VUL_PER_EVAL_SPLIT, 1)) ** 2
                short_penalty += (ben_def / max(MIN_SHORT_BENIGN_PER_EVAL_SPLIT, 1)) ** 2
                short_penalty *= SHORT_BUCKET_DEFICIT_WEIGHT

            score = (
                100.0 * row_penalty
                + 8.0 * vul_penalty
                + 3.0 * cwe_penalty
                + 1.0 * bucket_penalty
                + short_penalty
                + split_tie_noise[split][idx]
            )

            if project_count_targets:
                candidate_assignments = dict(project_to_split)
                candidate_assignments[str(project["project"])] = split
                score += _project_count_score(candidate_assignments)

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
        for k in short_keys:
            state[best_split][k] += float(project[k])

    # Deterministic local refinement: move whole projects when it improves the global objective.
    project_sizes = project_stats.set_index("project")["total_rows"].to_dict()
    current_score = score_state(build_state(project_to_split), project_to_split)
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
                candidate_score = score_state(build_state(candidate), candidate)
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

    n_projects = len(project_stats)
    target_unseen_val = max(1, round(n_projects * UNSEEN_VAL_FRAC))
    target_unseen_test = n_projects - target_unseen_val
    project_count_targets = {
        "unseen_val": target_unseen_val,
        "unseen_test": target_unseen_test,
    }

    project_to_split = _greedy_project_split(
        project_stats,
        split_fracs={"unseen_val": UNSEEN_VAL_FRAC, "unseen_test": 1.0 - UNSEEN_VAL_FRAC},
        seed=seed,
        project_count_targets=project_count_targets,
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
    expected_splits: list[str] | None = None,
) -> list[str]:
    """Validate split properties across all five splits. Returns list of warnings."""
    warnings: list[str] = []

    merged = df.merge(assignments, on="sample_id")

    # Check all rows assigned
    if len(merged) != len(df):
        warnings.append(f"Row count mismatch: {len(df)} input rows, {len(merged)} assigned")

    # Collect projects per split
    active_splits = expected_splits if expected_splits is not None else ALL_SPLITS
    split_projects: dict[str, set[str]] = {}
    for split_name in active_splits:
        split_df = merged[merged["split"] == split_name]
        split_projects[split_name] = set(split_df["project"].unique())

    # Check all 10 split pairs for project overlap
    for s1, s2 in combinations(active_splits, 2):
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
    for split_name in active_splits:
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

    # Unseen split balance checks
    total_unseen_projects = len(unseen_projects)
    min_proj_threshold = MIN_UNSEEN_PROJECTS
    if total_unseen_projects < 2 * MIN_UNSEEN_PROJECTS:
        min_proj_threshold = max(1, total_unseen_projects // 2)
        if total_unseen_projects > 0:
            warnings.append(
                f"Unseen pool has only {total_unseen_projects} projects — "
                f"relaxing minimum project threshold to {min_proj_threshold}"
            )

    for split_name in ["unseen_val", "unseen_test"]:
        split_df = merged[merged["split"] == split_name]
        if len(split_df) == 0:
            continue

        n_proj = split_df["project"].nunique()
        if n_proj < min_proj_threshold:
            warnings.append(
                f"UNSEEN PROJECT COUNT: '{split_name}' has only {n_proj} projects "
                f"(minimum {min_proj_threshold})"
            )

        n_vul = int((split_df["vul"] == 1).sum())
        if n_vul < MIN_UNSEEN_VUL_ROWS:
            warnings.append(
                f"UNSEEN VUL ROWS: '{split_name}' has only {n_vul} vulnerable rows "
                f"(minimum {MIN_UNSEEN_VUL_ROWS})"
            )

        proj_sizes = split_df.groupby("project").size()
        largest_proj = proj_sizes.idxmax()
        largest_share = proj_sizes.max() / len(split_df)
        if largest_share > 0.5:
            warnings.append(
                f"UNSEEN DOMINANCE: project '{largest_proj}' has {largest_share:.1%} of "
                f"'{split_name}' rows"
            )

    return warnings


def _validate_short_bucket_coverage(
    df: pd.DataFrame,
    assignments: pd.DataFrame,
    min_short_vul: int = MIN_SHORT_VUL_PER_EVAL_SPLIT,
    min_short_benign: int = MIN_SHORT_BENIGN_PER_EVAL_SPLIT,
) -> list[str]:
    """Check eval splits for short-bucket (0-64 token) coverage. Returns warnings."""
    warnings: list[str] = []
    if "code_token_count_bucket" not in df.columns or "vul" not in df.columns:
        return warnings

    merged = df.merge(assignments, on="sample_id")
    eval_splits = [s for s in ALL_SPLITS if s != "train" and s in merged["split"].values]

    for split_name in eval_splits:
        split_df = merged[merged["split"] == split_name]
        short_df = split_df[split_df["code_token_count_bucket"].astype(str) == "0-64"]
        n_vul = int((short_df["vul"] == 1).sum())
        n_benign = int((short_df["vul"] == 0).sum())

        if n_vul < min_short_vul:
            warnings.append(
                f"SHORT-BUCKET VUL DEFICIT: '{split_name}' has {n_vul} short vulnerable "
                f"rows (minimum {min_short_vul})"
            )
        if n_benign < min_short_benign:
            warnings.append(
                f"SHORT-BUCKET BENIGN DEFICIT: '{split_name}' has {n_benign} short benign "
                f"rows (minimum {min_short_benign})"
            )
        if n_vul < 10:
            warnings.append(
                f"SHORT-BUCKET UNRELIABLE: '{split_name}' has only {n_vul} short vulnerable "
                f"rows — too few for reliable short-function metrics"
            )

    return warnings


def _validate_derived_unseen_assignments(
    df: pd.DataFrame,
    derived_assignments: pd.DataFrame,
) -> list[str]:
    warnings: list[str] = []
    if derived_assignments.empty:
        warnings.append("Derived unseen subsets are empty")
        return warnings

    merged = df.merge(derived_assignments, on="sample_id", how="inner")
    for split_name in ["unseen_val", "unseen_test"]:
        split_df = merged[merged["split"] == split_name]
        if len(split_df) == 0:
            warnings.append(f"Derived split '{split_name}' is empty")
            continue
        if split_df["vul"].nunique() < 2:
            warnings.append(f"Derived split '{split_name}' has only one class")
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
    mode: str = "derived",
    derived_unseen_assignments: pd.DataFrame | None = None,
) -> str:
    """Generate a Markdown diagnostics report."""
    merged = df.merge(assignments, on="sample_id")
    active_splits = _active_splits(assignments)

    lines = [
        "# Split Diagnostics Report",
        "",
        f"- Split mode: {mode}",
        "",
        "## Split Sizes",
        "",
        "| Split | Total | Vulnerable | Benign | Vul Ratio | Projects | CWEs |",
        "|-------|-------|------------|--------|-----------|----------|------|",
    ]

    for split_name in active_splits:
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
        for split_name in active_splits:
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
        for split_name in active_splits:
            split_df = merged[merged["split"] == split_name]
            if len(split_df) == 0:
                continue
            cves = sorted(split_df["cve_id"].dropna().astype(str).unique().tolist())
            lines.append(f"| {split_name} | {len(cves)} | {', '.join(str(c) for c in cves)} |")

    # Project coverage
    lines.extend(["", "## Project Coverage", ""])
    lines.append("| Split | Unique Projects | Project List |")
    lines.append("|-------|-----------------|--------------|")
    for split_name in active_splits:
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

    for s1, s2 in combinations(active_splits, 2):
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

    # Unseen Split Balance
    unseen_split_names = [s for s in ["unseen_val", "unseen_test"] if s in merged["split"].values]
    if unseen_split_names:
        lines.extend(["", "## Unseen Split Balance", ""])
        for split_name in unseen_split_names:
            split_df = merged[merged["split"] == split_name]
            n_proj = split_df["project"].nunique()
            n_rows = len(split_df)
            n_vul = int((split_df["vul"] == 1).sum())
            n_benign = n_rows - n_vul
            proj_sizes = split_df.groupby("project").size()
            largest_share = proj_sizes.max() / n_rows if n_rows else 0.0
            lines.append(f"### {split_name}")
            lines.append(f"- Projects: {n_proj}")
            lines.append(f"- Total rows: {n_rows} ({n_vul} vul, {n_benign} benign)")
            lines.append(f"- Largest-project share: {largest_share:.1%}")
            lines.append("")
            lines.append("| Project | Rows | Vulnerable | Benign |")
            lines.append("|---------|------|------------|--------|")
            for proj in sorted(split_df["project"].unique()):
                proj_df = split_df[split_df["project"] == proj]
                vul = int((proj_df["vul"] == 1).sum())
                benign = len(proj_df) - vul
                lines.append(f"| {proj} | {len(proj_df)} | {vul} | {benign} |")
            lines.append("")

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
                    cves = sorted(proj_df["cve_id"].dropna().astype(str).unique().tolist())
                    lines.append(f"- **{proj}** ({len(proj_df)} rows, {vul} vul, {benign} benign, CVEs: {', '.join(str(c) for c in cves)})")
                else:
                    lines.append(f"- **{proj}** ({len(proj_df)} rows, {vul} vul, {benign} benign)")

    # Metadata distributions (qcut-based char/line buckets)
    for col in ("code_char_len_bucket", "code_line_count_bucket"):
        if col not in merged.columns:
            continue
        lines.extend(["", f"## {col} by Split", ""])
        ct = pd.crosstab(merged["split"], merged[col])
        lines.append("| Split | " + " | ".join(str(c) for c in ct.columns) + " |")
        lines.append("|-------" + "".join("| --- " for _ in ct.columns) + "|")
        for split_name in active_splits:
            if split_name in ct.index:
                row_vals = " | ".join(str(ct.loc[split_name, c]) for c in ct.columns)
                lines.append(f"| {split_name} | {row_vals} |")

    # Token Length Buckets (fixed boundaries)
    if "code_token_count_bucket" in merged.columns and "vul" in merged.columns:
        lines.extend(["", "## Token Length Buckets", ""])
        bucket_order = TOKEN_BUCKET_LABELS
        lines.append("| Split | Bucket | Total | Vulnerable | Benign |")
        lines.append("|-------|--------|-------|------------|--------|")
        for split_name in active_splits:
            split_df = merged[merged["split"] == split_name]
            if len(split_df) == 0:
                continue
            for bucket in bucket_order:
                bdf = split_df[split_df["code_token_count_bucket"].astype(str) == bucket]
                n_vul = int((bdf["vul"] == 1).sum())
                n_benign = int((bdf["vul"] == 0).sum())
                lines.append(f"| {split_name} | {bucket} | {len(bdf)} | {n_vul} | {n_benign} |")

        # Short-Function Evaluation Viability
        lines.extend(["", "## Short-Function Evaluation Viability", ""])
        lines.append("| Split | Short Vul | Short Benign | Vul Status | Benign Status |")
        lines.append("|-------|-----------|--------------|------------|---------------|")
        eval_split_names = [s for s in ALL_SPLITS if s != "train" and s in merged["split"].values]
        for split_name in eval_split_names:
            split_df = merged[merged["split"] == split_name]
            short_df = split_df[split_df["code_token_count_bucket"].astype(str) == "0-64"]
            n_vul = int((short_df["vul"] == 1).sum())
            n_benign = int((short_df["vul"] == 0).sum())
            vul_status = "PASS" if n_vul >= MIN_SHORT_VUL_PER_EVAL_SPLIT else "WARN"
            benign_status = "PASS" if n_benign >= MIN_SHORT_BENIGN_PER_EVAL_SPLIT else "WARN"
            if n_vul < 10:
                vul_status = "UNRELIABLE"
            lines.append(f"| {split_name} | {n_vul} | {n_benign} | {vul_status} | {benign_status} |")

        # Per-project short-bucket contributions in eval splits
        lines.extend(["", "## Short-Bucket Project Contributions (Eval Splits)", ""])
        lines.append("| Split | Project | Short Vul | Short Benign |")
        lines.append("|-------|---------|-----------|--------------|")
        for split_name in eval_split_names:
            split_df = merged[merged["split"] == split_name]
            short_df = split_df[split_df["code_token_count_bucket"].astype(str) == "0-64"]
            if len(short_df) == 0:
                continue
            for proj in sorted(short_df["project"].unique()):
                proj_short = short_df[short_df["project"] == proj]
                sv = int((proj_short["vul"] == 1).sum())
                sb = int((proj_short["vul"] == 0).sum())
                lines.append(f"| {split_name} | {proj} | {sv} | {sb} |")

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

    if derived_unseen_assignments is not None and len(derived_unseen_assignments) > 0:
        derived = df.merge(derived_unseen_assignments, on="sample_id", how="inner")
        lines.extend(["", "## Derived Unseen Splits", ""])
        lines.append("| Split | Total | Vulnerable | Benign | Unique CWEs |")
        lines.append("|-------|-------|------------|--------|-------------|")
        for split_name in ["unseen_val", "unseen_test"]:
            split_df = derived[derived["split"] == split_name]
            if len(split_df) == 0:
                continue
            vul = int((split_df["vul"] == 1).sum())
            benign = len(split_df) - vul
            cwes = int(split_df["cwe_id"].dropna().astype(str).nunique()) if "cwe_id" in split_df.columns else 0
            lines.append(f"| {split_name} | {len(split_df)} | {vul} | {benign} | {cwes} |")

        if "cwe_id" in derived.columns:
            lines.extend(["", "### Derived Unseen CWE Breakdown", ""])
            lines.append("| Split | CWE | Total | Vulnerable | Benign |")
            lines.append("|-------|-----|-------|------------|--------|")
            for split_name in ["unseen_val", "unseen_test"]:
                split_df = derived[derived["split"] == split_name]
                for cwe_id in sorted(split_df["cwe_id"].dropna().astype(str).unique().tolist()):
                    cwe_rows = split_df[split_df["cwe_id"].astype(str) == cwe_id]
                    vul = int((cwe_rows["vul"] == 1).sum())
                    benign = len(cwe_rows) - vul
                    lines.append(f"| {split_name} | {cwe_id} | {len(cwe_rows)} | {vul} | {benign} |")

        if "code_token_count_bucket" in derived.columns:
            lines.extend(["", "### Derived Unseen Short-Function Viability", ""])
            lines.append("| Split | Short Vul | Short Benign | Vul Status | Benign Status |")
            lines.append("|-------|-----------|--------------|------------|---------------|")
            for split_name in ["unseen_val", "unseen_test"]:
                split_df = derived[derived["split"] == split_name]
                short_df = split_df[split_df["code_token_count_bucket"].astype(str) == "0-64"]
                n_vul = int((short_df["vul"] == 1).sum())
                n_benign = int((short_df["vul"] == 0).sum())
                vul_status = "PASS" if n_vul >= MIN_SHORT_VUL_PER_EVAL_SPLIT else "WARN"
                benign_status = "PASS" if n_benign >= MIN_SHORT_BENIGN_PER_EVAL_SPLIT else "WARN"
                if n_vul < 10:
                    vul_status = "UNRELIABLE"
                lines.append(f"| {split_name} | {n_vul} | {n_benign} | {vul_status} | {benign_status} |")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def split_data(
    input_csv: str = str(SUBSET_CSV),
    output_dir: str = str(SPLITS_DIR),
    seed: int = SEED,
    split_mode: str = "project_disjoint",
) -> None:
    df = pd.read_csv(input_csv)
    official_split_mode = "source_split" in df.columns and df["source_split"].notna().any()
    derived_unseen_assignments = pd.DataFrame(columns=["sample_id", "split"])

    # Derive metadata for reporting
    code = df["func_before"].fillna("")
    df["code_char_len"] = code.str.len()
    df["code_line_count"] = code.str.count("\n") + 1
    df["code_token_count"] = code.str.split().str.len().fillna(0).astype(int)
    for col in ("code_char_len", "code_line_count"):
        try:
            df[col + "_bucket"] = pd.qcut(
                df[col], q=4, labels=["short", "medium", "long", "very_long"],
                duplicates="drop",
            )
        except ValueError:
            df[col + "_bucket"] = "medium"

    # Fixed token-count buckets (not data-dependent)
    edges = TOKEN_BUCKET_EDGES + [np.inf]
    df["code_token_count_bucket"] = pd.cut(
        df["code_token_count"],
        bins=edges,
        labels=TOKEN_BUCKET_LABELS,
        right=False,
    ).astype(str)

    # Validate held-out CWE eligibility
    has_cwe = "cwe_id" in df.columns
    if not official_split_mode and has_cwe and HELDOUT_CWE_IDS:
        _validate_heldout_eligibility(df, HELDOUT_CWE_IDS)

    # Build seen/unseen pools using full-project holdout
    migration_info: dict[str, list[str]] = {}
    if official_split_mode:
        if split_mode == "project_disjoint":
            df_seen = df.copy()
            df_unseen = pd.DataFrame(columns=df.columns)
            assignments = _dejoin_official_splits(df_seen, seed)
            derived_unseen_assignments = _derive_unseen_official_assignments(df_seen, assignments)
        else:
            df_seen = df.copy()
            df_unseen = pd.DataFrame(columns=df.columns)
            assignments = _materialize_official_split_assignments(df_seen)
            derived_unseen_assignments = _derive_unseen_official_assignments(df_seen, assignments)
    elif has_cwe and HELDOUT_CWE_IDS:
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
        assignments = _assign_splits(df_seen, seed)
    else:
        df_seen = df.copy()
        df_unseen = pd.DataFrame(columns=df.columns)
        assignments = _assign_splits(df_seen, seed)

    # Assign unseen splits
    if not official_split_mode and len(df_unseen) > 0:
        unseen_assignments = _assign_unseen_splits(df_unseen, seed)
        assignments = pd.concat([assignments, unseen_assignments], ignore_index=True)

    # Validate all splits
    df_all = pd.concat([df_seen, df_unseen], ignore_index=True) if len(df_unseen) > 0 else df_seen
    if not official_split_mode:
        _enforce_min_split_sizes(
            df_seen,
            assignments[assignments["split"].isin(["train", "val", "test"])].copy(),
            min_rows=MIN_SPLIT_ROWS,
            min_frac=MIN_SPLIT_FRAC,
        )
    expected_splits = ["train", "val", "test"] if official_split_mode else ALL_SPLITS
    warnings = _validate_splits(
        df_all,
        assignments,
        heldout_cwe_ids=None if official_split_mode else (HELDOUT_CWE_IDS if has_cwe else None),
        expected_splits=expected_splits,
    )
    warnings.extend(_validate_short_bucket_coverage(df_all, assignments))
    if official_split_mode:
        warnings.append("Official split mode active: unseen_val and unseen_test are derived from val/test CWEs absent from train")
        warnings.extend(_validate_derived_unseen_assignments(df_all, derived_unseen_assignments))
    for w in warnings:
        logger.warning(w)

    # Write outputs
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    assignments_path = out / "assignments.csv"
    assignments.to_csv(assignments_path, index=False)
    logger.info("Wrote split assignments to %s", assignments_path)
    if official_split_mode:
        derived_path = out / DERIVED_UNSEEN_ASSIGNMENTS_FILENAME
        derived_unseen_assignments.to_csv(derived_path, index=False)
        logger.info("Wrote derived unseen assignments to %s", derived_path)

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

    # Generate unseen audit CSV
    if not official_split_mode and len(df_unseen) > 0:
        unseen_merged = df_unseen.merge(assignments, on="sample_id")
        audit_rows = []
        for split_name in ["unseen_val", "unseen_test"]:
            split_df = unseen_merged[unseen_merged["split"] == split_name]
            split_total = len(split_df)
            split_vul_total = int((split_df["vul"] == 1).sum())
            for proj, proj_df in split_df.groupby("project"):
                total_rows = len(proj_df)
                vul_rows = int((proj_df["vul"] == 1).sum())
                benign_rows = total_rows - vul_rows
                unique_cwes = proj_df["cwe_id"].nunique() if "cwe_id" in proj_df.columns else 0
                row_share = total_rows / split_total if split_total else 0.0
                vul_share = vul_rows / split_vul_total if split_vul_total else 0.0
                audit_rows.append({
                    "split": split_name,
                    "project": proj,
                    "total_rows": total_rows,
                    "vulnerable_rows": vul_rows,
                    "benign_rows": benign_rows,
                    "unique_cwes": unique_cwes,
                    "row_share_in_split": round(row_share, 4),
                    "vul_share_in_split": round(vul_share, 4),
                })
        if audit_rows:
            audit_df = pd.DataFrame(audit_rows)
            audit_path = out / "unseen_audit.csv"
            audit_df.to_csv(audit_path, index=False)
            logger.info("Wrote unseen audit CSV to %s", audit_path)

    # Generate report
    if official_split_mode:
        report_mode = split_mode if split_mode == "project_disjoint" else "official"
    else:
        report_mode = "derived_group_split"
    report = _generate_split_report(
        df_all,
        assignments,
        warnings,
        migration_info,
        weights_df,
        mode=report_mode,
        derived_unseen_assignments=derived_unseen_assignments if official_split_mode else None,
    )
    report_path = out / "split_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info("Wrote split report to %s", report_path)

    # Log summary
    for split_name in _active_splits(assignments, expected_splits):
        n = (assignments["split"] == split_name).sum()
        if n > 0:
            logger.info("Split '%s': %d rows", split_name, n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data with stratified group-aware assignment")
    parser.add_argument("--input", type=str, default=str(SUBSET_CSV))
    parser.add_argument("--output", type=str, default=str(SPLITS_DIR))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=["project_disjoint", "official"],
        default="project_disjoint",
        help="Split strategy: 'project_disjoint' (default) enforces no project overlap; 'official' uses PrimeVul source splits as-is",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    split_data(input_csv=args.input, output_dir=args.output, seed=args.seed, split_mode=args.split_mode)
