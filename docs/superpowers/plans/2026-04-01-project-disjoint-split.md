# Project-Disjoint Split Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--split-mode` CLI argument to `split_data.py` that enforces project-disjoint splits on PrimeVul data by default, with an `official` mode for A/B comparison.

**Architecture:** The new `_dejoin_official_splits()` function anchors single-split projects to their current assignment, pools multi-split projects, and feeds them to the existing `_greedy_project_split` allocator with deficit-adjusted targets. The `split_data()` entry point gains a `split_mode` parameter that selects between this new path and the existing official passthrough.

**Tech Stack:** Python, pandas, numpy (all already in use)

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `graph_vul/pipeline/split_data.py` | Modify | Add `_dejoin_official_splits()`, update `split_data()` signature and routing, update `__main__` CLI |
| `tests/test_split_data.py` | Modify | Add tests for dejoin logic and split_mode argument |
| `tests/conftest.py` | Modify | Add fixture for data with `source_split` column and multi-split projects |
| `dvc.yaml` | Modify | Add `--split-mode` to `func_split` stage command |

---

### Task 1: Add test fixture for multi-split-project data with `source_split`

**Files:**
- Modify: `tests/conftest.py` (after line 151)

- [ ] **Step 1: Write the fixture**

Add this fixture at the end of `tests/conftest.py`. It creates data where some projects appear in multiple official splits (mimicking PrimeVul's leakage) and some are single-split:

```python
@pytest.fixture
def official_split_df_with_overlap():
    """Dataset with source_split column where some projects span multiple splits.

    Layout:
    - ProjectA: train only (80 rows)
    - ProjectB: val only (40 rows)
    - ProjectC: test only (40 rows)
    - ProjectD: train + val (60 train, 20 val) — multi-split
    - ProjectE: train + test (50 train, 15 test) — multi-split
    - ProjectF: train + val + test (40 train, 15 val, 10 test) — multi-split
    - ProjectG: val + test (20 val, 20 test) — multi-split
    """
    rng = np.random.RandomState(99)
    rows = []

    project_layout = [
        ("ProjectA", [("train", 80)]),
        ("ProjectB", [("val", 40)]),
        ("ProjectC", [("test", 40)]),
        ("ProjectD", [("train", 60), ("val", 20)]),
        ("ProjectE", [("train", 50), ("test", 15)]),
        ("ProjectF", [("train", 40), ("val", 15), ("test", 10)]),
        ("ProjectG", [("val", 20), ("test", 20)]),
    ]

    sample_id = 0
    for project, split_counts in project_layout:
        for source_split, count in split_counts:
            for i in range(count):
                token_count = rng.randint(10, 400)
                rows.append({
                    "sample_id": sample_id,
                    "project": project,
                    "source_split": source_split,
                    "func_before": "t " * token_count,
                    "cwe_id": rng.choice(["CWE-119", "CWE-125", "CWE-476"]),
                    "vul": 1 if i < max(count // 5, 2) else 0,
                })
                sample_id += 1

    return pd.DataFrame(rows)
```

- [ ] **Step 2: Verify fixture loads**

Run: `python -c "import tests.conftest"`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "Add test fixture for official splits with project overlap"
```

---

### Task 2: Write tests for `_dejoin_official_splits`

**Files:**
- Modify: `tests/test_split_data.py`

- [ ] **Step 1: Write the failing tests**

Add this test class to `tests/test_split_data.py`. First, update the import block at the top to include the new function (it doesn't exist yet, so the import will fail):

Add `_dejoin_official_splits` to the import list in the existing `from graph_vul.pipeline.split_data import (...)` block.

Then add this test class after `TestOfficialSplitAssignments`:

```python
class TestDejoinOfficialSplits:
    def test_returns_project_disjoint_assignments(self, official_split_df_with_overlap):
        df = official_split_df_with_overlap
        assignments = _dejoin_official_splits(df, seed=42)

        merged = df.merge(assignments, on="sample_id")
        train_p = set(merged[merged["split"] == "train"]["project"])
        val_p = set(merged[merged["split"] == "val"]["project"])
        test_p = set(merged[merged["split"] == "test"]["project"])

        assert train_p & val_p == set(), f"train/val overlap: {train_p & val_p}"
        assert train_p & test_p == set(), f"train/test overlap: {train_p & test_p}"
        assert val_p & test_p == set(), f"val/test overlap: {val_p & test_p}"

    def test_all_rows_assigned(self, official_split_df_with_overlap):
        df = official_split_df_with_overlap
        assignments = _dejoin_official_splits(df, seed=42)
        assert len(assignments) == len(df)
        assert set(assignments.columns) == {"sample_id", "split"}

    def test_single_split_projects_keep_assignment(self, official_split_df_with_overlap):
        df = official_split_df_with_overlap
        assignments = _dejoin_official_splits(df, seed=42)
        merged = df.merge(assignments, on="sample_id")

        # ProjectA is train-only, ProjectB is val-only, ProjectC is test-only
        for proj, expected_split in [("ProjectA", "train"), ("ProjectB", "val"), ("ProjectC", "test")]:
            proj_splits = merged.loc[merged["project"] == proj, "split"].unique()
            assert list(proj_splits) == [expected_split], (
                f"Single-split {proj} should stay in {expected_split}, got {proj_splits}"
            )

    def test_deterministic(self, official_split_df_with_overlap):
        df = official_split_df_with_overlap
        a1 = _dejoin_official_splits(df, seed=42)
        a2 = _dejoin_official_splits(df, seed=42)
        pd.testing.assert_frame_equal(a1, a2)

    def test_all_three_splits_present(self, official_split_df_with_overlap):
        df = official_split_df_with_overlap
        assignments = _dejoin_official_splits(df, seed=42)
        assert set(assignments["split"].unique()) == {"train", "val", "test"}

    def test_eval_splits_not_empty(self, official_split_df_with_overlap):
        df = official_split_df_with_overlap
        assignments = _dejoin_official_splits(df, seed=42)
        for split in ["val", "test"]:
            assert (assignments["split"] == split).sum() > 0, f"{split} is empty"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_split_data.py::TestDejoinOfficialSplits -v 2>&1 | head -30`
Expected: ImportError for `_dejoin_official_splits`

- [ ] **Step 3: Commit**

```bash
git add tests/test_split_data.py
git commit -m "Add failing tests for _dejoin_official_splits"
```

---

### Task 3: Implement `_dejoin_official_splits`

**Files:**
- Modify: `graph_vul/pipeline/split_data.py` (insert after `_materialize_official_split_assignments`, around line 92)

- [ ] **Step 1: Implement the function**

Insert the following function after `_materialize_official_split_assignments` (after line 91) in `split_data.py`:

```python
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
    project_stats = _build_project_stats(df_multi, major_cwes=major_cwes)
    project_to_split = _greedy_project_split(
        project_stats,
        split_fracs=deficit_fracs,
        seed=seed,
    )
    reallocated_assignments = _expand_project_assignments(df_multi, project_to_split)

    # Merge anchored + reallocated
    combined = pd.concat([anchored_assignments, reallocated_assignments], ignore_index=True)
    df.drop(columns=["_norm_split"], inplace=True)
    return combined[["sample_id", "split"]]
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `python -m pytest tests/test_split_data.py::TestDejoinOfficialSplits -v`
Expected: All 6 tests PASS

- [ ] **Step 3: Run existing tests to check for regressions**

Run: `python -m pytest tests/test_split_data.py -v`
Expected: All existing tests still PASS

- [ ] **Step 4: Commit**

```bash
git add graph_vul/pipeline/split_data.py
git commit -m "Implement _dejoin_official_splits for project-disjoint PrimeVul splits"
```

---

### Task 4: Write tests for `split_mode` parameter in `split_data()`

**Files:**
- Modify: `tests/test_split_data.py`

- [ ] **Step 1: Write the failing tests**

Add this test class at the end of `tests/test_split_data.py`:

```python
class TestSplitModeArgument:
    def test_official_mode_allows_project_overlap(self, tmp_path):
        """Official mode preserves original splits, including project overlap."""
        rows = []
        sample_id = 0
        # ProjectX in both train and val
        for split in ["train", "val"]:
            for i in range(30):
                rows.append({
                    "sample_id": sample_id, "project": "ProjectX",
                    "source_split": split, "func_before": f"px-{split}-{i}",
                    "cwe_id": "CWE-119", "vul": 1 if i < 5 else 0,
                })
                sample_id += 1
        # Single-split projects for test and padding
        for i in range(30):
            rows.append({
                "sample_id": sample_id, "project": f"T{i}",
                "source_split": "test", "func_before": f"test-{i}",
                "cwe_id": "CWE-119", "vul": 1 if i < 5 else 0,
            })
            sample_id += 1

        csv_path = tmp_path / "data.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        out = tmp_path / "splits"
        split_data(input_csv=str(csv_path), output_dir=str(out), seed=42, split_mode="official")

        assignments = pd.read_csv(out / "assignments.csv")
        merged = pd.DataFrame(rows).merge(assignments, on="sample_id")
        train_p = set(merged[merged["split"] == "train"]["project"])
        val_p = set(merged[merged["split"] == "val"]["project"])
        # ProjectX should be in both (official mode doesn't dejoin)
        assert "ProjectX" in train_p
        assert "ProjectX" in val_p

    def test_project_disjoint_mode_removes_overlap(self, tmp_path):
        """Project disjoint mode ensures no project appears in multiple splits."""
        rows = []
        sample_id = 0
        for split in ["train", "val"]:
            for i in range(30):
                rows.append({
                    "sample_id": sample_id, "project": "ProjectX",
                    "source_split": split, "func_before": f"px-{split}-{i}",
                    "cwe_id": "CWE-119", "vul": 1 if i < 5 else 0,
                })
                sample_id += 1
        for proj_idx in range(15):
            for i in range(8):
                rows.append({
                    "sample_id": sample_id, "project": f"T{proj_idx}",
                    "source_split": "train", "func_before": f"train-{proj_idx}-{i}",
                    "cwe_id": "CWE-119", "vul": 1 if i < 2 else 0,
                })
                sample_id += 1
        for proj_idx in range(10):
            for i in range(8):
                rows.append({
                    "sample_id": sample_id, "project": f"V{proj_idx}",
                    "source_split": "val", "func_before": f"val-{proj_idx}-{i}",
                    "cwe_id": "CWE-125", "vul": 1 if i < 2 else 0,
                })
                sample_id += 1
        for proj_idx in range(10):
            for i in range(8):
                rows.append({
                    "sample_id": sample_id, "project": f"S{proj_idx}",
                    "source_split": "test", "func_before": f"test-{proj_idx}-{i}",
                    "cwe_id": "CWE-476", "vul": 1 if i < 2 else 0,
                })
                sample_id += 1

        csv_path = tmp_path / "data.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        out = tmp_path / "splits"
        split_data(input_csv=str(csv_path), output_dir=str(out), seed=42, split_mode="project_disjoint")

        assignments = pd.read_csv(out / "assignments.csv")
        merged = pd.DataFrame(rows).merge(assignments, on="sample_id")

        from itertools import combinations as combs
        for s1, s2 in combs(["train", "val", "test"], 2):
            p1 = set(merged[merged["split"] == s1]["project"])
            p2 = set(merged[merged["split"] == s2]["project"])
            assert p1 & p2 == set(), f"Overlap between {s1} and {s2}: {p1 & p2}"

    def test_default_mode_is_project_disjoint(self, tmp_path):
        """split_data() without split_mode should default to project_disjoint."""
        rows = []
        sample_id = 0
        for split in ["train", "val"]:
            for i in range(30):
                rows.append({
                    "sample_id": sample_id, "project": "SharedProj",
                    "source_split": split, "func_before": f"sp-{split}-{i}",
                    "cwe_id": "CWE-119", "vul": 1 if i < 5 else 0,
                })
                sample_id += 1
        for proj_idx in range(15):
            for i in range(8):
                rows.append({
                    "sample_id": sample_id, "project": f"T{proj_idx}",
                    "source_split": "train", "func_before": f"t-{proj_idx}-{i}",
                    "cwe_id": "CWE-119", "vul": 1 if i < 2 else 0,
                })
                sample_id += 1
        for proj_idx in range(10):
            for i in range(8):
                rows.append({
                    "sample_id": sample_id, "project": f"V{proj_idx}",
                    "source_split": "val", "func_before": f"v-{proj_idx}-{i}",
                    "cwe_id": "CWE-125", "vul": 1 if i < 2 else 0,
                })
                sample_id += 1
        for proj_idx in range(10):
            for i in range(8):
                rows.append({
                    "sample_id": sample_id, "project": f"S{proj_idx}",
                    "source_split": "test", "func_before": f"test-{proj_idx}-{i}",
                    "cwe_id": "CWE-476", "vul": 1 if i < 2 else 0,
                })
                sample_id += 1

        csv_path = tmp_path / "data.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        out = tmp_path / "splits"
        split_data(input_csv=str(csv_path), output_dir=str(out), seed=42)

        assignments = pd.read_csv(out / "assignments.csv")
        merged = pd.DataFrame(rows).merge(assignments, on="sample_id")

        from itertools import combinations as combs
        for s1, s2 in combs(["train", "val", "test"], 2):
            p1 = set(merged[merged["split"] == s1]["project"])
            p2 = set(merged[merged["split"] == s2]["project"])
            assert p1 & p2 == set(), f"Default mode has overlap between {s1}/{s2}: {p1 & p2}"

    def test_report_includes_split_mode(self, tmp_path):
        rows = []
        sample_id = 0
        for proj_idx in range(10):
            for i in range(10):
                rows.append({
                    "sample_id": sample_id, "project": f"P{proj_idx}",
                    "source_split": "train" if proj_idx < 6 else ("val" if proj_idx < 8 else "test"),
                    "func_before": f"f-{proj_idx}-{i}",
                    "cwe_id": "CWE-119", "vul": 1 if i < 2 else 0,
                })
                sample_id += 1

        csv_path = tmp_path / "data.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        for mode in ["official", "project_disjoint"]:
            out = tmp_path / f"splits_{mode}"
            split_data(input_csv=str(csv_path), output_dir=str(out), seed=42, split_mode=mode)
            report = (out / "split_report.md").read_text()
            assert mode in report, f"Report should mention mode '{mode}'"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_split_data.py::TestSplitModeArgument -v 2>&1 | head -30`
Expected: TypeError — `split_data()` got unexpected keyword argument `split_mode`

- [ ] **Step 3: Commit**

```bash
git add tests/test_split_data.py
git commit -m "Add failing tests for split_mode parameter"
```

---

### Task 5: Wire `split_mode` into `split_data()` and CLI

**Files:**
- Modify: `graph_vul/pipeline/split_data.py` — `split_data()` function (line ~1227) and `__main__` block (line ~1401)

- [ ] **Step 1: Update `split_data()` signature and routing**

Change the `split_data` function signature at line 1227 from:

```python
def split_data(
    input_csv: str = str(SUBSET_CSV),
    output_dir: str = str(SPLITS_DIR),
    seed: int = SEED,
) -> None:
```

to:

```python
def split_data(
    input_csv: str = str(SUBSET_CSV),
    output_dir: str = str(SPLITS_DIR),
    seed: int = SEED,
    split_mode: str = "project_disjoint",
) -> None:
```

Then update the routing logic. Replace the block at lines 1266-1291 (the `if official_split_mode:` section for seen/unseen pool and assignment) with:

```python
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
```

Then update the report mode string. Replace line 1379:

```python
    report_mode = "official_source_split" if official_split_mode else "derived_group_split"
```

with:

```python
    if official_split_mode:
        report_mode = split_mode if split_mode == "project_disjoint" else "official"
    else:
        report_mode = "derived_group_split"
```

- [ ] **Step 2: Update the `__main__` CLI block**

Replace the `__main__` block (lines 1401-1409) with:

```python
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
```

- [ ] **Step 3: Run the new tests**

Run: `python -m pytest tests/test_split_data.py::TestSplitModeArgument -v`
Expected: All 4 tests PASS

- [ ] **Step 4: Run the full test suite**

Run: `python -m pytest tests/test_split_data.py -v`
Expected: All tests PASS (existing + new)

- [ ] **Step 5: Commit**

```bash
git add graph_vul/pipeline/split_data.py
git commit -m "Wire split_mode parameter into split_data() and CLI"
```

---

### Task 6: Update DVC pipeline

**Files:**
- Modify: `dvc.yaml` (line ~114)

- [ ] **Step 1: Update the `func_split` stage command**

Change line 114 from:

```yaml
    cmd: python -m graph_vul.pipeline.split_data --input data/func_subset.csv 
      --output data/func_splits
```

to:

```yaml
    cmd: python -m graph_vul.pipeline.split_data --input data/func_subset.csv 
      --output data/func_splits --split-mode project_disjoint
```

- [ ] **Step 2: Verify DVC parses correctly**

Run: `dvc repro --dry func_split 2>&1 | head -10`
Expected: Shows the stage would run with the updated command (no YAML parse errors)

- [ ] **Step 3: Commit**

```bash
git add dvc.yaml
git commit -m "Set func_split DVC stage to project_disjoint mode"
```

---

### Task 7: Integration smoke test on real data

**Files:** None (read-only verification)

- [ ] **Step 1: Run split in project_disjoint mode**

Run: `python -m graph_vul.pipeline.split_data --input data/func_subset.csv --output /tmp/splits_disjoint --split-mode project_disjoint`
Expected: Completes without error, logs show "Dejoin: N single-split projects (anchored), M multi-split projects (to reallocate)"

- [ ] **Step 2: Run split in official mode**

Run: `python -m graph_vul.pipeline.split_data --input data/func_subset.csv --output /tmp/splits_official --split-mode official`
Expected: Completes without error (same as current behavior)

- [ ] **Step 3: Verify project disjointness**

Run:
```bash
python -c "
import pandas as pd
from itertools import combinations

df = pd.read_csv('data/func_subset.csv', usecols=['sample_id', 'project'])

for mode, path in [('disjoint', '/tmp/splits_disjoint'), ('official', '/tmp/splits_official')]:
    assignments = pd.read_csv(f'{path}/assignments.csv')
    merged = df.merge(assignments, on='sample_id')
    overlaps = []
    for s1, s2 in combinations(['train', 'val', 'test'], 2):
        p1 = set(merged[merged['split'] == s1]['project'])
        p2 = set(merged[merged['split'] == s2]['project'])
        overlap = p1 & p2
        if overlap:
            overlaps.append(f'{s1}/{s2}: {len(overlap)} projects')
    if overlaps:
        print(f'{mode}: OVERLAPS — {overlaps}')
    else:
        print(f'{mode}: NO OVERLAP (project disjoint)')

    # Print split sizes
    for split in ['train', 'val', 'test']:
        n = (merged['split'] == split).sum()
        print(f'  {split}: {n} rows')
"
```
Expected:
- `disjoint: NO OVERLAP (project disjoint)` 
- `official: OVERLAPS — [...]`

- [ ] **Step 4: Check split report mentions mode**

Run: `head -5 /tmp/splits_disjoint/split_report.md && echo --- && head -5 /tmp/splits_official/split_report.md`
Expected: First report says `project_disjoint`, second says `official`

- [ ] **Step 5: Commit (no file changes — just verify)**

No commit needed. All verification is read-only.
