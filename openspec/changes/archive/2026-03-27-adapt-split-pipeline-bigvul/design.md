## Context

The split pipeline (`graph_vul/pipeline/split_data.py`) currently separates held-out CWE **rows** from the seen pool, then splits each pool independently. This allows a project to appear in both `train` (via its non-held-out CWE rows) and `unseen_val`/`unseen_test` (via its held-out CWE rows), leaking project-level signal across regimes.

The pipeline is used by two DVC stages (`split_data` for `data/subset.csv` → `data/splits/`, and `func_split` for `data/func_subset.csv` → `data/func_splits/`), both invoking the same `split_data()` entry point. The existing test suite references a removed `_select_held_out_projects` function and needs updating regardless.

Training is highly imbalanced (~8% vulnerable). There is no rebalancing mechanism; the training loop receives raw split assignments.

## Goals / Non-Goals

**Goals:**
- Enforce full-project holdout: if any row in a project has a held-out CWE, the entire project is excluded from seen splits.
- Produce five splits (`train`, `val`, `test`, `unseen_val`, `unseen_test`) with zero project overlap across all pairs.
- Validate held-out CWE eligibility before splitting (configurable thresholds).
- Produce a per-row sample-weight artifact (`train_sampler_weights.csv`) for balanced training.
- Extend the diagnostics report with project exclusivity details and train balancing stats.
- Preserve the existing `assignments.csv` format and CLI interface for backward compatibility.

**Non-Goals:**
- Rebalancing evaluation splits (val, test, unseen_val, unseen_test remain natural).
- Modifying downstream training code to consume weights (that is a separate change).
- Changing the subset extraction or embedding stages.
- Supporting `(project, cwe)` as the leakage boundary — project alone is the unit.
- Auto-selecting which CWEs to hold out (the list remains config-driven).

## Decisions

### 1. Pool construction: project-level holdout from held-out CWEs

**Choice**: Compute `unseen_projects = {p for p in projects if any row in p has cwe_id in HELDOUT_CWE_IDS}`. All rows from these projects form the unseen pool; all remaining rows form the seen pool.

**Alternative considered**: Continue filtering by CWE ID only (current behavior). Rejected because it allows project leakage across regimes.

**Alternative considered**: Use `(project, cwe)` pairs as the grouping unit. Rejected because it fragments project-level signal and complicates leakage analysis.

### 2. Seen split method: keep StratifiedGroupKFold

**Choice**: Retain the two-stage `StratifiedGroupKFold` approach for train/val/test assignment within the seen pool. It already handles project-disjoint stratification on the binary label.

**Rationale**: The current method works well for the seen pool. The new requirement is about *which rows enter the seen pool*, not how the seen pool is split. Default ratios change to 80/10/10 (from 70/15/15).

**Alternative considered**: Greedy stratified assignment optimizing CWE distribution and metadata buckets. Deferred — the current method is simpler and the marginal improvement is small given project-level granularity.

### 3. Unseen split method: keep StratifiedGroupKFold(n_splits=2)

**Choice**: Split unseen projects 50/50 into `unseen_val` and `unseen_test` using `StratifiedGroupKFold(n_splits=2)` on the binary label, grouped by project.

**Rationale**: Same logic as today, just applied to the correctly constructed unseen pool.

### 4. Held-out CWE eligibility validation

**Choice**: Before splitting, validate each held-out CWE against configurable thresholds: `min_heldout_total_rows`, `min_heldout_vul_rows`, `min_heldout_projects`. Fail with a clear error if any threshold is not met.

**Alternative considered**: Silently drop ineligible CWEs. Rejected because it could produce unexpectedly small or empty unseen splits.

**Default thresholds**: `min_heldout_total_rows=30`, `min_heldout_vul_rows=5`, `min_heldout_projects=2`.

### 5. Train balancing: per-row sample weights

**Choice**: Produce `train_sampler_weights.csv` with columns `sample_id, split, project, cwe_id, vul, sample_weight`. Weights are computed per training row using a configurable policy.

**Weight formula** (default):
1. Compute target counts: given `target_vul_ratio` (default 0.25 → 1:3), compute the effective weight ratio `w_vul / w_benign` that achieves the target ratio.
2. Vulnerable rows: base weight × CWE inverse-frequency factor (smoothed). `cwe_weight = (N_vul_total / N_vul_cwe) ^ cwe_smoothing_strength` where `cwe_smoothing_strength` defaults to 0.5 (square root smoothing).
3. Benign rows: base weight × project inverse-size factor. `proj_weight = (median_project_size / project_size) ^ project_downweight_strength` where `project_downweight_strength` defaults to 0.3.
4. Cap all weights at `max_weight` (default 10.0) relative to the median weight.
5. Normalize so weights sum to len(train).

**Alternative considered**: Materialized balanced manifest with duplicated rows. Not preferred because weighted sampling is more flexible and avoids inflating dataset size. However, we also produce a `train_balanced_manifest.csv` by sampling with the weights (for pipelines that expect a flat file).

**Decision**: Produce both artifacts — weights CSV (primary) and materialized manifest (convenience). The manifest is a deterministic sample using the weights and a fixed seed.

### 6. Config additions in `config.py`

New constants:
```python
# Split ratios (seen regime)
VAL_SPLIT = 0.10    # changed from 0.15
TEST_SPLIT = 0.10   # changed from 0.15

# Held-out CWE eligibility
MIN_HELDOUT_TOTAL_ROWS = 30
MIN_HELDOUT_VUL_ROWS = 5
MIN_HELDOUT_PROJECTS = 2

# Unseen split ratio
UNSEEN_VAL_FRAC = 0.5

# Train balancing
BALANCE_TARGET_VUL_RATIO = 0.25
BALANCE_CWE_SMOOTHING = 0.5
BALANCE_PROJECT_DOWNWEIGHT = 0.3
BALANCE_MAX_WEIGHT = 10.0
BALANCE_MAX_PROJECT_SHARE = 0.15
```

### 7. Report structure

Extend the existing `_generate_split_report` with three new sections:
- **Project Exclusivity**: all-pairs overlap check, seen/unseen pool sizes, which CWEs caused migration.
- **Train Balancing**: raw vs balanced vul ratio, CWE distribution before/after, top project contribution before/after, weight statistics.
- Keep all existing sections (split sizes, per-CWE, CVE coverage, project coverage, metadata buckets, warnings).

### 8. File organization

Keep everything in `split_data.py`. The file is currently ~387 lines; the additions (eligibility validation, weight computation, extended reporting) will grow it to ~600-700 lines. This is manageable for a single-purpose pipeline module. If it grows beyond ~800 lines, consider extracting the balancing logic to a helper module.

## Risks / Trade-offs

- **Smaller seen pool**: Full-project holdout may move more rows to the unseen pool than the previous row-level approach, shrinking train/val/test. → Mitigation: the eligibility thresholds ensure held-out CWEs are meaningful; monitor pool sizes in the report.
- **StratifiedGroupKFold limitations**: With fewer projects in the seen pool or highly skewed project sizes, the k-fold stratification may produce uneven splits. → Mitigation: the report warns on small splits (<5% of total) and single-class splits. A greedy assignment can be added later if needed.
- **Weight explosion**: CWE inverse-frequency on tiny CWEs could produce extreme weights. → Mitigation: `max_weight` cap and smoothing exponent < 1.0.
- **Backward compatibility**: Changing `VAL_SPLIT`/`TEST_SPLIT` defaults from 0.15 to 0.10 changes split sizes for existing users. → Mitigation: values are configurable; document the change.
- **Test suite needs rewrite**: Current tests import `_select_held_out_projects` which no longer exists. → This is already broken; the new tests will cover the new API.
