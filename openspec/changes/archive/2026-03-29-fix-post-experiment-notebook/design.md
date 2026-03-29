## Context

The `notebooks/post_experiment_analysis.ipynb` notebook is the standard post-run analysis tool. It currently merges `test` and `unseen_test` into a single `eval_df` for per-CWE/per-project analysis, runs threshold sweeps on test splits (leakage), lacks confusion matrices, and has no dedicated section for diagnosing the unseen_val vs unseen_test gap we observed (F1 0.577 vs 0.882, driven entirely by precision).

The notebook uses helper functions from `graph_vul/analysis/post_experiment_analysis.py` (the `pea` module).

## Goals / Non-Goals

**Goals:**
- Eliminate data leakage: threshold analysis uses only val/unseen_val, never test/unseen_test
- Split-specific breakdowns: per-CWE and per-project tables computed per split, not on merged eval_df
- Add confusion matrices (counts + normalized) per evaluation split
- Add unseen gap analysis section: side-by-side unseen_val vs unseen_test with per-project FP diagnostics
- Per-project full metrics (accuracy, F1, precision, recall) per split
- Correct score_kind propagation through all threshold/score sections

**Non-Goals:**
- Changing the training pipeline or split logic
- Adding new model evaluation capabilities beyond what's listed
- Refactoring the pea module architecture

## Decisions

### 1. Threshold analysis restricted to val splits only

Threshold sweep cell (currently cell 20) iterates over `["test", "unseen_test"]`. Change to `["val", "unseen_val"]`. The tuned threshold can then be *applied* to test splits for comparison, but the sweep/optimization itself never touches test data.

**Alternative**: Keep test in sweep as "diagnostic only" — rejected because it creates a temptation to select thresholds from test results.

### 2. Split-specific eval_df construction

Instead of one merged `eval_df`, build a dict `{split_name: df}` for per-CWE and per-project analysis. Each split gets its own table and CSV output (e.g., `per_cwe_metrics_test.csv`, `per_cwe_metrics_unseen_test.csv`).

The existing merged `eval_df` is kept for cross-split comparison views only (e.g., the unseen gap section).

### 3. Confusion matrix via sklearn + seaborn heatmap

Use `sklearn.metrics.confusion_matrix` and `seaborn.heatmap` for display. One raw-count matrix and one row-normalized matrix per split. Added as a new section after Aggregate Metrics.

No new pea helper needed — the display logic is ~10 lines inline in the notebook cell using standard sklearn/seaborn.

### 4. Unseen gap analysis as a new notebook section

New section after "Seen vs Unseen" that:
- Shows side-by-side aggregate metrics for unseen_val vs unseen_test
- Computes per-project metrics in each unseen split separately
- Highlights projects with high FP counts in unseen_val
- Shows class distribution (positive rate) per project per split
- Computes project concentration (% of split from top-N projects)

This is notebook-only logic; no new pea module functions needed beyond existing `per_project_metrics` and `compute_binary_metrics`.

### 5. Score handling fix

The `score_kind` variable must be consistently used everywhere scores are compared to thresholds. When `score_kind == "logit"`, use `score_raw` for threshold comparisons (sigmoid is monotonic so ordering is preserved, but the threshold value must be in logit space). Audit all cells that reference `score` to ensure they respect `effective_score_kind`.

## Risks / Trade-offs

- **More verbose output**: split-specific tables multiply the number of displayed tables. Mitigated by collapsing less important splits behind conditionals.
- **Existing saved outputs change format**: per-CWE/per-project CSVs will have different names. Downstream consumers (if any) need updating. Low risk — these are analysis artifacts, not pipeline inputs.
