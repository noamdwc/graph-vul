## Why

The post-experiment analysis notebook has several issues that compromise result integrity and diagnostic value: threshold analysis runs on test splits (data leakage), per-CWE/per-project breakdowns are not split-specific, confusion matrices are missing, and there is no dedicated section to compare unseen_val vs unseen_test to diagnose the F1 gap we observed (0.577 vs 0.882, driven by precision differences).

## What Changes

- **Remove test leakage in threshold analysis**: threshold sweep and tuning must use only `val` or `unseen_val` — never `test` or `unseen_test`. Test splits are evaluate-only.
- **Split-specific per-CWE and per-project tables**: currently `eval_df` merges test + unseen_test, hiding split-level patterns. Add breakdowns per split.
- **Correct score/probability handling**: ensure `score_kind` (prob vs logit) is propagated correctly through threshold sweep, histogram, and reliability sections.
- **Add confusion matrices**: display per-split confusion matrices (counts and normalized).
- **Add unseen_val vs unseen_test comparison section**: side-by-side metrics, per-project FP analysis, class-rate comparison to diagnose why unseen_val has lower precision.
- **Per-project full metric tables**: accuracy, F1, recall, precision per project per split.

## Capabilities

### New Capabilities
- `unseen-gap-analysis`: dedicated section comparing unseen_val and unseen_test splits — side-by-side aggregate metrics, per-project FP/FN rates, class distribution comparison, project dominance analysis.
- `confusion-matrix-display`: per-split confusion matrix (raw counts and row-normalized) with visual heatmap.
- `split-specific-breakdowns`: per-CWE and per-project metric tables computed and displayed separately for each evaluation split rather than on a merged eval_df.

### Modified Capabilities
<!-- No existing specs to modify -->

## Impact

- `notebooks/post_experiment_analysis.ipynb` — cells 10, 12, 14, 16, 20 modified; new cells added for confusion matrices and unseen gap analysis.
- `graph_vul/analysis/post_experiment_analysis.py` — may need new helper functions for confusion matrix display and split-specific grouped metrics.
- Output CSVs will change: per-CWE and per-project CSVs will be emitted per-split instead of merged.
