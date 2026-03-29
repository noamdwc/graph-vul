## 1. Fix threshold analysis leakage

- [x] 1.1 Change threshold sweep cell to iterate over `["val", "unseen_val"]` instead of `["test", "unseen_test"]`
- [x] 1.2 Update threshold comparison: find best threshold from val, then apply to test/unseen_test for display only
- [x] 1.3 Audit all score references to ensure `effective_score_kind` is respected (use `score_raw` for logit, `score` for prob)

## 2. Add confusion matrices

- [x] 2.1 Add new notebook section after Aggregate Metrics with per-split confusion matrices (raw counts + row-normalized heatmaps) for all loaded eval splits

## 3. Split-specific per-CWE and per-project breakdowns

- [x] 3.1 Refactor per-CWE analysis cell: loop over each eval split, compute and display per-CWE metrics separately, save as `per_cwe_metrics_{split}.csv`
- [x] 3.2 Refactor per-project analysis cell: loop over each eval split, compute and display per-project metrics (accuracy, F1, precision, recall, support, TP, FP, TN, FN) separately, save as `per_project_metrics_{split}.csv`
- [x] 3.3 Update eval_df construction: keep merged eval_df for cross-split sections, but per-CWE/per-project sections use split-specific frames

## 4. Unseen gap analysis section

- [x] 4.1 Add new section after Seen vs Unseen: side-by-side aggregate metrics for unseen_val vs unseen_test
- [x] 4.2 Add per-project FP diagnostics: per-project metrics for each unseen split sorted by FP count
- [x] 4.3 Add project concentration analysis: top-3 and top-5 row share per unseen split
- [x] 4.4 Add class distribution comparison: positive rate per project across unseen_val vs unseen_test, sorted by difference

## 5. Verify and clean up

- [x] 5.1 Run notebook end-to-end on current prediction files to verify no errors
- [x] 5.2 Verify output CSVs are written with correct split-specific names
