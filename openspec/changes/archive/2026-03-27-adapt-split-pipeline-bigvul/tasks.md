## 1. Config Updates

- [x] 1.1 Add new config constants to `graph_vul/config.py`: `MIN_HELDOUT_TOTAL_ROWS`, `MIN_HELDOUT_VUL_ROWS`, `MIN_HELDOUT_PROJECTS`, `UNSEEN_VAL_FRAC`, `BALANCE_TARGET_VUL_RATIO`, `BALANCE_CWE_SMOOTHING`, `BALANCE_PROJECT_DOWNWEIGHT`, `BALANCE_MAX_WEIGHT`, `BALANCE_MAX_PROJECT_SHARE`
- [x] 1.2 Update `VAL_SPLIT` to 0.10 and `TEST_SPLIT` to 0.10 in config
- [x] 1.3 Remove `HELD_OUT_PROJECT_COUNT` from config (no longer used)

## 2. Held-out CWE Eligibility Validation

- [x] 2.1 Add `_validate_heldout_eligibility(df, heldout_cwe_ids, min_total, min_vul, min_projects)` function that checks each held-out CWE against thresholds and raises `ValueError` with a clear message if any fail
- [x] 2.2 Wire eligibility validation into `split_data()` before pool construction

## 3. Full-Project Holdout Pool Construction

- [x] 3.1 Add `_build_unseen_pool(df, heldout_cwe_ids)` function that identifies all projects touching held-out CWEs and returns `(df_seen, df_unseen, migration_info)` where `migration_info` maps each unseen project to the held-out CWEs that caused its migration
- [x] 3.2 Replace the current row-level CWE filtering in `split_data()` with the new full-project holdout logic
- [x] 3.3 Add logging showing pool sizes and migration summary

## 4. Seen Pool Splitting

- [x] 4.1 Update `_assign_splits()` to use the new default ratios (80/10/10) and accept only seen-pool rows
- [x] 4.2 Verify the two-stage `StratifiedGroupKFold` still works correctly with the filtered seen pool

## 5. Unseen Pool Splitting

- [x] 5.1 Keep `_assign_unseen_splits()` largely as-is, operating on the full unseen pool (all rows from unseen projects, not just held-out CWE rows)
- [x] 5.2 Add `UNSEEN_VAL_FRAC` config support if non-50/50 split is needed

## 6. Cross-Regime Validation

- [x] 6.1 Update `_validate_splits()` to check all 10 split pairs for project overlap (train/val, train/test, val/test, unseen_val/unseen_test, and all 6 cross-regime pairs)
- [x] 6.2 Add validation that no held-out CWE IDs appear in train/val/test rows
- [x] 6.3 Add validation that unseen-pool projects do not appear in seen splits
- [x] 6.4 Merge `_validate_unseen_splits()` logic into the unified validation or keep it as a second pass — ensure no duplicate checks

## 7. Train Balancing

- [x] 7.1 Add `_compute_sample_weights(df_train, config_params)` function implementing the weight formula: class ratio targeting, CWE inverse-frequency smoothing for vulnerable rows, project downweighting for benign rows, max weight capping, max project share capping, normalization
- [x] 7.2 Add `_generate_balanced_manifest(df_train, weights, seed)` function that samples with replacement using the weights to produce a materialized manifest
- [x] 7.3 Wire weight computation and manifest generation into `split_data()`, writing `train_sampler_weights.csv` and `train_balanced_manifest.csv` to the output directory

## 8. Extended Reporting

- [x] 8.1 Add "Project Exclusivity" section to report: all-pairs overlap check results, seen/unseen pool project counts, migration reasons per project
- [x] 8.2 Add "Train Balancing" section to report: raw vs balanced vul ratio, CWE distribution before/after, top project contribution before/after, weight statistics (min, median, max, std)
- [x] 8.3 Extend overlap verification section to cover all 10 pairs instead of current 4
- [x] 8.4 Ensure all existing report sections (split sizes, per-CWE, CVE coverage, project coverage, metadata buckets, warnings) still work with the five-split output

## 9. Tests

- [x] 9.1 Update `tests/conftest.py` fixture to include multiple CWEs (both seen and held-out) across shared projects, so the full-project holdout logic can be tested
- [x] 9.2 Rewrite `tests/test_split_data.py` to match the new API: remove `_select_held_out_projects` references, test `_build_unseen_pool`, test `_validate_heldout_eligibility`, test full-project holdout, test all-pairs overlap, test weight computation
- [x] 9.3 Add end-to-end test verifying: five splits produced, zero project overlap across all pairs, held-out CWEs absent from seen splits, weights file and manifest produced
- [x] 9.4 Add test for held-out CWE eligibility failure (small CWE raises error)

## 10. DVC and Integration

- [x] 10.1 Verify `dvc.yaml` split_data and func_split stages still work (output dir `data/splits/` captures all new files automatically since it's a directory output)
- [x] 10.2 Update imports in `split_data.py` to include new config constants
- [x] 10.3 Run the full split pipeline on the actual dataset and inspect the report for correctness
