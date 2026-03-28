## 1. Allocator: Project-count penalty

- [x] 1.1 Add module-level constant `UNSEEN_PROJECT_COUNT_PENALTY = 20.0`
- [x] 1.2 Add optional `project_count_targets: dict[str, int] | None` parameter to `_greedy_project_split()`
- [x] 1.3 Add `project_count_penalty` term to `score_state()` — compute `(actual_count / target_count - 1.0)^2` per split, weighted by `UNSEEN_PROJECT_COUNT_PENALTY`. Only active when `project_count_targets` is provided.
- [x] 1.4 Track per-split project counts in the allocator state alongside row/vul/CWE accumulators
- [x] 1.5 Update `_assign_unseen_splits()` to compute project-count targets as `target_unseen_val = max(1, round(n_projects * UNSEEN_VAL_FRAC))`, `target_unseen_test = n_projects - target_unseen_val`, and pass to `_greedy_project_split()`

## 2. Hard validation

- [x] 2.1 Add module-level constants `MIN_UNSEEN_PROJECTS = 8` and `MIN_UNSEEN_VUL_ROWS = 10`
- [x] 2.2 In `_validate_splits()`, add checks for minimum project count per unseen split (relaxed proportionally when total unseen projects < 2 × threshold)
- [x] 2.3 In `_validate_splits()`, add checks for minimum vulnerable rows per unseen split
- [x] 2.4 Add dominance warning: if any single project has > 50% of rows in an unseen split, emit a warning naming the project and its share

## 3. Extended split report

- [x] 3.1 Add "Unseen Split Balance" section to `_generate_split_report()` showing per-project row counts, vulnerable/benign counts, and largest-project share for each unseen split
- [x] 3.2 Include project count summary line for `unseen_val` and `unseen_test` at the top of the section

## 4. Audit CSV

- [x] 4.1 In `split_data()`, after assignments are complete, generate `unseen_audit.csv` with columns: `split`, `project`, `total_rows`, `vulnerable_rows`, `benign_rows`, `unique_cwes`, `row_share_in_split`, `vul_share_in_split`
- [x] 4.2 Skip writing the file when the unseen pool is empty

## 5. Verification

- [x] 5.1 Run the split pipeline end-to-end and verify `unseen_val` has more projects than before
- [x] 5.2 Verify split report contains the new unseen balance section and dominance warnings (if applicable)
- [x] 5.3 Verify `unseen_audit.csv` is written with correct columns including `row_share_in_split` and `vul_share_in_split`
- [x] 5.4 Verify no project overlap across all five splits
- [x] 5.5 Verify no held-out CWE leakage into train/val/test
