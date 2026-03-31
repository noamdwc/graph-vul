## 1. Config Constants

- [x] 1.1 Add `TOKEN_BUCKET_EDGES`, `TOKEN_BUCKET_LABELS`, `MIN_SHORT_VUL_PER_EVAL_SPLIT`, `MIN_SHORT_BENIGN_PER_EVAL_SPLIT`, and `SHORT_BUCKET_DEFICIT_WEIGHT` to `graph_vul/config.py`

## 2. Fixed Token Bucketing

- [x] 2.1 Replace `pd.qcut` for `code_token_count` with `pd.cut` using `TOKEN_BUCKET_EDGES` / `TOKEN_BUCKET_LABELS` in `split_data()` (lines ~957-965). Keep qcut for `code_char_len` and `code_line_count` only
- [x] 2.2 Update the import block in `split_data.py` to import the new config constants

## 3. Project Stats Extension

- [x] 3.1 Extend `_build_project_stats()` to compute `short_vul` and `short_benign` columns per project (vulnerable and benign row counts in the 0–64 token bucket)

## 4. Greedy Allocator Scoring

- [x] 4.1 Add short-bucket deficit penalty to `score_state()` — penalize eval splits (val, test, unseen_val, unseen_test) when `short_vul` or `short_benign` are below configured minimums, weighted by `SHORT_BUCKET_DEFICIT_WEIGHT`
- [x] 4.2 Add the same short-bucket deficit penalty to the per-project scoring loop (lines ~304-349) so the greedy assignment considers short-bucket impact when evaluating candidate splits
- [x] 4.3 Track `short_vul` and `short_benign` in the allocator state dict alongside `rows` and `vul_rows`, updating them as projects are assigned

## 5. Post-Split Validation

- [x] 5.1 Add `_validate_short_bucket_coverage()` function that checks each eval split's 0–64 bucket against `MIN_SHORT_VUL_PER_EVAL_SPLIT` and `MIN_SHORT_BENIGN_PER_EVAL_SPLIT`, emitting warnings (not errors) for deficits
- [x] 5.2 Call `_validate_short_bucket_coverage()` from `split_data()` after `_validate_splits()`, passing results into the warnings list

## 6. Split Report Updates

- [x] 6.1 Add "Token Length Buckets" section to `_generate_split_report()` showing per-split × per-fixed-bucket × per-label (vul/benign) counts table
- [x] 6.2 Add "Short-Function Evaluation Viability" section showing 0–64 bucket vul/benign counts per eval split with pass/warn status
- [x] 6.3 Add per-project short-bucket contribution table for eval splits (which projects contribute short rows, how many vul/benign)

## 7. Verification

- [x] 7.1 Run the split pipeline on the current dataset and verify: fixed bucket labels appear in assignments metadata, split_report.md contains the new sections, no downstream training code changes needed
- [x] 7.2 Verify that project-disjointness, held-out CWE guarantees, and all existing validation checks still pass
