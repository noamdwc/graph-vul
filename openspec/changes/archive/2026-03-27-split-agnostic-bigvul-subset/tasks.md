## 1. Config and contract cleanup

- [x] 1.1 Remove or deprecate `FUNC_MIN_PROJECT_ROWS` in `graph_vul/config.py` and update imports so `func_extract_subset.py` depends only on dedup, max token, and max `(project, cwe_id)` settings
- [x] 1.2 Review `CWE_IDS` and `HELDOUT_CWE_IDS` references and ensure subset extraction no longer treats them as filtering inputs while preserving later split-stage compatibility
- [x] 1.3 Update any inline docstrings, comments, or user-facing config/report labels that still describe the subset stage as curated-CWE or min/max support filtering

## 2. Subset extraction refactor

- [x] 2.1 Refactor `graph_vul/pipeline/func_extract_subset.py` to keep all labeled `cwe_id` rows from BigVul instead of filtering to configured CWE lists
- [x] 2.2 Preserve the current metadata derivation, exact-code deduplication, and token-length filtering order and semantics based on `func_before`
- [x] 2.3 Simplify the `(project, cwe_id)` group filter to drop only groups whose row count exceeds `FUNC_MAX_PROJECT_ROWS`
- [x] 2.4 Keep the output CSV shape and output paths stable while ensuring the subset remains split-agnostic

## 3. Manifest and report updates

- [x] 3.1 Extend the manifest schema to record before/after row counts and surviving CWE counts for each major filter stage
- [x] 3.2 Update the markdown report to show stage-by-stage filter counts, top CWEs by final row count, max-row group removals, and post-filter token-length summary
- [x] 3.3 Preserve the existing machine-readable manifest plus markdown report generation pattern and keep backward-compatible fields where practical

## 4. Downstream compatibility and tests

- [x] 4.1 Update or add tests covering all-labeled-CWE retention, exact-code dedup preservation, token filtering at the 512-token threshold, and max-only `(project, cwe_id)` filtering
- [x] 4.2 Review downstream code, manifests, and reports for assumptions that `func_subset.csv` contains only the curated training plus held-out CWE subset, and adjust callers or docs where needed
- [x] 4.3 Confirm `dvc.yaml` and related stage wiring do not require topology changes and document any additive manifest/report schema changes

## 5. Validation

- [x] 5.1 Run the relevant subset extraction tests and fix any regressions
- [x] 5.2 Run `python -m graph_vul.pipeline.func_extract_subset` on the current BigVul input and verify output files are produced successfully
- [x] 5.3 Validate that the generated manifest/report include stage counts, surviving CWE counts, top final CWEs, oversized-group removal counts, and token-length summary
