## Why

The current function-level subset builder still bakes split policy into subset construction by restricting BigVul to configured training and held-out CWE lists and by dropping low-support `(project, cwe_id)` groups. That makes the subset stage unsuitable as the clean post-cleanup universe needed for the next dataset-analysis and split-design steps.

## What Changes

- Refactor `graph_vul/pipeline/func_extract_subset.py` so subset construction keeps all labeled CWEs that survive the existing cleanup flow instead of filtering to `CWE_IDS` and `HELDOUT_CWE_IDS`.
- Remove the minimum `(project, cwe_id)` support rule from the function subset stage and retain only the existing maximum-size protection for oversized `(project, cwe_id)` buckets.
- Preserve the current exact-code deduplication, max token-length filter, metadata derivation, machine-readable manifest generation, markdown report generation, and overall filter ordering.
- Expand manifest and report content to make stage-by-stage row and CWE survival counts explicit, report `(project, cwe_id)` groups removed by the max-row filter, surface top final CWEs by row count, and summarize post-filter token-length distributions.
- Update configuration and pipeline documentation so subset extraction becomes split-agnostic, while later split selection remains responsible for training-vs-held-out CWE policy.

## Capabilities

### New Capabilities
- `split-agnostic-func-subset-reporting`: Stage-by-stage reporting for the function subset pipeline, including filter survival counts and final CWE/token summaries.

### Modified Capabilities
- `func-subset-sampling`: Change the subset stage from min/max `(project, cwe_id)` filtering to max-only filtering while preserving deduplication, token filtering, metadata derivation, and output generation.
- `multi-cwe-subset-sampling`: Remove the requirement that subset extraction filters to the configured union of `CWE_IDS` and `HELDOUT_CWE_IDS`; subset extraction now keeps the full labeled CWE universe and leaves split selection to later stages.

## Impact

- `graph_vul/pipeline/func_extract_subset.py` will change most substantially.
- `graph_vul/config.py` will remove or deprecate the minimum project-row setting for function subset extraction while preserving the max token and max `(project, cwe_id)` settings.
- `tests/test_extract_subset.py` and any function-subset-specific tests will need updated expectations around retained CWEs and small `(project, cwe_id)` groups.
- `openspec/specs/func-subset-sampling/spec.md` and `openspec/specs/multi-cwe-subset-sampling/spec.md` will require requirement updates; a new reporting capability spec will document the stronger manifest/report contract.
- Existing `dvc.yaml` stage wiring should remain stable, but downstream code and docs that assume `func_subset.csv` contains only a curated training plus held-out CWE subset must be reviewed.
