## Why

The current split pipeline uses `pd.qcut`-based length buckets (short/medium/long/very_long) that produce data-dependent bin edges. This makes it impossible to guarantee stable, meaningful evaluation on short functions (0-64 tokens) because: (1) the bucket boundaries shift with each dataset, (2) the greedy allocator only softly matches overall bucket proportions without ensuring minimum counts in any specific bucket, and (3) val/test may end up with too few short-function positives or negatives for reliable metrics. Since short functions are a critical evaluation slice for vulnerability detection, we need explicit fixed-boundary token buckets and allocation-time awareness of short-function evaluation viability.

## What Changes

- Replace `pd.qcut`-based length bucketing with fixed token-count buckets: 0-64, 64-128, 128-256, 256-512, 512+ (the last bucket is for reporting only).
- Add configurable minimum short-bucket targets (both vulnerable and benign) for val, test, unseen_val, and unseen_test.
- Extend `_build_project_stats()` to track per-project short-bucket vulnerable/benign counts.
- Modify the greedy allocator scoring function to include a short-bucket deficit penalty that strongly prefers assignments satisfying minimum short-function evaluation targets.
- Add post-split validation checks: per-split x per-bucket x per-label count tables, explicit warnings when short-bucket counts are too small for reliable metrics.
- Extend `split_report.md` with fixed-bucket distribution tables and per-split short-bucket positive/negative counts.
- Add fallback behavior with clear warnings when project-level grouping makes it impossible to satisfy short-bucket minimums.

## Capabilities

### New Capabilities
- `short-function-eval-buckets`: Fixed token-count bucketing (0-64, 64-128, 128-256, 256-512, 512+) replacing qcut-based buckets, with per-project short-bucket stats and evaluation-viability validation.

### Modified Capabilities
- `data-split`: Greedy allocator gains short-bucket deficit penalty; split report gains fixed-bucket distribution tables and short-function evaluation diagnostics.

## Impact

- **Code**: `graph_vul/pipeline/split_data.py` (bucketing, project stats, greedy scorer, validation, report generation), `graph_vul/config.py` (new constants for bucket boundaries and minimums).
- **Outputs**: `split_report.md` will have new sections; `assignments.csv` format is unchanged; column `code_token_count_bucket` values change from qcut labels to fixed-boundary labels.
- **Downstream**: No changes required to training code. The `code_token_count_bucket` column is only used in split diagnostics, not in model training.
- **Dependencies**: None new. Removes reliance on `pd.qcut` for the primary bucketing (may keep for secondary diagnostics if desired).
