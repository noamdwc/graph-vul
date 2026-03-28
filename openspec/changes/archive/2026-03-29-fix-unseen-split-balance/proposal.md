## Why

The `unseen_val` split currently contains only ~7 projects because the greedy allocator optimizes for row-count targets, not project-count balance. A few large projects dominate the allocation, making `unseen_val` unstable and potentially unrepresentative of `unseen_test`. This undermines the reliability of unseen-domain validation metrics.

## What Changes

- Update the greedy allocator's scoring function to include a **project-count balance** penalty when splitting the unseen pool, so `unseen_val` and `unseen_test` get a more even share of projects — not just rows.
- Add **hard validation** for unseen splits: minimum project count, minimum vulnerable rows, and dominance warnings when a single project exceeds a threshold share.
- Extend the **split report** with unseen-specific diagnostics: project counts, per-project row breakdown, vulnerable/benign counts, and largest-project share.
- Save an **unseen audit CSV** with per-project, per-split statistics (total rows, vulnerable rows, benign rows, unique CWEs).

## Capabilities

### New Capabilities
- `unseen-split-audit`: Audit CSV and extended reporting for unseen split composition (project balance, dominance checks, per-project stats).

### Modified Capabilities
- `data-split`: Update the unseen pool allocator to optimize for project-count balance alongside row/vulnerability/CWE balance; add hard validation constraints for unseen splits.

## Impact

- **Code**: `graph_vul/pipeline/split_data.py` — `_assign_unseen_splits()`, `_greedy_project_split()`, `_validate_splits()`, `_generate_split_report()`.
- **Outputs**: New `unseen_audit.csv` artifact alongside existing split outputs.
- **Behavior**: Existing split results will change (unseen_val will have more projects, unseen_test fewer rows per project on average). DVC pipeline will need re-run.
