## Why

Short functions (0-64 tokens) exhibit significantly lower F1 on unseen splits compared to longer functions. These two populations have fundamentally different characteristics — short functions are often wrappers, single-statement guards, or trivial helpers, while longer functions carry richer semantic and structural signals. Training a single model on both forces the model to compromise between these regimes. Splitting the problem into two independent sub-problems (short vs. non-short) lets us evaluate, tune, and potentially train specialized models for each population.

## What Changes

- Add a pipeline stage that partitions the func_subset into two disjoint subsets based on the existing `code_token_count` column: **short** (0-64 tokens) and **other** (65+ tokens).
- Produce separate `assignments.csv` files for each partition so that downstream training and evaluation stages can operate on each independently.
- Run the existing `split_data` pipeline on each partition independently, producing per-partition train/val/test/unseen_val/unseen_test splits.
- Update the DVC pipeline to orchestrate the two-partition flow alongside (or replacing) the current single-partition flow.
- Provide per-partition post-experiment analysis so metrics are reported separately for each sub-problem.

## Capabilities

### New Capabilities
- `function-length-partition`: Pipeline stage that splits func_subset.csv into short (0-64 tokens) and other (65+ tokens) partitions, each with its own subset CSV.

### Modified Capabilities
- `data-split`: The split stage must be runnable on each partition independently, producing separate assignments and reports per partition.
- `dvc-pipeline`: DVC stages added for the two-partition flow (partition → split → train → eval per partition).

## Impact

- **New file**: `graph_vul/pipeline/partition_by_length.py` — partitions func_subset into short/other CSVs
- **Modified file**: `dvc.yaml` — new stages for partition, per-partition split, train, and eval
- **Modified file**: `graph_vul/pipeline/split_data.py` — must work correctly on smaller, filtered subsets (may need no changes if already robust)
- **Modified file**: `graph_vul/config.py` — partition threshold constant (`SHORT_FUNCTION_MAX_TOKENS = 64`)
- **Data outputs**: `data/partitions/short/func_subset.csv`, `data/partitions/other/func_subset.csv`, per-partition split directories
