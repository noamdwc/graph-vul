## Context

The fine-tuned UniXcoder model shows degraded performance on short functions (0-64 tokens) on unseen splits. The current pipeline processes all functions as a single population through `func_extract_subset` → `func_split` → train → eval. The `code_token_count_bucket` column and short-function eval buckets already exist in the split stage, but training and evaluation treat all functions uniformly.

Current active pipeline flow (from `dvc.yaml`):
1. `func_extract_subset` → `data/func_subset.csv`
2. `func_split` → `data/func_splits/assignments.csv`
3. `func_finetune` (commented out, run manually) → checkpoint
4. `func_export_predictions` → prediction CSVs
5. `post_experiment_analysis` → notebook PDF

The `split_data.py` module already computes `code_token_count` (whitespace-split token count) and assigns `code_token_count_bucket` using fixed boundaries from config (`TOKEN_BUCKET_EDGES`).

## Goals / Non-Goals

**Goals:**
- Partition the func_subset into short (0-64 tokens) and other (65+ tokens) subsets before splitting.
- Run the existing split pipeline independently on each partition.
- Enable independent training and evaluation per partition.
- Keep the single-population flow intact as the default; the partitioned flow is an additional parallel path.

**Non-Goals:**
- Modifying the model architecture or training logic — each partition uses the same model code.
- Changing the split algorithm itself — `split_data.py` runs as-is on each partition.
- Building ensemble or routing logic to combine short/other model predictions — that's a future change.
- Modifying `func_extract_subset` — partitioning happens after subset extraction.

## Decisions

### 1. Partition as a standalone pipeline stage

**Decision**: Add a new script `graph_vul/pipeline/partition_by_length.py` that reads `func_subset.csv`, computes `code_token_count` if not present, and writes two CSVs: `data/partitions/short/func_subset.csv` and `data/partitions/other/func_subset.csv`.

**Rationale**: Clean separation of concerns. The partition stage is a simple filter — no need to modify `func_extract_subset` or `split_data`. Each downstream stage receives a standard func_subset CSV and works without modification.

**Alternatives considered**:
- Filtering inside `split_data.py` with a `--token-filter` flag → rejected; mixes partitioning concerns into the split logic and complicates the split report.
- Partitioning in the notebook after training → rejected; we need separate training runs, not just separate evaluation.

### 2. Token count computation

**Decision**: Reuse the same whitespace-split token counting already in `split_data.py` (line ~1395: `code.str.split().str.len()`). The partition script imports or reimplements this logic. Threshold: `code_token_count <= 64` → short, `> 64` → other.

**Rationale**: Consistent with existing `code_token_count_bucket` definitions. The `TOKEN_BUCKET_EDGES[1] = 64` boundary is already established.

### 3. DVC stage structure

**Decision**: Add these new DVC stages:
- `partition_by_length`: depends on `data/func_subset.csv`, outputs `data/partitions/{short,other}/func_subset.csv`
- `func_split_short`: runs `split_data` on `data/partitions/short/func_subset.csv` → `data/partitions/short/splits/`
- `func_split_other`: runs `split_data` on `data/partitions/other/func_subset.csv` → `data/partitions/other/splits/`

Training and eval stages per partition follow the same pattern but are left commented out (like existing `func_finetune`), to be uncommented when ready.

**Rationale**: Mirrors the existing `func_split` stage pattern. Each partition gets its own output directory tree. The existing single-population stages remain unchanged.

**Alternatives considered**:
- Parameterized DVC stages with `foreach` → cleaner but DVC foreach has quirks with path interpolation; explicit stages are more debuggable for two partitions.

### 4. Config constants

**Decision**: Add `SHORT_FUNCTION_MAX_TOKENS = 64` to `graph_vul/config.py`. The partition script uses this constant. Partition output paths are derived: `DATA_DIR / "partitions" / {partition_name}`.

**Rationale**: Single source of truth for the threshold. Matches `TOKEN_BUCKET_EDGES[1]`.

### 5. Split robustness on small partitions

**Decision**: The short-function partition may have very few projects or CWEs. If `split_data` raises errors due to insufficient data for stratified splitting, the partition stage should emit a warning and the split stage should gracefully degrade (e.g., skip unseen splits if held-out CWE rows are absent from the partition).

**Rationale**: The short partition may have only a few hundred rows. The split algorithm already has minimum thresholds (`MIN_SPLIT_ROWS`, `MIN_HELDOUT_TOTAL_ROWS`) — we rely on these existing guardrails.

## Risks / Trade-offs

- **[Risk] Short partition too small for held-out CWEs** → Mitigation: `split_data.py` already handles missing held-out CWEs gracefully (skips unseen splits). Validate with a dry run.
- **[Risk] Short partition has heavily imbalanced label distribution** → Mitigation: train balancing already handles class imbalance via `train_sampler_weights.csv`. Monitor metrics.
- **[Trade-off] Two parallel pipeline paths increase DVC complexity** → Acceptable; only 3 new stages, and the structure is repetitive/predictable.
- **[Trade-off] Duplicated data on disk (func_subset rows appear in partition CSVs)** → Acceptable; CSVs are small (tens of MB). Clarity over disk savings.
