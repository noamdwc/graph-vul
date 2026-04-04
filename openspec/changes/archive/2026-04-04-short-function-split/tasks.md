## 1. Config

- [x] 1.1 Add `SHORT_FUNCTION_MAX_TOKENS = 64` to `graph_vul/config.py`

## 2. Partition Script

- [x] 2.1 Create `graph_vul/pipeline/partition_by_length.py` with CLI (`--input`, `--output-dir`), token counting, and two-way partition logic
- [x] 2.2 Add partition summary logging (row count, vul count, project count, CWE count per partition)
- [x] 2.3 Write unit tests for `partition_by_length` in `tests/test_partition_by_length.py`

## 3. Split Robustness

- [x] 3.1 Verify `split_data.py` handles input with no held-out CWE rows gracefully (skips unseen splits without error)
- [x] 3.2 If needed, add graceful fallback in `split_data.py` for missing held-out CWEs
- [x] 3.3 Add test for split_data with no held-out CWE rows in input

## 4. DVC Pipeline

- [x] 4.1 Add `partition_by_length` stage to `dvc.yaml` with correct deps and outs
- [x] 4.2 Add `func_split_short` stage to `dvc.yaml` (split_data on short partition)
- [x] 4.3 Add `func_split_other` stage to `dvc.yaml` (split_data on other partition)

## 5. Integration Validation

- [x] 5.1 Run `partition_by_length` on actual `func_subset.csv` and verify row counts and output files
- [x] 5.2 Run `func_split_short` and `func_split_other` and verify assignments and split reports are produced
- [x] 5.3 Verify `dvc repro` dependency chain works end-to-end for the partitioned flow
