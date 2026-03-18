## Context

The current pipeline has five DVC stages ending with `train`. The train/val split is performed inline in `train.py` using `sklearn.model_selection.train_test_split` with `VAL_SPLIT=0.2`. There is no test set, so model evaluation only happens on validation data. The split is not versioned by DVC — changing split parameters doesn't invalidate the training cache correctly because the split logic is embedded in the training script.

## Goals / Non-Goals

**Goals:**
- Add a `split_data` DVC stage that produces deterministic, versioned train/val/test index files.
- Introduce a held-out test set for final evaluation.
- Make `train.py` consume pre-computed split indices instead of splitting inline.

**Non-Goals:**
- Adding a separate test/evaluation DVC stage (future work).
- Changing the split algorithm (stratified `train_test_split` is kept).
- K-fold cross-validation support.

## Decisions

### 1. Split output format: JSON index files

Write three JSON files (`train.json`, `val.json`, `test.json`) each containing a list of integer indices into the subset CSV.

**Why over CSV**: Indices are simple integer lists; JSON is human-readable and trivially loaded with `json.load()`. No need for pandas just to read splits.

**Why over parquet/feather**: Overkill for a list of integers.

### 2. Two-step stratified split (train → val+test → val, test)

Use `train_test_split` twice:
1. Split full dataset into train (70%) and temp (30%).
2. Split temp into val (15%) and test (15%).

All splits are stratified on the `vul` label column.

**Why over a single three-way split**: `sklearn` doesn't natively support stratified three-way splits. Two sequential calls is the standard pattern and keeps the code simple.

### 3. Stage placement in DVC graph

`split_data` depends on `data/subset_cwe119.csv` (the subset CSV) only — it doesn't need embeddings. It runs after `extract_subset` and before `train`. The `train` stage gains `data/splits/` as a dependency.

**Why not depend on embeddings**: The split operates on sample indices/labels, not on feature data. This lets it run in parallel with `embed_code` and `embed_cve`.

### 4. Config constants

Add `TEST_SPLIT = 0.15` to `config.py`. Adjust `VAL_SPLIT` to `0.15` (from `0.2`). Combined with training at 70%, this gives a 70/15/15 split.

## Risks / Trade-offs

- **Existing DVC cache invalidation** → Running `dvc repro` after this change will re-run `train` since its dependencies change. This is expected and correct.
- **Train.py must handle missing splits gracefully during transition** → The split stage must run before training. DVC enforces this via the dependency graph, so no special fallback is needed.
