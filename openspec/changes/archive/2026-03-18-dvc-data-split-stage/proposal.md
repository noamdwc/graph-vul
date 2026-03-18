## Why

The train/val split currently happens inside `train.py` at training time, with no test set at all. This means evaluation metrics are only computed on validation data, and the split is not reproducible as a standalone DVC stage. Adding a dedicated `split_data` DVC stage produces deterministic train/val/test index files that are versioned by DVC, enables proper held-out test evaluation, and decouples data splitting from training.

## What Changes

- Add a new `graph_vul/pipeline/split_data.py` script that reads the subset CSV, performs a stratified train/val/test split, and writes index files (JSON or CSV) to `data/splits/`.
- Add a `split_data` stage to `dvc.yaml` between the embedding stages and the `train` stage.
- Update `train.py` to load split indices from disk instead of calling `train_test_split` inline.
- Add `TEST_SPLIT` config constant alongside the existing `VAL_SPLIT`.

## Capabilities

### New Capabilities
- `data-split`: Deterministic stratified train/val/test splitting as a standalone DVC pipeline stage.

### Modified Capabilities
- `training-loop`: Training will consume pre-computed split indices instead of splitting data at runtime.
- `dvc-pipeline`: A new `split_data` stage is inserted between embeddings and training.

## Impact

- **Code**: New `graph_vul/pipeline/split_data.py`; modified `graph_vul/train.py`, `graph_vul/config.py`.
- **Pipeline**: `dvc.yaml` gains a `split_data` stage; `train` stage gains a dependency on `data/splits/`.
- **Data**: New `data/splits/` directory with train/val/test index files tracked by DVC.
- **Dependencies**: No new packages — `sklearn.model_selection.train_test_split` is already used.
