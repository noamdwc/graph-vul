## 1. Config updates

- [x] 1.1 Add `TEST_SPLIT = 0.15` to `graph_vul/config.py` and update `VAL_SPLIT` from `0.2` to `0.15`
- [x] 1.2 Add `SPLITS_DIR = DATA_DIR / "splits"` path constant to `graph_vul/config.py`

## 2. Split data script

- [x] 2.1 Create `graph_vul/pipeline/split_data.py` with CLI (`--input`, `--output`, `--seed`) that performs two-step stratified split (70/15/15) and writes `train.json`, `val.json`, `test.json`

## 3. DVC pipeline

- [x] 3.1 Add `split_data` stage to `dvc.yaml` depending on `data/subset_cwe119.csv` and `graph_vul/pipeline/split_data.py`, outputting `data/splits/`
- [x] 3.2 Add `data/splits/` as a dependency of the `train` stage in `dvc.yaml`

## 4. Training script update

- [x] 4.1 Update `train.py` to load split indices from `data/splits/train.json` and `data/splits/val.json` instead of calling `train_test_split` inline
- [x] 4.2 Remove the `sklearn.model_selection.train_test_split` import and inline split logic from `train.py`
