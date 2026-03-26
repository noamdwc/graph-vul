## 1. Data Splitting

- [x] 1.1 Update `split_data.py` (or the func_split equivalent) to split held-out CWE samples into `unseen_val` and `unseen_test` using StratifiedGroupKFold with project as group
- [x] 1.2 Write unseen split assignments into the same `assignments.csv` with split values `unseen_val` and `unseen_test`
- [x] 1.3 Add validation: assert no project overlap between unseen_val and unseen_test, and no held-out CWE samples in train/val/test

## 2. Dataset

- [x] 2.1 Update `FuncFinetuneDataset` to accept `unseen_val` and `unseen_test` as valid split names (no code change needed if it already filters by split string — verify and test)

## 3. Training & Evaluation

- [x] 3.1 Update `train_finetune.py` to create `unseen_val` and `unseen_test` dataloaders after training
- [x] 3.2 Run `trainer.test()` on each unseen dataloader and collect metrics
- [x] 3.3 Update `func_finetune_lit_module.py` to support multiple test dataloaders and log metrics with `unseen_val_` and `unseen_test_` prefixes

## 4. Config & Pipeline

- [x] 4.1 Add any new config constants needed (e.g., unseen split ratio) in `config.py`
- [x] 4.2 Update `dvc.yaml` to reflect new deps/outs if needed (assignments.csv already tracked — verify unseen metrics flow into metrics file)

## 5. Verification

- [x] 5.1 Run `func_split` stage and verify `assignments.csv` contains unseen_val and unseen_test rows with correct CWE IDs
- [x] 5.2 ~~Run `func_finetune` stage and verify metrics~~ (skipped — will verify on GPU)
