## 1. Config changes

- [x] 1.1 Update `TARGET_VULNERABLE_SAMPLES` to 1200 and `TARGET_BENIGN_SAMPLES` to 1200 in `graph_vul/config.py`
- [x] 1.2 Add `EARLY_STOPPING_PATIENCE = 7` and `EARLY_STOPPING_MIN_DELTA = 0.005` to `graph_vul/config.py`

## 2. Balanced split logic

- [x] 2.1 In `graph_vul/pipeline/split_data.py`, after the project-disjoint split, undersample the majority class in train and val to achieve ~1:1 ratio while leaving test untouched
- [x] 2.2 Log the post-balance class counts for train, val, and test in the split report

## 3. Early stopping

- [x] 3.1 Add `EarlyStopping(monitor="val_f1", mode="max", patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA)` callback in `graph_vul/train.py`

## 4. Cleanup

- [x] 4.1 Verify `WeightedRandomSampler` still works correctly with balanced train split (no-op or near-no-op is fine)
- [x] 4.2 Run the pipeline end-to-end and confirm total samples ~3 000, balanced train/val, natural-ratio test, and early stopping triggers
