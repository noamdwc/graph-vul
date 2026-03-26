## Why

Training currently runs for all 50 epochs with no early stopping, wasting compute and risking overfitting past the best validation checkpoint. Additionally, the train and val splits preserve the natural (imbalanced) class distribution—only the DataLoader's `WeightedRandomSampler` balances classes at batch level. We want balanced class counts baked into the train and val subsets themselves, while keeping test at its natural distribution so evaluation reflects real-world prevalence. The total sample budget should remain ~3 000.

## What Changes

- **Add early stopping** on `val_f1` to the Trainer callbacks, halting training when validation F1 stops improving.
- **Change subset sampling** so that train and val splits contain roughly equal numbers of vulnerable and benign samples, while the test split retains its natural class ratio.
- Adjust sampling targets so the total across train + val + test stays around 3 000 samples.

## Capabilities

### New Capabilities
- `early-stopping`: EarlyStopping callback configuration monitoring `val_f1` with patience and min-delta parameters.

### Modified Capabilities
- `structured-subset-sampling`: Sampling targets change so train+val are class-balanced while test keeps natural ratio. Total budget stays ~3 000.

## Impact

- `graph_vul/train.py` — add `EarlyStopping` callback to Trainer.
- `graph_vul/pipeline/split_data.py` / `graph_vul/config.py` — adjust sampling targets and split logic for balanced train/val and natural-ratio test.
- `openspec/specs/structured-subset-sampling/spec.md` — spec update for new sampling requirements.
- Training runs will terminate earlier, reducing compute cost.
