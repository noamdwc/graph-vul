### Requirement: Early stopping on validation F1
The Trainer SHALL include an `EarlyStopping` callback that monitors `val_f1` in `max` mode. Training SHALL halt when `val_f1` has not improved by at least `EARLY_STOPPING_MIN_DELTA` for `EARLY_STOPPING_PATIENCE` consecutive validation epochs.

#### Scenario: Training stops after patience exhausted
- **WHEN** `val_f1` does not improve by more than `EARLY_STOPPING_MIN_DELTA` for `EARLY_STOPPING_PATIENCE` consecutive epochs
- **THEN** the Trainer stops training early and the best checkpoint (saved by `ModelCheckpoint`) is the final model artifact

#### Scenario: Training continues while F1 improves
- **WHEN** `val_f1` improves by at least `EARLY_STOPPING_MIN_DELTA` within the patience window
- **THEN** training continues up to `NUM_EPOCHS`

---

### Requirement: Configurable early stopping parameters
`config.py` SHALL define `EARLY_STOPPING_PATIENCE` (default 7) and `EARLY_STOPPING_MIN_DELTA` (default 0.005). The Trainer SHALL read these values from config.

#### Scenario: Default early stopping config
- **WHEN** no overrides are provided
- **THEN** `EARLY_STOPPING_PATIENCE` is 7 and `EARLY_STOPPING_MIN_DELTA` is 0.005

---

### Requirement: Early stopping coexists with ModelCheckpoint
The `EarlyStopping` callback SHALL NOT interfere with the existing `ModelCheckpoint` callback. The best checkpoint SHALL still be saved based on `val_f1` regardless of when training stops.

#### Scenario: Best checkpoint preserved on early stop
- **WHEN** training stops early at epoch 20 and the best `val_f1` was at epoch 15
- **THEN** the saved checkpoint corresponds to epoch 15
