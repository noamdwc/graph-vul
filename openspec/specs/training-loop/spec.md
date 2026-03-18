### Requirement: Train/validation split
The training script SHALL load pre-computed split indices from `data/splits/train.json` and `data/splits/val.json` instead of performing an inline `train_test_split`. The split files SHALL be produced by the upstream `split_data` DVC stage.

#### Scenario: Stratified split
- **WHEN** the training script starts
- **THEN** it SHALL read `data/splits/train.json` and `data/splits/val.json`, and use those indices to construct the training and validation DataLoaders

### Requirement: Training loop with BCE loss
The training loop SHALL be managed by the Lightning Trainer rather than a manual for-loop. The `GraphVulLitModule.training_step` SHALL compute BCEWithLogitsLoss. Per-epoch train loss and validation metrics SHALL be logged via Lightning's `self.log()` mechanism, which routes to the configured loggers (including WandbLogger).

#### Scenario: Single epoch execution
- **WHEN** one training epoch completes
- **THEN** the Trainer SHALL have invoked `training_step` for each batch and the mean training loss SHALL be logged

### Requirement: Validation evaluation
After each training epoch, the Trainer SHALL run a validation loop calling `validation_step` on each batch without gradient computation. Validation metrics (accuracy, precision, recall, F1) SHALL be computed using `torchmetrics` and logged via `self.log()`.

#### Scenario: Validation metrics reported
- **WHEN** a validation pass completes
- **THEN** accuracy, precision, recall, and F1 SHALL be logged to all configured Lightning loggers

### Requirement: Model checkpoint saving
The Lightning `ModelCheckpoint` callback SHALL save a full Lightning checkpoint (model state, optimizer state, epoch) whenever `val_f1` improves. The checkpoint path SHALL be `data/best_model.ckpt`.

#### Scenario: Improved validation F1
- **WHEN** the current epoch's validation F1 exceeds the best recorded F1
- **THEN** a Lightning checkpoint SHALL be saved to `data/best_model.ckpt`

### Requirement: Reproducibility
The training script SHALL call `pytorch_lightning.seed_everything(seed)` to seed PyTorch, NumPy, and Python's random module, replacing the manual `seed_everything` function.

#### Scenario: Deterministic training
- **WHEN** two training runs use the same seed and data
- **THEN** they SHALL produce identical loss and metric sequences

### Requirement: W&B experiment logging
The training script SHALL initialize a W&B run, log all hyperparameters to the run config, and log training loss, validation loss, accuracy, precision, recall, and F1 score after each epoch.

#### Scenario: Metrics visible in W&B
- **WHEN** a training run completes
- **THEN** all per-epoch metrics SHALL be visible as time-series charts in the W&B run dashboard

### Requirement: Metrics JSON output
The training script SHALL write the best validation metrics (accuracy, precision, recall, F1) to a JSON file at a configurable path for DVC metrics tracking.

#### Scenario: Metrics file written
- **WHEN** training completes
- **THEN** a `data/metrics.json` file SHALL exist containing keys `accuracy`, `precision`, `recall`, `f1` with the best validation values

### Requirement: Load pre-computed embeddings
The training script SHALL load pre-computed code embeddings and CVE embeddings from disk rather than computing them inline. The `CPGDataset` class SHALL be refactored to accept paths to pre-computed embedding directories.

#### Scenario: Training with pre-computed embeddings
- **WHEN** the training stage runs
- **THEN** it SHALL read `.pt` files from `data/processed/code_embeddings/` and `data/processed/cve_embeddings/` without invoking DeBERTa
