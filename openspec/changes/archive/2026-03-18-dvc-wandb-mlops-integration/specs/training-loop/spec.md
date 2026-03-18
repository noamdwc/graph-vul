## ADDED Requirements

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

## MODIFIED Requirements

### Requirement: Training loop with BCE loss
The training script SHALL run a configurable number of epochs, each consisting of a full pass over the training set using mini-batches. The loss function SHALL be Binary Cross-Entropy with logits (BCEWithLogitsLoss). Additionally, the training script SHALL log per-epoch train loss and validation metrics to W&B.

#### Scenario: Single epoch execution
- **WHEN** one training epoch completes
- **THEN** the script SHALL report the mean training loss for that epoch and log it to W&B

### Requirement: Validation evaluation
After each training epoch, the script SHALL evaluate the model on the validation set without gradient computation and report: accuracy, precision, recall, and F1 score. These metrics SHALL also be logged to W&B.

#### Scenario: Validation metrics reported
- **WHEN** a validation pass completes
- **THEN** accuracy, precision, recall, and F1 are printed to stdout for that epoch AND logged to W&B
