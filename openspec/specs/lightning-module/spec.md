### Requirement: LightningModule wrapper for GraphVulModel
The project SHALL provide a `GraphVulLitModule` class in `graph_vul/model/lit_module.py` that inherits from `pytorch_lightning.LightningModule` and composes an instance of `GraphVulModel`. The underlying `GraphVulModel` SHALL remain a plain `nn.Module` and SHALL NOT be modified.

#### Scenario: Module composition
- **WHEN** `GraphVulLitModule` is instantiated
- **THEN** it SHALL contain a `GraphVulModel` instance accessible as an attribute, and `GraphVulModel` SHALL NOT inherit from `LightningModule`

### Requirement: Training step
The `GraphVulLitModule` SHALL implement `training_step` that accepts a PyG `Batch` object, runs the forward pass through `GraphVulModel`, computes `BCEWithLogitsLoss`, logs the training loss, and returns the loss tensor.

#### Scenario: Single training step
- **WHEN** `training_step` is called with a batch
- **THEN** it SHALL return a scalar loss tensor and log `train_loss` via `self.log()`

### Requirement: Validation step
The `GraphVulLitModule` SHALL implement `validation_step` that accepts a PyG `Batch` object, runs the forward pass, computes predictions, and updates `torchmetrics` metric objects for accuracy, precision, recall, and F1 score.

#### Scenario: Single validation step
- **WHEN** `validation_step` is called with a batch
- **THEN** it SHALL update the module's torchmetrics metric instances with predictions and labels from that batch

### Requirement: Validation epoch end metric logging
The `GraphVulLitModule` SHALL compute and log validation metrics (accuracy, precision, recall, F1) at the end of each validation epoch using `self.log()`. The F1 metric SHALL be logged with the key `val_f1` for use by the `ModelCheckpoint` callback.

#### Scenario: End of validation epoch
- **WHEN** a validation epoch completes
- **THEN** `val_accuracy`, `val_precision`, `val_recall`, and `val_f1` SHALL be logged and the torchmetrics objects SHALL be reset

### Requirement: Optimizer configuration
The `GraphVulLitModule` SHALL implement `configure_optimizers` returning an `Adam` optimizer. The learning rate SHALL be configurable via a constructor parameter with a default from `config.LEARNING_RATE`.

#### Scenario: Optimizer setup
- **WHEN** the Lightning Trainer initializes optimizers
- **THEN** an Adam optimizer SHALL be created with the configured learning rate over all model parameters
