## MODIFIED Requirements

### Requirement: W&B run initialization
The training script SHALL configure a Lightning `WandbLogger` with project name `graph-vul`. All hyperparameters (learning rate, GAT hidden channels, number of epochs, batch size, random seed) SHALL be logged to the W&B run config via the logger. Manual calls to `wandb.init()` SHALL be removed.

#### Scenario: Training start with W&B
- **WHEN** the training script begins execution
- **THEN** a W&B run SHALL be created via `WandbLogger` with all hyperparameters visible in the run's config panel

### Requirement: Per-epoch metric logging
Lightning's `self.log()` calls in the `GraphVulLitModule` SHALL route metrics to the `WandbLogger`. The following metrics SHALL be logged after each epoch: training loss, validation accuracy, validation precision, validation recall, and validation F1.

#### Scenario: Epoch completion logging
- **WHEN** a training epoch and its validation pass complete
- **THEN** all metrics SHALL appear in the W&B run dashboard as time-series charts

### Requirement: W&B run cleanup
The `WandbLogger` SHALL handle run cleanup automatically when the Trainer finishes. Manual `wandb.finish()` calls SHALL be removed.

#### Scenario: Normal training completion
- **WHEN** all epochs finish
- **THEN** the W&B run SHALL be properly closed and appear as "finished" in the dashboard

#### Scenario: Training interruption
- **WHEN** training is interrupted (e.g., KeyboardInterrupt)
- **THEN** the W&B run SHALL still be properly closed by the Trainer's shutdown logic
