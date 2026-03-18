## ADDED Requirements

### Requirement: W&B run initialization
The training script SHALL initialize a W&B run at startup with project name `graph-vul` and log all hyperparameters (learning rate, GAT hidden channels, number of epochs, batch size, random seed) to the run config.

#### Scenario: Training start with W&B
- **WHEN** the training script begins execution
- **THEN** a W&B run SHALL be created with all hyperparameters visible in the run's config panel

### Requirement: Per-epoch metric logging
The training script SHALL log the following metrics to W&B after each epoch: training loss, validation loss, accuracy, precision, recall, and F1 score.

#### Scenario: Epoch completion logging
- **WHEN** a training epoch and its validation pass complete
- **THEN** all six metrics (train_loss, val_loss, accuracy, precision, recall, f1) SHALL be logged to W&B with the epoch number as the step

### Requirement: W&B run cleanup
The training script SHALL call `wandb.finish()` when training completes or is interrupted, ensuring the run is properly closed.

#### Scenario: Normal training completion
- **WHEN** all epochs finish
- **THEN** `wandb.finish()` SHALL be called and the run SHALL appear as "finished" in the W&B dashboard

#### Scenario: Training interruption
- **WHEN** training is interrupted (e.g., KeyboardInterrupt)
- **THEN** `wandb.finish()` SHALL still be called to avoid orphaned runs

### Requirement: W&B login prerequisite
The project documentation SHALL instruct users to run `wandb login` before their first training run.

#### Scenario: First-time setup
- **WHEN** a new developer reads the project setup instructions
- **THEN** they SHALL find a step instructing them to run `wandb login` and provide their API key
