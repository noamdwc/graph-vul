## ADDED Requirements

### Requirement: Train/validation split
The training script SHALL split the dataset into training and validation sets using a stratified split (preserving label balance). The default split ratio SHALL be 80/20.

#### Scenario: Stratified split
- **WHEN** the dataset is loaded with N samples
- **THEN** the training set SHALL contain ~80% of samples and the validation set ~20%, with approximately equal label ratios in both splits

### Requirement: Training loop with BCE loss
The training script SHALL run a configurable number of epochs, each consisting of a full pass over the training set using mini-batches. The loss function SHALL be Binary Cross-Entropy with logits (BCEWithLogitsLoss).

#### Scenario: Single epoch execution
- **WHEN** one training epoch completes
- **THEN** the script SHALL report the mean training loss for that epoch

### Requirement: Validation evaluation
After each training epoch, the script SHALL evaluate the model on the validation set without gradient computation and report: accuracy, precision, recall, and F1 score.

#### Scenario: Validation metrics reported
- **WHEN** a validation pass completes
- **THEN** accuracy, precision, recall, and F1 are printed to stdout for that epoch

### Requirement: Model checkpoint saving
The training script SHALL save the model state dict to disk whenever the validation F1 score improves over the previous best.

#### Scenario: Improved validation F1
- **WHEN** the current epoch's validation F1 exceeds the best recorded F1
- **THEN** the model checkpoint SHALL be saved to a configured path and the best F1 SHALL be updated

### Requirement: Reproducibility
The training script SHALL accept a random seed parameter and use it to seed PyTorch, NumPy, and Python's random module for reproducible runs.

#### Scenario: Deterministic training
- **WHEN** two training runs use the same seed and data
- **THEN** they SHALL produce identical loss and metric sequences
