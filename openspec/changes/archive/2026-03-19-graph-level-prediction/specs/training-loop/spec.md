## MODIFIED Requirements

### Requirement: Training loop with BCE loss
The `GraphVulLitModule.training_step` SHALL compute BCEWithLogitsLoss on graph-level logits `[batch_size]` against graph-level labels `[batch_size]`, instead of per-node loss.

#### Scenario: Single epoch execution
- **WHEN** one training epoch completes
- **THEN** the loss SHALL be computed on graph-level predictions against graph-level labels

### Requirement: Validation evaluation
Validation metrics (accuracy, precision, recall, F1) SHALL be computed directly on graph-level model output without post-hoc scatter aggregation from node predictions.

#### Scenario: Validation metrics reported
- **WHEN** a validation pass completes
- **THEN** metrics SHALL be computed directly on the model's graph-level output
