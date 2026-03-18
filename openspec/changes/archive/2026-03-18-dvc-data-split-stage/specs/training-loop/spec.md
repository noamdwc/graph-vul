## MODIFIED Requirements

### Requirement: Train/validation split
The training script SHALL load pre-computed split indices from `data/splits/train.json` and `data/splits/val.json` instead of performing an inline `train_test_split`. The split files SHALL be produced by the upstream `split_data` DVC stage.

#### Scenario: Stratified split
- **WHEN** the training script starts
- **THEN** it SHALL read `data/splits/train.json` and `data/splits/val.json`, and use those indices to construct the training and validation DataLoaders
