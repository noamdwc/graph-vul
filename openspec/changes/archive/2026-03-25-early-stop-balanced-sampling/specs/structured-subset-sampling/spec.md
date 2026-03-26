## MODIFIED Requirements

### Requirement: Configurable sampling targets
The subset sampler SHALL accept `target_vulnerable_samples` and `target_benign_samples` as configuration parameters. The sampler SHALL support equal counts for class-balanced sampling. Default values SHALL be defined in `config.py` with `TARGET_VULNERABLE_SAMPLES=1200` and `TARGET_BENIGN_SAMPLES=1200`, producing ~2 400 samples pre-split (~3 000 after accounting for caps and availability).

#### Scenario: Equal class sampling with 1:1 ratio
- **WHEN** target_vulnerable_samples=1200 and target_benign_samples=1200
- **THEN** the sampler produces approximately 1200 vulnerable and 1200 benign rows, totaling ~2400

#### Scenario: Infeasible benign target
- **WHEN** the requested benign count exceeds available benign rows for selected CVEs
- **THEN** the sampler samples all available benign rows, logs a warning with requested vs achieved counts, and completes without error

## ADDED Requirements

### Requirement: Balanced train and val splits
The split step SHALL undersample the majority class within the train and val splits to achieve approximately 1:1 vulnerable-to-benign ratio. The test split SHALL retain all assigned rows without rebalancing.

#### Scenario: Train split is class-balanced
- **WHEN** the project-disjoint split assigns rows to train
- **THEN** the train split contains approximately equal numbers of vulnerable and benign samples (within 5% tolerance)

#### Scenario: Val split is class-balanced
- **WHEN** the project-disjoint split assigns rows to val
- **THEN** the val split contains approximately equal numbers of vulnerable and benign samples (within 5% tolerance)

#### Scenario: Test split preserves natural ratio
- **WHEN** the project-disjoint split assigns rows to test
- **THEN** the test split retains all assigned rows without class rebalancing

### Requirement: Total sample budget
The total number of samples across train, val, and test splits SHALL be approximately 3 000 (±300). The undersampling in train/val reduces the effective count but the pre-split sampling targets SHALL be set to compensate.

#### Scenario: Total samples within budget
- **WHEN** sampling and splitting complete
- **THEN** the sum of train + val + test rows is between 2 700 and 3 300
