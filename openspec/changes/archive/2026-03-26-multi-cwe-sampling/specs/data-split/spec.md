## ADDED Requirements

### Requirement: Held-out CWE routing to test split
Rows belonging to held-out CWEs (defined by `HELDOUT_CWE_IDS` in config) SHALL be assigned exclusively to the test split. They SHALL NOT appear in train or val.

#### Scenario: Held-out CWE row assigned to test
- **WHEN** a row has `cwe_id = "CWE-416"` (a held-out CWE)
- **THEN** it SHALL be assigned to the "test" split

#### Scenario: Held-out CWE row never in train
- **WHEN** the split is complete
- **THEN** no row with a held-out CWE SHALL appear in the "train" or "val" splits

### Requirement: Training CWEs split normally
Rows belonging to training CWEs (`CWE_IDS`) SHALL be split across train/val/test using the existing `StratifiedGroupKFold` logic, stratified on the `vul` label and grouped by `project`.

#### Scenario: Training CWE rows in all splits
- **WHEN** the split is complete
- **THEN** train, val, and test SHALL each contain rows from the training CWEs

### Requirement: Split report marks held-out CWEs
The split diagnostics report SHALL include a section identifying which test rows come from held-out CWEs versus training CWEs.

#### Scenario: Report shows held-out breakdown
- **WHEN** the split report is generated
- **THEN** it SHALL show the count of test rows per CWE, distinguishing held-out CWEs from training CWEs
