### Requirement: Deterministic split output
The split SHALL be seeded with a configurable random seed (default 42) so that identical inputs produce identical splits across runs.

#### Scenario: Reproducible splits
- **WHEN** `split_data` runs twice with the same seed and input CSV
- **THEN** the resulting train, val, and test index lists SHALL be identical

### Requirement: CLI interface
The script SHALL accept `--input` (path to subset CSV), `--output` (output directory), and `--seed` (random seed) as command-line arguments and SHALL be runnable as `python -m graph_vul.pipeline.split_data`.

#### Scenario: Custom output directory
- **WHEN** `split_data` is invoked with `--output /tmp/splits`
- **THEN** the split output files SHALL be written to `/tmp/splits/`

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
