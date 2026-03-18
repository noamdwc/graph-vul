## ADDED Requirements

### Requirement: Stratified three-way split
The `split_data` script SHALL split the dataset indices into train, validation, and test sets using stratified sampling on the `vul` label. The default ratio SHALL be 70% train, 15% validation, 15% test.

#### Scenario: Default split ratios
- **WHEN** `split_data` runs on a dataset with N samples
- **THEN** the train set SHALL contain ~70% of samples, validation ~15%, and test ~15%, with approximately equal label ratios in all three sets

#### Scenario: Stratification preserves label balance
- **WHEN** the dataset contains 50% vulnerable and 50% benign samples
- **THEN** each split (train, val, test) SHALL independently contain approximately 50% vulnerable and 50% benign samples

### Requirement: Deterministic split output
The split SHALL be seeded with a configurable random seed (default 42) so that identical inputs produce identical splits across runs.

#### Scenario: Reproducible splits
- **WHEN** `split_data` runs twice with the same seed and input CSV
- **THEN** the resulting train, val, and test index lists SHALL be identical

### Requirement: JSON index file output
The script SHALL write three JSON files — `train.json`, `val.json`, `test.json` — to an output directory (default `data/splits/`). Each file SHALL contain a sorted list of integer indices referencing rows in the subset CSV.

#### Scenario: Output files created
- **WHEN** `split_data` completes successfully
- **THEN** `data/splits/train.json`, `data/splits/val.json`, and `data/splits/test.json` SHALL exist and each SHALL be valid JSON containing a list of integers

### Requirement: CLI interface
The script SHALL accept `--input` (path to subset CSV), `--output` (output directory), and `--seed` (random seed) as command-line arguments and SHALL be runnable as `python -m graph_vul.pipeline.split_data`.

#### Scenario: Custom output directory
- **WHEN** `split_data` is invoked with `--output /tmp/splits`
- **THEN** the three JSON files SHALL be written to `/tmp/splits/`
