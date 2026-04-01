## MODIFIED Requirements

### Requirement: CLI interface
The split stage SHALL continue to accept `--input`, `--output`, and `--seed`, and it SHALL additionally support a mode that materializes assignments from official dataset split labels already present in the input subset.

#### Scenario: Official split mode
- **WHEN** `split_data` is invoked on a normalized subset containing official source split labels
- **THEN** it writes `assignments.csv` from those labels instead of deriving new train/validation/test partitions

### Requirement: Split assignments output
The split stage SHALL produce a CSV file with columns `sample_id` and `split`. For PrimeVul-backed official-split mode, the allowed split values SHALL be `train`, `val`, and `test`, and every input row SHALL appear exactly once.

#### Scenario: PrimeVul assignments file written
- **WHEN** the split stage completes in official-split mode
- **THEN** `assignments.csv` contains one row per normalized PrimeVul sample
- **AND** each row is assigned to exactly one of `train`, `val`, or `test`

### Requirement: Split diagnostics report
The split stage SHALL write the standard split artifacts and SHALL explain whether the splits were derived by the BigVul allocator or copied from official dataset partitions. When official splits are used, the report SHALL note that unseen splits are intentionally absent.

#### Scenario: Report documents official PrimeVul splits
- **WHEN** the PrimeVul-backed split stage completes
- **THEN** the split report states that assignments came from PrimeVul's official split definition
- **AND** it documents the train/val/test counts after normalization filters
