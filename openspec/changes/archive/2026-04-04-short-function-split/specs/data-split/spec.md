## MODIFIED Requirements

### Requirement: Split assignments output
The split stage SHALL produce a CSV file with columns `sample_id` and `split` (values: `train`, `val`, `test`, `unseen_val`, `unseen_test`). Every input row SHALL appear exactly once. The unseen pool allocator SHALL optimize for project-count balance alongside row-count, vulnerability-ratio, and CWE-distribution balance. When the input subset lacks rows for any held-out CWE, the split stage SHALL skip unseen split creation and produce only `train`, `val`, and `test` splits without error.

#### Scenario: Assignments file written
- **WHEN** the split completes
- **THEN** a CSV file exists at the configured path with one row per sample

#### Scenario: Five split values present
- **WHEN** held-out CWEs are configured and meet eligibility thresholds
- **THEN** the assignments file SHALL contain rows with all five split values

#### Scenario: Unseen splits have balanced project counts
- **WHEN** the unseen pool contains at least 20 projects
- **THEN** `unseen_val` SHALL contain at least 8 projects
- **AND** `unseen_test` SHALL contain at least 8 projects

#### Scenario: No held-out CWE rows in input
- **WHEN** the input subset contains zero rows for any configured held-out CWE
- **THEN** the split stage SHALL produce only `train`, `val`, and `test` splits
- **AND** no error SHALL be raised
- **AND** the split report SHALL note that unseen splits were skipped due to insufficient held-out CWE data
