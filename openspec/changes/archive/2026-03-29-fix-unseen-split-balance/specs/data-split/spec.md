## MODIFIED Requirements

### Requirement: Split assignments output
The split stage SHALL produce a CSV file with columns `sample_id` and `split` (values: `train`, `val`, `test`, `unseen_val`, `unseen_test`). Every input row SHALL appear exactly once. The unseen pool allocator SHALL optimize for project-count balance alongside row-count, vulnerability-ratio, and CWE-distribution balance.

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

---

## ADDED Requirements

### Requirement: Project-count-aware unseen allocation
The greedy allocator, when invoked for unseen splits, SHALL include a project-count balance penalty in its scoring function so that splits receive a balanced number of projects, not only balanced row counts.

#### Scenario: Project count penalty active for unseen splits
- **WHEN** `_greedy_project_split` is called with unseen split fractions
- **THEN** the scoring function SHALL penalize deviation from the target project count per split
- **AND** the penalty SHALL be weighted high enough to prevent extreme project-count imbalance

#### Scenario: Row balance still maintained
- **WHEN** unseen allocation completes
- **THEN** row-count balance SHALL still be optimized alongside project-count balance (not ignored)

---

### Requirement: Hard validation for unseen splits
The validation step SHALL enforce minimum thresholds on unseen split composition and fail early if they are not met.

#### Scenario: Minimum project count enforced
- **WHEN** the unseen pool has >= 20 projects
- **THEN** `unseen_val` SHALL have at least `MIN_UNSEEN_PROJECTS` projects (configurable, default 8)
- **AND** `unseen_test` SHALL have at least `MIN_UNSEEN_PROJECTS` projects
- **AND** if either threshold is not met, a validation warning SHALL be emitted

#### Scenario: Minimum vulnerable rows enforced
- **WHEN** unseen allocation completes
- **THEN** both `unseen_val` and `unseen_test` SHALL contain at least `MIN_UNSEEN_VUL_ROWS` vulnerable rows (configurable, default 10)
- **AND** if either threshold is not met, a validation warning SHALL be emitted

#### Scenario: Small unseen pool graceful handling
- **WHEN** the unseen pool has fewer than 20 projects
- **THEN** the minimum project count threshold SHALL be relaxed proportionally
- **AND** a warning SHALL note that the unseen pool is too small for ideal balance
