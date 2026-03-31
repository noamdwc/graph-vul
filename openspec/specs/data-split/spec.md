### Requirement: Deterministic split output
The split SHALL be seeded with a configurable random seed (default 42) so that identical inputs produce identical splits across runs. This applies to all five splits and the train balancing artifacts.

#### Scenario: Reproducible splits
- **WHEN** `split_data` runs twice with the same seed and input CSV
- **THEN** the resulting `train`, `val`, `test`, `unseen_val`, and `unseen_test` assignments SHALL be identical
- **AND** `train_sampler_weights.csv` SHALL be identical
- **AND** `train_balanced_manifest.csv` SHALL be identical

---

### Requirement: CLI interface
The script SHALL accept `--input` (path to subset CSV), `--output` (output directory), and `--seed` (random seed) as command-line arguments and SHALL be runnable as `python -m graph_vul.pipeline.split_data`.

#### Scenario: Custom output directory
- **WHEN** `split_data` is invoked with `--output /tmp/splits`
- **THEN** the split output files (`assignments.csv`, `split_report.md`, `train_sampler_weights.csv`, `train_balanced_manifest.csv`) SHALL be written to `/tmp/splits/`

---

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

### Requirement: Held-out CWE routing to unseen splits
Rows belonging to held-out CWEs (defined by `HELDOUT_CWE_IDS` in config) and ALL other rows from projects containing held-out CWE rows SHALL be assigned exclusively to `unseen_val` or `unseen_test`. They SHALL NOT appear in `train`, `val`, or `test`.

#### Scenario: Held-out CWE row in unseen split
- **WHEN** a row has `cwe_id = "CWE-416"` (a held-out CWE)
- **THEN** it SHALL be assigned to `unseen_val` or `unseen_test`

#### Scenario: Same-project non-held-out CWE row also in unseen split
- **WHEN** a row has `cwe_id = "CWE-119"` (seen CWE) but its project also has CWE-416 rows
- **THEN** it SHALL be assigned to `unseen_val` or `unseen_test`, not to `train`/`val`/`test`

---

### Requirement: Training CWEs split normally
Rows belonging to training CWEs from projects NOT in the unseen pool SHALL be split across `train`/`val`/`test` using `StratifiedGroupKFold` logic, stratified on the `vul` label and grouped by `project`.

#### Scenario: Training CWE rows in seen splits
- **WHEN** the split is complete
- **THEN** `train`, `val`, and `test` SHALL each contain rows from the training CWEs

---

### Requirement: Split report marks held-out CWEs
The split diagnostics report SHALL include a section identifying which unseen split rows come from held-out CWEs versus other CWEs (co-located in unseen projects). It SHALL also show which projects were migrated to unseen and why.

#### Scenario: Report shows held-out breakdown
- **WHEN** the split report is generated
- **THEN** it SHALL show the count of unseen rows per CWE, distinguishing held-out CWEs from co-located CWEs

---

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

---

### Requirement: Split diagnostics report
The split stage SHALL produce a Markdown report containing: split sizes with vulnerable/benign counts and ratios for all five splits, per-CWE breakdown per split, CVE coverage per split, project coverage per split, project overlap verification for all 10 pairs, per-split project details, metadata bucket distributions, project exclusivity section, train balancing section, fixed token-count bucket distributions per split with vulnerable/benign breakdown, short-function evaluation viability per eval split, per-project short-bucket contributions in eval splits, and warnings.

#### Scenario: Report covers all five splits
- **WHEN** the report is generated
- **THEN** it SHALL include statistics for train, val, test, unseen_val, and unseen_test

#### Scenario: Warning on skewed split
- **WHEN** a split has fewer than 5% of total rows or has only one class
- **THEN** the report includes a warning

#### Scenario: Report includes fixed token bucket distributions
- **WHEN** the report is generated
- **THEN** it SHALL include a table showing per-split row counts for each fixed token bucket (0–64, 64–128, 128–256, 256–512, 512+) with vulnerable and benign sub-counts

#### Scenario: Report includes short-function viability assessment
- **WHEN** the report is generated
- **THEN** it SHALL include a section assessing whether each eval split has enough short-function samples for reliable evaluation, with pass/warn indicators
