## MODIFIED Requirements

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
The split stage SHALL produce a CSV file with columns `sample_id` and `split` (values: `train`, `val`, `test`, `unseen_val`, `unseen_test`). Every input row SHALL appear exactly once.

#### Scenario: Assignments file written
- **WHEN** the split completes
- **THEN** a CSV file exists at the configured path with one row per sample

#### Scenario: Five split values present
- **WHEN** held-out CWEs are configured and meet eligibility thresholds
- **THEN** the assignments file SHALL contain rows with all five split values

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
