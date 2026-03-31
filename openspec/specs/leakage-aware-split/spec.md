### Requirement: Project-disjoint splits
The split stage SHALL assign every row to exactly one of `train`, `val`, `test`, `unseen_val`, or `unseen_test`. No project SHALL appear in more than one split. All rows from a given project SHALL be assigned to the same split.

#### Scenario: Zero project overlap across all pairs
- **WHEN** the split completes
- **THEN** the intersection of projects across any two of the five splits is empty

#### Scenario: All rows assigned
- **WHEN** the split completes
- **THEN** every row in the input subset has a split assignment in exactly one of the five splits

---

### Requirement: Held-out project selection for evaluation
The split stage SHALL designate `HELD_OUT_PROJECT_COUNT` projects as held-out. All rows from held-out projects SHALL be assigned to val or test (never train). Held-out projects SHALL be selected from projects with both vulnerable and benign samples and with sufficient row count for meaningful evaluation.

#### Scenario: Held-out projects excluded from train
- **WHEN** the split completes with HELD_OUT_PROJECT_COUNT=3
- **THEN** 3 projects appear only in val/test and have zero rows in train

#### Scenario: Held-out projects have both classes
- **WHEN** held-out projects are selected
- **THEN** each held-out project contains at least one vulnerable and one benign row

---

### Requirement: Val/test split of held-out projects
Unseen-pool projects SHALL be split into `unseen_val` and `unseen_test` by project, using `StratifiedGroupKFold(n_splits=2)` on the binary label. The default split is 50/50 by project count, configurable via `UNSEEN_VAL_FRAC`.

#### Scenario: Unseen splits are project-disjoint
- **WHEN** the split completes
- **THEN** no project appears in both `unseen_val` and `unseen_test`

#### Scenario: Unseen splits have held-out CWEs
- **WHEN** the split completes
- **THEN** both `unseen_val` and `unseen_test` contain rows with held-out CWE IDs

---

### Requirement: Remaining projects assigned via seen splits
All rows from projects NOT in the unseen pool SHALL be assigned to `train`, `val`, or `test` using `StratifiedGroupKFold`, stratified on the `vul` label and grouped by `project`. Default ratios: train ~80%, val ~10%, test ~10% (configurable via `VAL_SPLIT` and `TEST_SPLIT`).

#### Scenario: Seen pool split into three
- **WHEN** the split completes
- **THEN** `train`, `val`, and `test` contain only projects from the seen pool

#### Scenario: Approximate ratio achieved
- **WHEN** the split completes with default ratios
- **THEN** train contains roughly 80% of seen-pool rows, val ~10%, test ~10%

---

### Requirement: Stratification within constraints
The split stage SHALL attempt to preserve class ratio and metadata distribution across splits, subject to the project-disjoint constraint. The split SHALL NOT artificially force class balance.

#### Scenario: Class ratio approximately preserved
- **WHEN** the subset has 33% vulnerable rows
- **THEN** each split's vulnerable ratio is within a reasonable range of 33%, subject to project-level granularity

#### Scenario: Natural skew preserved
- **WHEN** the subset has imbalanced classes
- **THEN** the split does not resample or reweight to force 50/50 balance

---

### Requirement: Reproducible split with seed
The split stage SHALL use the configured `SEED` for all random operations. Given identical input and configuration, the split SHALL produce identical assignments.

#### Scenario: Deterministic split
- **WHEN** the split runs twice with the same seed and input
- **THEN** the assignments are identical

---

### Requirement: Split assignments output
The split stage SHALL produce a CSV file with columns `sample_id` and `split` (values: "train", "val", "test").

#### Scenario: Assignments file written
- **WHEN** the split completes
- **THEN** a CSV file exists at the configured path with one row per sample

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

---

### Requirement: Held-out CWE implies full project holdout
The split stage SHALL identify all projects that contain any row with a held-out CWE ID. ALL rows from those projects (regardless of their CWE) SHALL be moved to the unseen pool. No row from an unseen-pool project SHALL appear in `train`, `val`, or `test`.

#### Scenario: Project with mixed CWEs fully excluded from seen splits
- **WHEN** project-X has rows with CWE-119 (seen) and CWE-416 (held-out)
- **THEN** ALL rows from project-X (including CWE-119 rows) SHALL be in `unseen_val` or `unseen_test`
- **AND** zero rows from project-X SHALL appear in `train`, `val`, or `test`

#### Scenario: Unseen pool contains only projects touching held-out CWEs
- **WHEN** the split completes
- **THEN** every project in `unseen_val` or `unseen_test` SHALL have at least one row with a held-out CWE ID

---

### Requirement: Held-out CWE eligibility validation
Before splitting, the stage SHALL validate each configured held-out CWE against thresholds: `MIN_HELDOUT_TOTAL_ROWS`, `MIN_HELDOUT_VUL_ROWS`, `MIN_HELDOUT_PROJECTS`. If any held-out CWE fails a threshold, the stage SHALL raise an error with a clear message identifying the failing CWE and which threshold was violated.

#### Scenario: CWE too small rejected
- **WHEN** held-out CWE-999 has only 5 total rows and `MIN_HELDOUT_TOTAL_ROWS=30`
- **THEN** the stage SHALL fail with an error naming CWE-999 and the threshold

#### Scenario: CWE with insufficient projects rejected
- **WHEN** held-out CWE-999 appears in only 1 project and `MIN_HELDOUT_PROJECTS=2`
- **THEN** the stage SHALL fail with an error naming CWE-999 and the threshold

#### Scenario: All held-out CWEs eligible
- **WHEN** all held-out CWEs meet all thresholds
- **THEN** the stage SHALL proceed to splitting without error

---

### Requirement: Strict cross-regime overlap validation
The validation SHALL check project overlap across ALL 10 split pairs: train/val, train/test, val/test, unseen_val/unseen_test, train/unseen_val, train/unseen_test, val/unseen_val, val/unseen_test, test/unseen_val, test/unseen_test. ALL overlaps SHALL be zero.

#### Scenario: All pairs checked
- **WHEN** the validation runs
- **THEN** it SHALL check and report on all 10 split pairs

#### Scenario: Cross-regime overlap detected
- **WHEN** a project appears in both `train` and `unseen_val`
- **THEN** a validation warning SHALL be emitted naming the overlapping project

---

### Requirement: Seen regime excludes held-out CWEs
No row with a held-out CWE ID SHALL appear in `train`, `val`, or `test`. This is a consequence of full-project holdout but SHALL be independently verified.

#### Scenario: Held-out CWE absent from seen splits
- **WHEN** the split completes
- **THEN** zero rows in `train`, `val`, or `test` have a `cwe_id` matching any held-out CWE

---

### Requirement: Project exclusivity report
The split report SHALL include a "Project Exclusivity" section showing: zero-overlap verification for all 10 split pairs, count of projects in the seen pool, count of projects in the unseen pool, and which held-out CWEs caused each project to migrate to the unseen pool.

#### Scenario: Report shows migration reasons
- **WHEN** the report is generated
- **THEN** it SHALL list each unseen-pool project and the held-out CWE(s) that caused its migration
