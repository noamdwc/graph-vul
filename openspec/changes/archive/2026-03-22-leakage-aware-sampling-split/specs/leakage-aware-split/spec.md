## ADDED Requirements

### Requirement: Project-disjoint splits
The split stage SHALL assign every row to exactly one of train, val, or test. No project SHALL appear in more than one split. All rows from a given project SHALL be assigned to the same split.

#### Scenario: Zero project overlap
- **WHEN** the split completes
- **THEN** the intersection of projects across any two splits is empty

#### Scenario: All rows assigned
- **WHEN** the split completes
- **THEN** every row in the input subset has a split assignment

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
Held-out project rows SHALL be divided between val and test splits, approximately equally by row count. If there are enough held-out projects, different projects SHALL go to different splits when feasible.

#### Scenario: Held-out rows distributed to val and test
- **WHEN** 3 projects are held out
- **THEN** some held-out project rows appear in val and some in test, with roughly balanced sizes

---

### Requirement: Remaining projects assigned to train
All rows from non-held-out projects SHALL be assigned to the train split.

#### Scenario: Train contains only seen projects
- **WHEN** the split completes
- **THEN** train contains rows exclusively from non-held-out projects

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
The split stage SHALL produce a Markdown report containing: split sizes, class counts and ratios per split, unique CVEs per split, unique projects per split, project overlap verification (must be zero), held-out project summary, metadata bucket distributions per split, and warnings if constraints are infeasible.

#### Scenario: Report written with overlap check
- **WHEN** the split completes
- **THEN** the report explicitly states that project overlap is zero and lists held-out projects

#### Scenario: Warning on skewed split
- **WHEN** a split has fewer than 5% of total rows or has only one class
- **THEN** the report includes a warning about the skewed split
