## ADDED Requirements

### Requirement: Per-row sample weights for balanced training
The split stage SHALL produce a `train_sampler_weights.csv` file containing one row per training sample with columns: `sample_id`, `split`, `project`, `cwe_id`, `vul`, `sample_weight`. Only rows assigned to the `train` split SHALL appear in this file. Evaluation splits SHALL NOT be rebalanced.

#### Scenario: Weights file produced
- **WHEN** the split stage completes
- **THEN** `train_sampler_weights.csv` exists in the output directory with one row per training sample and all required columns

#### Scenario: Evaluation splits untouched
- **WHEN** the split stage completes
- **THEN** no rows from `val`, `test`, `unseen_val`, or `unseen_test` appear in the weights file

---

### Requirement: Configurable target vulnerable-to-benign ratio
The weight computation SHALL target a configurable effective vulnerable:benign ratio (default 1:3, i.e., 25% vulnerable). The weights SHALL be computed so that the weighted sum of vulnerable rows divided by total weighted sum approximates the target ratio.

#### Scenario: Default ratio targeting
- **WHEN** the balancing runs with default config (`target_vul_ratio=0.25`)
- **THEN** the effective weighted vulnerable ratio SHALL be within 5 percentage points of 25%

#### Scenario: Custom ratio
- **WHEN** `BALANCE_TARGET_VUL_RATIO` is set to 0.5
- **THEN** the effective weighted vulnerable ratio SHALL approximate 50%

---

### Requirement: CWE inverse-frequency smoothing for vulnerable rows
Vulnerable training rows SHALL receive a CWE-based weight factor proportional to `(N_vul_total / N_vul_cwe) ^ cwe_smoothing_strength`, where `cwe_smoothing_strength` is configurable (default 0.5). This flattens the vulnerable CWE distribution so rare CWEs get higher weight.

#### Scenario: Rare CWE upweighted
- **WHEN** CWE-A has 10 vulnerable rows and CWE-B has 100 vulnerable rows
- **THEN** CWE-A vulnerable rows SHALL have a higher per-row CWE weight factor than CWE-B rows

#### Scenario: Smoothing strength zero disables CWE reweighting
- **WHEN** `cwe_smoothing_strength=0`
- **THEN** all vulnerable rows SHALL have equal CWE weight factors regardless of CWE frequency

---

### Requirement: Project downweighting for benign rows
Benign training rows SHALL receive a project-based weight factor proportional to `(median_project_size / project_size) ^ project_downweight_strength`, where `project_downweight_strength` is configurable (default 0.3). This reduces dominance of oversized projects.

#### Scenario: Large project downweighted
- **WHEN** project-X has 5000 benign rows and median project size is 100
- **THEN** project-X benign rows SHALL have lower per-row project weight than median-sized projects

#### Scenario: Downweight strength zero disables project reweighting
- **WHEN** `project_downweight_strength=0`
- **THEN** all benign rows SHALL have equal project weight factors

---

### Requirement: Weight capping
All sample weights SHALL be capped at `max_weight` times the median weight (default `max_weight=10.0`). After capping, weights SHALL be renormalized so they sum to the number of training rows.

#### Scenario: Extreme weight capped
- **WHEN** a row's computed weight exceeds 10x the median weight
- **THEN** its weight SHALL be clamped to 10x the median weight before normalization

---

### Requirement: Optional max project share in train
When `BALANCE_MAX_PROJECT_SHARE` is set (default 0.15), no single project's total weight SHALL exceed that fraction of the total training weight. Rows from over-represented projects SHALL have their weights reduced proportionally.

#### Scenario: Giant project capped
- **WHEN** a project would contribute 30% of total training weight and max share is 15%
- **THEN** that project's rows SHALL have weights scaled down so their total is at most 15% of total weight

---

### Requirement: Materialized balanced manifest
The split stage SHALL also produce `train_balanced_manifest.csv` by sampling from training rows using the computed weights with a fixed seed. This file SHALL contain the same columns as `assignments.csv` plus `sample_weight`, and MAY contain duplicate `sample_id` values (oversampled rows).

#### Scenario: Manifest produced
- **WHEN** the split stage completes
- **THEN** `train_balanced_manifest.csv` exists in the output directory

#### Scenario: Manifest is deterministic
- **WHEN** the split runs twice with the same seed and input
- **THEN** `train_balanced_manifest.csv` SHALL be identical

---

### Requirement: Train balancing report section
The split report SHALL include a "Train Balancing" section showing: raw train vulnerable/benign ratio, balanced effective ratio, vulnerable CWE distribution before and after weighting, top project contribution before and after weighting, weight statistics (min, median, max, std), and any caps applied.

#### Scenario: Balancing section in report
- **WHEN** the split report is generated
- **THEN** it SHALL contain a "Train Balancing" section with before/after ratio comparison
