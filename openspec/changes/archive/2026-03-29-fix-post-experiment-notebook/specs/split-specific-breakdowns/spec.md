## ADDED Requirements

### Requirement: Per-CWE metrics computed per split
The notebook SHALL compute and display per-CWE metrics separately for each evaluation split rather than on a merged eval_df. Each split's per-CWE table SHALL be saved as a separate CSV (e.g., `per_cwe_metrics_test.csv`).

#### Scenario: Multiple eval splits loaded
- **WHEN** test and unseen_test are both loaded with `cwe_id` column
- **THEN** the notebook displays separate per-CWE tables for each split and saves separate CSV files

#### Scenario: Single eval split loaded
- **WHEN** only one eval split is loaded
- **THEN** the notebook displays one per-CWE table for that split

### Requirement: Per-project metrics computed per split
The notebook SHALL compute and display per-project metrics (accuracy, F1, precision, recall, support, TP, FP, TN, FN) separately for each evaluation split. Each split's per-project table SHALL be saved as a separate CSV.

#### Scenario: Per-project table per split
- **WHEN** an eval split is loaded with a `project` column
- **THEN** the notebook displays per-project metrics for that split including accuracy, F1, precision, recall columns

#### Scenario: Per-project CSV output per split
- **WHEN** per-project metrics are computed for split `X`
- **THEN** a CSV is saved as `per_project_metrics_X.csv` in the output directory

### Requirement: Threshold analysis restricted to validation splits
The threshold sweep and threshold optimization SHALL use only `val` and `unseen_val` splits. The `test` and `unseen_test` splits SHALL NOT be used for threshold selection or sweep visualization. The tuned threshold MAY be applied to test splits for comparison display only.

#### Scenario: Threshold sweep uses val splits
- **WHEN** threshold analysis runs
- **THEN** sweep is computed on `val` and/or `unseen_val` only

#### Scenario: Tuned threshold applied to test for comparison
- **WHEN** a best threshold is found from val
- **THEN** it is applied to test and unseen_test to show comparison metrics, but the test data is not used to select the threshold

### Requirement: Score kind consistency
All cells that compare scores to thresholds SHALL respect the `effective_score_kind` variable. When `score_kind` is `logit`, threshold comparisons SHALL use `score_raw` (logit values). When `score_kind` is `prob`, threshold comparisons SHALL use the `score` column (probability values).

#### Scenario: Logit score handling
- **WHEN** `effective_score_kind` is `logit` and threshold sweep runs
- **THEN** thresholds are compared against `score_raw` column values

#### Scenario: Probability score handling
- **WHEN** `effective_score_kind` is `prob` or `None` and threshold sweep runs
- **THEN** thresholds are compared against `score` column values
