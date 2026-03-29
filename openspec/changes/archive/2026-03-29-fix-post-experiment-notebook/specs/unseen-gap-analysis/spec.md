## ADDED Requirements

### Requirement: Unseen split comparison section
The notebook SHALL include a dedicated section comparing unseen_val and unseen_test when both splits are available. This section SHALL display side-by-side aggregate metrics (accuracy, precision, recall, F1, support, positive_rate) for both unseen splits.

#### Scenario: Both unseen splits present
- **WHEN** both `unseen_val` and `unseen_test` are loaded in `frames`
- **THEN** the notebook displays a comparison table with aggregate metrics for both splits side by side

#### Scenario: Only one unseen split present
- **WHEN** only one of `unseen_val` or `unseen_test` is loaded
- **THEN** the section prints a skip message and does not error

### Requirement: Per-project FP diagnostics in unseen splits
The notebook SHALL compute per-project metrics separately for unseen_val and unseen_test and highlight projects with the highest false positive counts and rates in each split.

#### Scenario: Per-project FP table for unseen_val
- **WHEN** unseen_val is loaded and has a `project` column
- **THEN** the notebook displays per-project metrics (accuracy, precision, recall, F1, support, TP, FP, TN, FN) sorted by FP count descending

#### Scenario: Per-project FP table for unseen_test
- **WHEN** unseen_test is loaded and has a `project` column
- **THEN** the notebook displays per-project metrics sorted by FP count descending

### Requirement: Project concentration analysis
The notebook SHALL show how concentrated each unseen split is — what fraction of total rows comes from the top-3 and top-5 projects.

#### Scenario: Concentration stats displayed
- **WHEN** both unseen splits are loaded with a `project` column
- **THEN** the notebook displays concentration metrics (top-3 and top-5 row share) for each unseen split

### Requirement: Class distribution comparison
The notebook SHALL compare the positive (vulnerable) rate per project across unseen_val and unseen_test to identify whether class imbalance differences explain the precision gap.

#### Scenario: Positive rate comparison
- **WHEN** both unseen splits are loaded with `project` and `y_true` columns
- **THEN** the notebook displays a table showing positive_rate per project per split, sorted by the difference in positive rate
