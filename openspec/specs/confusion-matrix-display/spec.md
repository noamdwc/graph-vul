## ADDED Requirements

### Requirement: Per-split confusion matrix display
The notebook SHALL display a confusion matrix for each evaluation split (val, test, unseen_val, unseen_test — whichever are loaded). Each matrix SHALL show both raw counts and row-normalized (percentage) values.

#### Scenario: Confusion matrix for a loaded split
- **WHEN** a split is loaded with `y_true` and `y_pred` columns
- **THEN** the notebook displays a 2x2 confusion matrix heatmap with raw counts annotated, and a second heatmap with row-normalized percentages

#### Scenario: Split not loaded
- **WHEN** a split is not present in `frames`
- **THEN** no confusion matrix is displayed for that split and no error is raised

### Requirement: Confusion matrix placement
The confusion matrix section SHALL appear after the Aggregate Metrics section and before the Seen vs Unseen section.

#### Scenario: Section ordering
- **WHEN** the notebook is run end to end
- **THEN** confusion matrices appear immediately after aggregate metrics display
