## ADDED Requirements

### Requirement: Held-out CWE data split into unseen_val and unseen_test
The pipeline SHALL split samples belonging to held-out CWEs (CWE-200, CWE-416) into two disjoint sets: `unseen_val` and `unseen_test`. The split SHALL be project-disjoint (no project appears in both sets). The split SHALL be stratified by vulnerability label.

#### Scenario: Unseen splits are created during func_split
- **WHEN** the `func_split` pipeline stage runs
- **THEN** the `assignments.csv` file SHALL contain rows with split values `unseen_val` and `unseen_test` for held-out CWE samples

#### Scenario: Project disjointness in unseen splits
- **WHEN** unseen_val and unseen_test splits are created
- **THEN** no project SHALL appear in both `unseen_val` and `unseen_test`

#### Scenario: Unseen CWEs excluded from training splits
- **WHEN** the split is created
- **THEN** no sample with a held-out CWE id SHALL appear in `train`, `val`, or `test` splits

### Requirement: Model evaluation on unseen CWE splits
The training pipeline SHALL evaluate the trained model on `unseen_val` and `unseen_test` splits after training completes. Evaluation SHALL produce binary classification metrics (accuracy, precision, recall, F1) for each unseen split.

#### Scenario: Unseen metrics are computed after training
- **WHEN** the `func_finetune` stage completes training
- **THEN** the model SHALL be evaluated on both `unseen_val` and `unseen_test` dataloaders

#### Scenario: Unseen metrics are logged
- **WHEN** evaluation on unseen splits completes
- **THEN** the metrics file SHALL contain keys prefixed with `unseen_val_` and `unseen_test_` (e.g., `unseen_val_f1`, `unseen_test_f1`)

### Requirement: Dataset supports unseen split loading
The `FuncFinetuneDataset` SHALL accept `unseen_val` and `unseen_test` as valid split names and load the corresponding samples from `assignments.csv`.

#### Scenario: Loading unseen_val split
- **WHEN** `FuncFinetuneDataset` is instantiated with `split="unseen_val"`
- **THEN** it SHALL return only samples assigned to the `unseen_val` split

#### Scenario: Loading unseen_test split
- **WHEN** `FuncFinetuneDataset` is instantiated with `split="unseen_test"`
- **THEN** it SHALL return only samples assigned to the `unseen_test` split
