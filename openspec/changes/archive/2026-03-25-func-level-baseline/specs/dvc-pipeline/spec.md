## ADDED Requirements

### Requirement: Function baseline DVC stages
The `dvc.yaml` file SHALL define four additional stages: `func_extract_subset`, `func_embed`, `func_split`, and `func_train`. These stages SHALL be independent of the existing six graph pipeline stages.

#### Scenario: Independent execution
- **WHEN** `dvc repro func_train` is run
- **THEN** only the four `func_*` stages SHALL execute; no existing graph pipeline stages SHALL be triggered

#### Scenario: Full pipeline still works
- **WHEN** `dvc repro` is run without specifying a stage
- **THEN** both the existing graph pipeline stages and the new `func_*` stages SHALL execute

### Requirement: Function baseline stage dependency graph
The `func_*` stages SHALL encode the following dependency relationships:
- `func_embed` depends on `func_extract_subset`
- `func_split` depends on `func_extract_subset`
- `func_train` depends on `func_embed` and `func_split`

#### Scenario: Parallel-eligible stages
- **WHEN** `func_extract_subset` completes
- **THEN** `func_embed` and `func_split` SHALL be eligible to run, as they have no mutual dependency

### Requirement: Function baseline metrics tracking
The `func_train` stage SHALL declare `data/func_metrics.json` as a metrics file with `cache: false`, enabling `dvc metrics show` and `dvc metrics diff`.

#### Scenario: View function baseline metrics
- **WHEN** the user runs `dvc metrics show` after `func_train`
- **THEN** the output SHALL include metrics from `data/func_metrics.json`
