## MODIFIED Requirements

### Requirement: Five-stage DVC pipeline
The `dvc.yaml` file SHALL define six stages: `extract_subset`, `joern_export`, `embed_code`, `embed_cve`, `split_data`, and `train`. Each stage SHALL declare its command, dependencies, and outputs.

#### Scenario: Full pipeline execution
- **WHEN** `dvc repro` is run from the project root
- **THEN** all six stages SHALL execute in dependency order, skipping any stage whose dependencies have not changed since the last run

#### Scenario: Partial re-execution
- **WHEN** only the training script is modified
- **THEN** `dvc repro` SHALL re-run only the `train` stage and skip all upstream stages

### Requirement: Stage dependency graph
The stages SHALL encode the following dependency relationships:
- `joern_export` depends on `extract_subset`
- `embed_code` depends on `joern_export`
- `embed_cve` depends on `extract_subset`
- `split_data` depends on `extract_subset`
- `train` depends on `embed_code`, `embed_cve`, and `split_data`

#### Scenario: Parallel-eligible stages
- **WHEN** `extract_subset` completes
- **THEN** `joern_export`, `embed_cve`, and `split_data` SHALL be eligible to run, as they have no mutual dependency
