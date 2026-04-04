## ADDED Requirements

### Requirement: DVC initialization
The repository SHALL be initialized with DVC (`dvc init`) and the resulting `.dvc/` directory SHALL be committed to version control.

#### Scenario: Fresh DVC init
- **WHEN** a developer clones the repository
- **THEN** the `.dvc/` configuration directory SHALL be present and DVC commands SHALL work without re-initialization

### Requirement: Six-stage DVC pipeline
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

### Requirement: DVC-tracked outputs
Each stage SHALL declare its output artifacts so DVC can track them. The `.gitignore` SHALL be updated to exclude DVC-tracked data directories (`data/`) while keeping `.dvc` and `dvc.yaml` / `dvc.lock` under version control.

#### Scenario: Git status after pipeline run
- **WHEN** `dvc repro` completes and the user runs `git status`
- **THEN** `data/` contents SHALL NOT appear as untracked files, but `dvc.lock` changes SHALL appear as modified

### Requirement: Extract subset stage outputs
The `extract_subset` DVC stage SHALL declare outputs for: subset CSV, subset manifest JSON, and subset report MD. It SHALL depend on the Big-Vul CSV and the extract_subset module.

#### Scenario: DVC tracks subset artifacts
- **WHEN** dvc.yaml is inspected
- **THEN** the extract_subset stage lists subset CSV, manifest JSON, and report MD as outputs

---

### Requirement: Split data stage dependencies
The `split_data` DVC stage SHALL depend on the subset CSV and the split_data module. It SHALL output the split assignments CSV and split report. It SHALL run after embedding stages in the pipeline.

#### Scenario: DVC tracks split artifacts
- **WHEN** dvc.yaml is inspected
- **THEN** the split_data stage lists assignments CSV and report as outputs

---

### Requirement: Downstream stages use updated subset path
All DVC stages that depend on the subset CSV (joern_export, embed_code, embed_cve) SHALL reference the new subset path (`data/subset.csv`) instead of the old path (`data/subset_cwe119.csv`).

#### Scenario: Old path not referenced
- **WHEN** dvc.yaml is inspected
- **THEN** no stage references `subset_cwe119.csv`

---

### Requirement: Metrics tracking via DVC
The `train` stage SHALL declare a metrics file (`data/metrics.json`) that DVC tracks, enabling `dvc metrics show` and `dvc metrics diff`.

#### Scenario: View metrics after training
- **WHEN** the user runs `dvc metrics show` after a training run
- **THEN** the output SHALL display the best validation accuracy, precision, recall, and F1 from the training run

---

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

---

### Requirement: Partition-by-length DVC stage
The `dvc.yaml` file SHALL define a `partition_by_length` stage that runs `python -m graph_vul.pipeline.partition_by_length`, depends on `data/func_subset.csv`, `graph_vul/pipeline/partition_by_length.py`, and `graph_vul/config.py`, and outputs `data/partitions/short/func_subset.csv` and `data/partitions/other/func_subset.csv`.

#### Scenario: Stage triggers on subset change
- **WHEN** `data/func_subset.csv` is updated and `dvc repro partition_by_length` is run
- **THEN** the partition stage SHALL re-execute and produce updated partition CSVs

---

### Requirement: Per-partition split DVC stages
The `dvc.yaml` file SHALL define two split stages:
- `func_split_short`: runs `split_data` on `data/partitions/short/func_subset.csv` with output `data/partitions/short/splits/`
- `func_split_other`: runs `split_data` on `data/partitions/other/func_subset.csv` with output `data/partitions/other/splits/`

Both stages SHALL depend on their respective partition CSV, `graph_vul/pipeline/split_data.py`, and `graph_vul/config.py`.

#### Scenario: Short partition split
- **WHEN** `dvc repro func_split_short` is run
- **THEN** it SHALL produce `data/partitions/short/splits/assignments.csv` and `data/partitions/short/splits/split_report.md`

#### Scenario: Other partition split
- **WHEN** `dvc repro func_split_other` is run
- **THEN** it SHALL produce `data/partitions/other/splits/assignments.csv` and `data/partitions/other/splits/split_report.md`

#### Scenario: Dependency chain
- **WHEN** `dvc repro func_split_short` is run and `partition_by_length` has not yet run
- **THEN** DVC SHALL first run `partition_by_length` before running `func_split_short`
