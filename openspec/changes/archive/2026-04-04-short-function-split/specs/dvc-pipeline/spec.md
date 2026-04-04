## ADDED Requirements

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
