## MODIFIED Requirements

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
