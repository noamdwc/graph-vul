## MODIFIED Requirements

### Requirement: Function baseline DVC stages
The `dvc.yaml` file SHALL define the function-level baseline pipeline around PrimeVul-backed artifacts. The function-level path SHALL include `func_extract_subset`, `func_split`, `func_finetune`, and `func_export_predictions`, and it SHALL remain compatible with the downstream `eval_external_tier1` stage.

#### Scenario: PrimeVul fine-tune pipeline executes independently
- **WHEN** `dvc repro func_export_predictions` is run
- **THEN** DVC executes the PrimeVul-backed `func_extract_subset`, `func_split`, `func_finetune`, and `func_export_predictions` stages in dependency order
- **AND** no BigVul-specific extraction step is required for that function-level path

#### Scenario: External evaluation remains downstream-compatible
- **WHEN** `dvc repro eval_external_tier1` is run after the PrimeVul fine-tune pipeline
- **THEN** the stage can consume the PrimeVul-trained checkpoint and exported predictions without additional manual artifact translation

### Requirement: Function baseline stage dependency graph
The function-level DVC stages SHALL encode the following dependency relationships:
- `func_split` depends on `func_extract_subset`
- `func_finetune` depends on `func_extract_subset` and `func_split`
- `func_export_predictions` depends on `func_finetune`, `func_extract_subset`, and `func_split`

#### Scenario: Stage ordering reflects PrimeVul artifacts
- **WHEN** `func_extract_subset` completes
- **THEN** `func_split` becomes eligible to run
- **AND** `func_finetune` remains blocked until both the normalized subset artifacts and split assignments are available

### Requirement: Function baseline metrics tracking
The PrimeVul-backed function fine-tuning stage SHALL declare `data/func_finetune_metrics.json` as a DVC metrics file with `cache: false`, and the prediction export stage SHALL write per-split prediction CSVs for later analysis.

#### Scenario: PrimeVul metrics visible in DVC
- **WHEN** the PrimeVul fine-tune stage completes
- **THEN** `dvc metrics show` includes the metrics from `data/func_finetune_metrics.json`
- **AND** prediction files are present under `data/analysis/func_finetune_predictions/`
