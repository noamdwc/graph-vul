## ADDED Requirements

### Requirement: DVC stage definition
A new DVC stage `eval_external_tier1` SHALL be added to `dvc.yaml`. It MUST depend on the trained checkpoint, CWE embeddings, the evaluation script, and the adapter module. It MUST produce an output directory containing predictions, metrics, and a summary report.

#### Scenario: Stage runs after training
- **WHEN** `dvc repro eval_external_tier1` is executed and the checkpoint exists
- **THEN** the stage loads each external dataset, runs inference, and writes outputs to `data/analysis/external_eval_tier1/`

#### Scenario: Stage skipped when checkpoint unchanged
- **WHEN** `dvc repro` is run and neither the checkpoint nor evaluation code has changed
- **THEN** the stage is skipped (cached)

### Requirement: Per-dataset inference
The evaluation script SHALL run inference on each registered external dataset independently, using the same trained checkpoint and tokenizer. Graph features MUST be disabled (`use_graph=False`) for external datasets.

#### Scenario: Inference on PrimeVul
- **WHEN** the evaluation script processes PrimeVul
- **THEN** it loads the PrimeVul adapter output, tokenizes with UniXcoder (max_length=512), runs the model in eval mode with `torch.inference_mode()`, and collects logits, probabilities, and binary predictions

#### Scenario: Inference on DiverseVul
- **WHEN** the evaluation script processes DiverseVul
- **THEN** it loads the DiverseVul adapter output, tokenizes with UniXcoder (max_length=512), runs the model in eval mode with `torch.inference_mode()`, and collects logits, probabilities, and binary predictions

### Requirement: Per-dataset prediction files
The script SHALL save a CSV file per dataset at `data/analysis/external_eval_tier1/<dataset>_predictions.csv` containing at minimum: `sample_id`, `cwe_id`, `vul` (ground truth), `y_pred` (predicted label), `prob` (predicted probability), `logit` (raw logit), and `token_length`.

#### Scenario: Prediction file contents
- **WHEN** inference completes for a dataset
- **THEN** a CSV is written with one row per sample and the required columns

### Requirement: Per-dataset metrics
The script SHALL compute and save per-dataset metrics as JSON at `data/analysis/external_eval_tier1/<dataset>_metrics.json`. Metrics MUST include: `accuracy`, `precision`, `recall`, `f1`, `pr_auc`, and `roc_auc`.

#### Scenario: Metrics computation
- **WHEN** predictions are available for a dataset
- **THEN** metrics are computed using sklearn and saved as a JSON object with keys `accuracy`, `precision`, `recall`, `f1`, `pr_auc`, `roc_auc`, plus `num_samples`, `num_vulnerable`, and `num_benign`

### Requirement: Combined metrics file
A combined `metrics.json` SHALL be written at `data/analysis/external_eval_tier1/metrics.json` with top-level keys per dataset (e.g., `primevul`, `diversevul`), each containing the metric values. This file is DVC-tracked as a metrics file.

#### Scenario: DVC metrics tracking
- **WHEN** the stage completes
- **THEN** `dvc metrics show` displays the external evaluation metrics from `metrics.json`

### Requirement: Markdown summary report
The script SHALL generate `data/analysis/external_eval_tier1/summary.md` containing a comparison table with rows for BigVul test, PrimeVul, and DiverseVul, and columns for each metric. BigVul test metrics SHALL be loaded from the existing predictions if available, or marked as N/A.

#### Scenario: Summary table format
- **WHEN** the summary is generated
- **THEN** it contains a markdown table with dataset names as rows and metrics (accuracy, precision, recall, F1, PR-AUC, ROC-AUC) as columns, plus sample counts

### Requirement: Unseen CWE logging
The script SHALL log the count and fraction of samples per dataset whose CWE ID does not have a pre-computed embedding (i.e., falls back to zero vector).

#### Scenario: CWE coverage reported
- **WHEN** inference runs on an external dataset
- **THEN** the log output includes the number and percentage of samples with unseen CWE IDs
