## Why

The current pipeline trains and evaluates exclusively on BigVul. We have no signal on how the model generalises to functions from other vulnerability datasets with different collection methodologies, project distributions, and labelling conventions. Without external benchmarks, we cannot distinguish genuine vulnerability detection from dataset-specific pattern matching. PrimeVul and DiverseVul are two widely-used, publicly available alternatives that provide complementary coverage — PrimeVul is curated for label quality, DiverseVul for scale and CWE diversity.

## What Changes

- Add dataset adapters that load PrimeVul and DiverseVul from Hugging Face and normalise them into the same function-level schema used by the existing pipeline (`func_before`, `vul`, `cwe_id`, `sample_id`).
- Add inference logic that runs the trained BigVul checkpoint against each external dataset and saves per-dataset predictions.
- Add per-dataset metric computation (accuracy, precision, recall, F1, PR-AUC, ROC-AUC).
- Add a new DVC stage (`eval_external_tier1`) that orchestrates loading, inference, metrics, and reporting.
- Generate a markdown summary comparing BigVul test-set results with PrimeVul and DiverseVul results.

## Capabilities

### New Capabilities
- `external-dataset-adapter`: Adapter interface for loading and normalising external HuggingFace datasets into the pipeline's function-level schema.
- `external-eval-stage`: DVC stage that runs inference on external datasets using the trained checkpoint, computes metrics, saves predictions, and produces a comparison report.

### Modified Capabilities

_(none — the BigVul training pipeline is unchanged)_

## Impact

- **New files**: dataset adapters (≈1 module), external evaluation entry point (≈1 module), DVC stage definition.
- **Modified files**: `dvc.yaml` (new stage added).
- **Dependencies**: `datasets` (Hugging Face) added as a runtime dependency.
- **No changes** to training, existing evaluation, model architecture, or BigVul data pipeline.
