## Why

The current function-level baseline is anchored to BigVul-specific extraction and split logic, which makes the main training pipeline inherit BigVul's noisier labels and project/CWE assumptions. We want a minimally invasive way to swap the baseline to PrimeVul first, so we can measure whether cleaner labels improve in-distribution performance while keeping the current model, tokenizer, metrics, DVC flow, and downstream comparison points intact.

## What Changes

- Replace the function-level training data source with PrimeVul loaded from Hugging Face, while keeping the current UniXcoder fine-tuning architecture, preprocessing, truncation, metrics, and reporting flow.
- Add a PrimeVul training adapter that normalizes PrimeVul rows into the same function-level schema currently consumed by `FuncFinetuneDataset` and related reporting/export code.
- Reuse PrimeVul's official train/validation/test split for the main baseline pipeline, instead of forcing the current BigVul-specific held-out-CWE split regime.
- Keep the function-level DVC stage names and downstream artifact contracts as stable as possible, so the existing train, prediction export, post-analysis, and later external-eval steps continue to work with minimal churn.
- Update external evaluation expectations so a PrimeVul-trained checkpoint can still be compared later on BigVul and DiverseVul without treating PrimeVul itself as an external benchmark.

## Capabilities

### New Capabilities
- `primevul-training-adapter`: Load PrimeVul from Hugging Face, normalize it into the function-level training schema, and materialize official split assignments for the existing baseline pipeline.

### Modified Capabilities
- `dvc-pipeline`: Function-level DVC stages use PrimeVul-backed normalized artifacts and the existing fine-tune/export/eval flow remains runnable end-to-end.
- `func-subset-sampling`: Function-level subset preparation becomes dataset-adapter-driven instead of assuming labeled BigVul rows as the only source.
- `data-split`: The split stage can materialize official dataset splits for the function-level pipeline instead of only deriving BigVul-style seen/unseen project splits.
- `external-eval-stage`: External evaluation for the PrimeVul-first baseline compares the trained checkpoint against BigVul and DiverseVul, not PrimeVul-as-external.

## Impact

- **Primary files to modify**: `dvc.yaml`, `graph_vul/config.py`, `graph_vul/pipeline/func_extract_subset.py`, `graph_vul/pipeline/split_data.py`, `graph_vul/dataset/func_finetune_dataset.py`, `graph_vul/train_finetune.py`, `graph_vul/export_finetune_predictions.py`, `graph_vul/eval_external.py`, `graph_vul/dataset/external_datasets.py`.
- **Primary new file**: a small PrimeVul adapter module under `graph_vul/dataset/` or `graph_vul/pipeline/`.
- **Data dependency change**: the main function-level baseline now depends on Hugging Face PrimeVul downloads instead of `data/bigvul.csv`.
- **Artifact behavior**: keep canonical function-level output paths where possible, but add PrimeVul provenance to manifests/reports and update any stale BigVul-specific cache/report naming that would cause ambiguity.
