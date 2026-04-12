## Why

The current function-level pipeline conditions on the ground-truth CWE embedding but does not directly train the model to distinguish the correct CWE from plausible wrong CWEs. That leaves a gap in exactly the behavior we want to probe for short functions: whether the representation actually learns "this function matches this CWE" rather than only using CWE as extra classifier context.

## What Changes

- Add an optional CWE-matching supervision block to the existing function-level fine-tuning pipeline while keeping the current classifier head and binary vulnerability objective unchanged.
- Introduce shared projection heads for function and CWE embeddings, plus two auxiliary supervision modes: wrong-CWE negative classification and vulnerable-only triplet loss.
- Support a small configurable ablation matrix:
  - `E0`: baseline
  - `E1`: wrong-CWE negatives only
  - `E2`: triplet loss only
  - `E3`: wrong-CWE negatives + triplet loss
  - `E4`: same as `E3`, but with hard negatives from similar or same-family CWEs
- Keep the initial settings conservative and easy to disable: one wrong-CWE negative per positive, `lambda=0.1`, `margin=0.2`, and defaults that preserve baseline behavior.
- Extend evaluation/reporting so each ablation run emits comparable overall metrics, per-CWE metrics, Top-1/Top-3 CWE retrieval, unseen/external results, and fixed function-length slices (`0-64`, `64-128`, `128-256`, `256+`).

## Capabilities

### New Capabilities
- `cwe-matching-supervision`: Configurable auxiliary supervision for matching functions to CWEs using wrong-CWE negatives, shared projection heads, and vulnerable-only triplet loss.
- `cwe-matching-ablation-reporting`: Comparable experiment reporting for E0-E4 with retrieval metrics, per-CWE breakdowns, unseen/external outputs, and fixed function-length slices.

### Modified Capabilities
- `model-architecture`: `FuncFinetuneModel` and its Lightning/training wrappers expose optional matching-space parameters and outputs while preserving the existing classifier head and baseline path when disabled.
- `dvc-pipeline`: The function-level DVC flow can run named CWE-matching ablations and write per-variant metrics/reports without disrupting the current baseline pipeline.

## Impact

- **Primary files to modify**: `graph_vul/config.py`, `graph_vul/dataset/func_finetune_dataset.py`, `graph_vul/model/func_finetune_model.py`, `graph_vul/model/func_finetune_lit_module.py`, `graph_vul/train_finetune.py`, `graph_vul/export_finetune_predictions.py`, `graph_vul/eval_external.py`, `dvc.yaml`.
- **Likely new helper modules**: a small CWE-matching loss/sampling helper under `graph_vul/model/` or `graph_vul/dataset/`.
- **Training impact**: auxiliary loss terms and negative sampling are opt-in and configured so `E0` remains the exact baseline path.
- **Evaluation impact**: prediction exports and summaries need additional retrieval/ranking outputs and fixed length-slice reporting for both in-distribution and external runs.
