## 1. Dataset Adapters

- [x] 1.1 Create `graph_vul/dataset/external_datasets.py` with `load_primevul()` adapter that loads from HuggingFace, normalises to `[sample_id, func_before, vul, cwe_id]`, filters by 512-token limit, and normalises CWE IDs
- [x] 1.2 Add `load_diversevul()` adapter in the same module with the same normalisation contract
- [x] 1.3 Add `EXTERNAL_DATASETS` registry dict mapping `"primevul"` and `"diversevul"` to their loader functions

## 2. External Evaluation Dataset

- [x] 2.1 Create `ExternalEvalDataset` class (in `external_datasets.py` or a new file) that accepts a DataFrame, tokenizes with UniXcoder (max_length=512), loads CWE embeddings with zero-vector fallback, and returns the same item dict as `FuncFinetuneDataset`

## 3. Evaluation Entry Point

- [x] 3.1 Create `graph_vul/eval_external.py` with inference loop: load checkpoint, iterate registered datasets, run model in eval/inference_mode with `use_graph=False`, collect logits/probs/preds per dataset
- [x] 3.2 Add per-dataset prediction CSV export (`<dataset>_predictions.csv`) with columns: `sample_id`, `cwe_id`, `vul`, `y_pred`, `prob`, `logit`, `token_length`
- [x] 3.3 Add per-dataset metrics computation (accuracy, precision, recall, F1, PR-AUC, ROC-AUC) using sklearn, saved as `<dataset>_metrics.json`
- [x] 3.4 Write combined `metrics.json` with per-dataset keys for DVC metrics tracking
- [x] 3.5 Generate `summary.md` with markdown comparison table (BigVul test vs PrimeVul vs DiverseVul)
- [x] 3.6 Log unseen-CWE counts and fractions per dataset

## 4. DVC Integration

- [x] 4.1 Add `eval_external_tier1` stage to `dvc.yaml` with deps on checkpoint, CWE embeddings, eval script, and adapter module; outputs to `data/analysis/external_eval_tier1`; metrics file at `metrics.json`

## 5. Dependencies

- [x] 5.1 Ensure `datasets` (HuggingFace) is listed in project dependencies

## 6. Verification

- [x] 6.1 Dry-run: verify adapters load and return correct schema (smoke test with small slice)
- [x] 6.2 Run `dvc repro eval_external_tier1` end-to-end and confirm all output files are produced
