## 1. PrimeVul Data Preparation

- [ ] 1.1 Add a small PrimeVul training adapter module that loads official Hugging Face splits and normalizes rows into the current function-level schema
- [ ] 1.2 Update `graph_vul/pipeline/func_extract_subset.py` to use the PrimeVul adapter, preserve metadata/dedup/token filtering, and emit the canonical subset CSV, manifest, and report with PrimeVul provenance
- [ ] 1.3 Update or extend `graph_vul/config.py` with any minimal dataset-source constants needed for the PrimeVul-first path

## 2. Split Materialization

- [ ] 2.1 Extend `graph_vul/pipeline/split_data.py` so it can materialize `train`/`val`/`test` assignments from PrimeVul official split labels
- [ ] 2.2 Ensure the split report clearly documents official-split mode and the absence of unseen splits for PrimeVul

## 3. Training And Prediction Compatibility

- [ ] 3.1 Update `graph_vul/dataset/func_finetune_dataset.py` only as needed to consume the normalized PrimeVul subset without changing tokenization or label semantics
- [ ] 3.2 Update `graph_vul/train_finetune.py` so validation/test execution tolerates missing `unseen_val` and `unseen_test` splits
- [ ] 3.3 Update `graph_vul/export_finetune_predictions.py` so default export behavior skips unavailable unseen splits cleanly and still saves PrimeVul in-distribution predictions for later analysis

## 4. DVC And External Evaluation

- [ ] 4.1 Update `dvc.yaml` so the function-level pipeline stages (`func_extract_subset`, `func_split`, `func_finetune`, `func_export_predictions`) run end-to-end on PrimeVul-backed artifacts
- [ ] 4.2 Update `graph_vul/eval_external.py` and any external dataset registry usage so the PrimeVul-first baseline compares later against BigVul and DiverseVul rather than PrimeVul-as-external
- [ ] 4.3 Verify artifact names, manifests, and metrics clearly indicate PrimeVul as the training source without forcing a broad path refactor

## 5. Verification

- [ ] 5.1 Smoke-test PrimeVul loading and schema normalization, including label values, CWE normalization, and split mapping
- [ ] 5.2 Run the function-level DVC path through fine-tuning and prediction export on PrimeVul end-to-end
- [ ] 5.3 Confirm the resulting artifacts include PrimeVul train/val/test metrics plus saved predictions for later error analysis
