## 1. Configuration

- [x] 1.1 Add UniXcoder and function baseline constants to `config.py` (model name, embedding dim, paths, sampling params, MLP params, training params)

## 2. Embedding Utilities

- [x] 2.1 Add `load_unixcoder()` to `embed_utils.py` — load tokenizer and model from `microsoft/unixcoder-base`
- [x] 2.2 Add `embed_mean_pool()` to `embed_utils.py` — tokenize, encode, mean-pool over non-padding tokens, return 768-dim tensor

## 3. Subset Sampling

- [x] 3.1 Create `graph_vul/pipeline/func_extract_subset.py` — fork from `extract_subset.py` with project-size filter (50–2000 rows), min vulnerable filter (≥3), token-length filter (<513 whitespace tokens), CVE params (15–30), no top2 penalty
- [x] 3.2 Verify subset report: 24 projects, 2220 functions (342 vul, 1878 benign)

## 4. Function Embedding

- [x] 4.1 Create `graph_vul/pipeline/embed_functions.py` — iterate subset CSV, compute mean-pooled UniXcoder embeddings, save to `data/processed/func_embeddings/{sample_id}.pt`, skip existing
- [x] 4.2 Spot-check saved tensors have shape `[768]`

## 5. Dataset and Model

- [x] 5.1 Create `graph_vul/dataset/func_dataset.py` — `torch.utils.data.Dataset` loading `.pt` embeddings + labels, filtered by split assignments CSV
- [x] 5.2 Create `graph_vul/model/func_classifier.py` — `FuncBaselineModel` MLP: Linear(768,128) → ReLU → Dropout(0.3) → Linear(128,1)
- [x] 5.3 Create `graph_vul/model/func_lit_module.py` — Lightning module wrapping FuncBaselineModel with AdamW, LambdaLR, BCEWithLogitsLoss, metric logging

## 6. Training Script

- [x] 6.1 Create `graph_vul/train_baseline.py` — standard DataLoader, WeightedRandomSampler, ModelCheckpoint on val_f1, WandbLogger, trainer.test(), metrics to `data/func_metrics.json`
- [x] 6.2 Add CLI args for `--epochs`, `--batch-size`, `--lr` overriding config defaults

## 7. DVC Pipeline

- [x] 7.1 Add `func_extract_subset`, `func_embed`, `func_split`, `func_train` stages to `dvc.yaml` with correct deps/outs/metrics

## 8. End-to-End Verification

- [x] 8.1 Run `dvc repro func_train` end-to-end
- [x] 8.2 Verify W&B logs contain accuracy, precision, recall, F1 (val_f1=0.35, overfitting — to be addressed in follow-up)
- [x] 8.3 Verify split report: val has 1 project (gpac, 48 rows), test has 5 projects (606 rows). Project-disjoint confirmed.
