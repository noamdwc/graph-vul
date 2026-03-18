## 1. Dependencies & DVC Init

- [x] 1.1 Add `dvc` and `wandb` to `requirements.txt`
- [x] 1.2 Run `dvc init` and commit the `.dvc/` directory
- [x] 1.3 Update `.gitignore` to exclude `data/` artifacts while keeping `dvc.yaml`, `dvc.lock`, and `.dvc/` tracked

## 2. Extract Embedding Logic

- [x] 2.1 Create `graph_vul/pipeline/embed_utils.py` with shared DeBERTa embedding helper (load model, tokenize, mean-pool)
- [x] 2.2 Create `graph_vul/pipeline/embed_code.py` — standalone script that reads CPG exports from `data/raw/`, computes per-node DeBERTa embeddings, and writes `.pt` files (node features, edge_index, edge_type) to `data/processed/code_embeddings/`. Accept `--input` and `--output` CLI args. Skip existing outputs.
- [x] 2.3 Create `graph_vul/pipeline/embed_cve.py` — standalone script that reads `subset_cwe119.csv`, computes DeBERTa embeddings for CVE descriptions, and writes `.pt` files to `data/processed/cve_embeddings/`. Accept `--input` and `--output` CLI args. Use zero vector for missing descriptions. Skip existing outputs.

## 3. Parameterize Existing Pipeline Scripts

- [x] 3.1 Add `--input` and `--output` CLI arguments to `extract_subset.py` (defaulting to existing config paths)
- [x] 3.2 Add `--input` and `--output` CLI arguments to `joern_export.py` (defaulting to existing config paths)

## 4. Refactor CPGDataset for Pre-computed Embeddings

- [x] 4.1 Refactor `CPGDataset` in `cpg_dataset.py` to load pre-computed code and CVE embeddings from disk instead of running DeBERTa inline. Accept `code_embeddings_dir` and `cve_embeddings_dir` constructor parameters.

## 5. W&B Integration in Training

- [x] 5.1 Add `wandb.init()` at training start with project name `graph-vul` and hyperparameters (lr, hidden channels, epochs, batch size, seed) in config
- [x] 5.2 Log `train_loss`, `val_loss`, `accuracy`, `precision`, `recall`, `f1` to W&B via `wandb.log()` after each epoch
- [x] 5.3 Add `wandb.finish()` call on normal completion and in a `KeyboardInterrupt` handler
- [x] 5.4 Write best validation metrics to `data/metrics.json` at end of training for DVC metrics tracking

## 6. DVC Pipeline Definition

- [x] 6.1 Create `dvc.yaml` with the five stages (`extract_subset`, `joern_export`, `embed_code`, `embed_cve`, `train`) including correct `cmd`, `deps`, `outs`, and `metrics` declarations
- [x] 6.2 Verify dependency graph: `extract_subset` → `joern_export` → `embed_code` → `train`; `extract_subset` → `embed_cve` → `train`

## 7. Documentation

- [x] 7.1 Update project README or execution docs to document: `wandb login`, `dvc repro` workflow, and `dvc metrics show` usage
