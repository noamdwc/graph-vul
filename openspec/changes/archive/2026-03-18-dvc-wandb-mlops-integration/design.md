## Context

The Graph-Vul POC has a five-step pipeline: subset extraction → Joern CPG export → DeBERTa code embeddings → DeBERTa CVE embeddings → GATv2 training. These steps are currently run as independent Python scripts with implicit ordering. Experiment results are printed to stdout and lost between runs.

The codebase uses `graph_vul/config.py` for centralized path and hyperparameter configuration. The `cpg_dataset.py` file currently combines CPG parsing, code node embedding, and CVE text embedding in a single `CPGDataset` class with disk caching.

## Goals / Non-Goals

**Goals:**
- Define a `dvc.yaml` with five stages that encode the full dependency graph from raw data to trained model.
- Separate the DeBERTa embedding logic from `cpg_dataset.py` into two standalone scripts (code embeddings, CVE embeddings) that DVC can invoke independently.
- Integrate W&B into the training stage to log hyperparameters and per-epoch metrics.
- Keep all existing functionality intact — this is additive tooling, not an architectural rewrite.

**Non-Goals:**
- Remote DVC storage configuration (S3, GCS, etc.) — local-only for now.
- W&B sweeps or hyperparameter search automation.
- Changing the model architecture or loss function.
- DVC-tracking the DeBERTa model weights themselves.

## Decisions

### 1. Split embedding logic into two standalone scripts

**Decision**: Create `graph_vul/pipeline/embed_code.py` and `graph_vul/pipeline/embed_cve.py` that extract the DeBERTa embedding logic currently inside `CPGDataset`.

**Rationale**: DVC stages need discrete scripts with clear file-level inputs and outputs. The current `CPGDataset.__init__` interleaves parsing, embedding, and caching. Separating code and CVE embeddings also enables independent re-runs (e.g., re-embed CVE text without reprocessing code nodes).

**Alternatives considered**: Wrapping `CPGDataset` with CLI flags to run partial steps — rejected because it overloads a PyG Dataset class with pipeline orchestration concerns.

### 2. DVC stage dependency graph

**Decision**: Five stages with this dependency chain:
```
extract_subset → joern_export → embed_code ─┐
                                              ├→ train
extract_subset ──────────────→ embed_cve ────┘
```

- `extract_subset`: deps=[`bigvul.csv`, script], outs=[`subset_cwe119.csv`]
- `joern_export`: deps=[`subset_cwe119.csv`, script], outs=[`data/raw/`]
- `embed_code`: deps=[`data/raw/`, script], outs=[`data/processed/code_embeddings/`]
- `embed_cve`: deps=[`subset_cwe119.csv`, script], outs=[`data/processed/cve_embeddings/`]
- `train`: deps=[`data/processed/code_embeddings/`, `data/processed/cve_embeddings/`, script], outs=[`data/best_model.pt`], metrics=[`data/metrics.json`]

**Rationale**: Mirrors the actual data flow. CVE embedding depends only on the CSV (not on Joern), so it can run in parallel with graph extraction + code embedding.

### 3. W&B integration point

**Decision**: Initialize W&B at the top of the training script with `wandb.init(project="graph-vul", config={...})`. Log per-epoch train loss, val loss, and all four classification metrics via `wandb.log()`. Finish with `wandb.finish()`.

**Rationale**: Minimal invasion — adds ~15 lines to `train.py`. No W&B in pipeline stages since those are deterministic transforms, not experiments.

**Alternatives considered**: Using W&B Artifacts for data versioning alongside DVC — rejected as redundant; DVC already handles data lineage.

### 4. Parameterization approach

**Decision**: Pipeline scripts accept CLI arguments (via `argparse`) for input/output paths, falling back to `config.py` defaults. DVC `params.yaml` is not used in this iteration; parameters are passed directly in `dvc.yaml` cmd strings using values from config.

**Rationale**: The codebase already centralizes config in `config.py`. Adding `params.yaml` introduces a second source of truth. Direct CLI args are sufficient for DVC dependency tracking and keep the change minimal.

### 5. Training stage outputs a metrics JSON

**Decision**: The training stage writes final best metrics to `data/metrics.json` in addition to the model checkpoint. DVC tracks this as a `metrics` file enabling `dvc metrics show` / `dvc metrics diff`.

**Rationale**: Enables CLI-level experiment comparison without needing W&B access.

## Risks / Trade-offs

- **Embedding script duplication**: Extracting embedding logic from `CPGDataset` into standalone scripts may create some code duplication. → Mitigation: Extract shared DeBERTa utility functions into a common module (`graph_vul/pipeline/embed_utils.py`).
- **CPGDataset still needed for DataLoader**: The training stage still needs a PyG Dataset to create batched graphs. → Mitigation: `CPGDataset` will be refactored to load pre-computed embeddings from disk rather than computing them inline.
- **DVC + large binary files**: CPG exports and embeddings can be large. → Mitigation: `.gitignore` updated to exclude `data/` artifacts; DVC tracks them with `.dvc` files. Remote storage is a future enhancement.
- **W&B network dependency**: W&B logging requires internet. → Mitigation: `wandb` supports offline mode (`WANDB_MODE=offline`); document this as a fallback.
