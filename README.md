# Graph-Vul

Multi-modal vulnerability detection combining Code Property Graphs with CVE text descriptions. A GATv2 encoder processes the code structure while DeBERTa embeddings capture semantics for both code nodes and vulnerability descriptions. The two branches are fused via an MLP for binary classification of CWE-119 (buffer overflow) vulnerabilities.

## Architecture

```
Branch A: C/C++ code → Joern CPG (AST+CFG+PDG) → DeBERTa node embeddings → GATv2 → global mean pool → code_vector (128-d)
Branch B: CVE description → DeBERTa → text_vector (768-d) → linear projection → text_proj (128-d)
Fusion:   [code_vector ∥ text_proj] → MLP (256 → 1) → binary prediction
```

## Project Structure

```
graph_vul/
├── config.py                  # Hyperparameters, paths, sampling/split config
├── pipeline/
│   ├── extract_subset.py      # Leakage-aware structured subset sampling
│   ├── split_data.py          # Project-disjoint train/val/test split
│   ├── joern_export.py        # Joern CLI wrapper for CPG extraction
│   ├── embed_code.py          # DeBERTa embeddings for CPG nodes
│   ├── embed_cve.py           # DeBERTa embeddings for CVE descriptions
│   └── embed_utils.py         # Shared DeBERTa embedding helpers
├── dataset/
│   └── cpg_dataset.py         # PyG Dataset: loads pre-computed embeddings
├── model/
│   ├── gat_encoder.py         # 2-layer GATv2 with edge-type-aware attention
│   ├── text_encoder.py        # Linear projection for DeBERTa text vectors
│   └── classifier.py          # Fusion MLP + top-level GraphVulModel
└── train.py                   # Training loop with W&B tracking and F1 checkpointing
tests/                         # Subset sampling and split tests
dvc.yaml                       # DVC pipeline definition (7 stages)
```

## Prerequisites

- Python 3.10+
- [Joern](https://joern.io/) CLI installed and on `PATH`
- The [Big-Vul](https://github.com/ZeoVan/MSR_20_Code_Vulnerability_CSV_Dataset) dataset CSV placed at `data/bigvul.csv`

## Setup

```bash
pip install -r requirements.txt
wandb login          # authenticate with Weights & Biases (one-time)
```

## Usage

The full pipeline is managed by [DVC](https://dvc.org/). A single command runs all stages in dependency order, skipping any that are already up to date:

```bash
dvc repro
```

This executes the pipeline stages:
1. **extract_subset** — Structured CVE-diverse sampling from Big-Vul → `data/subset.csv` + manifest + report
2. **joern_export** — Runs Joern CPG extraction → `data/raw/`
3. **embed_code** — DeBERTa embeddings for CPG nodes → `data/processed/code_embeddings/`
4. **embed_cve** — DeBERTa embeddings for CVE descriptions → `data/processed/cve_embeddings/`
5. **split_data** — Project-disjoint train/val/test split → `data/splits/`
6. **train** — Model training with W&B logging → `data/best_model.ckpt` + `data/metrics.json`

### Viewing Metrics

```bash
dvc metrics show              # display best validation metrics
dvc metrics diff              # compare metrics across commits/experiments
```

Training metrics (loss, accuracy, precision, recall, F1) are also logged per-epoch to [Weights & Biases](https://wandb.ai/) for interactive visualization.

### Running Individual Stages

```bash
python -m graph_vul.pipeline.extract_subset --input data/bigvul.csv --output data/subset.csv
python -m graph_vul.pipeline.joern_export --input data/subset.csv --output data/raw
python -m graph_vul.pipeline.embed_code --input data/raw --output data/processed/code_embeddings
python -m graph_vul.pipeline.embed_cve --input data/subset.csv --output data/processed/cve_embeddings
python -m graph_vul.pipeline.split_data --input data/subset.csv --output data/splits
python -m graph_vul.train --epochs 35 --batch-size 32 --lr 1e-4
```

### Offline W&B Mode

If you don't have internet access, set `WANDB_MODE=offline` before training:

```bash
WANDB_MODE=offline dvc repro
```

## Leakage-Aware Sampling and Splitting

### Why project is a leakage source

Functions from the same project share coding style, naming conventions, build patterns, and repository structure. If the same project appears in both training and evaluation, the model can exploit these superficial signals instead of learning vulnerability patterns. EDA confirmed that project-level features are a dominant confound.

### Why CVE is a key dimension

Each CVE groups related vulnerability instances. Within a CVE, functions share the same vulnerability type and context. Holding out entire CVEs for evaluation tests whether the model generalizes to unseen vulnerability patterns, not just unseen functions from a known vulnerability.

### Subset and split are separate stages

The **subset stage** selects a diverse, training-oriented sample from the full Big-Vul dataset:
- Selects 7–12 CVEs using diversity scoring (project diversity, metadata spread, class balance)
- Supports configurable benign oversampling (default: 2:1 benign-to-vulnerable) for training
- Enforces per-project, per-CVE, and per-group caps to avoid dominance
- Uses metadata-aware sampling (code length, line count, token count buckets)
- Produces a manifest JSON and human-readable report alongside the subset CSV

The **split stage** assigns rows to train/val/test with strict project disjointness:
- Held-out projects are fully excluded from training (zero project overlap)
- Held-out projects are selected from smaller/mid-size projects with both classes
- Val and test contain only held-out projects
- Produces an assignments CSV and a diagnostics report with overlap verification

### Why class balance is not enforced in splits

Real-world vulnerability data is heavily imbalanced. Forcing 50/50 balance in evaluation sets would produce unrealistic metrics. The subset uses benign oversampling for training effectiveness, and the split preserves whatever class ratio exists naturally within each project.

### Artifacts produced

| Stage | Artifact | Description |
|-------|----------|-------------|
| extract_subset | `data/subset.csv` | Subset with project, code, CVE, and label columns |
| extract_subset | `data/subset_manifest.json` | Machine-readable selection metadata |
| extract_subset | `data/subset_report.md` | Human-readable sampling summary |
| split_data | `data/splits/assignments.csv` | Per-row split assignments (sample_id, split) |
| split_data | `data/splits/split_report.md` | Split diagnostics with overlap verification |

## Key Design Decisions

- **DeBERTa for both branches** — `microsoft/deberta-v3-base` (768-d) is used for code node embeddings and CVE text vectorization, providing a shared semantic space.
- **Edge-type-aware GAT** — An `nn.Embedding` maps edge types (AST=0, CFG=1, PDG=2) to learned edge attributes fed into GATv2Conv.
- **Disk caching** — Processed PyG `Data` objects are saved as `.pt` files to avoid redundant DeBERTa inference on subsequent runs.
- **Project-disjoint splits** — No project appears in more than one split, preventing leakage of coding style and repository signals.
