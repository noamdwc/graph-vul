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
├── config.py                  # Hyperparameters and paths
├── pipeline/
│   ├── extract_subset.py      # Big-Vul → balanced CWE-119 subset (~1000 samples)
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
dvc.yaml                       # DVC pipeline definition (5 stages)
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

This executes the five pipeline stages:
1. **extract_subset** — Filters Big-Vul for CWE-119 → `data/subset_cwe119.csv`
2. **joern_export** — Runs Joern CPG extraction → `data/raw/`
3. **embed_code** — DeBERTa embeddings for CPG nodes → `data/processed/code_embeddings/`
4. **embed_cve** — DeBERTa embeddings for CVE descriptions → `data/processed/cve_embeddings/`
5. **train** — Model training with W&B logging → `data/best_model.pt` + `data/metrics.json`

### Viewing Metrics

```bash
dvc metrics show              # display best validation metrics
dvc metrics diff              # compare metrics across commits/experiments
```

Training metrics (loss, accuracy, precision, recall, F1) are also logged per-epoch to [Weights & Biases](https://wandb.ai/) for interactive visualization.

### Running Individual Stages

You can also run stages manually:

```bash
python -m graph_vul.pipeline.extract_subset --input data/bigvul.csv --output data/subset_cwe119.csv
python -m graph_vul.pipeline.joern_export --input data/subset_cwe119.csv --output data/raw
python -m graph_vul.pipeline.embed_code --input data/raw --output data/processed/code_embeddings
python -m graph_vul.pipeline.embed_cve --input data/subset_cwe119.csv --output data/processed/cve_embeddings
python -m graph_vul.train --epochs 50 --batch-size 32 --lr 1e-3
```

### Offline W&B Mode

If you don't have internet access, set `WANDB_MODE=offline` before training:

```bash
WANDB_MODE=offline dvc repro
```

## Key Design Decisions

- **DeBERTa for both branches** — `microsoft/deberta-v3-base` (768-d) is used for code node embeddings and CVE text vectorization, providing a shared semantic space.
- **Edge-type-aware GAT** — An `nn.Embedding` maps edge types (AST=0, CFG=1, PDG=2) to learned edge attributes fed into GATv2Conv.
- **Disk caching** — Processed PyG `Data` objects are saved as `.pt` files to avoid redundant DeBERTa inference on subsequent runs.
