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
│   └── joern_export.py        # Joern CLI wrapper for CPG extraction
├── dataset/
│   └── cpg_dataset.py         # PyG Dataset: parses CPGs, computes DeBERTa embeddings
├── model/
│   ├── gat_encoder.py         # 2-layer GATv2 with edge-type-aware attention
│   ├── text_encoder.py        # Linear projection for DeBERTa text vectors
│   └── classifier.py          # Fusion MLP + top-level GraphVulModel
└── train.py                   # Training loop with stratified split and F1 checkpointing
```

## Prerequisites

- Python 3.10+
- [Joern](https://joern.io/) CLI installed and on `PATH`
- The [Big-Vul](https://github.com/ZeoVan/MSR_20_Code_Vulnerability_CSV_Dataset) dataset CSV placed at `data/bigvul.csv`

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Extract subset

```bash
python -m graph_vul.pipeline.extract_subset
```

Filters Big-Vul for CWE-119 and creates a balanced subset at `data/subset_cwe119.csv`.

### 2. Export CPGs

```bash
python -m graph_vul.pipeline.joern_export
```

Runs Joern on each sample to produce CPG exports under `data/raw/`.

### 3. Train

```bash
python -m graph_vul.train --epochs 50 --batch-size 32 --lr 1e-3
```

The first run builds the PyG dataset (DeBERTa embeddings are computed and cached to `data/processed/`). Best model checkpoint is saved to `data/best_model.pt` based on validation F1.

## Key Design Decisions

- **DeBERTa for both branches** — `microsoft/deberta-v3-base` (768-d) is used for code node embeddings and CVE text vectorization, providing a shared semantic space.
- **Edge-type-aware GAT** — An `nn.Embedding` maps edge types (AST=0, CFG=1, PDG=2) to learned edge attributes fed into GATv2Conv.
- **Disk caching** — Processed PyG `Data` objects are saved as `.pt` files to avoid redundant DeBERTa inference on subsequent runs.
