## 1. Project Setup

- [x] 1.1 Create the project directory structure (`graph_vul/`, `pipeline/`, `dataset/`, `model/`, `data/raw/`, `data/processed/`)
- [x] 1.2 Create `config.py` with hyperparameters, paths, and DeBERTa model name
- [x] 1.3 Add `requirements.txt` with pinned versions for torch, torch-geometric, transformers, pandas, numpy, scikit-learn

## 2. Data Pipeline

- [x] 2.1 Implement `pipeline/extract_subset.py`: read Big-Vul CSV, filter to CWE-119, create balanced ~1K subset, write filtered CSV with function code, CVE ID, CVE description, and label columns
- [x] 2.2 Implement `pipeline/joern_export.py`: iterate over subset, write each function to a temp `.c` file, invoke Joern CLI to export CPG, store output in `data/raw/<sample_id>/`
- [x] 2.3 Add idempotency check in `joern_export.py` — skip samples with existing non-empty export directories
- [x] 2.4 Add error handling for Joern parse failures — log and continue

## 3. PyG Dataset

- [x] 3.1 Implement `dataset/cpg_dataset.py` with a class extending `torch_geometric.data.Dataset` — implement `len()`, `get(idx)`, `processed_file_names`
- [x] 3.2 Add CPG parsing logic: read Joern node/edge files, build `edge_index` tensor, assign `edge_type` for AST/CFG/PDG edges
- [x] 3.3 Add DeBERTa node embedding: for each node, extract code text, run through DeBERTa, store as node feature matrix `x`
- [x] 3.4 Add embedding caching: save pre-computed node embeddings and text vectors to disk, load from cache on subsequent runs
- [x] 3.5 Add CVE text embedding: encode CVE description via DeBERTa, store as `text_vec` on the `Data` object; use zero vector for benign samples without descriptions
- [x] 3.6 Add binary label (`y`) to each `Data` object from the subset CSV

## 4. Model Architecture

- [x] 4.1 Implement `model/gat_encoder.py`: 2-layer GATv2 with 4 heads, hidden dim 128, edge-type-aware attention, global mean pooling → code vector
- [x] 4.2 Implement `model/text_encoder.py`: linear projection layer to map DeBERTa text vector (768-d) to projected dimension
- [x] 4.3 Implement `model/classifier.py`: fusion MLP (concatenation → 256 → 1) with dropout 0.3 and LeakyReLU, sigmoid output; top-level `GraphVulModel` composing all three modules

## 5. Training Loop

- [x] 5.1 Implement `train.py`: dataset loading, stratified 80/20 train/val split, PyG DataLoader setup
- [x] 5.2 Add training loop: epoch iteration, mini-batch forward/backward, BCEWithLogitsLoss, optimizer step, epoch loss logging
- [x] 5.3 Add validation evaluation: no-grad forward pass, compute accuracy, precision, recall, F1
- [x] 5.4 Add best-model checkpointing: save state dict when validation F1 improves
- [x] 5.5 Add reproducibility seeding: accept seed parameter, seed torch/numpy/random at startup
