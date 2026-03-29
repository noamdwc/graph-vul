## Why

The current function baseline and finetune paths fuse only code embeddings and CWE embeddings (`[emb_code || emb_cwe]`), discarding the structural information available in Joern CPGs. Adding a lightweight graph encoder over the local AST+CFG lets us measure whether function-level structural features improve vulnerability detection without the complexity of the full graph model (which uses node-level DeBERTa embeddings, multi-head GAT, and mid-path text fusion). This is the minimal structural baseline needed before exploring richer graph features (PDG, node text, target-node logic).

## What Changes

- Add a preprocessing stage that extracts function-local graphs (AST+CFG edges only) from existing Joern CPG exports, storing node type IDs and edge type IDs per sample.
- Add stable vocabulary mappings (Joern node type → int, edge type → int) so IDs are consistent across runs.
- Add a `FunctionGraphEncoder` module: `nn.Embedding` for node types → 2-layer RGCN → `global_mean_pool` → fixed-size graph vector.
- Update `FuncDataset` and `FuncFinetuneDataset` to optionally load and return graph data alongside existing code/CWE embeddings.
- Update `FuncBaselineModel` and `FuncFinetuneModel` to optionally accept and fuse the graph embedding: `[emb_code || emb_graph_ast_cfg || emb_cwe]`.
- Update `FuncLitModule` and `FuncFinetuneLitModule` to forward graph data through the model.
- Update `train_baseline.py` and `train_finetune.py` to support a `--use-graph` flag (or config toggle) that enables graph features.
- Update `colab_finetune.ipynb` to support graph-augmented finetune training.
- Add basic logging: node/edge type vocab sizes, average nodes/edges per graph, missing/empty graph count.

## Capabilities

### New Capabilities
- `graph-preprocessing`: Extract function-local AST+CFG graphs from Joern exports into per-sample `.pt` artifacts with node type IDs, edge index, and edge type IDs, plus stable vocab mappings.
- `graph-encoder`: `FunctionGraphEncoder` module — node type embeddings → 2-layer RGCN → global mean pool → fixed graph vector.

### Modified Capabilities
- `func-baseline-model`: Add optional graph embedding fusion so the model supports `[emb_code || emb_graph || emb_cwe]` in addition to the existing `[emb_code || emb_cwe]`.
- `func-embedding`: Update `FuncDataset` to optionally load graph artifacts alongside code/CWE embeddings.

## Impact

- **New files**: graph preprocessing script (`pipeline/extract_graphs.py`), vocab mapping, `FunctionGraphEncoder` module (`model/graph_encoder.py`).
- **Modified files**: `config.py` (new paths/hyperparams), `dataset/func_dataset.py`, `dataset/func_finetune_dataset.py`, `model/func_classifier.py`, `model/func_finetune_model.py`, `model/func_lit_module.py`, `model/func_finetune_lit_module.py`, `train_baseline.py`, `train_finetune.py`, `notebooks/colab_finetune.ipynb`.
- **New dependencies**: `torch-geometric` RGCN (`RGCNConv`), already available since the project uses PyG for `CPGDataset`.
- **Backwards compatible**: All existing code paths still work when graph features are not enabled.
