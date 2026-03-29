## MODIFIED Requirements

### Requirement: FuncBaselineModel supports optional graph embedding fusion
The `FuncBaselineModel` SHALL accept an optional `graph_dim` parameter. When `graph_dim > 0`, the model SHALL instantiate a `FunctionGraphEncoder` and fuse `[emb_code || emb_graph || emb_cwe]` through the MLP head. When `graph_dim == 0` (default), the model SHALL behave identically to the current implementation (`[emb_code || emb_cwe]`).

#### Scenario: Forward pass without graph (backwards compatible)
- **WHEN** the model is instantiated with default parameters (`graph_dim=0`) and `forward()` is called with `code_emb` and `cwe_emb` only
- **THEN** it SHALL return logits computed from `MLP(cat([emb_code, emb_cwe]))`, identical to current behavior

#### Scenario: Forward pass with graph embedding
- **WHEN** the model is instantiated with `graph_dim > 0` and `forward()` is called with `code_emb`, `cwe_emb`, and a PyG `Batch` of graph data
- **THEN** it SHALL return logits computed as `MLP(cat([emb_code, graph_encoder(graph_data), emb_cwe]))` with shape `[batch_size, 1]`

### Requirement: FuncFinetuneModel supports optional graph embedding fusion
The `FuncFinetuneModel` SHALL accept an optional `graph_dim` parameter. When `graph_dim > 0`, the model SHALL instantiate a `FunctionGraphEncoder` and fuse `[encoder(code) || emb_graph || emb_cwe]` through the MLP head. When `graph_dim == 0` (default), the model SHALL behave identically to the current implementation.

#### Scenario: Forward pass without graph (backwards compatible)
- **WHEN** the model is instantiated with default parameters and `forward()` is called without graph data
- **THEN** it SHALL return logits computed from `MLP(cat([pooled_code, emb_cwe]))`, identical to current behavior

#### Scenario: Forward pass with graph embedding
- **WHEN** the model is instantiated with `graph_dim > 0` and `forward()` receives graph data
- **THEN** it SHALL return logits computed as `MLP(cat([pooled_code, graph_encoder(graph_data), emb_cwe]))` with shape `[batch_size, 1]`

### Requirement: FuncDataset supports optional graph loading
The `FuncDataset` SHALL accept a `use_graph: bool = False` parameter and a `graph_artifacts_dir` path. When `use_graph=True`, `__getitem__` SHALL also load the graph artifact and return `(code_emb, cwe_emb, graph_data, label)`.

#### Scenario: Dataset without graph (backwards compatible)
- **WHEN** `FuncDataset` is instantiated with `use_graph=False` (default)
- **THEN** `__getitem__` SHALL return `(code_emb, cwe_emb, label)` as before

#### Scenario: Dataset with graph enabled
- **WHEN** `FuncDataset` is instantiated with `use_graph=True`
- **THEN** `__getitem__` SHALL return `(code_emb, cwe_emb, graph_data, label)` where `graph_data` is a PyG `Data` object

#### Scenario: Missing graph artifact uses fallback
- **WHEN** `use_graph=True` and the graph artifact `.pt` does not exist for a sample
- **THEN** the dataset SHALL substitute a fallback graph with a single UNK node (type ID 0) and no edges

### Requirement: FuncFinetuneDataset supports optional graph loading
The `FuncFinetuneDataset` SHALL accept a `use_graph: bool = False` parameter. When `use_graph=True`, `__getitem__` SHALL include a `"graph_data"` key in the returned dict.

#### Scenario: Dataset without graph (backwards compatible)
- **WHEN** `FuncFinetuneDataset` is instantiated with `use_graph=False` (default)
- **THEN** `__getitem__` SHALL return the same dict as before (no `graph_data` key)

#### Scenario: Dataset with graph enabled
- **WHEN** `FuncFinetuneDataset` is instantiated with `use_graph=True`
- **THEN** `__getitem__` SHALL return a dict including `"graph_data"` as a PyG `Data` object

### Requirement: Custom collate function for graph batching
The system SHALL provide a collate function that batches variable-size graphs using PyG's `Batch.from_data_list` while stacking other tensors normally. This collate SHALL be used by both training scripts when `--use-graph` is enabled.

#### Scenario: Collate for func baseline batch
- **WHEN** a list of `(code_emb, cwe_emb, graph_data, label)` tuples is collated
- **THEN** the result SHALL contain stacked `code_emb` `[B, 768]`, stacked `cwe_emb` `[B, 768]`, a PyG `Batch` of graphs, and stacked `labels` `[B]`

#### Scenario: Collate for finetune batch
- **WHEN** a list of dicts with `graph_data` keys is collated
- **THEN** the result SHALL contain a PyG `Batch` under the `"graph_data"` key, with all other keys stacked normally

### Requirement: Training scripts support --use-graph flag
Both `train_baseline.py` and `train_finetune.py` SHALL accept a `--use-graph` CLI flag. When set, they SHALL load graph vocab, pass `use_graph=True` to datasets, use the custom collate, instantiate models with `graph_dim > 0`, and log graph statistics to WandB.

#### Scenario: train_baseline.py without --use-graph
- **WHEN** `python -m graph_vul.train_baseline` is run without `--use-graph`
- **THEN** training SHALL behave identically to before

#### Scenario: train_baseline.py with --use-graph
- **WHEN** `python -m graph_vul.train_baseline --use-graph` is run
- **THEN** training SHALL use `FuncDataset(use_graph=True)`, `FuncBaselineModel(graph_dim=GRAPH_HIDDEN_DIM)`, custom collate, and log graph statistics

#### Scenario: train_finetune.py with --use-graph
- **WHEN** `python -m graph_vul.train_finetune --use-graph` is run
- **THEN** training SHALL use `FuncFinetuneDataset(use_graph=True)`, `FuncFinetuneModel(graph_dim=GRAPH_HIDDEN_DIM)`, custom collate, and log graph statistics

### Requirement: Lightning modules forward graph data
`FuncLitModule` and `FuncFinetuneLitModule` SHALL pass graph data through to the model when present in the batch. When graph data is not present, they SHALL behave identically to current.

#### Scenario: FuncLitModule with graph data
- **WHEN** a batch contains graph data (4-tuple instead of 3-tuple)
- **THEN** the lit module SHALL pass graph data to `model.forward()` as the graph argument

#### Scenario: FuncFinetuneLitModule with graph data
- **WHEN** a batch dict contains a `"graph_data"` key
- **THEN** the lit module SHALL pass it to `model.forward()` as the graph argument

### Requirement: Colab notebook supports graph-augmented finetune
The `colab_finetune.ipynb` notebook SHALL include a `USE_GRAPH` toggle. When enabled, the notebook SHALL validate graph artifacts exist on Drive, install PyG dependencies, and pass the flag to the training call.

#### Scenario: Notebook with USE_GRAPH = False
- **WHEN** `USE_GRAPH = False` in the config cell
- **THEN** the notebook SHALL behave identically to current

#### Scenario: Notebook with USE_GRAPH = True
- **WHEN** `USE_GRAPH = True` in the config cell
- **THEN** the notebook SHALL validate `graph_artifacts/` and vocab files exist on Drive, install `torch-geometric` and `torch-scatter`/`torch-sparse`, and call `train()` with `use_graph=True`

### Requirement: Config constants for graph baseline
`config.py` SHALL define: `GRAPH_ARTIFACTS_DIR`, `GRAPH_NODE_VOCAB_PATH`, `GRAPH_EDGE_VOCAB_PATH`, `GRAPH_NODE_EMBED_DIM`, `GRAPH_HIDDEN_DIM`, `GRAPH_NUM_RGCN_LAYERS`, `GRAPH_DROPOUT`.

#### Scenario: Config constants defined
- **WHEN** `config.py` is imported
- **THEN** the graph-related constants SHALL be available with sensible defaults (node_embed_dim=64, hidden_dim=64, num_rgcn_layers=2, dropout=0.3)
