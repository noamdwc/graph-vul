## ADDED Requirements

### Requirement: GAT encoder for code graphs
The model SHALL include a Graph Attention Network branch that accepts a batched PyG `Data` object (node features, edge index, edge type, batch vector) and produces a fixed-size code vector per graph. The GAT SHALL use GATv2 convolution layers with edge-type awareness.

#### Scenario: Forward pass through GAT branch
- **WHEN** a batch of graph `Data` objects is passed to the GAT encoder
- **THEN** the encoder SHALL output a tensor of shape `[batch_size, code_vector_dim]` representing each graph's structural embedding

#### Scenario: Variable-size graphs in batch
- **WHEN** graphs in a batch have different numbers of nodes and edges
- **THEN** the GAT encoder SHALL handle them correctly via PyG batching and produce one code vector per graph through global pooling

### Requirement: DeBERTa text vector ingestion
The model SHALL accept pre-computed DeBERTa text vectors (from the dataset's `text_vec` attribute) as input to the text branch. The text branch SHALL apply a linear projection to align the text vector dimensionality with the fusion layer's expectations.

#### Scenario: Text vector projection
- **WHEN** a batch of `text_vec` tensors is provided
- **THEN** the text branch SHALL output a tensor of shape `[batch_size, projected_text_dim]`

### Requirement: Multi-modal fusion via MLP
The model SHALL produce per-node logits via the NodeClassifier MLP head, then aggregate to a single graph-level logit via max-pooling over the batch vector. The final output of `GraphVulModel.forward()` SHALL be a tensor of shape `[batch_size]` containing one raw logit per graph. The early fusion mechanism (CVE embedding injected between GAT layers) SHALL remain unchanged.

#### Scenario: Binary classification output
- **WHEN** the node features are passed through the MLP head and aggregated
- **THEN** the model SHALL output a tensor of shape `[batch_size]` with one logit per graph, computed as the max of per-node logits within each graph

### Requirement: Model composability
The GAT encoder, text projection, and fusion MLP SHALL be separate `nn.Module` subclasses composed within a top-level model class. This allows independent testing and future replacement of individual branches.

#### Scenario: Independent module instantiation
- **WHEN** the GAT encoder module is instantiated standalone
- **THEN** it SHALL be usable independently for forward passes without requiring the text branch or fusion head

### Requirement: Model supports optional internal graph fusion
`FuncFinetuneModel` SHALL accept two new optional parameters:
- `fusion_layers: list[int] | None = None` — encoder layer indices where gated residual fusion is applied. `None` means no internal fusion (preserves current behavior).
- `final_graph_concat: bool = True` — whether to include graph_emb in the late concatenation before the MLP head. `True` preserves current behavior.

When `fusion_layers` is not None and `graph_dim > 0`:
- The model SHALL create one `GatedResidualFusion` module per specified layer
- The model SHALL register forward hooks on the corresponding `encoder.encoder.layer[i]` modules
- The pooled `graph_emb` SHALL be computed once per forward pass (from `FunctionGraphEncoder`) and stored for hook access

When `final_graph_concat` is False:
- The model SHALL exclude `graph_emb` from the concatenation before the MLP head
- The MLP head input_dim SHALL be `encoder_dim + cwe_dim` (no graph_dim term)
- CWE embedding concatenation SHALL always be included regardless of this flag

#### Scenario: Default behavior unchanged
- **WHEN** `FuncFinetuneModel` is constructed with default parameters (`fusion_layers=None`, `final_graph_concat=True`)
- **THEN** the model behaves identically to the current implementation (late concat only)

#### Scenario: Internal fusion with late concat (E1–E3)
- **WHEN** `fusion_layers=[10, 11]` and `final_graph_concat=True` and `graph_dim=64`
- **THEN** the model applies gated residual fusion at layers 10 and 11, AND concatenates graph_emb into the late fusion before the MLP head

#### Scenario: Internal fusion without late concat (E4)
- **WHEN** `fusion_layers=[8, 9, 10, 11]` and `final_graph_concat=False` and `graph_dim=64`
- **THEN** the model applies gated residual fusion at layers 8–11, but does NOT include graph_emb in the late concatenation. The MLP head input_dim is `768 + 768 = 1536`.

### Requirement: Lightning module passes fusion parameters
`FuncFinetuneLitModule` SHALL accept `fusion_layers` and `final_graph_concat` parameters and forward them to `FuncFinetuneModel`. These parameters SHALL be included in `save_hyperparameters()` for checkpoint reproducibility.

#### Scenario: Checkpoint contains fusion config
- **WHEN** a model is trained with `fusion_layers=[10, 11]` and saved
- **THEN** loading the checkpoint restores the model with the same fusion configuration

### Requirement: Train function accepts fusion parameters
The `train()` function in `train_finetune.py` SHALL accept optional `fusion_layers`, `final_graph_concat`, `checkpoint_path`, and `metrics_path` parameters, forwarding them appropriately to the model and file I/O.

#### Scenario: Train with fusion config
- **WHEN** `train(use_graph=True, fusion_layers=[10, 11], final_graph_concat=True)` is called
- **THEN** the model is created with internal fusion at layers 10 and 11, and late graph concat enabled

#### Scenario: Train with custom output paths
- **WHEN** `train(checkpoint_path=Path("custom/model.ckpt"), metrics_path=Path("custom/metrics.json"))` is called
- **THEN** the checkpoint and metrics are written to the specified paths instead of the defaults
