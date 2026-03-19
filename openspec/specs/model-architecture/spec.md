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
