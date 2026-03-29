## ADDED Requirements

### Requirement: FunctionGraphEncoder module
The system SHALL provide a `FunctionGraphEncoder` PyTorch module that converts a batch of variable-size graphs (with node type IDs and typed edges) into fixed-size graph-level vectors.

#### Scenario: Forward pass on a batch of graphs
- **WHEN** the encoder receives a PyG `Batch` object containing `node_type_ids`, `edge_index`, `edge_type_ids`, and `batch` assignment
- **THEN** it SHALL produce a tensor of shape `[batch_size, graph_dim]` by: (1) looking up node type embeddings via `nn.Embedding`, (2) passing through two `RGCNConv` layers each followed by ReLU and dropout, (3) applying `global_mean_pool` over nodes per graph

### Requirement: FunctionGraphEncoder uses RGCN with relation-aware convolution
The encoder SHALL use `torch_geometric.nn.RGCNConv` for message passing, with `num_relations` set to the number of edge types in the edge type vocabulary (AST=0, CFG=1 for this baseline, so `num_relations=2`).

#### Scenario: Edge types are respected during message passing
- **WHEN** a graph contains both AST and CFG edges
- **THEN** the RGCNConv layers SHALL use separate learned weight matrices for each edge type

### Requirement: FunctionGraphEncoder handles empty graphs
The encoder SHALL handle graphs with zero edges gracefully.

#### Scenario: Graph with single node and no edges
- **WHEN** a fallback graph with one UNK node and no edges is passed
- **THEN** the encoder SHALL return the global_mean_pool of the single node's embedding (the UNK embedding vector), without errors
