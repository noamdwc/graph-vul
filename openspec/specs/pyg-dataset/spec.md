## ADDED Requirements

### Requirement: Parse Joern CPG exports into PyG Data objects
The dataset class SHALL parse each Joern CPG export directory and construct a PyTorch Geometric `Data` object containing: an `edge_index` tensor representing all graph edges, an `edge_type` tensor distinguishing AST, CFG, and PDG edges, and a node feature matrix `x`.

#### Scenario: Valid CPG export parsed
- **WHEN** the dataset loads a sample with a valid Joern CPG export
- **THEN** a `Data` object is produced with `edge_index` of shape `[2, num_edges]`, `edge_type` of shape `[num_edges]`, and `x` of shape `[num_nodes, embedding_dim]`

#### Scenario: Empty or corrupt CPG export
- **WHEN** a sample's CPG export directory is empty or contains unparseable files
- **THEN** the dataset SHALL skip that sample and log a warning

### Requirement: Initialize node features with DeBERTa embeddings
The dataset SHALL use a pre-trained DeBERTa model to embed each node's code content (token text, operator name, or type identifier) into a dense vector. The embeddings SHALL be pre-computed once and cached to disk for reuse.

#### Scenario: Node embedding computation
- **WHEN** the dataset processes a CPG for the first time
- **THEN** each node's textual content is passed through DeBERTa and the resulting embedding vector is stored as that node's feature row in the `x` matrix

#### Scenario: Cached embeddings exist
- **WHEN** pre-computed embeddings already exist on disk for a sample
- **THEN** the dataset SHALL load them from cache instead of re-running DeBERTa inference

### Requirement: Pair each graph with a DeBERTa CVE text vector
The dataset SHALL embed the CVE description text for each sample using DeBERTa and store the resulting vector as a `text_vec` attribute on the `Data` object.

#### Scenario: CVE text embedding
- **WHEN** a sample has an associated CVE description
- **THEN** the `Data` object SHALL include a `text_vec` tensor of shape `[embedding_dim]` representing the DeBERTa encoding of that description

#### Scenario: Missing CVE description
- **WHEN** a benign sample has no CVE description
- **THEN** the `text_vec` SHALL be a zero vector of the same dimensionality

### Requirement: Provide standard PyG Dataset interface
The dataset class SHALL extend `torch_geometric.data.Dataset` and implement `len()`, `get(idx)`, and `processed_file_names` so it integrates with PyG DataLoaders. Each `Data` object SHALL carry a graph-level label `y` as a scalar float (not per-node broadcasted), representing the vulnerability status of the entire function.

#### Scenario: DataLoader integration
- **WHEN** the dataset is passed to a PyG `DataLoader` with a batch size
- **THEN** it SHALL produce batched `Data` objects with correctly collated graphs, node features, text vectors, and graph-level labels where `batch.y` has shape `[batch_size]`

#### Scenario: Graph-level label assignment
- **WHEN** the dataset loads a sample with vulnerability label `vul`
- **THEN** the `Data` object's `y` attribute SHALL be a scalar `float(vul)` instead of `torch.full((num_nodes,), float(vul))`
