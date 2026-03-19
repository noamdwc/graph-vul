## MODIFIED Requirements

### Requirement: Provide standard PyG Dataset interface
Each `Data` object SHALL carry a graph-level label `y` as a scalar float (not per-node broadcasted), representing the vulnerability status of the entire function.

#### Scenario: Graph-level label assignment
- **WHEN** the dataset loads a sample with vulnerability label `vul`
- **THEN** the `Data` object's `y` attribute SHALL be a scalar `float(vul)` instead of `torch.full((num_nodes,), float(vul))`
