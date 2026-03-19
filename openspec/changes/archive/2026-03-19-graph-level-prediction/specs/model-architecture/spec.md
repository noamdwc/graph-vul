## MODIFIED Requirements

### Requirement: Multi-modal fusion via MLP
The model SHALL produce per-node logits via the NodeClassifier MLP head, then aggregate to a single graph-level logit via max-pooling over the batch vector. The final output of `GraphVulModel.forward()` SHALL be a tensor of shape `[batch_size]` containing one raw logit per graph. The early fusion mechanism (CVE embedding injected between GAT layers) SHALL remain unchanged.

#### Scenario: Binary classification output
- **WHEN** the node features are passed through the MLP head and aggregated
- **THEN** the model SHALL output a tensor of shape `[batch_size]` with one logit per graph, computed as the max of per-node logits within each graph
