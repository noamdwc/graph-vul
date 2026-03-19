## ADDED Requirements

### Requirement: Max-pool node logits to graph-level prediction
The model SHALL aggregate per-node logits to a single graph-level logit by computing the maximum node logit per graph using the batch vector. A graph SHALL be predicted positive if its graph-level logit exceeds the classification threshold.

#### Scenario: Single graph with mixed node logits
- **WHEN** a graph has node logits `[-1.2, 0.8, -0.3]`
- **THEN** the graph-level logit SHALL be `0.8` (the maximum)

#### Scenario: Batched graphs with different sizes
- **WHEN** a batch contains graphs with varying numbers of nodes
- **THEN** the max aggregation SHALL produce exactly one logit per graph, with shape `[batch_size]`

#### Scenario: All negative node logits
- **WHEN** all node logits in a graph are below the classification threshold
- **THEN** the graph SHALL be predicted as benign (negative)

#### Scenario: At least one positive node logit
- **WHEN** at least one node logit in a graph exceeds the classification threshold
- **THEN** the graph SHALL be predicted as vulnerable (positive)
