## Context

The model currently operates at node-level: `GraphVulModel.forward()` returns per-node logits `[N]`, the dataset broadcasts graph labels to every node (`y=torch.full((num_nodes,), label)`), and `BCEWithLogitsLoss` is computed per-node. Validation already aggregates to graph-level via `scatter(..., reduce="max")` for metrics. The ground truth is fundamentally graph-level (a function is vulnerable or not), so the node-level weak supervision introduces noise.

## Goals / Non-Goals

**Goals:**
- Compute loss at graph level using the actual ground truth granularity
- Aggregate node logits → graph logit via max-pooling (any positive node → positive graph)
- Simplify validation logic (no post-hoc aggregation needed)

**Non-Goals:**
- Changing the GAT encoder, fusion mechanism, or embedding pipeline
- Adding node-level explainability (future work — can still extract per-node logits if needed)
- Changing training hyperparameters or schedule

## Decisions

### Decision 1: Max-pool aggregation over learned graph-level readout
Use `scatter(logits, batch, reduce="max")` to aggregate node logits to graph level rather than adding a learnable global pooling layer (e.g., `global_mean_pool` or attention-based readout).

**Rationale**: The semantics match the task — "if any node is vulnerable, the graph is vulnerable." Max-pooling directly encodes this inductive bias. A learnable readout adds parameters and complexity without clear benefit given the binary detection task. The model already produces meaningful per-node logits via the `NodeClassifier` MLP; max simply selects the most suspicious node.

**Alternative considered**: `global_mean_pool` — averages all node representations before classification. Rejected because it dilutes signal from the few vulnerable nodes among many benign ones.

### Decision 2: Keep NodeClassifier, add aggregation in GraphVulModel
Rather than replacing `NodeClassifier`, keep it producing per-node logits and add the scatter-max aggregation as the final step in `GraphVulModel.forward()`. This preserves the ability to inspect per-node scores for future explainability.

**Rationale**: Minimal code change. The `NodeClassifier` MLP is still useful — it transforms node features into a vulnerability relevance score. Only the loss boundary shifts from per-node to per-graph.

### Decision 3: Graph-level label in dataset
The `CPGDataset.get()` method will return `y` as a scalar `float(row["vul"])` instead of `torch.full((num_nodes,), float(row["vul"]))`. PyG's batching will concatenate these into `[batch_size]`.

**Rationale**: Cleaner data representation that matches the ground truth. Eliminates the need to derive graph labels from node labels during validation.

## Risks / Trade-offs

- **Loss of node-level gradient signal** → The per-node MLP still receives gradients through the max operation, but only the "winning" (max-logit) node per graph gets gradient flow. This is acceptable — it encourages the model to identify the single most vulnerable node rather than labeling all nodes.
- **Sparse gradient flow** → If most graphs have many nodes, gradient only flows through one node per graph per batch. Mitigation: The GAT layers still propagate information across all nodes; only the classifier head has sparse gradients.
- **Future explainability** → Per-node logits are still available at inference time by calling the model without the final aggregation step. No capability is lost.
