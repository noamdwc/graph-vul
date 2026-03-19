## Why

The current architecture performs **late fusion** — the CVE text embedding is only combined with the code representation after the GAT has already pooled node features into a single graph-level vector. This means the GAT layers never see the vulnerability context described in the CVE while computing node representations. Additionally, the model predicts a single binary label per graph, offering no insight into *which* nodes are involved in the vulnerability.

By fusing the CVE embedding into node features **after the first GAT layer** and switching to **per-node prediction**, the model can:
1. Let the second GAT layer propagate vulnerability-aware signals through the graph structure.
2. Identify which specific nodes (statements, expressions) are most likely to be part of a vulnerable pattern.
3. Provide explainability — node-level scores highlight the vulnerability location within the CPG.

## What Changes

- Refactor `GATEncoder` to expose intermediate node representations after the first GAT layer (before the second layer) instead of directly pooling to a graph vector.
- Add a fusion step that broadcasts the projected CVE embedding to every node and concatenates (or adds) it to the node features between GAT layers.
- Add a linear projection to align the fused dimension back to the expected input dimension for subsequent GAT layers.
- Replace the graph-level `FusionMLP` classifier with a per-node classification head that produces a logit for each node.
- Update `GraphVulModel` to orchestrate the new flow: GAT layer 1 → fuse CVE → GAT layer 2 → per-node classifier.
- Update `GraphVulLitModule` to handle per-node labels and loss computation (BCEWithLogitsLoss applied per node).
- Update `CPGDataset` to provide per-node labels (propagated from the graph-level `vul` label to all nodes in the graph).
- Update training metrics to work with node-level predictions (aggregate per-graph for reporting).

## Capabilities

### Modified Capabilities
- `model-architecture`: GATEncoder split into layered stages with mid-stream CVE fusion; FusionMLP replaced by per-node classifier head.
- `pyg-dataset`: CPGDataset extended to produce per-node labels.
- `lightning-module`: LitModule updated for node-level loss and metrics.
- `training-loop`: Metrics aggregation adapted for node-level predictions.

## Impact

- **Model checkpoint**: Existing `best_model.ckpt` is incompatible — retraining required.
- **Metrics**: Node-level metrics will differ from graph-level; graph-level metrics can still be derived by aggregating node predictions (e.g., max or mean).
- **Data**: No new data required. Per-node labels are derived from the existing graph-level label at load time.
- **Dependencies**: No new dependencies.
