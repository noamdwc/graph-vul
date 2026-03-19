## Why

The current node-level prediction approach uses weak supervision (broadcasting graph labels to all nodes), which introduces label noise — benign nodes in vulnerable functions are mislabeled as positive. Returning to graph-level prediction with a clear aggregation rule (any positive node → positive graph) provides cleaner supervision and aligns the loss function with the actual ground truth granularity.

## What Changes

- **BREAKING**: Replace per-node BCE loss with graph-level BCE loss computed after max-pooling node logits per graph
- Replace `NodeClassifier` per-node output with graph-level aggregation: max over node logits per graph produces a single scalar per graph
- Labels return to graph-level (`y` shape `[batch_size]` instead of `[num_nodes]`)
- Validation metrics computed directly on graph-level predictions (no post-hoc aggregation needed)
- Dataset returns graph-level labels instead of broadcasting to nodes

## Capabilities

### New Capabilities
- `graph-level-aggregation`: Max-pool node logits to produce a single graph-level prediction; graph is positive if any node logit exceeds threshold

### Modified Capabilities
- `model-architecture`: Final output changes from per-node logits `[N]` to per-graph logit `[batch_size]` via max aggregation over nodes; early fusion unchanged
- `training-loop`: Loss computed on graph-level predictions with graph-level labels instead of per-node weak supervision
- `pyg-dataset`: Dataset returns graph-level `y` label (scalar per graph) instead of per-node broadcasted labels

## Impact

- **Model code**: `classifier.py` — replace `NodeClassifier` forward to include scatter-max aggregation; update `GraphVulModel` forward pass
- **Lightning module**: `lit_module.py` — update training/validation steps for graph-level loss and metrics
- **Dataset**: `cpg_dataset.py` — change label assignment from per-node broadcast to per-graph scalar
- **Specs**: `model-architecture`, `training-loop`, `pyg-dataset` specs need requirement updates
- **No impact on**: embedding pipeline, DVC stages, data splits, W&B integration (only metric semantics change)
