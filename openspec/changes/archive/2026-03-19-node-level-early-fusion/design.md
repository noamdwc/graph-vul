## Context

The current model architecture follows a late-fusion pattern: GATEncoder processes the CPG to produce a single graph-level vector (via global mean pooling), which is then concatenated with the projected CVE text vector and classified by a FusionMLP. This refactor changes to early fusion with node-level prediction.

**Current flow:**
```
nodes → [GAT L1] → [GAT L2] → global_mean_pool → concat(graph_vec, text_proj) → MLP → 1 logit
```

**New flow:**
```
nodes → [GAT L1] → concat(node_feats, text_proj_broadcast) → project → [GAT L2] → per-node MLP → N logits
```

## Goals / Non-Goals

**Goals:**
- Fuse CVE embedding into node representations between GAT layer 1 and layer 2.
- Predict a vulnerability score per node in the CPG.
- Maintain compatibility with the existing DVC pipeline and data format (no changes to embedding stages).
- Derive graph-level predictions from node-level predictions for backward-comparable metrics.

**Non-Goals:**
- Fine-grained per-node ground truth labels (we propagate graph-level labels to all nodes as a weak supervision signal).
- Changes to the data pipeline, embedding stages, or DVC stages.
- Changes to the DeBERTa embedding or Joern export process.

## Decisions

### 1. Fusion strategy: concatenation + linear projection

**Choice:** After GAT layer 1, broadcast the projected CVE vector (128-d) to every node and concatenate it with the node features (512-d after layer 1 with 4 heads × 128 hidden). Then apply a linear projection to reduce back to `hidden_dim * heads` (512-d) before feeding into GAT layer 2.

**Why:** Concatenation preserves both signals without information loss. A linear projection after fusion keeps the dimensionality manageable and matches GAT layer 2's expected input. Additive fusion was considered but concatenation is more expressive for a POC.

### 2. Per-node labels via weak supervision

**Choice:** Assign the graph-level vulnerability label to every node in the graph. All nodes in a vulnerable function get label 1; all nodes in a benign function get label 0.

**Why:** We don't have ground truth per-node vulnerability annotations. Weak supervision from the graph label is the simplest approach and still teaches the model to associate vulnerability context with code structure. Future work could use attention scores or gradient-based attribution to refine which nodes are truly vulnerable.

### 3. Node-level classifier head

**Choice:** A small MLP (hidden_dim → hidden_dim → 1) applied independently to each node's final representation after GAT layer 2.

**Why:** Simple per-node classification head. The GAT layers already capture structural context, so the classifier just needs to map each node's enriched representation to a vulnerability score.

### 4. Graph-level metric derivation

**Choice:** For reporting graph-level metrics (to compare with the previous architecture), take the **max** of node-level predictions per graph. A graph is predicted vulnerable if any node exceeds the threshold.

**Why:** Max-pooling aligns with the intuition that a function is vulnerable if it contains at least one vulnerable node/statement. Mean-pooling would dilute signal from a few vulnerable nodes in a large graph.

### 5. GATEncoder refactor approach

**Choice:** Split the `forward` method into explicit stages rather than a loop. The encoder exposes a method that returns node features after each layer, and the top-level model handles fusion between layers.

**Why:** Moving fusion logic into the model (not the encoder) keeps GATEncoder focused on graph convolutions and makes the fusion point explicit and configurable.

## File Changes

| File | Change |
|------|--------|
| `graph_vul/model/gat_encoder.py` | Remove global_mean_pool. Split forward into `forward_layer(layer_idx, x, ...)` or expose layers directly. Return node-level features. |
| `graph_vul/model/classifier.py` | Replace `FusionMLP` with `NodeClassifier` (per-node MLP). Rewrite `GraphVulModel.forward` for early-fusion flow. Add fusion projection layer. |
| `graph_vul/model/text_encoder.py` | No changes needed — `TextProjection` already projects 768→128. |
| `graph_vul/model/lit_module.py` | Update loss to be per-node BCE. Update metrics for node-level preds. Add graph-level aggregation for val metrics. |
| `graph_vul/dataset/cpg_dataset.py` | Change `y` from `[1]` shape to `[num_nodes]` shape (all nodes share the graph label). |
| `graph_vul/config.py` | Add `NODE_CLASSIFIER_HIDDEN_DIM` constant. Update `MLP_HIDDEN_DIM` usage if needed. |
| `graph_vul/train.py` | No structural changes expected — DataLoader and training loop are handled by Lightning. |

## Risks / Trade-offs

- **[Weak supervision noise]** Labeling all nodes with the graph label is noisy — benign nodes in vulnerable functions get positive labels. → Mitigation: The model can still learn via attention to distinguish informative nodes. This is a known weak-supervision pattern.
- **[Class imbalance at node level]** Vulnerable graphs may have many nodes, amplifying positive labels. → Mitigation: BCEWithLogitsLoss per node is already balanced within each graph; monitor per-graph metrics alongside per-node ones.
- **[Dimension mismatch]** Fusion concatenation changes the feature dimension mid-network. → Mitigation: Explicit linear projection layer after fusion ensures dimensions are correct for GAT layer 2.
- **[Checkpoint incompatibility]** The model architecture changes break saved checkpoints. → Mitigation: Retraining is expected; old checkpoint is still available in git history / DVC cache.
