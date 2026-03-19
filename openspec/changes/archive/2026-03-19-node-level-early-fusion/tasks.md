## 1. Refactor GATEncoder for layer-by-layer access

- [x] 1.1 Remove `global_mean_pool` import and usage from `gat_encoder.py`
- [x] 1.2 Change `forward()` to return **node-level** features (not pooled graph vector) — iterate through conv layers with activation, return final node embeddings
- [x] 1.3 Add a `forward_one_layer(layer_idx, x, edge_index, edge_attr)` method (or split `convs` into named attributes `conv1`, `conv2`) so the caller can invoke layers individually
- [x] 1.4 Update `output_dim` to reflect the node-level output dimension after the last layer (128 with `concat=False`)

## 2. Update GraphVulModel for early fusion

- [x] 2.1 In `classifier.py`, add a `FusionProjection` layer: `nn.Linear(GAT_HIDDEN_DIM * GAT_HEADS + GAT_HIDDEN_DIM, GAT_HIDDEN_DIM * GAT_HEADS)` to project concatenated node+text features back to the GAT layer 2 input dimension
- [x] 2.2 Replace `FusionMLP` with `NodeClassifier`: a per-node MLP (`hidden_dim → hidden_dim → 1`) with LeakyReLU and dropout
- [x] 2.3 Rewrite `GraphVulModel.forward()`:
  - Run GAT layer 1 on input node features → node embeddings (N × 512)
  - Project CVE text via `TextProjection` → (batch_size × 128)
  - Broadcast text_proj to each node using `batch` tensor → (N × 128)
  - Concatenate node embeddings with broadcast text → (N × 640)
  - Apply `FusionProjection` → (N × 512)
  - Run GAT layer 2 → node embeddings (N × 128)
  - Apply `NodeClassifier` → per-node logits (N,)
  - Return logits and batch tensor for downstream aggregation

## 3. Update config

- [x] 3.1 Add `NODE_CLASSIFIER_HIDDEN_DIM = 128` to `config.py`

## 4. Update CPGDataset for per-node labels

- [x] 4.1 In `cpg_dataset.py`, change `y` from `torch.tensor([float(row["vul"])])` (shape `[1]`) to `torch.full((num_nodes,), float(row["vul"]))` where `num_nodes = x.size(0)`

## 5. Update LitModule for node-level training

- [x] 5.1 Update `training_step` to apply `BCEWithLogitsLoss` on per-node logits against per-node labels (`batch.y`)
- [x] 5.2 Update `validation_step`:
  - Compute per-node predictions
  - Aggregate to graph-level: for each graph in the batch, take `max` of node logits → graph prediction
  - Compare graph-level predictions against graph-level labels (derived from node labels via `scatter_max` or manual grouping using `batch` tensor)
  - Update metrics with graph-level preds/labels
- [x] 5.3 Verify that `batch.y` now has shape `(total_nodes_in_batch,)` and adjust `.squeeze()` calls accordingly

## 6. Verify and test

- [x] 6.1 Run a quick smoke test: instantiate the model, create a dummy batch, verify forward pass produces per-node logits with correct shape
- [x] 6.2 Run `dvc repro train` (or `python -m graph_vul.train --epochs 1`) to verify end-to-end training works
- [x] 6.3 Check that validation metrics are computed and logged correctly
