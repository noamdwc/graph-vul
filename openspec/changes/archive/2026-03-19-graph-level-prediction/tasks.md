## 1. Dataset

- [x] 1.1 Change `CPGDataset.get()` to return scalar `y=float(row["vul"])` instead of `torch.full((num_nodes,), float(row["vul"]))`

## 2. Model

- [x] 2.1 Add scatter-max aggregation in `GraphVulModel.forward()` to reduce per-node logits to per-graph logits using the batch vector
- [x] 2.2 Update `GraphVulModel.forward()` return shape from `[N]` to `[batch_size]`

## 3. Lightning Module

- [x] 3.1 Update `training_step` to compute BCEWithLogitsLoss on graph-level logits against `batch.y` (now `[batch_size]`)
- [x] 3.2 Simplify `validation_step` to use graph-level model output directly instead of scatter-aggregating node predictions

## 4. Specs

- [x] 4.1 Archive updated specs to `openspec/specs/` after implementation is verified
