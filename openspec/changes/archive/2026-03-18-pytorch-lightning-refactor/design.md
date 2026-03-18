## Context

The Graph-Vul project trains a multi-modal vulnerability detection model (GATv2 + DeBERTa text projection + fusion MLP) using a hand-rolled PyTorch training loop in `graph_vul/train.py`. The loop manually handles device placement, gradient computation, checkpointing, W&B logging, and metric computation. The project uses PyTorch Geometric for graph batching and DVC for pipeline orchestration.

The current loop is ~230 lines and tightly couples training logic, experiment tracking, and checkpoint management. Adding features like early stopping, learning rate scheduling, or mixed precision would require significant manual code.

## Goals / Non-Goals

**Goals:**
- Wrap the existing `GraphVulModel` in a `LightningModule` without changing the underlying model architecture
- Replace the manual training loop with Lightning's `Trainer`, preserving all current behavior (stratified split, BCE loss, F1-based checkpointing, DVC metrics output)
- Use Lightning's `WandbLogger` and `ModelCheckpoint` callbacks instead of manual implementations
- Maintain compatibility with the DVC pipeline (same CLI entry point, same output artifacts)
- Use `torchmetrics` for metric computation instead of sklearn in the training loop

**Non-Goals:**
- Changing the model architecture (GATEncoder, TextProjection, FusionMLP)
- Introducing a LightningDataModule (the PyG DataLoader setup is simple enough to keep inline)
- Adding distributed training, mixed precision, or other advanced Trainer features (though the refactor makes these trivially addable later)
- Changing the dataset class or data pipeline stages

## Decisions

### Decision 1: Single LightningModule wrapping GraphVulModel

**Choice:** Create a `GraphVulLitModule` in `graph_vul/model/lit_module.py` that composes `GraphVulModel` and adds Lightning lifecycle methods.

**Alternatives considered:**
- Make `GraphVulModel` itself inherit from `LightningModule` — rejected because it couples the model architecture to the training framework, making the model harder to use outside Lightning (e.g., inference scripts)
- Separate LightningModules per sub-model — rejected as over-engineering; the model is small enough for a single module

**Rationale:** Composition keeps `GraphVulModel` as a plain `nn.Module`, usable anywhere. The LightningModule is purely a training wrapper.

### Decision 2: Use torchmetrics for validation metrics

**Choice:** Use `torchmetrics.classification` (Accuracy, Precision, Recall, F1Score) instead of accumulating predictions and calling sklearn.

**Alternatives considered:**
- Keep sklearn metrics with manual accumulation in `on_validation_epoch_end` — works but doesn't leverage Lightning's metric aggregation or device handling

**Rationale:** `torchmetrics` integrates natively with Lightning's `self.log()`, handles batched accumulation automatically, and stays on-device.

### Decision 3: Stratified split stays in train.py

**Choice:** Keep the `train_test_split` logic in `train.py` (the script), not inside the LightningModule or a DataModule.

**Alternatives considered:**
- Create a `LightningDataModule` — adds a file and class for what is currently 6 lines of split + DataLoader code

**Rationale:** The data loading is straightforward (pre-computed embeddings, single dataset class). A DataModule would be ceremony without benefit at this scale.

### Decision 4: DVC metrics JSON written via a Lightning Callback

**Choice:** Create a small callback that writes `data/metrics.json` from the best checkpoint's metrics at the end of training.

**Alternatives considered:**
- Write metrics in `on_train_end` inside the LightningModule — mixes training logic with pipeline concerns
- Write metrics in `train.py` after `trainer.fit()` by reading `trainer.callback_metrics` — simpler, and this is the preferred approach

**Rationale:** After `trainer.fit()` completes, `train.py` reads the best metrics from the checkpoint callback and writes the JSON. This keeps the LightningModule clean and the DVC integration in the script where it belongs.

### Decision 5: Checkpoint format compatibility

**Choice:** Use Lightning's `ModelCheckpoint` which saves full Lightning checkpoints (model state, optimizer state, epoch, etc.) to `data/best_model.ckpt`.

**Alternatives considered:**
- Save only `state_dict` via a custom callback to maintain the current `.pt` format — adds complexity and loses Lightning's resume-from-checkpoint capability

**Rationale:** The Lightning checkpoint format is strictly more useful (supports training resumption). The output path changes from `data/best_model.pt` to `data/best_model.ckpt`, requiring a `dvc.yaml` update.

## Risks / Trade-offs

- **[Checkpoint format change]** → Existing `data/best_model.pt` is incompatible with new format. Mitigation: this is a POC with no deployed models; the DVC pipeline reruns from scratch.
- **[New dependencies]** → `pytorch-lightning` and `torchmetrics` add ~50MB to the environment. Mitigation: acceptable for a training project; both are well-maintained PyTorch ecosystem libraries.
- **[PyG + Lightning interaction]** → PyTorch Geometric's `DataLoader` and `Data` batching must work with Lightning's training loop. Mitigation: Lightning supports custom DataLoaders; PyG's DataLoader is a standard `torch.utils.data.DataLoader` subclass. No known incompatibilities.
- **[MPS compatibility]** → Some PyTorch operations may not be fully supported on MPS. Mitigation: Lightning handles MPS via `accelerator="mps"`; PyG operations used here (GATv2Conv, global_mean_pool) have MPS support in recent PyTorch versions.
- **[W&B double-init]** → Lightning's `WandbLogger` handles `wandb.init()` internally. Must not call `wandb.init()` manually. Mitigation: remove manual init; let `WandbLogger` manage the lifecycle.
