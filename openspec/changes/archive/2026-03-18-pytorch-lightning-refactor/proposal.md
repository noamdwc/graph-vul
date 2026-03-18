## Why

The training loop in `graph_vul/train.py` is a hand-rolled PyTorch loop with manual device management, gradient handling, checkpointing, and metric computation. This makes it harder to add features like multi-GPU training, mixed precision, early stopping, or learning rate scheduling without significant boilerplate. PyTorch Lightning provides these out of the box while keeping the model code clean and testable.

## What Changes

- Wrap `GraphVulModel` in a `LightningModule` that encapsulates the training step, validation step, optimizer configuration, and metric logging
- Replace the manual training loop in `train.py` with a Lightning `Trainer`
- Integrate existing W&B logging through Lightning's `WandbLogger` callback instead of manual `wandb.log()` calls
- Add Lightning's built-in `ModelCheckpoint` callback to replace manual best-model saving
- Add `pytorch-lightning` and `torchmetrics` to `requirements.txt`
- **BREAKING**: The `train.py` CLI interface will change — arguments now map to Lightning Trainer flags

## Capabilities

### New Capabilities
- `lightning-module`: LightningModule wrapper for GraphVulModel with training/validation steps, optimizer config, and metric computation
- `lightning-trainer`: Trainer setup with callbacks (checkpointing, early stopping), W&B logger integration, and CLI argument mapping

### Modified Capabilities
- `training-loop`: Training/validation split and epoch logic moves into Lightning's Trainer and DataModule pattern
- `wandb-tracking`: W&B initialization and metric logging shifts from manual calls to Lightning's WandbLogger callback

## Impact

- **Code**: `graph_vul/train.py` rewritten; new module(s) under `graph_vul/model/` for the LightningModule
- **Dependencies**: `pytorch-lightning`, `torchmetrics` added to `requirements.txt`
- **DVC pipeline**: The `train` stage command in `dvc.yaml` may need updated arguments
- **Model checkpoints**: Checkpoint format changes from manual `torch.save` state dict to Lightning checkpoint format (includes optimizer state, epoch, etc.)
