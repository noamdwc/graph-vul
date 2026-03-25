## ADDED Requirements

### Requirement: MLP classifier architecture
The `FuncBaselineModel` SHALL be a 2-layer MLP: Linear(768, FUNC_MLP_HIDDEN_DIM) → ReLU → Dropout(FUNC_MLP_DROPOUT) → Linear(FUNC_MLP_HIDDEN_DIM, 1). Default hidden dim is 128, dropout is 0.3.

#### Scenario: Forward pass output shape
- **WHEN** a batch of N embeddings (shape `[N, 768]`) is passed through the model
- **THEN** the output SHALL have shape `[N, 1]` (raw logits, no sigmoid)

### Requirement: Dataset loading
The `FuncDataset` SHALL be a standard `torch.utils.data.Dataset` (not PyG) that loads pre-computed `.pt` embeddings and returns `(embedding: Tensor[768], label: Tensor[scalar])` tuples. It SHALL filter functions by split using the split assignments CSV.

#### Scenario: Dataset item format
- **WHEN** `__getitem__` is called
- **THEN** it SHALL return a tuple of (768-dim float tensor, scalar float tensor)

#### Scenario: Split filtering
- **WHEN** a FuncDataset is created with split="train"
- **THEN** it SHALL only include functions assigned to the train split

### Requirement: Loss function
Training SHALL use `BCEWithLogitsLoss` applied to the raw model output.

#### Scenario: Loss computation
- **WHEN** the model produces logits
- **THEN** `BCEWithLogitsLoss` SHALL be applied directly without a prior sigmoid

### Requirement: Class-balanced sampling
Training SHALL use `WeightedRandomSampler` to address class imbalance, with weights inversely proportional to class frequency.

#### Scenario: Balanced batches
- **WHEN** the training DataLoader samples a batch
- **THEN** vulnerable and non-vulnerable functions SHALL appear in approximately equal proportion

### Requirement: Lightning training module
A `FuncLitModule` SHALL wrap the MLP model with AdamW optimizer and LambdaLR scheduler, following the same pattern as the existing `lit_module.py`.

#### Scenario: Optimizer configuration
- **WHEN** the Lightning module configures optimizers
- **THEN** it SHALL use AdamW with learning rate FUNC_LEARNING_RATE (1e-3)

### Requirement: Training script
`train_baseline.py` SHALL orchestrate training with:
- Standard `torch.utils.data.DataLoader` (no PyG dependency)
- `ModelCheckpoint` on `val_f1`
- `WandbLogger` for metric tracking
- `trainer.test()` after fit
- Metrics saved to `data/func_metrics.json`

#### Scenario: End-to-end training
- **WHEN** `python -m graph_vul.train_baseline` is run
- **THEN** the model SHALL train for FUNC_NUM_EPOCHS (50) epochs, save the best checkpoint by val_f1, run test evaluation, and write metrics to `data/func_metrics.json`

#### Scenario: Metrics output format
- **WHEN** training completes
- **THEN** `data/func_metrics.json` SHALL contain at minimum: test accuracy, precision, recall, and F1

### Requirement: Training hyperparameters from config
Training SHALL use FUNC_BATCH_SIZE (64), FUNC_LEARNING_RATE (1e-3), and FUNC_NUM_EPOCHS (50) from config, overridable via CLI args.

#### Scenario: CLI override
- **WHEN** `--epochs 100` is passed on the command line
- **THEN** training SHALL run for 100 epochs instead of the config default
