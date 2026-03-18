### Requirement: Trainer instantiation with callbacks
The training script SHALL instantiate a `pytorch_lightning.Trainer` with a `ModelCheckpoint` callback configured to monitor `val_f1` in `max` mode, saving the best model to `data/best_model.ckpt`.

#### Scenario: Trainer setup
- **WHEN** the training script configures the Trainer
- **THEN** the Trainer SHALL have a `ModelCheckpoint` callback monitoring `val_f1` with `mode="max"` and `save_top_k=1`

### Requirement: Epoch and device configuration
The Trainer SHALL accept `max_epochs` from the CLI arguments and SHALL automatically select the accelerator based on hardware availability, supporting CUDA GPUs, Apple Silicon MPS, and CPU fallback.

#### Scenario: CUDA GPU available
- **WHEN** CUDA is available on the system
- **THEN** the Trainer SHALL use the GPU accelerator

#### Scenario: MPS available (Apple Silicon)
- **WHEN** no CUDA is available but MPS is available
- **THEN** the Trainer SHALL use the MPS accelerator

#### Scenario: CPU only
- **WHEN** neither CUDA nor MPS is available
- **THEN** the Trainer SHALL use the CPU accelerator

### Requirement: Reproducibility via seed_everything
The training script SHALL call `pytorch_lightning.seed_everything` with the configured seed before training to ensure reproducible results, replacing the manual seed function.

#### Scenario: Deterministic training
- **WHEN** two training runs use the same seed and data
- **THEN** they SHALL produce identical loss and metric sequences

### Requirement: DVC metrics JSON output after training
After `trainer.fit()` completes, the training script SHALL write the best validation metrics (accuracy, precision, recall, F1) to `data/metrics.json` for DVC tracking. The metrics SHALL be read from `trainer.callback_metrics`.

#### Scenario: Metrics file written
- **WHEN** training completes
- **THEN** `data/metrics.json` SHALL exist containing keys `accuracy`, `precision`, `recall`, `f1` with the best validation values

### Requirement: CLI argument compatibility
The training script SHALL accept `--epochs`, `--batch-size`, `--lr`, and `--seed` arguments. The DVC pipeline stage command SHALL work with the updated script.

#### Scenario: DVC pipeline execution
- **WHEN** `dvc repro` runs the train stage
- **THEN** the training script SHALL accept the same CLI arguments as before and produce `data/best_model.ckpt` and `data/metrics.json`
