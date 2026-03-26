## Context

Training runs for a fixed 50 epochs with no early stopping—only a `ModelCheckpoint` saves the best `val_f1` checkpoint. The subset sampler targets 1 000 vulnerable + 2 000 benign rows (~3 000 total) and hands them to a `StratifiedGroupKFold` splitter that preserves the natural 1:2 class ratio across train/val/test. A `WeightedRandomSampler` in the DataLoader partially compensates during training but doesn't affect the val split's imbalance.

## Goals / Non-Goals

**Goals:**
- Stop training automatically when `val_f1` plateaus, saving compute and reducing overfit risk.
- Produce class-balanced train and val splits so the model sees equal positive/negative examples and validation metrics are not skewed by class prevalence.
- Keep test at its natural class ratio for realistic evaluation.
- Maintain ~3 000 total samples across all splits.

**Non-Goals:**
- Changing the model architecture, loss function, or optimizer.
- Modifying the CVE selection / diversity-scoring logic.
- Adjusting per-group caps or deduplication.

## Decisions

### 1. Early stopping callback

**Choice:** Add PyTorch Lightning `EarlyStopping(monitor="val_f1", mode="max", patience=7, min_delta=0.005)` alongside the existing `ModelCheckpoint`.

**Rationale:** `val_f1` is already logged and used for checkpointing, so no new metric plumbing is needed. Patience of 7 gives the LR scheduler time to recover from temporary plateaus. `min_delta=0.005` ignores trivially small improvements.

**Alternative considered:** Monitoring `val_loss`—rejected because F1 is more directly aligned with our evaluation goal and is already the checkpoint metric.

### 2. Balanced train/val via sampling targets

**Choice:** Change the subset sampling step to produce roughly equal vulnerable and benign samples overall (~1 500 each, ~3 000 total). Then in the split step, apply class-balanced allocation to train and val while letting test take whatever natural ratio falls out of the project-disjoint split.

Concretely:
- `TARGET_VULNERABLE_SAMPLES = 1200`, `TARGET_BENIGN_SAMPLES = 1200` (equal).
- After the project-disjoint split assigns rows to train/val/test, **undersample the majority class in train and val** to achieve 1:1 ratio within each. Test keeps all its assigned rows untouched.
- The `WeightedRandomSampler` in the DataLoader becomes unnecessary for the graph pipeline (train is already balanced) but can remain as a no-op safety net.

**Rationale:** Balancing at the split level (not just the DataLoader) ensures val metrics reflect balanced evaluation and the model trains on genuinely balanced batches. Keeping test natural means our final metrics reflect real-world-like prevalence.

**Alternative considered:** Only balance via DataLoader sampler—rejected because val would remain imbalanced, skewing F1 reporting.

### 3. Config parameters

New constants in `config.py`:
- `EARLY_STOPPING_PATIENCE = 7`
- `EARLY_STOPPING_MIN_DELTA = 0.005`

Updated constants:
- `TARGET_VULNERABLE_SAMPLES = 1200`
- `TARGET_BENIGN_SAMPLES = 1200`

## Risks / Trade-offs

- **Smaller effective training set**: Undersampling majority class in train reduces total training rows. Mitigation: the model already trains with balanced batches via `WeightedRandomSampler`; making the split itself balanced just makes this explicit and consistent.
- **Test set class ratio varies by run**: Because test keeps whatever projects fall into it, the ratio may shift across re-samples. Mitigation: `StratifiedGroupKFold` still stratifies by label, limiting drift; report test class distribution in split report.
- **Patience tuning**: 7 may be too aggressive or too conservative. Mitigation: expose as config constant so it's easy to adjust.
