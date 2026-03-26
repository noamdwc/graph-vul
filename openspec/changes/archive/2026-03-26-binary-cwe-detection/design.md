## Context

The model already performs binary vulnerability detection with CWE description embeddings as auxiliary input (code + CWE embedding → vulnerable/benign). Training uses 6 CWEs; held-out CWEs (CWE-200, CWE-416) are filtered out in `split_data.py` but never evaluated against. The architecture is already set up for zero-shot CWE generalization — the CWE embedding provides the semantic bridge — but we lack the evaluation infrastructure to measure it.

## Goals / Non-Goals

**Goals:**
- Split held-out CWE samples into `unseen_val` and `unseen_test` (project-disjoint)
- Evaluate the trained model on all three test sets: `test`, `unseen_val`, `unseen_test`
- Report per-split binary metrics in the DVC metrics file
- Keep training data and model architecture unchanged

**Non-Goals:**
- Changing the model architecture or loss function
- Adding new CWEs beyond the existing 6 + 2 held-out
- Few-shot or meta-learning approaches
- Per-CWE metric breakdowns within splits

## Decisions

### 1. Unseen CWE splitting strategy
**Decision**: Use StratifiedGroupKFold on held-out CWE rows (same as main split), with project as group, producing a 50/50 val/test split.

**Rationale**: Project disjointness prevents data leakage. A 50/50 split maximizes sample count in both sets given the smaller held-out pool. The same splitting logic reuses existing infrastructure.

**Alternative considered**: Random split without project grouping — rejected because it allows the same project's functions to appear in both unseen_val and unseen_test.

### 2. Evaluation integration
**Decision**: Add unseen dataloaders to `train_finetune.py` and call `trainer.test()` on each after training completes. Log metrics with `unseen_val_` and `unseen_test_` prefixes.

**Rationale**: Keeps evaluation in the same training script, avoids a separate evaluation stage. The Lightning module already supports multiple test dataloaders via `test_step` with a `dataloader_idx`.

**Alternative considered**: Separate DVC `evaluate` stage — rejected as over-engineering for two additional test calls.

### 3. Split file format
**Decision**: Extend the existing `assignments.csv` to include `unseen_val` and `unseen_test` as split values (alongside `train`, `val`, `test`).

**Rationale**: Single file, same schema, same loading code in the dataset class. The dataset already filters by split name — adding new split names requires no structural changes.

### 4. Metrics output
**Decision**: Extend `func_finetune_metrics.json` with `unseen_val_*` and `unseen_test_*` keys.

**Rationale**: DVC tracks a single metrics file. Adding keys is simpler than adding files.

## Risks / Trade-offs

- **Small unseen set size** → Metrics may have high variance. Mitigation: report sample counts alongside metrics so results can be interpreted with appropriate confidence.
- **Project distribution skew in held-out CWEs** → If held-out CWEs come from few projects, the project-disjoint split may be unbalanced. Mitigation: log split statistics in the split report.
- **Class imbalance in unseen splits** → Held-out CWEs may have different vul/benign ratios. Mitigation: use the same WeightedRandomSampler logic is not needed for test-only splits; just report the class distribution.
