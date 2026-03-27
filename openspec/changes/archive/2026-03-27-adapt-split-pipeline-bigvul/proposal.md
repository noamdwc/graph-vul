## Why

The current split pipeline separates held-out CWE **rows** but does not exclude the **projects** that contain those rows from the seen regime. This means a project can appear in both `train` and `unseen_*` splits through different CWEs, leaking project-level signal across regimes. Additionally, the training set is highly imbalanced (~8% vulnerable) with no rebalancing mechanism, and evaluation splits lack clear separation between seen-CWE and unseen-CWE regimes. We need a stricter split design for the BigVul function-level subset that enforces full-project holdout for held-out CWEs and provides a balanced training view.

## What Changes

- **Full-project holdout for held-out CWEs**: If any row in a project has a held-out CWE, the entire project moves to the unseen pool — no rows from that project appear in train/val/test. **BREAKING**: projects that previously leaked across regimes will now be fully excluded from seen splits.
- **Five-way split**: Produce `train`, `val`, `test`, `unseen_val`, `unseen_test` assignments, with seen/unseen regimes fully disjoint by project.
- **Held-out CWE eligibility validation**: Add configurable thresholds (min rows, min vulnerable rows, min projects) to reject held-out CWEs that are too small for meaningful evaluation.
- **Strict cross-regime overlap validation**: Check all 10 split pairs for zero project overlap, not just within-regime pairs.
- **Train balancing artifact**: Produce a separate `train_sampler_weights.csv` with per-row sample weights for weighted sampling during training. Evaluation splits remain in their natural distribution. **BREAKING**: downstream training code must adopt weighted sampling or ignore the new artifact.
- **Extended reporting**: Add project exclusivity report (which CWEs caused project migration), train balancing report (before/after ratios, CWE distribution, project contribution), and all-pairs overlap verification.

## Capabilities

### New Capabilities
- `train-balancing`: Configurable per-row weight computation for balanced training via weighted sampling. Covers vulnerable/benign ratio targeting, CWE inverse-frequency smoothing, project downweighting, and weight capping.

### Modified Capabilities
- `leakage-aware-split`: Requirements change from row-level CWE holdout to full-project holdout for held-out CWEs. Adds held-out CWE eligibility thresholds, all-pairs overlap validation, and five-way split output.
- `data-split`: Requirements change to support five splits instead of three, add unseen regime, and include the train balancing artifact in stage output.

## Impact

- **Code**: `graph_vul/pipeline/split_data.py` (major changes), `graph_vul/config.py` (new config params)
- **DVC pipeline**: `dvc.yaml` split_data and func_split stages may need updated output declarations
- **Tests**: `tests/test_split_data.py` needs rewrite to match new API (tests already reference removed functions)
- **Downstream**: Training code (`train.py`, `train_finetune.py`, dataset classes) should adopt weighted sampling using the new artifact; existing `assignments.csv` format is preserved
- **Data**: `data/splits/` and `data/func_splits/` will contain additional files (`train_sampler_weights.csv`)
