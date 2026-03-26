## Why

The current model trains on a fixed set of 6 CWEs and evaluates on the same CWEs. The held-out CWEs (CWE-200, CWE-416) exist in the data pipeline but are never used during evaluation. To move toward a generalizable vulnerability detector, we need to verify the model can detect vulnerabilities for CWE types it has never seen during training — using the CWE description embedding as the bridge between seen and unseen vulnerability classes.

## What Changes

- **Dataset splits**: Add `unseen_val` and `unseen_test` splits containing only held-out CWE samples (CWE-200, CWE-416), split by project disjointness
- **Training data**: Train only on the 6 existing CWEs (CWE-119, CWE-125, CWE-20, CWE-476, CWE-190, CWE-787) — no change to training CWEs
- **Evaluation**: Evaluate on 3 test sets:
  1. `test` — standard test split (seen CWEs)
  2. `unseen_val` — validation on held-out CWEs for hyperparameter decisions
  3. `unseen_test` — final evaluation on held-out CWEs
- **Metrics**: Report binary classification metrics (accuracy, precision, recall, F1) separately for each evaluation split
- **DVC pipeline**: Update `func_split` and `func_finetune` stages to produce and consume the new splits

## Capabilities

### New Capabilities
- `unseen-cwe-evaluation`: Splitting held-out CWE data into unseen_val/unseen_test and evaluating the trained model on these splits

### Modified Capabilities

## Impact

- `graph_vul/pipeline/split_data.py` or `func_split` — must produce unseen_val/unseen_test assignments
- `graph_vul/dataset/func_finetune_dataset.py` — must support loading unseen splits
- `graph_vul/train_finetune.py` — must create dataloaders for unseen splits and run evaluation
- `graph_vul/model/func_finetune_lit_module.py` — must log metrics for unseen splits
- `dvc.yaml` — update pipeline deps/outs
- `graph_vul/config.py` — add paths for unseen split outputs
