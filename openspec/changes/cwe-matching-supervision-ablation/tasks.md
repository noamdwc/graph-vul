## 1. Matching Supervision Core

- [x] 1.1 Add config and training-entrypoint support for named ablation modes `E0` through `E4`, defaulting to the current baseline behavior
- [x] 1.2 Add shared function/CWE projection heads and optional matching outputs to `FuncFinetuneModel` without changing the current classifier head path
- [x] 1.3 Add wrong-CWE negative sampling with default `1` negative per vulnerable positive and fallback behavior for missing hard negatives
- [x] 1.4 Add vulnerable-only triplet loss with configurable `lambda=0.1` and `margin=0.2`, and log each enabled loss component separately

## 2. Pipeline and Evaluation

- [x] 2.1 Thread the matching-supervision configuration through `FuncFinetuneLitModule`, `train_finetune.py`, checkpoint hyperparameters, and DVC stage commands
- [x] 2.2 Extend prediction export and evaluation to compute Top-1/Top-3 CWE retrieval from the shared matching space
- [x] 2.3 Extend summary outputs to include overall metrics, per-CWE metrics, unseen/external results, and fixed length slices `0-64`, `64-128`, `128-256`, `256+`
- [x] 2.4 Ensure per-variant manifests, metrics, and reports are distinguishable so E0-E4 outputs can be compared cleanly

## 3. Verification

- [x] 3.1 Run at least one baseline (`E0`) and one combined (`E3` or `E4`) smoke test to verify the feature is easy to disable and does not break the current pipeline
- [x] 3.2 Verify exported artifacts include retrieval metrics, per-CWE outputs, unseen/external summaries, and the required function-length slices
