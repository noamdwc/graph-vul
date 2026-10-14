# E3 reconstruction — 14 October 2026

The preparation is rebuilt in `data/e3-reproduction-20261014/prepared`. Existing ZIPs, datasets, checkpoints, Joern exports, and caches were read without modification or deletion. The original E3 checkpoint and predictions were copied to `data/e3-reproduction-20261014/reference/E3_all_layers`.

## What is verified

The local BigVul archive matches the historical MD5 `95436ff6b6eb8b95c44f42dee5fe7541`. Running the filtering code from `f9867245` produced a **byte-identical** `func_subset.csv`: MD5 `4f08bfe20f5c8e1ad79140990706f719`.

The rebuilt 77,300-function corpus has 34,191 train, 8,465 validation, 4,727 test, 15,487 unseen-validation, and 14,430 unseen-test rows. Every original evaluation sample ID, source function, project, CWE, vulnerability label, and split matches. There are zero project overlaps and zero exact-code overlaps between the rebuilt partitions. The historical training sampler order also matches the dataset order.

Recomputed metrics from the original prediction CSVs:

| Cohort | Rows | Positive support | F1 | Precision | Recall | Benign false-positive rate | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Test | 4,727 | 111 | 0.882629 | 0.921569 | 0.846847 | 0.001733 | 0.887223 |
| Unseen test | 14,430 | 870 | 0.891875 | 0.951760 | 0.839080 | 0.002729 | 0.908995 |

These are recalculations from saved predictions, not newly executed model predictions. The historical task supplies annotated CWE embeddings during inference, so these results describe CWE-conditioned classification.

Machine-readable evidence: `preparation_audit.json`, `feature_audit.json`, and `reference_metrics.json` in the reconstruction directory. Prepared input hashes are recorded in `prepared/reconstruction.json` and verified by the runner before execution.

## Reconstructed features

5,313 Joern CPGs were recovered from the existing cache by SHA-256 of the stripped original source function. The historical `extract_graphs` producer generated AST+CFG tensors in the new preparation. Samples without cached exports retain the original code's empty-graph behavior.

| Partition | Functions | Recovered graphs | Empty graphs |
|---|---:|---:|---:|
| Train | 34,191 | 2,028 | 32,163 |
| Validation | 8,465 | 153 | 8,312 |
| Test | 4,727 | 872 | 3,855 |
| Unseen validation | 15,487 | 911 | 14,576 |
| Unseen test | 14,430 | 1,349 | 13,081 |

**Original graph identity is not recovered.** The rebuilt vocabulary has 9,678 node types and two relations (AST/CFG); the original checkpoint has 2,822 node embeddings. Original vocabulary mappings and graph-presence masks are unavailable. The recovered checkpoint cannot be scored against the rebuilt vocabulary. A fresh E3 run on this preparation must report its own measured results, not inherit the historical 0.891875 F1.

85 required CWE vectors were copied from the downloaded archive; their historical identity is unverified. Four missing vectors (`CWE-436`, `CWE-494`, `CWE-769`, `CWE-90`) were regenerated with the cached DeBERTa revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3` and description-only texts derived from the pinned local CWE catalog. Their texts are retained in `prepared/regenerated_cwe_descriptions.json`. Historical embedding identity remains unverified.

## Completed execution checks

A real UniXcoder CPU smoke run used both labels, present and missing graphs, all 12 fusion layers, and one optimizer step. Its loss and inference scores were finite; graph-encoder gradients were nonzero. It used four functions and 128-token sequences and does not estimate performance. Evidence is in `smoke-01/smoke.json` and `smoke-01/run.json`.

Python scripts and notebook cells parse successfully. The local runtime has no CUDA GPU, so full training has not been executed.

## Run E3 on Colab

1. Upload `data/e3-reproduction-20261014/e3-inputs-20261014.zip` (about 23 MB) to `My Drive/data/graph-vul/e3-reproduction-20261014/`.
2. Open `notebooks/colab_e3_reproduction.ipynb` in Colab and select a GPU runtime.
3. Run its cells. It clones `portfolio/appsflyer`, whose model code is based on `f9867245`, runs its E3-only runner, verifies every prepared file hash, and writes a fresh timestamped output folder to Drive.
4. Return the entire run folder: checkpoint, `run.json`, training metrics, test/unseen-test prediction CSVs, export manifest, and recomputed `metrics.json`.

The full run uses seed 42, 10 epochs, batch size 64, accumulation 8, encoder LR 2e-5, head LR 1e-3, graph fusion in all 12 layers, and late graph concatenation. It selects the checkpoint by validation F1 and evaluates at threshold 0.5. UniXcoder is pinned to revision `5604afdc964f6c53782a6813140ade5216b99006`. The notebook pins the core versions recorded with the historical fine-tuning environment and installs the [official matching PyTorch/CUDA package trio](https://pytorch.org/get-started/previous-versions/). The exact March 30 Colab environment was not archived. Dependency versions are recorded for each run. No other ablation configurations are launched.

On a CUDA host, the equivalent command from this worktree is:

```bash
python scripts/run_e3_reproduction.py \
  --data-root data/e3-reproduction-20261014/prepared \
  --output-dir data/e3-reproduction-20261014/runs/E3_all_layers-01
```

The output directory must be new. The runner never reuses the recovered original checkpoint for fresh training.

## Rebuild the preparation again

The following commands require new output paths. They preserve the original source repository's artifacts:

```bash
python scripts/prepare_e3_reproduction.py \
  --repo /Users/noamc/repos/graph-vul-appsflyer \
  --source-repo /Users/noamc/repos/graph-vul \
  --output /Users/noamc/repos/graph-vul-appsflyer/data/e3-reproduction-NEW
python scripts/rebuild_e3_features.py \
  --repo /Users/noamc/repos/graph-vul-appsflyer \
  --source-repo /Users/noamc/repos/graph-vul \
  --output /Users/noamc/repos/graph-vul-appsflyer/data/e3-reproduction-NEW
```

The recovery scripts expect the two downloaded ZIP names used on 14 October. The existing preparation already includes its hash manifest and Colab package; rebuilding again is unnecessary for the pending GPU run.
