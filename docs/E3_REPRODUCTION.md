# E3 reconstruction — 14 October 2026

The preparation is rebuilt in `data/e3-reproduction-20261014/prepared`. Existing ZIPs, datasets, checkpoints, Joern exports, and caches were read without modification or deletion. The original E3 checkpoint and predictions were copied to `data/e3-reproduction-20261014/reference/E3_all_layers`.

**Current Colab package:** `data/e3-reproduction-20261014/downloads/e3-inputs-full-joern-20261014.zip` contains graph tensors for all 77,300 functions. The earlier E3 and E2 runs documented below used only 5,313 cached graphs; they remain archived but are not the result of this full-coverage preparation. The new package still cannot reproduce the historical graph representation: its rebuilt vocabulary has 61,763 labels, versus 2,822 embeddings in the historical checkpoint, and the original mapping is unavailable.

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

## Earlier partial cache-only preparation (retained)

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

Python scripts and notebook cells parse successfully. The earlier cache-only GPU run completed on Colab; its results are labeled below. The full-coverage GPU run is pending the new Colab execution.

## Full-coverage graph rebuild

Joern 4.0.506 regenerated the 71,987 functions without exact cache hits in one reused JVM/C2Cpg frontend. The run took 16 minutes 24 seconds. It wrote one CPG binary and cpg14 DOT exports for every function; all expected sample IDs completed. Pilot comparisons against the standard CLI produced identical parsed node order and indexed AST/CFG edges for samples `74305` and `74832`.

Combining those exports with 5,313 exact source-hash cache hits yielded 77,300/77,300 AST/CFG tensors, with zero empty-graph fallbacks. The new vocabulary contains 61,763 node labels and two relations. The source functions, labels, and split assignments remain byte-verified, but the lost historical vocabulary and graph-presence mask mean the new graph IDs cannot be proven identical to the original. Treat the next GPU run as a fresh full-coverage E3 measurement.

The full preparation and audit are in `data/e3-reproduction-20261014/recovery/full-joern-20261014/`. `feature_audit.json` records graph coverage and sources; `prepared/graph_export_provenance.json` records Joern version, timing, coverage, and the CLI-equivalence pilot. The existing project parser produced 77,300 `.pt` tensors totaling about 672 MB. A real-model CPU smoke run passed on the full preparation with finite loss and nonzero graph gradients; this is an execution check, not a performance result.

## Run E3 on Colab

1. Upload `data/e3-reproduction-20261014/downloads/e3-inputs-full-joern-20261014.zip` (about 146 MB) to `My Drive/data/graph-vul/e3-reproduction-20261014/`.
2. Open `notebooks/colab_e3_reproduction.ipynb` in Colab and select a GPU runtime.
3. Run its cells. It clones `portfolio/appsflyer`, whose model code is based on `f9867245`, runs its E3-only runner, verifies every prepared file hash, and writes a fresh timestamped output folder to Drive.
4. Return the entire run folder: checkpoint, `run.json`, training metrics, test/unseen-test prediction CSVs, export manifest, and recomputed `metrics.json`.

The full run uses seed 42, 10 epochs, batch size 64, accumulation 8, encoder LR 2e-5, head LR 1e-3, graph fusion in all 12 layers, and late graph concatenation. It selects the checkpoint by validation F1 and evaluates at threshold 0.5. UniXcoder is pinned to revision `5604afdc964f6c53782a6813140ade5216b99006`. The notebook pins the core versions recorded with the historical fine-tuning environment and installs the [official matching PyTorch/CUDA package trio](https://pytorch.org/get-started/previous-versions/). The exact March 30 Colab environment was not archived. Dependency versions are recorded for each run. No other ablation configurations are launched.

On a CUDA host, the equivalent command from this worktree is:

```bash
python scripts/run_e3_reproduction.py \
  --data-root data/e3-reproduction-20261014/recovery/full-joern-20261014/prepared \
  --output-dir data/e3-reproduction-20261014/recovery/full-joern-20261014/runs/E3_all_layers-01
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
  --output /Users/noamc/repos/graph-vul-appsflyer/data/e3-reproduction-NEW \
  --joern-raw /path/to/persistent-joern-exports
```

The `--joern-raw` directory should contain one sample-ID folder per regenerated function. When supplied, the feature rebuild stops if any function lacks a DOT export or graph tensor.

The recovery scripts expect the two downloaded ZIP names used on 14 October. The existing preparation already includes its hash manifest and Colab package; rebuilding again is unnecessary for further runs using this preparation.

## Completed Colab run — partial cache-only features — 14 October 2026

Run `E3_all_layers-20261014T091812Z` completed with return code 0 on an NVIDIA A100-SXM4-40GB. Its source revision is `acaab932945c41eb09a48b9f680b4fcfb5f9a291`, seed 42. All 5,413 prepared file hashes verify, and the run's preparation-manifest hash matches the local preparation. Every exported evaluation function, annotation, label, and split matches. Predictions agree with the fixed 0.5 threshold and contain finite scores and unique sample IDs.

The selected checkpoint is epoch 9 (zero-based), global step 670: the tenth epoch of the configured budget. Its callback selected on validation F1, with best score **0.877980**. The checkpoint contains all 12 fusion layers, late graph concatenation, 9,678 node types, and two graph relations. Its SHA-256 is `4673f949d91c44e4896bff0e8405f9c53974b92855ea878d6ca8a20eae5c85b8`.

| Cohort | Rows | Positive support | F1 | Precision | Recall | Benign false-positive rate | PR-AUC (average precision) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Test | 4,727 | 111 | 0.867925 | 0.910891 | 0.828829 | 0.001950 | 0.868150 |
| Unseen test | 14,430 | 870 | 0.886573 | 0.950066 | 0.831034 | 0.002802 | 0.926603 |

Confusion counts: test TP=92, FP=9, TN=4,607, FN=19; unseen test TP=723, FP=38, TN=13,522, FN=147. These are fresh, single-seed results on reconstructed features for CWE-conditioned classification. They do not establish a graph-fusion improvement over a baseline or exact reproduction of the March graph inputs.

The downloaded `metrics.json` agrees with recomputed F1/precision/recall but reports NaN for ranking/calibration metrics. The historical helper at `graph_vul/analysis/post_experiment_analysis.py:315` checks `score.notna().nunique()` rather than uniqueness of the scores themselves. Its complete finite score columns therefore skip those computations. PR-AUC, ROC-AUC, and Brier score were recalculated directly from the exported scores in `runs/E3_all_layers-20261014T091812Z/audit.json`. This is a reporting issue; no additional training is needed. The original downloaded files and extracted raw results are preserved.

The executed notebook contains a successful subprocess return and final metrics table. It does not retain the complete epoch-by-epoch child-process logs. Installation reported dependency conflicts with unused Colab packages, but training and export completed successfully.

A defensible project description, pending the full-coverage run's metrics:

> Built a CWE-conditioned vulnerability classifier combining pretrained code representations with AST/CFG graph fusion and evaluated it on held-out projects.

Do not attach the partial-cache E3 F1 to this project description. Replace this sentence with the full-coverage run's measured metrics only after that Colab run completes.

## Short-function check — 14 October 2026

The historical short-function bucket includes exported `token_length` from 0 through 64, inclusive. Length is the inference attention-mask sum, including special tokens; predictions use threshold 0.5. Fresh and original E3 exports have identical sample IDs, token lengths, labels, and short-function membership.

| Cohort | Short functions | Vulnerable functions | Original E3 F1 | Fresh E3 F1 |
|---|---:|---:|---:|---:|
| Test | 1,054 | 6 | 0.285714 | 0.285714 |
| Unseen test | 1,938 | 19 | 0.647059 | 0.400000 |

For unseen short functions, original E3 has TP=11, FP=4, TN=1,915, FN=8. Fresh E3 has TP=6, FP=5, TN=1,914, FN=13: precision 0.545455, recall 0.315789. These metrics are independently recomputed from both prediction CSVs. They describe different feature preparations and do not isolate a causal effect of a model change. The short cohort has only 19 positive examples.

Two historical summaries explain the earlier improvement figures:

| Historical comparison | Unseen short F1 | Evidence status |
|---|---:|---|
| E2 last-four-layer fusion → E3 all-layer fusion | 0.411765 → 0.647059 | `outputs/fusion_ablation_results.csv`; original E3 predictions verified, E2 predictions not recovered |
| E0 late-only fusion → E6 gated concat adapter | 0.457143 → 0.744186 | Main repository `outputs/fusion_method_ablation_detailed_results.csv`, introduced at `3336331`; E0/E6 predictions not recovered |

The 0.457→0.744 figure belongs to the later E6 experiment, not the E3 architecture selected at `f9867245`. Neither historical baseline can be compared directly with the fresh E3 run because original feature identity is unavailable. A CV claim that the reproduced model improved short-function F1 is currently unsupported. The controlled E2 comparison was subsequently completed on the exact reconstructed preparation, with the same seed, training budget, checkpoint selection, threshold, and cohort; its results are recorded below. No additional training was launched during the initial short-function check.

Machine-readable evidence, source hashes, counts, and cohort checks are saved in `runs/E3_all_layers-20261014T091812Z/short_function_audit.json`. Original downloaded artifacts remain unchanged.

## E2 controlled baseline on Colab

The archived E2/E3 comparison below used the partial cache-only preparation. To compare model depth on the full-coverage features later, rerun E2 using the new `e3-inputs-full-joern-20261014.zip` and the new full-coverage E3 `run.json` as its reference. The current E2 notebook still targets the archived partial-cache run.

The hypothesis was that all-layer fusion yields higher unseen short-function F1 than last-four-layer fusion. E2 changed only fusion layers to `[8, 9, 10, 11]`; it retained late graph concatenation, seed 42, ten epochs, batch 64, accumulation 8, encoder/head learning rates, validation-F1 checkpoint selection, and threshold 0.5. The 0.400 E3 short F1 below is from the partial cache-only preparation. It does not predict the full-coverage E3 result.

The shared runner defaults to E3 for existing callers. E2 requires `--experiment E2 --reference-run PATH/TO/E3/run.json`. CPU smoke mode exercises the requested fusion depth without requiring a comparable GPU runtime. Full E2 starts from the pinned pretrained encoder, not the E3 checkpoint.

The notebook writes a fresh `runs/E2_last4-YYYYMMDDTHHMMSSZ` directory, a sibling `E2_last4-YYYYMMDDTHHMMSSZ-training.log`, and `comparison.json` in the run directory. It verifies paired sample IDs, labels, and token lengths before comparing all functions and the inclusive 0–64-token bucket. Return the entire E2 directory, training log, and executed notebook. The primary comparison is E3 minus E2 short unseen-test F1; positive values support the hypothesis for this seed and cohort. Overall F1/precision/recall and confusion counts provide context. With only 19 vulnerable short functions, a positive delta alone does not establish a stable improvement across seeds or populations.

Local validation: the E2 CPU smoke performed one optimizer step on four real functions with both labels and present/missing graphs; loss 0.649392, finite inference scores, and 576 nonzero graph-embedding gradient entries. Its evidence is retained in `smoke-e2-01/`. Runner and notebook Python cells parse. The comparison cell was executed against identical reference predictions and returned the expected zero delta, with exact short-function support (1,938/19). Matching reference provenance is accepted; altered input manifest, runtime versions, seed, or reference fusion depth are rejected. These are execution and comparison checks, not full E2 performance results.

### Recover a final-cell import failure

If training completed but the notebook's final comparison cell fails during NumPy/SciPy import, preserve the completed E2 folder. The reported `numpy.ufunc.__module__` error is consistent with notebook modules remaining loaded across the pinned-package installation; the comparison originally imported analysis dependencies inside that long-lived kernel. `scripts/compare_e2_e3_runs.py` now performs the same comparison in a fresh Python subprocess, matching the successful training process strategy. This changes reporting execution only, not training, predictions, or metric definitions.

In the existing notebook session, pull the updated `portfolio/appsflyer` checkout and launch only the comparison script with `--e2-run OUTPUT --e3-run REFERENCE`. Do not rerun the training cell. The script reads both prediction exports and writes `comparison.json` only in the completed E2 directory.

## Completed E2/E3 controlled comparison — partial cache-only features — 14 October 2026

The downloaded E2 run `E2_last4-20261014T104751Z` completed ten epochs on an NVIDIA A100-SXM4-40GB. Training source is `43066abf1097d59745adc1de026e417efa83312e`. The complete `graph_vul` code is unchanged from the E3 source `acaab932945c41eb09a48b9f680b4fcfb5f9a291`; only the runner's requested fusion depth differs. Preparation-manifest hash, all 5,413 prepared file hashes, encoder revision, recorded dependency versions, seed, budget, learning rates, batch/accumulation, late graph concatenation, checkpoint-selection metric, and threshold match E3. Every evaluation sample ID and all exported metadata, including labels and token lengths, match exactly.

The selected E2 checkpoint is epoch 9 (zero-based), global step 670, with validation F1 0.882108. Its saved hyperparameters and fusion parameter keys confirm layers `[8, 9, 10, 11]`, late graph concatenation, 9,678 node types, and two graph relations. ZIP members passed CRC checks during extraction into a fresh local directory. Original ZIPs, notebooks, logs, and previous results remain preserved.

| Cohort | Functions | Vulnerable functions | E2 F1 | E3 F1 | E3 minus E2 |
|---|---:|---:|---:|---:|---:|
| Test, all | 4,727 | 111 | 0.840708 | 0.867925 | +0.027217 |
| Unseen test, all | 14,430 | 870 | 0.889559 | 0.886573 | −0.002987 |
| Test, 0–64 tokens | 1,054 | 6 | 0.500000 | 0.285714 | −0.214286 |
| Unseen test, 0–64 tokens | 1,938 | 19 | 0.387097 | 0.400000 | +0.012903 |

All values were recomputed independently from the prediction CSVs and agree with the downloaded `comparison.json`. On unseen short functions both models detect the same six vulnerable examples and miss the same thirteen. E2 has six false positives, E3 five. Four false positives are shared; E3 corrects two E2 false positives and introduces one. Thus the observed +0.012903 F1 difference is driven by one fewer false positive, with no recall improvement. On the short test slice E3 detects only one vulnerable example versus E2's two.

| All-function metric | E2 test | E3 test | E2 unseen test | E3 unseen test |
|---|---:|---:|---:|---:|
| Precision | 0.826087 | 0.910891 | 0.936468 | 0.950066 |
| Recall | 0.855856 | 0.828829 | 0.847126 | 0.831034 |
| False positives | 20 | 9 | 50 | 38 |
| True positives | 95 | 92 | 737 | 723 |
| PR-AUC (average precision) | 0.855779 | 0.868150 | 0.935525 | 0.926603 |

E3 improves test F1 by 0.027217 and reduces false positives, while also missing three additional vulnerable test functions. On unseen test it reduces false positives by twelve but misses fourteen additional vulnerable functions; overall F1 and average precision are slightly lower. These are single-seed measurements for CWE-conditioned classification on reconstructed graph/CWE features. The depth comparison does not reproduce the historical short-function gain, and it does not test the later E6 gated concat adapter.

**CV conclusion:** do not claim short-function F1 improved from about 0.4 to 0.7. The old 0.886573 F1 and 0.926603 average precision are real measurements from the partial cache-only E3 run, but they are not the full-coverage result. Until the new Colab run finishes, describe the model and held-out evaluation without attaching a performance number. E2's 0.889559 unseen F1 is likewise from the partial-cache comparison.

The notebook retains the original comparison-cell NumPy import error and a subsequent successful comparison subprocess (`returncode=0`). The error occurred after training/export and was recovered without repeating training. The full training log confirms the ten-epoch budget was reached and the selected checkpoint was restored for evaluation. Saved ranking/calibration metrics still contain NaN from the previously diagnosed historical analysis-helper bug; this audit recalculates them directly from finite probabilities without changing raw results.

Machine-readable verification, hashes, checkpoint settings, confusion counts, paired short predictions, all/short metrics, and E3-minus-E2 deltas are retained in `runs/E2_last4-20261014T104751Z/audit.json`.
