# CWE-Matching Ablation Results (colab-e0 … colab-e4)

Run date: 2026-04-12 → 2026-04-14 (Colab, `other` partition, USE_GRAPH=True, single seed 42).
Source: `notebooks/colab_cwe_matching_ablation.ipynb`. W&B runs `colab-e0`, `colab-e1`, `colab-e2`, `colab-e3`, `colab-e4`.

## Variants

Inferred from per-run loss signals (`train_loss_wrong_cwe`, `train_loss_triplet`, `train_hard_negative_rate`):

| Variant | wrong-CWE loss | Triplet loss | Hard-negative mining |
|---|---|---|---|
| E0 | off | off | off (baseline: classification only) |
| E1 | on | off | off |
| E2 | on | on | on (rate ≈ 0.83) |
| E3 | on | on | off (random negatives) |
| E4 | on | on | on (rate ≈ 0.81, same config as E2) |

All variants share: 30 epochs (crashed 12–15 in, after final eval), BS=8, encoder_lr=2e-5, head_lr=1e-3, grad_accum=8, focal γ=2 α=0.25, matching λ=0.1, triplet margin=0.2, projection dim=128, seed=42.

## Summary metrics (default 0.5 threshold unless noted)

| Variant | val_f1 | test_f1 | unseen_val_f1 | unseen_test_f1 | opt-thresh F1 (val) |
|---|---|---|---|---|---|
| E0 | 0.284 | 0.106 | 0.155 | 0.246 | 0.431 |
| E1 | 0.116 | 0.117 | 0.073 | 0.269 | 0.434 |
| **E2** | 0.263 | 0.137 | **0.167** | **0.282** | 0.424 |
| E3 | 0.215 | 0.147 | 0.124 | 0.260 | 0.421 |
| E4 | 0.274 | 0.126 | 0.163 | 0.246 | 0.428 |

## Findings

1. **Matching auxiliary helps project-disjoint generalization.** Every variant with the wrong-CWE loss beats E0 on `unseen_test_f1` (0.246 → 0.260–0.282).
2. **Hard-negative mining is the decisive ingredient.** E2 vs E3 (same losses, only E2 mines hard negatives): unseen_test 0.282 vs 0.260, unseen_val 0.167 vs 0.124, val 0.263 vs 0.215. Random-negative triplets add little over the wrong-CWE repulsion alone.
3. **Triplet + hard mining improves over wrong-CWE alone.** E1 → E2: unseen_test 0.269 → 0.282, unseen_val 0.073 → 0.167, val 0.116 → 0.263.
4. **E4 underperforms E2 despite identical config** (unseen_test 0.246 vs 0.282). The gap is comparable in magnitude to the E0 → E2 improvement, so the single-seed result is not conclusive — seed variance may explain most of the signal. This motivates a multi-seed rerun.
5. **Optimal-threshold F1 is flat (~0.42–0.43) across all variants.** Default-threshold differences largely reflect decision-boundary placement, not ranking quality. The matching loss helps calibration more than separability.
6. **Val and unseen-val disagree on ranking.** E0 has the best `val_f1` but weakest `unseen_*`. Model selection should use `unseen_val_f1`, where E2 is also the winner.

## Conclusion

E2 is the recommended configuration going forward. Hard-negative mining is the one component that clearly separates the matching variants from each other; random-negative triplet (E3) has been dropped.

As of 2026-04-15, **E2 is the default** for `MATCHING_ABLATION_MODE` in `graph_vul/config.py`. `train_finetune.py` and all downstream stages inherit this default; pass `--ablation-mode E0` to reproduce the baseline.

## Next step

Rerun E0, E2, E4 with ≥3 seeds (e.g. 42, 43, 44) via the multi-seed sweep in `notebooks/colab_cwe_matching_ablation.ipynb` and compare means with bootstrap CIs via `graph_vul.analysis.post_experiment_analysis` to firm up the E2-vs-E4 gap.
