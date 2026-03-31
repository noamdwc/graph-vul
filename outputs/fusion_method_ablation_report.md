# Fusion Method Ablation — Experiment Report

## 1. Introduction

This report summarizes the results of the **fusion method ablation study**, which investigates which graph-to-code fusion mechanism works best for injecting pooled graph embeddings (from an AST+CFG RGCN encoder) into UniXcoder transformer layers for vulnerability detection.

The central question is: **given that internal graph fusion improves detection (established in the prior depth ablation), which fusion method should be used — and does it matter for short functions (0–64 tokens)?**

Eight fusion methods were evaluated, all using the same training data, hyperparameters, and graph embeddings.

## 2. Experimental Setup

### 2.1 Configurations

| ID | Tag | Fusion Method | Description |
|----|-----|---------------|-------------|
| E0 | `E0_late_only` | Late only | Baseline: concatenate graph embedding after encoder (no internal fusion) |
| E1 | `E1_gated_residual` | Gated residual | `h + sigmoid(gate(h)) * proj(g)` — learned scalar gate per hidden dim |
| E2 | `E2_ungated_residual` | Ungated residual | `h + proj(g)` — direct additive injection, no gating |
| E3 | `E3_scalar_gated_residual` | Scalar gated residual | `h + sigmoid(scalar_gate) * proj(g)` — single learned scalar gate |
| E4 | `E4_film` | FiLM | Feature-wise Linear Modulation: `gamma(g) * h + beta(g)` |
| E5 | `E5_concat_adapter` | Concat adapter | Concatenate `[h; g]` and project back to hidden dim |
| E6 | `E6_gated_concat_adapter` | Gated concat adapter | Concat adapter with a learned gate controlling injection strength |
| E7 | `E7_cross_attention_graph_tokens` | Cross-attention | Cross-attend over graph token sequence |

All experiments use `USE_GRAPH=True` with identical training hyperparameters and late graph concatenation (except E0 which uses late concat only).

### 2.2 Training

| Parameter | Value |
|-----------|-------|
| Training time per experiment | 89–116 minutes |
| GPU | NVIDIA L4 (23 GB) |
| Total runtime | ~13 hours |

### 2.3 Evaluation Splits

- **Test (seen):** 114,442 samples — CWE types seen during training
- **Unseen test:** 465,822 samples — CWE types held out entirely from training

No validation-split predictions were exported; analysis is limited to test and unseen test.

## 3. Results

### 3.1 Overall Metrics

#### Seen Test (Known CWE Types)

| Experiment | F1 | Precision | Recall | Accuracy |
|------------|----|-----------|--------|----------|
| E0: Late only (baseline) | 0.880 | 0.939 | 0.829 | 0.9947 |
| E1: Gated residual | 0.875 | 0.938 | 0.820 | 0.9945 |
| **E2: Ungated residual** | **0.886** | 0.939 | **0.838** | **0.9949** |
| E3: Scalar gated residual | 0.845 | 0.853 | 0.838 | 0.9928 |
| E4: FiLM | 0.871 | 0.929 | 0.820 | 0.9943 |
| E5: Concat adapter | 0.047 | 0.024 | 0.712 | 0.3167 |
| E6: Gated concat adapter | 0.871 | 0.929 | 0.820 | 0.9943 |
| E7: Cross-attention | 0.000 | 0.000 | 0.000 | 0.9765 |

On seen CWEs, **E2 (ungated residual)** achieves the highest test F1 (0.886). Five of the six viable methods cluster tightly within ~1pp of each other (0.871–0.886). E3 trades some precision for recall, landing slightly lower overall (0.845).

#### Unseen Test (Generalization to Held-Out CWE Types)

| Experiment | F1 | Precision | Recall |
|------------|----|-----------|--------|
| E0: Late only (baseline) | 0.883 | 0.953 | 0.822 |
| E1: Gated residual | 0.874 | 0.956 | 0.805 |
| E2: Ungated residual | 0.882 | 0.966 | 0.811 |
| E3: Scalar gated residual | 0.870 | 0.895 | 0.846 |
| **E4: FiLM** | **0.887** | 0.961 | 0.824 |
| E5: Concat adapter | 0.104 | 0.056 | 0.667 |
| E6: Gated concat adapter | 0.884 | 0.957 | 0.822 |
| E7: Cross-attention | 0.019 | 0.052 | 0.011 |

On unseen CWEs, **E4 (FiLM)** achieves the highest F1 (0.887), followed closely by E6 (0.884) and the E0 baseline (0.883). The top four methods are separated by only 1.7pp. E3 again trades precision for recall but has the highest recall among viable methods (0.846).

#### Failed Experiments

Two methods catastrophically failed:

- **E5 (concat adapter):** F1 collapsed to 0.047 (test) / 0.104 (unseen). Accuracy dropped to 31.7% on the test split, indicating the model learned to predict the positive class almost indiscriminately (recall = 0.71 with precision = 0.024).
- **E7 (cross-attention graph tokens):** Complete failure with F1 = 0.000 (test) / 0.019 (unseen). The model essentially predicts everything as non-vulnerable (recall = 0.000 / 0.011).

### 3.2 Token-Length Bucket Analysis

#### Test Split (Seen CWEs)

| Experiment | 0–64 F1 | 64–128 F1 | 128–256 F1 | 256–512 F1 |
|------------|---------|-----------|------------|------------|
| E0: Late only | 0.286 | 0.933 | 0.902 | 0.897 |
| E1: Gated residual | 0.286 | 0.933 | 0.902 | 0.889 |
| E2: Ungated residual | 0.286 | **1.000** | 0.920 | 0.891 |
| E3: Scalar gated residual | 0.250 | **1.000** | 0.885 | 0.847 |
| E4: FiLM | 0.286 | **1.000** | 0.868 | 0.887 |
| E6: Gated concat adapter | 0.286 | 0.941 | 0.902 | 0.881 |

On the seen test split, short-function performance is nearly identical across all viable methods (F1 = 0.286, except E3 at 0.250). All achieve perfect precision but extremely low recall (1 of 6 vulnerable short functions detected in a pool of 1,054). This reflects the tiny vulnerable sample count rather than meaningful method differences.

#### Unseen Test Split (Held-Out CWEs)

| Experiment | 0–64 F1 | 64–128 F1 | 128–256 F1 | 256–512 F1 |
|------------|---------|-----------|------------|------------|
| E0: Late only | 0.457 | 0.923 | 0.955 | 0.872 |
| E1: Gated residual | 0.500 | 0.887 | 0.947 | 0.865 |
| E2: Ungated residual | 0.438 | 0.914 | 0.952 | 0.872 |
| E3: Scalar gated residual | 0.560 | 0.926 | 0.956 | 0.854 |
| E4: FiLM | 0.467 | 0.901 | 0.956 | 0.879 |
| **E6: Gated concat adapter** | **0.744** | **0.943** | **0.969** | 0.859 |

**This is the most important table.** On unseen CWEs, **E6 (gated concat adapter)** dramatically outperforms all other methods on short functions:

- **E6 achieves 0.744 F1 on 0–64 token unseen functions** — a +63% relative improvement over the baseline E0 (0.457) and +33% over the next-best E3 (0.560).
- E6 also leads in the 64–128 bucket (0.943) and 128–256 bucket (0.969).
- E6's short-function advantage is driven by **both high precision (0.667) and high recall (0.842)**, detecting 16 of 19 vulnerable short functions vs. 8/19 for E0 and 7/19 for E2.

### 3.3 Short Function Detail (0–64 Tokens)

#### Test Split

| Experiment | F1 | Precision | Recall | Support |
|------------|---:|----------:|-------:|--------:|
| E0: Late only | 0.286 | 1.000 | 0.167 | 1,054 |
| E1: Gated residual | 0.286 | 1.000 | 0.167 | 1,054 |
| E2: Ungated residual | 0.286 | 1.000 | 0.167 | 1,054 |
| E3: Scalar gated residual | 0.250 | 0.500 | 0.167 | 1,054 |
| E4: FiLM | 0.286 | 1.000 | 0.167 | 1,054 |
| E6: Gated concat adapter | 0.286 | 1.000 | 0.167 | 1,054 |

All viable methods achieve identical recall (0.167 = 1/6 vulnerable). The test split has too few vulnerable short functions to differentiate methods.

#### Unseen Test Split

| Experiment | F1 | Precision | Recall | Support |
|------------|---:|----------:|-------:|--------:|
| E0: Late only | 0.457 | 0.500 | 0.421 | 1,938 |
| E1: Gated residual | 0.500 | 0.529 | 0.474 | 1,938 |
| E2: Ungated residual | 0.438 | 0.538 | 0.368 | 1,938 |
| E3: Scalar gated residual | 0.560 | 0.452 | 0.737 | 1,938 |
| E4: FiLM | 0.467 | 0.636 | 0.368 | 1,938 |
| **E6: Gated concat adapter** | **0.744** | **0.667** | **0.842** | 1,938 |

E6 dominates on unseen short functions. It is the only method to surpass 0.60 F1, combining the highest precision among recall-oriented methods (0.667) with far-and-away the highest recall (0.842). By contrast, E3 achieves high recall (0.737) but much lower precision (0.452), and E4 has decent precision (0.636) but low recall (0.368).

## 4. Discussion

### 4.1 Why E5 and E7 Failed

**E5 (concat adapter)** directly concatenates graph and hidden representations, doubling the hidden dimension before projecting back. Without a gating mechanism, the graph signal overwhelms the pretrained hidden states from the first training step, destabilizing UniXcoder's representations. The model collapses to near-random prediction.

**E7 (cross-attention graph tokens)** adds graph token sequences as cross-attention keys/values. This fundamentally alters the attention pattern of the pretrained transformer, introducing new key-value pairs that the pretrained attention weights were never calibrated for. The model fails to learn useful representations within the training budget.

The contrast between **E5 (failure) and E6 (success)** is instructive: E6 is identical to E5 except for a learned gate that controls injection strength. This confirms that **gating is essential** for any fusion method that significantly perturbs the pretrained hidden states.

### 4.2 The Role of Gating

Three pairings illustrate gating's impact:

| Pair | Ungated | Gated | Delta (Unseen F1) |
|------|---------|-------|--------------------|
| Residual | E2 (0.882) | E1 (0.874) | −0.008 |
| Scalar residual | — | E3 (0.870) | — |
| Concat adapter | E5 (**0.104**) | E6 (0.884) | **+0.780** |

For residual methods, gating is not critical because the additive injection is inherently mild — a projection of the graph embedding is added to the hidden state, and the pretrained weights naturally dampen this small perturbation. For concat adapters, gating is the difference between catastrophic failure and top-tier performance.

### 4.3 Short Functions: Why E6 Excels

E6's gated concat adapter outperforms all other methods on short functions by a wide margin (0.744 vs. 0.560 next-best). The concat adapter provides a richer fusion: rather than simply adding a projection of the graph embedding (as in residual methods), it concatenates graph and hidden representations, allowing the projection layer to learn joint features. The gate prevents this from destabilizing training while still permitting stronger graph influence at inference time.

For short functions (0–64 tokens), the token sequence provides minimal context. The graph structural signal (AST + control flow) becomes proportionally more important. E6's architecture allows the model to rely more heavily on graph features when the token signal is weak.

### 4.4 Overall vs. Short-Function Rankings

The overall F1 rankings and short-function rankings diverge:

- **Overall best:** E4 FiLM (unseen F1 = 0.887) — but only 0.467 on short functions
- **Short-function best:** E6 gated concat adapter (unseen 0–64 F1 = 0.744) — with unseen F1 = 0.884 overall

E6 sacrifices only 0.3pp of overall unseen F1 relative to E4 while gaining +27.7pp on short functions. This trade-off strongly favors E6 when short-function detection is a priority.

## 5. Conclusion

1. **The choice of fusion method matters — but mostly for short functions.** On overall metrics, six of eight methods cluster within 1.7pp F1 on unseen CWEs. On short functions (0–64 tokens), the spread is 30.7pp (0.438–0.744), revealing that method choice disproportionately affects the hardest detection regime.

2. **Gated concat adapter (E6) is the recommended fusion method.** E6 achieves 0.744 F1 on unseen short functions — a +63% relative improvement over the baseline — while maintaining competitive overall unseen F1 (0.884, within 0.3pp of the best). No other method comes close to this short-function performance.

3. **Gating is essential for high-capacity fusion.** The concat adapter without gating (E5) fails catastrophically; with gating (E6), it becomes the best method. Cross-attention (E7) also fails, likely because it too lacks a mechanism to gradually introduce graph influence during training.

4. **FiLM (E4) is a strong alternative when short functions are not a priority.** E4 achieves the highest overall unseen F1 (0.887) with a simple, parameter-efficient mechanism. It may be preferred in settings where the function length distribution skews longer.

5. **Residual methods (E0–E2) offer no improvement over the baseline.** None of the residual variants meaningfully improve upon late-only fusion (E0), suggesting that simple additive injection is too mild to transfer useful structural information through the encoder.

6. **Recommendation:** Adopt E6 (gated concat adapter) as the default fusion method. It provides the best balance of overall and short-function performance, and its gating mechanism ensures stable training.
