## Context

The current function-level fine-tuning stack already has the right high-level ingredients for this ablation: tokenized function inputs, precomputed CWE embeddings, a stable classifier head, and export/evaluation scripts that operate on the same artifacts. What is missing is direct supervision that teaches the representation to separate the true CWE from wrong but plausible alternatives.

The user constraint is to keep this minimal, configurable, and easy to disable. That rules out replacing the classifier, redesigning the data pipeline, or introducing a large retrieval-only training path. The cleanest fit is to layer optional auxiliary supervision onto the existing function/CWE pair pipeline and thread a small amount of extra reporting through the current export/eval stages.

## Goals / Non-Goals

**Goals:**
- Preserve the current binary classifier head and baseline behavior when the feature is disabled.
- Add a shared matching space for function and CWE embeddings with minimal model-surface changes.
- Support the E0-E4 ablation matrix entirely through config or CLI flags rather than separate training codepaths.
- Apply triplet loss only to vulnerable samples.
- Start with conservative defaults: one wrong-CWE negative per positive, `lambda=0.1`, `margin=0.2`.
- Produce evaluation outputs that let us answer the experiment question, especially on short functions.

**Non-Goals:**
- Rebuilding the pipeline around a pure metric-learning or retrieval model.
- Changing split generation, rebalancing strategy, or the primary vulnerability label semantics.
- Introducing large-scale hard-negative mining infrastructure beyond a small similar/same-family heuristic for E4.
- Expanding the classifier to multi-label CWE prediction; the retrieval outputs are for evaluation and auxiliary supervision.

## Decisions

### 1. Keep the classifier head unchanged and add auxiliary matching heads

`FuncFinetuneModel` will keep the existing pooled-code + CWE + optional-graph classifier head exactly as the primary prediction path. A pair of small projection heads will map the pooled function representation and the CWE embedding into a shared matching space used only for auxiliary losses and retrieval reporting.

**Rationale:** This isolates the ablation to the supervision block. It preserves checkpoint shape and downstream behavior for the main classifier path as much as possible, and it makes E0 a true no-op configuration rather than a separate implementation.

**Alternative considered:** Replace the classifier head with similarity scoring in the shared space. Rejected because it would confound the ablation with a head redesign and create unnecessary implementation risk.

### 2. Generate wrong-CWE negatives inside the current dataset/batch flow

Wrong-CWE negatives will be sampled against vulnerable positives using existing subset metadata and the current CWE embedding cache. The default policy is one wrong-CWE negative per positive. E4 switches the sampling policy to prefer similar or same-family CWEs when available; otherwise it falls back to standard wrong-CWE sampling.

**Rationale:** This reuses the current `FuncFinetuneDataset` inputs and avoids a new offline preprocessing stage. Sampling negatives at dataset or collate time keeps the feature easy to disable and lets us reuse the current train/export split artifacts.

**Alternative considered:** Pre-materialize negative pairs as a new dataset artifact. Rejected because it would add DVC/data churn for a small ablation and make the feature less configurable.

### 3. Restrict triplet loss to vulnerable samples and combine losses additively

For vulnerable samples only, the model will compute an anchor-positive-negative triplet in the shared space and add `lambda * triplet_loss` to the existing BCE loss. When wrong-CWE negatives are enabled, the same sampled negative can be reused for both the negative-pair classification term and the triplet term. Benign samples continue to train only on the main BCE objective.

**Rationale:** This matches the experiment intent and avoids inventing pseudo-positive CWE semantics for benign samples.

**Alternative considered:** Apply triplet loss to all samples using synthetic benign anchors. Rejected because the semantics are weak and likely to add noise.

### 4. Evaluate retrieval from the same shared space used for training

Prediction export and evaluation will optionally rank candidate CWE embeddings by similarity to the projected function embedding and report Top-1/Top-3 retrieval. The reporting layer will aggregate these metrics overall, per split, per CWE, on unseen/external datasets, and across fixed length slices `0-64`, `64-128`, `128-256`, `256+`.

**Rationale:** This gives a direct measure of the behavior the ablation is meant to improve and keeps training and evaluation aligned around the same representation.

### 5. Represent E0-E4 as named configurations, not separate codepaths

Training/eval entrypoints should accept a small named ablation config or equivalent flags that expand to:
- `E0`: all auxiliary terms disabled
- `E1`: wrong-CWE negatives enabled
- `E2`: triplet enabled
- `E3`: wrong-CWE negatives + triplet enabled
- `E4`: E3 plus hard-negative preference

**Rationale:** This keeps the change easy to compare, reproduce, and disable, and it reduces branching in implementation code.

## Risks / Trade-offs

- **[Negative sampling noise]** -> Random wrong-CWE negatives may be too easy and fail to move retrieval quality. -> **Mitigation:** keep E4 as a hard-negative option with same-family/similar-CWE preference and report retrieval metrics separately by length slice.
- **[Training instability from auxiliary loss]** -> Even a small auxiliary term can hurt the main classifier if the scale is wrong. -> **Mitigation:** default `lambda=0.1`, log each loss component separately, and preserve `E0` as an exact disable path.
- **[Metric/reporting sprawl]** -> Adding retrieval, per-CWE, unseen/external, and length slices can create too many ad hoc outputs. -> **Mitigation:** keep outputs in the existing export/eval structure and standardize on one summary schema per run.
- **[Hard-negative heuristic ambiguity]** -> "Similar/same-family CWE" may not always be available or well-defined from current metadata. -> **Mitigation:** implement a simple deterministic preference order with fallback to any wrong CWE and record the negative source in reports/logs.
- **[Checkpoint compatibility drift]** -> New optional heads add parameters to the fine-tune checkpoint. -> **Mitigation:** keep defaults disabled, save hyperparameters/config in checkpoints, and ensure loading works when auxiliary heads are present or absent.
