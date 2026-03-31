## Context

The pipeline currently trains and evaluates on BigVul only. The model uses UniXcoder + CWE embeddings (+ optional graph) and stores predictions via `export_finetune_predictions.py`, which depends on `FuncFinetuneDataset` tied to BigVul's `func_subset.csv` and `assignments.csv` split files.

We want to run the same trained checkpoint against PrimeVul and DiverseVul — two external HuggingFace datasets — without modifying training or existing evaluation.

## Goals / Non-Goals

**Goals:**
- Run inference on PrimeVul and DiverseVul using the existing trained checkpoint.
- Produce per-dataset predictions and metrics comparable to BigVul test results.
- Make the adapter interface easy to extend for future external datasets.
- Add a single DVC stage that orchestrates the entire external evaluation.

**Non-Goals:**
- Training or fine-tuning on external datasets.
- Graph features for external datasets (no Joern/AST extraction — code-only inference).
- CWE-specific evaluation (binary only for Tier-1).
- Modifying the existing BigVul pipeline or model architecture.

## Decisions

### 1. Adapter as a simple function returning a DataFrame

Each adapter is a function `load_<dataset>() -> pd.DataFrame` returning columns `[sample_id, func_before, vul, cwe_id]`. This mirrors `func_subset.csv` columns exactly.

**Why not a class hierarchy?** The normalisation is stateless and dataset-specific. A function per dataset in a single `external_datasets.py` module is simpler and equally extensible — adding a new dataset means adding one function.

**Alternative considered:** Subclass-based adapter with `BaseAdapter.load()`. Rejected as premature — we only have two datasets and no shared state.

### 2. Lightweight ExternalDataset (not FuncFinetuneDataset)

Create a minimal `ExternalEvalDataset` that accepts a DataFrame directly instead of reading from `func_subset.csv` + `assignments.csv`. It reuses the same tokenizer and max_length, and returns the same item dict (`input_ids`, `attention_mask`, `cwe_embedding`, `label`, `sample_id`, `token_count`).

**Why not reuse FuncFinetuneDataset?** Its constructor is hard-wired to BigVul file paths and split-based filtering. Refactoring it to accept a DataFrame would change the training-path interface for no benefit.

### 3. CWE embedding handling for external datasets

External datasets may contain CWE IDs that were never seen during training and have no pre-computed embedding. The adapter will map CWE IDs to the `CWE-XXX` format used by our embeddings. Missing CWEs get a zero vector (same fallback already used in `FuncFinetuneDataset._get_cwe_embedding`).

### 4. Single entry-point script with dataset registry

A new module `graph_vul/eval_external.py` acts as the entry point. It imports adapter functions from `graph_vul/dataset/external_datasets.py`, runs inference per dataset, computes metrics, saves predictions, and writes the markdown summary.

**Why a registry dict?** A simple `DATASETS = {"primevul": load_primevul, "diversevul": load_diversevul}` dict makes it trivial to add datasets without touching the orchestration loop.

### 5. Metrics: reuse sklearn, not torchmetrics

The existing `export_finetune_predictions.py` computes logits/probs/preds and saves them. Metrics (accuracy, precision, recall, F1, PR-AUC, ROC-AUC) can be computed from the saved predictions using `sklearn.metrics`, which is already a project dependency. This avoids coupling to the Lightning module's torchmetrics.

### 6. DVC stage shape

```yaml
eval_external_tier1:
  cmd: python -m graph_vul.eval_external
  deps:
    - data/func_finetune_model.ckpt
    - data/processed/cwe_embeddings
    - graph_vul/eval_external.py
    - graph_vul/dataset/external_datasets.py
  outs:
    - data/analysis/external_eval_tier1
  metrics:
    - data/analysis/external_eval_tier1/metrics.json:
        cache: false
```

Output directory structure:
```
data/analysis/external_eval_tier1/
  primevul_predictions.csv
  diversevul_predictions.csv
  primevul_metrics.json
  diversevul_metrics.json
  metrics.json            # combined, DVC-tracked
  summary.md              # markdown comparison table
```

## Risks / Trade-offs

**[Label semantics mismatch]** PrimeVul uses a more conservative labelling methodology than BigVul. DiverseVul derives labels from commit-level diffs. Neither guarantees identical positive/negative semantics.
→ Mitigation: Document label source per dataset in the summary. Accept that metrics are cross-dataset comparisons, not absolute measures.

**[CWE distribution mismatch]** External datasets contain CWEs not in our training set. The model has never seen embeddings for these CWEs, so it falls back to zero vectors.
→ Mitigation: Log the fraction of samples with unseen CWEs per dataset. This is expected and informative — it measures generalisation.

**[Function granularity mismatch]** DiverseVul labels are function-level but extracted from a different tool pipeline. PrimeVul is explicitly function-level. Minor differences in extraction (e.g., whitespace, comments) could affect tokenization.
→ Mitigation: Apply the same whitespace-token filter (512 max) to external datasets for comparability. Log how many samples are filtered out.

**[HuggingFace download at DVC runtime]** The adapter downloads datasets from HuggingFace, which requires network access during DVC repro.
→ Mitigation: HuggingFace `datasets` caches locally after first download. Document that the first run requires network access.

**[No graph features]** External datasets won't have graph artifacts, so graph-enabled models will run with `use_graph=False` (or zero-vector graph fallback).
→ Mitigation: Explicitly set `use_graph=False` in the external eval path and note this in the summary.
