## Context

The current graph-based vulnerability detection pipeline builds Joern CPGs per CVE and trains a GAT model. The subset contains ~10 projects, leading to poor split diversity (single-project val/test sets) and domination by outlier projects (Chrome, Linux, Android with thousands of rows each). We need a cleaner subset and a simpler baseline model to establish a performance floor before iterating on graph architectures.

The existing pipeline has: `extract_subset.py` (CVE-based sampling), `embed_cve.py` / `embed_utils.py` (CodeBERT embeddings), `split_data.py` (StratifiedGroupKFold), and a PyG-based training loop. The function-level baseline runs in parallel to this pipeline — no existing stages are modified.

Each row in BigVul is a function (the `func_before` column). Throughout this change, "sample" and "function" are used interchangeably — one row = one function = one sample.

## Goals / Non-Goals

**Goals:**
- Produce a cleaner BigVul subset (~25 projects, ~3000 functions) by filtering on project size and function length
- Establish a function-level classification baseline using frozen UniXcoder mean-pooled embeddings + MLP
- Pre-compute function embeddings reusable as node features in future graph models
- Run end-to-end via DVC with W&B metric logging

**Non-Goals:**
- Fine-tuning UniXcoder (frozen encoder only — this is a baseline)
- Replacing or modifying the existing graph pipeline
- Multi-class classification (binary vulnerable/not-vulnerable only)
- Hyperparameter search (fixed config for initial baseline)

## Decisions

### 1. Fork extract_subset.py rather than parameterize it
**Choice**: Create a separate `func_extract_subset.py` instead of adding flags to the existing script.
**Why**: The filtering logic differs substantially (project-size filter, token-length filter, different CVE params, no top2 penalty). A fork keeps both pipelines independently reproducible via DVC. The existing graph pipeline subset must remain stable.
**Alternative**: Parameterize `extract_subset.py` with mode flags — rejected because it couples two pipelines and complicates the DVC DAG.

### 2. UniXcoder over CodeBERT for function embeddings
**Choice**: Use `microsoft/unixcoder-base` instead of the existing CodeBERT setup.
**Why**: UniXcoder is pre-trained on code understanding tasks including function-level classification. Its representations are better suited for whole-function encoding than CodeBERT's, which was designed for statement-level tasks. Both load via `transformers` — no new dependencies.
**Alternative**: Reuse CodeBERT — rejected because its embeddings are optimized for shorter sequences and we already have statement-level CodeBERT embeddings for comparison.

### 3. Mean pooling over CLS token
**Choice**: Use mean pooling over all non-padding token hidden states rather than the CLS token.
**Why**: Mean pooling captures information from the entire function body rather than relying on a single token. For frozen encoders (no fine-tuning to push classification signal into CLS), mean pooling consistently outperforms CLS in retrieval and classification benchmarks. This is especially relevant for longer functions where important tokens may be far from position 0.
**Alternative**: CLS token — rejected because frozen CLS tokens tend to be weaker representations when the model wasn't fine-tuned with a classification objective on the target task.

### 4. Pre-computed embeddings rather than online encoding
**Choice**: Save mean-pooled vectors as `.pt` files and load them in the Dataset.
**Why**: The encoder is frozen, so embeddings are deterministic. Pre-computing avoids GPU cost during training, simplifies the training script (pure MLP, no encoder in memory), and produces reusable artifacts for future models.
**Alternative**: Online encoding during training — rejected because it wastes GPU on redundant forward passes.

### 5. Separate DVC stages parallel to existing pipeline
**Choice**: Four new `func_*` stages that don't depend on any existing graph pipeline stages.
**Why**: Allows independent `dvc repro func_train` without triggering graph pipeline stages. The only shared dependency is `data/bigvul.csv`.

### 6. Reuse split_data.py as-is
**Choice**: Invoke `split_data.py` with `--input data/func_subset.csv --output data/func_splits`.
**Why**: The StratifiedGroupKFold logic is project-agnostic. With ~25 projects, it should produce 3–4 projects per val/test split — much better diversity than the current 10-project subset.

## Risks / Trade-offs

- **Frozen mean-pooled embeddings may be weak** → Expected for a baseline. If accuracy is near-random, it still validates the subset/split quality and provides a floor for graph model comparison.
- **Whitespace token filter is coarse** → Functions at 512 whitespace tokens may exceed 512 subword tokens. Mitigated by `truncation=True` in the tokenizer — worst case loses some tail tokens, acceptable for a baseline.
- **Embedding storage (~9MB)** → Negligible. Tracked by DVC.
- **Subset size sensitivity** → The 50–2000 row filter and ≥3 vulnerable filter are heuristics. If fewer than 20 projects pass, we may need to relax bounds. Verified against BigVul: ~34 eligible projects, targeting ~25.
