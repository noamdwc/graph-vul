## Why

The current GAT-on-Joern-CPG model suffers from poor train/val generalization because the subset contains only ~10 projects, leading to single-project val/test splits, and outlier projects (Chrome, Linux, Android) dominate the data. We need a cleaner subset and a non-graph baseline to establish a performance floor before iterating on graph architectures.

## What Changes

- **New sampling pipeline**: Filter BigVul to projects with 50–2000 CWE-119 rows and ≥3 vulnerable samples, targeting ~25 projects. Drop functions with ≥513 whitespace tokens. Bump CVE params to MIN_CVES=15, MAX_CVES=30.
- **Function-level baseline model**: Frozen UniXcoder mean-pooled embeddings (768-dim) fed into a 2-layer MLP (768→128→1) with BCEWithLogitsLoss. No graph dependency.
- **Pre-computed embeddings**: UniXcoder mean-pooled vectors saved as `.pt` files, reusable as node features in future call-graph models.
- **Dedicated DVC stages**: `func_extract_subset`, `func_embed`, `func_split`, `func_train` — fully parallel to existing graph pipeline stages.

## Capabilities

### New Capabilities
- `func-subset-sampling`: Project-size and token-length filtering for a cleaner BigVul subset (~25 projects, ~3000 samples)
- `func-embedding`: Frozen UniXcoder mean-pooled embedding extraction and storage for function-level source code
- `func-baseline-model`: MLP classifier over pre-computed embeddings with Lightning training, W&B logging, and metric export

### Modified Capabilities
- `dvc-pipeline`: Add four new stages for the function-level baseline pipeline

## Impact

- **New files**: `func_extract_subset.py`, `embed_functions.py`, `func_dataset.py`, `func_classifier.py`, `func_lit_module.py`, `train_baseline.py`
- **Modified files**: `config.py` (additive constants only), `dvc.yaml` (new stages), `embed_utils.py` (add `load_unixcoder`/`embed_mean_pool`)
- **Dependencies**: None new — `microsoft/unixcoder-base` loads via existing `transformers` install
- **Data**: New outputs under `data/func_subset.csv`, `data/func_splits/`, `data/processed/func_embeddings/` (~9MB)
