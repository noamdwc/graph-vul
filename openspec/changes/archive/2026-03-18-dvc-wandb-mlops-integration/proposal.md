## Why

The Graph-Vul POC currently runs its multi-stage pipeline (data ingestion → graph extraction → embedding → training) via manually invoked Python scripts with no reproducibility tracking. Training experiments are not logged, making it impossible to compare hyperparameter configurations or audit past runs. Integrating DVC and W&B establishes the MLOps foundation needed before scaling the POC beyond a single CWE.

## What Changes

- **DVC pipeline definition**: A `dvc.yaml` file codifying five pipeline stages with explicit dependency and output tracking:
  1. **Data ingestion** — extract Big-Vul CWE-119 subset
  2. **Graph extraction** — Joern CPG export
  3. **Code embedding** — DeBERTa node-level embeddings for CPG nodes
  4. **CVE embedding** — DeBERTa embeddings for CVE description text
  5. **Training** — model training with W&B logging
- **DVC initialization & gitignore updates**: Initialize DVC in the repo; update `.gitignore` to exclude DVC-tracked data artifacts while keeping `.dvc` metadata files under version control.
- **W&B experiment tracking in training**: Instrument the training stage to log hyperparameters at run start and record training loss, validation loss, accuracy, precision, recall, and F1 per epoch.
- **Script parameterization for DVC**: Ensure pipeline scripts accept CLI parameters and produce outputs in locations DVC can track as stage dependencies/outputs. Separate the current monolithic dataset building into distinct code-embedding and CVE-embedding stages.
- **Dependency updates**: Add `dvc` and `wandb` to `requirements.txt`.
- **Updated execution docs**: Document `wandb login` and `dvc repro` as the new entry points.

## Capabilities

### New Capabilities
- `dvc-pipeline`: DVC pipeline definition and data artifact tracking for the five-stage workflow (ingest, graph extraction, code embedding, CVE embedding, training).
- `wandb-tracking`: Weights & Biases integration for hyperparameter logging and per-epoch metric tracking during training.

### Modified Capabilities
- `data-pipeline`: Scripts must accept parameterized inputs/outputs so DVC can declare explicit deps and outs. The embedding step in `cpg_dataset.py` must be split into two separable stages (code node embeddings and CVE text embeddings).
- `training-loop`: Training loop must integrate W&B logging calls alongside existing metric computation, and become a DVC stage itself.

## Impact

- **Code**: `train.py`, `extract_subset.py`, `joern_export.py`, `cpg_dataset.py`, `config.py` modified for parameterization and stage separation. New embedding scripts or CLI entry points created.
- **New files**: `dvc.yaml`, `dvc.lock` (generated), `.dvc/` directory.
- **Dependencies**: `dvc` and `wandb` added to `requirements.txt`.
- **Developer workflow**: `dvc repro` replaces manual multi-step invocation; `wandb login` required before first training run.
- **Existing architecture**: Core multimodal model (GATEncoder + TextProjection + FusionMLP) unchanged.
