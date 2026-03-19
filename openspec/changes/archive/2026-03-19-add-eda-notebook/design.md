## Context

The Graph-Vul project has a fully automated DVC pipeline that produces a processed PyG dataset with CPG graphs, DeBERTa embeddings, and train/val/test splits. The source data includes raw C/C++ code and CVE text descriptions in `subset_cwe119.csv`. However, there is no tooling for interactively exploring either the raw text data or the processed graph/embedding data, which slows down debugging and modeling decisions.

## Goals / Non-Goals

**Goals:**
- Provide a single Jupyter notebook covering EDA for both raw text data and processed embeddings/graphs.
- Raw text analysis: code length distributions, CVE description statistics, token counts, vocabulary characteristics.
- Processed data analysis: class balance, graph structure statistics, embedding distributions, split statistics.
- Make it easy to re-run after pipeline changes to verify data quality.

**Non-Goals:**
- Modifying the pipeline or model code.
- Building a dashboard or web UI.
- Automating EDA as a DVC stage (this is an interactive tool).

## Decisions

### 1. Single notebook with two major sections
**Choice**: One `notebooks/eda.ipynb` with Part 1 (raw text) and Part 2 (processed graphs/embeddings).
**Rationale**: Keeps the full data story — from raw inputs to model-ready features — in one place. The dataset is small (~1000 samples) so runtime is not a concern.
**Alternative considered**: Separate notebooks per topic — rejected as unnecessary given dataset size.

### 2. Raw text loaded from CSV, processed data via CPGDataset
**Choice**: Load `data/subset_cwe119.csv` directly with pandas for raw text EDA. Use the existing `CPGDataset` PyG class for processed data EDA.
**Rationale**: Raw text analysis needs the original CSV columns (code, CVE descriptions). Processed data analysis should use the same loader the model uses to ensure consistency.

### 3. Visualization libraries: matplotlib + seaborn + networkx
**Choice**: Use `matplotlib` and `seaborn` for statistical plots, `networkx` for graph visualization.
**Rationale**: Standard, well-known libraries. `networkx` can render small CPG subgraphs via `torch_geometric.utils.to_networkx`.

### 4. Dev dependencies only
**Choice**: Add visualization libraries as optional/dev dependencies, not core requirements.
**Rationale**: The notebook is a development tool, not part of the training pipeline.

## Risks / Trade-offs

- **Large dataset in memory** → The full processed dataset may be large with all embeddings loaded. Mitigation: the notebook can load a subset or iterate lazily for aggregate stats.
- **Notebook staleness** → As the pipeline evolves, the notebook may become outdated. Mitigation: uses `CPGDataset` API and reads the same CSV the pipeline uses, so it stays aligned automatically.
