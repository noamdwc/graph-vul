## Why

The project has no exploratory data analysis tooling. Understanding both the raw text data (code snippets, CVE descriptions) and the processed features (graph structure, embeddings) is essential for informed modeling decisions and debugging pipeline issues. An EDA notebook provides interactive, visual insight into the full data lifecycle without modifying the pipeline.

## What Changes

- Add a Jupyter notebook (`notebooks/eda.ipynb`) with two major sections:
  - **Raw text EDA**: code length distributions, CVE description statistics, token counts, vocabulary characteristics, and class balance from `subset_cwe119.csv`.
  - **Processed data EDA**: graph structure statistics (node/edge counts, edge type ratios), embedding distributions and dimensionality checks, train/val/test split statistics, and sample graph visualizations.
- Add `jupyter`, `matplotlib`, `seaborn`, and `networkx` to development dependencies.

## Capabilities

### New Capabilities
- `eda-notebook`: Interactive Jupyter notebook for exploratory data analysis covering raw text data and processed CPG graphs/embeddings.

### Modified Capabilities

## Impact

- **notebooks/**: New `eda.ipynb` file.
- **Dependencies**: `jupyter`, `matplotlib`, `seaborn`, `networkx` added as dev/optional dependencies.
- No changes to pipeline code, model, or DVC stages.
