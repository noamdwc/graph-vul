## Why

The current POC is trained on a 1,000-sample CWE-119 subset. Before extending coverage to additional CVEs, we need a clear picture of the full Big-Vul dataset: how many distinct CVEs exist, how samples distribute across them, which CWEs have enough data for modeling, and whether the structural patterns observed on the subset hold at scale. Without this knowledge we risk picking CVEs that are too sparse or too imbalanced.

## What Changes

- Add a new Jupyter notebook (`notebooks/full_dataset_eda.ipynb`) that loads `data/bigvul.csv` and produces:
  - Total sample count, vulnerable vs benign class balance
  - Number of distinct CVE IDs and CWE IDs
  - Sample count per CVE (distribution + top-N table)
  - Sample count per CWE (distribution + top-N table)
  - Code length statistics across the full dataset
  - CVE description statistics across the full dataset
  - Recommendations for which CVEs/CWEs to target next based on data availability

## Capabilities

### New Capabilities
- `full-dataset-eda`: Notebook that performs exploratory data analysis on the complete Big-Vul CSV, focusing on CVE/CWE coverage and data distribution to inform POC expansion decisions.

### Modified Capabilities

_(none)_

## Impact

- New file: `notebooks/full_dataset_eda.ipynb`
- Reads existing `data/bigvul.csv` (already downloaded via DVC pipeline)
- No changes to existing code, model, or pipeline
- Depends on pandas, matplotlib, seaborn (already in project dependencies)
