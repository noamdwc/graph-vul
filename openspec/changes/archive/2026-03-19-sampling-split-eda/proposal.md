## Why

The current 1000-row subset is sampled randomly from BigVul (~188K rows, 3,539 CVEs, 91 CWEs, ~5.8% vulnerable). Random sampling likely over-represents a few dominant CVEs/CWEs and under-represents rare ones. Row-level random train/val/test splits risk data leakage because the same CVE, commit, or even identical code can appear across splits. We need evidence-based sampling and splitting policies before building the graph dataset.

## What Changes

- Add a new **Section 7: Sampling and Split Design EDA** to `notebooks/full_dataset_eda.ipynb` with seven subsections:
  - 7.1 Group structure and potential leakage sources
  - 7.2 Label balance by group
  - 7.3 Diversity diagnostics for candidate 1000-row sampling strategies
  - 7.4 Concentration and dominance analysis
  - 7.5 Split design diagnostics
  - 7.6 Outlier and truncation-relevant diagnostics
  - 7.7 Final recommendation cell
- All new cells reuse the existing `df` dataframe and libraries (pandas, numpy, matplotlib, seaborn).
- Cells are robust to missing columns, degrading gracefully with clear messages.
- No new dependencies introduced.

## Capabilities

### New Capabilities
- `sampling-split-eda`: EDA section analyzing dataset group structure, label balance, sampling strategies, concentration, split leakage, and outliers to recommend a smarter 1000-row sampling policy and leakage-aware split strategy.

### Modified Capabilities
<!-- None — this adds new notebook cells without changing existing specs. -->

## Impact

- **Code**: `notebooks/full_dataset_eda.ipynb` — new cells appended after existing Section 6.
- **Dependencies**: None new (pandas, numpy, matplotlib, seaborn already used).
- **Downstream**: Findings will inform changes to `graph_vul/pipeline/extract_subset.py` and split logic in future work.
