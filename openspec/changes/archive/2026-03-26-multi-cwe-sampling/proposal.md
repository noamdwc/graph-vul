## Why

The current pipeline is hard-coded to CWE-119 only, limiting the model to binary vulnerability detection within a single weakness class. Extending to multi-CWE classification enables the model to distinguish between different vulnerability types — a more realistic and useful capability. Analysis of the full Big-Vul dataset (documented in `docs/cwe_selection_analysis.md`) identified 8 CWEs that survive the pipeline filters with sufficient data, project diversity, and semantic distinctness.

## What Changes

- **Multi-CWE subset extraction**: Replace single-CWE filtering with a configurable list of 8 target CWEs. 6 CWEs are used for train/val/test; 2 are held out exclusively for zero-shot CWE generalization testing.
- **CWE descriptions**: Add descriptions for the 7 new CWEs to `data/cwe_descriptions.json` (CWE-119 already exists).
- **Config updates**: Replace `CWE_ID` scalar with `CWE_IDS` list and `HELDOUT_CWE_IDS` list in `config.py`.
- **Sampling adaptation**: Update `func_extract_subset.py` to filter across all 8 CWEs, apply per-CWE project-size filters, and tag each row with its CWE class.
- **Split adaptation**: Ensure held-out CWEs only appear in the test split, while the 6 training CWEs are split normally across train/val/test.

## Capabilities

### New Capabilities
- `multi-cwe-subset-sampling`: Subset extraction across multiple CWEs with held-out CWE support
- `cwe-descriptions`: CWE description data for all 8 target CWEs

### Modified Capabilities
- `func-subset-sampling`: Project-size filtering now applies per-CWE within projects, not globally for one CWE. Output includes `cwe_id` column for all 8 CWEs.
- `data-split`: Test split must include held-out CWE rows that never appear in train/val.

## Impact

- `graph_vul/config.py` — `CWE_ID` replaced by `CWE_IDS` + `HELDOUT_CWE_IDS`
- `graph_vul/pipeline/func_extract_subset.py` — multi-CWE filtering logic
- `graph_vul/pipeline/split_data.py` — held-out CWE routing to test-only
- `data/cwe_descriptions.json` — 7 new entries
- `dvc.yaml` — stage dependencies updated
- Downstream stages (embed, train) unaffected structurally — they already consume `func_subset.csv` with a `cwe_id` column
