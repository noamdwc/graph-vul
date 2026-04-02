## Why

Short functions (0–64 tokens) show significantly lower F1 on unseen_test compared to longer functions. We need a dedicated error-analysis notebook to understand *why* — are errors driven by wrapper/helper functions, missing caller/callee context, CWE mismatches, or calibration issues? The findings will directly inform the next round of experiments (context expansion, length-aware fusion, threshold tuning, graph changes).

## What Changes

- Add a new Jupyter notebook (`notebooks/short_function_error_analysis.ipynb`) that loads prediction CSVs + metadata, builds TP/TN/FP/FN subsets filtered by token bucket, and produces aggregate metrics, per-CWE/project error breakdowns, confidence distributions, qualitative review tables, and a hypothesis/recommendations section.
- Add reusable helper functions to `graph_vul/analysis/post_experiment_analysis.py` for error-subset construction, per-bucket metrics, and lexical-marker extraction — so future notebooks can import them.
- Optionally extend `export_finetune_predictions.py` to include graph-level statistics (node count, edge count, density) in prediction CSVs if not already present.

## Capabilities

### New Capabilities
- `short-function-error-notebook`: Jupyter notebook for short-function FP/FN analysis with aggregate metrics, qualitative tables, and hypothesis-driven recommendations.

### Modified Capabilities
- `unseen-gap-analysis`: Extended with short-function-specific error breakdowns and confidence distribution analysis for the 0–64 token bucket.

## Impact

- **New file**: `notebooks/short_function_error_analysis.ipynb`
- **Modified file**: `graph_vul/analysis/post_experiment_analysis.py` (new helper functions)
- **Possibly modified**: `graph_vul/export_finetune_predictions.py` (graph stats columns)
- **Data inputs**: existing `data/analysis/func_finetune_predictions/*.csv`, `data/func_subset.csv`, `data/processed/graph_artifacts/*.pt`
- **Output directory**: `outputs/short_function_error_analysis/`
