## Context

The fine-tuned UniXcoder model shows degraded performance on short functions (0–64 tokens), particularly on unseen_test. The existing `post_experiment_analysis.ipynb` provides aggregate metrics and per-split breakdowns but does not drill into *why* short functions fail — whether the errors are systematic (e.g., wrapper functions, missing context) or statistical (e.g., calibration drift, threshold mismatch).

Current infrastructure:
- Prediction CSVs already include `token_length`, `prob`, `y_pred`, `y_true`, `cwe_id`, `project`, `func_before`, and `sample_id`.
- `graph_vul.analysis.post_experiment_analysis` provides `canonicalize_prediction_frame`, `aggregate_metrics_table`, and `bootstrap_ci`.
- Graph artifacts (`.pt` files per sample) are available but graph-level statistics (node count, edge count) are not yet columns in the prediction CSVs.

## Goals / Non-Goals

**Goals:**
- Produce a self-contained notebook that quantifies short-function errors by split, CWE, project, and confidence band.
- Surface qualitative examples (top FN/FP by confidence) so a human reviewer can form hypotheses.
- Add reusable helpers to `post_experiment_analysis.py` so future notebooks can call the same error-analysis logic.
- Output clear, actionable recommendations for next experiments.

**Non-Goals:**
- Retraining or modifying the model in this change.
- Building a full dashboard or interactive tool — the notebook is the deliverable.
- Analyzing functions longer than 128 tokens in depth (only as a comparison baseline).
- Modifying the training or evaluation pipeline itself.

## Decisions

### 1. Notebook structure: single notebook with helper imports

**Decision**: One notebook (`notebooks/short_function_error_analysis.ipynb`) that imports helpers from `graph_vul.analysis.post_experiment_analysis`.

**Rationale**: Matches existing notebook conventions (see `post_experiment_analysis.ipynb`). New generic helpers go into the module; notebook-specific display logic stays in cells. Avoids a new module just for this analysis.

**Alternatives considered**:
- Standalone script → rejected; notebooks are better for exploratory analysis with inline plots.
- Separate `short_function_analysis.py` module → premature; add helpers to existing module first.

### 2. Error subset construction

**Decision**: Build error subsets as filtered DataFrames with a `quadrant` column (`TP`, `TN`, `FP`, `FN`) added by a new helper `add_error_quadrant(df, threshold=0.5)`.

**Rationale**: A single column makes groupby/pivot operations trivial. The helper is reusable.

### 3. Graph statistics: lazy loading from .pt files

**Decision**: Add a helper `load_graph_stats(sample_ids, graph_dir)` that reads each `{sample_id}.pt`, extracts `num_nodes`, `num_edges`, and `density = num_edges / (num_nodes * (num_nodes - 1))` (or 0 for degenerate graphs), and returns a DataFrame. The notebook left-joins this onto the prediction frame.

**Rationale**: Avoids modifying the export pipeline — graph stats are derived on-the-fly. For ~5k test samples this takes seconds. If we later want them in the CSV, we can add them to `export_finetune_predictions.py` separately.

**Alternatives considered**:
- Adding graph columns to the export script → heavier change, not needed for analysis-only work.

### 4. Lexical marker extraction

**Decision**: Add a helper `extract_top_tokens(code_series, top_k=30)` that tokenizes code strings (simple whitespace + punctuation split), counts term frequency, and returns the top-k tokens. The notebook compares FN vs FP token distributions.

**Rationale**: Lightweight heuristic — enough to surface patterns like `return`, `if`, `NULL`, `sizeof` dominance in error subsets. Not a full NLP pipeline.

### 5. Output directory

**Decision**: `outputs/short_function_error_analysis/` for all saved figures, tables, and summary markdown.

**Rationale**: Matches existing convention (`outputs/post_experiment_analysis/`, `outputs/fusion_ablation_report.md`).

### 6. Notebook sections

The notebook follows this section order:

1. **Config** — editable paths, threshold, bucket edges
2. **Load & Canonicalize** — load prediction CSVs, add `quadrant` column, add token buckets
3. **Aggregate Metrics** — metrics by split × bucket (table + heatmap)
4. **Error Breakdowns** — FN/FP counts by CWE, by project (bar charts)
5. **Confidence Distributions** — histogram of `prob` by quadrant for short functions
6. **Short vs Non-Short Comparison** — error rate comparison across buckets
7. **Graph Statistics** — node count, edge count, density comparisons (if graph artifacts available)
8. **Lexical Markers** — top tokens in FN vs FP code
9. **Qualitative Review** — top 50 FN and top 50 FP on unseen_test 0–64, sorted by confidence
10. **Hypotheses & Recommendations** — markdown cells summarizing findings

## Risks / Trade-offs

- **[Risk] Graph artifacts may not exist for all samples** → Mitigation: graceful fallback; graph stats section is conditional on file availability, with a warning count.
- **[Risk] Lexical tokenization is naive** → Mitigation: acceptable for exploratory analysis; we explicitly label it as heuristic, not a rigorous NLP analysis.
- **[Risk] Notebook may be slow on large unseen_test (~31k rows)** → Mitigation: graph stats are only loaded for the short-function subset (~few thousand rows), not the full split.
- **[Trade-off] Helpers in `post_experiment_analysis.py` grow the module** → Acceptable; the module is already the designated home for analysis utilities.
