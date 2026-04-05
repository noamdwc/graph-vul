# tree-sitter Graph Extraction Design

## Problem

The Joern-based CPG export stage (`joern_export.py`) takes days to run on 153K samples. Each sample spawns two JVM subprocesses (`joern-parse` + `joern-export`), and with 4 workers the JVM cold-start overhead dominates. This blocks graph-based model iteration.

## Solution

Replace Joern with tree-sitter for AST extraction. A single new module parses C source directly in-process (no subprocesses, no JVM) and writes `.pt` graph artifact files. This is an AST-only ablation — CFG edges are omitted. If the ablation shows CFG matters, heuristic CFG construction can be added as a follow-up.

## Architecture

### New module: `graph_vul/pipeline/treesitter_graphs.py`

A single-stage pipeline that replaces both `joern_export.py` (Joern CLI invocation) and the graph-extraction portion of `extract_graphs.py` (CPG parsing to tensors).

**Input:** Subset CSV (`func_subset.csv`) with `sample_id` and `func_before` columns.

**Output:** Per-sample `.pt` files in `GRAPH_ARTIFACTS_DIR`, each containing:
- `node_type_ids`: `torch.long` tensor of node type indices
- `edge_index`: `[2, num_edges]` `torch.long` tensor of AST parent-child edges
- `edge_type_ids`: `torch.long` tensor, all zeros (AST edge type)

Additionally writes `node_type_vocab.json` and `edge_type_vocab.json` to the same directory.

### Processing approach

Two-pass sequential loop (no multiprocessing needed — tree-sitter parses in microseconds):

1. **Pass 1 (vocab scan):** Parse all samples, collect the set of all tree-sitter node types. Build and save the node type vocabulary. Edge vocab is trivial (single entry: AST=0).
2. **Pass 2 (extraction):** Parse each sample again, convert to tensors, write `.pt` files.

Skip logic:
- Skip if `.pt` file already exists
- Skip if `func_before` is empty or not a string

No caching layer — re-parsing is cheap enough that caching adds unnecessary complexity.

### Node type mapping

tree-sitter C grammar produces node types like `function_definition`, `if_statement`, `identifier`, `binary_expression`, etc. These differ from Joern's vocabulary (`METHOD`, `CONTROL_STRUCTURE`, `IDENTIFIER`). A new vocabulary is built from scratch. This means the model must be retrained with the new vocab.

### Edge types

AST edges only for this ablation. `edge_type_ids` are all `0`. The model config `NUM_EDGE_TYPES=3` stays unchanged — the RGCN can handle unused edge types; it simply won't see CFG/PDG edges during training.

### AST edge construction

For each tree-sitter node, create a directed edge from parent to each **named** child (i.e., `node.is_named == True`). Anonymous nodes (punctuation like `{`, `;`, `)`) are skipped — they add noise without structural information. This produces a clean AST comparable to Joern's AST edges.

## Dependencies

New packages added to `requirements.txt`:
- `tree-sitter`
- `tree-sitter-c`

## DVC integration

New stage `func_treesitter_graphs` in `dvc.yaml`:
- **cmd:** `python -m graph_vul.pipeline.treesitter_graphs --input data/func_subset.csv --output data/processed/graph_artifacts`
- **deps:** `data/func_subset.csv`, `graph_vul/pipeline/treesitter_graphs.py`, `graph_vul/config.py`
- **outs:** `data/processed/graph_artifacts`

## What stays unchanged

- `extract_graphs.py` — kept as-is for backward compatibility with existing Joern CPG exports
- `joern_export.py` — kept as-is, just no longer used in the active pipeline
- `func_dataset.py` / `func_finetune_dataset.py` — already load `.pt` files with the target schema
- `cpg_dataset.py` — different format, untouched
- Model code — RGCN architecture unchanged

## CLI interface

```
python -m graph_vul.pipeline.treesitter_graphs --input <csv> --output <dir>
```

Same pattern as other pipeline modules. Falls back to `FUNC_SUBSET_CSV` and `GRAPH_ARTIFACTS_DIR` defaults from config.

## Expected performance

~153K samples processed in minutes (vs. days with Joern). tree-sitter C parsing is microseconds per file; the bottleneck shifts to disk I/O for writing `.pt` files.

## Future work

If AST-only ablation shows CFG edges are needed, add heuristic CFG construction by walking the AST for control-flow constructs (`if`, `while`, `for`, `return`, `goto`) and linking basic blocks. This would remain in-process and fast.
