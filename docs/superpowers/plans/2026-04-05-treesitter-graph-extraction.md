# tree-sitter Graph Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the slow Joern-based CPG export (days) with tree-sitter AST extraction (minutes) for 153K C function samples.

**Architecture:** Single new module `graph_vul/pipeline/treesitter_graphs.py` reads the subset CSV, parses each function with tree-sitter's C parser, and writes per-sample `.pt` files directly to `GRAPH_ARTIFACTS_DIR`. Two-pass approach: first scan to build node type vocabulary, then extraction pass to write tensors.

**Tech Stack:** `tree-sitter`, `tree-sitter-c`, PyTorch, pandas

---

### Task 1: Add tree-sitter dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add tree-sitter packages to requirements.txt**

Add these two lines to the end of the main dependencies section (before the `# Dev / EDA` comment):

```
tree-sitter>=0.21.0
tree-sitter-c>=0.21.0
```

- [ ] **Step 2: Install dependencies**

Run: `python -m pip install tree-sitter tree-sitter-c`
Expected: Successful installation

- [ ] **Step 3: Verify tree-sitter works**

Run: `python -c "import tree_sitter_c; from tree_sitter import Language, Parser; parser = Parser(Language(tree_sitter_c.language())); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "Add tree-sitter and tree-sitter-c dependencies"
```

---

### Task 2: Write failing tests for tree-sitter graph extraction

**Files:**
- Create: `tests/test_treesitter_graphs.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for tree-sitter-based graph extraction."""

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from graph_vul.pipeline.treesitter_graphs import (
    extract_treesitter_graphs,
    parse_function_ast,
)

SIMPLE_C_FUNC = """\
int add(int a, int b) {
    return a + b;
}
"""

IF_C_FUNC = """\
int max(int a, int b) {
    if (a > b) {
        return a;
    }
    return b;
}
"""

EMPTY_FUNC = ""


class TestParseFunctionAst:
    """Unit tests for the single-function AST parser."""

    def test_returns_nodes_and_edges(self):
        node_types, edges = parse_function_ast(SIMPLE_C_FUNC)
        assert len(node_types) > 0
        assert len(edges) > 0

    def test_node_types_are_strings(self):
        node_types, _ = parse_function_ast(SIMPLE_C_FUNC)
        assert all(isinstance(nt, str) for nt in node_types)

    def test_edges_are_parent_child_tuples(self):
        node_types, edges = parse_function_ast(SIMPLE_C_FUNC)
        num_nodes = len(node_types)
        for src, dst in edges:
            assert 0 <= src < num_nodes
            assert 0 <= dst < num_nodes

    def test_root_node_is_function_definition(self):
        node_types, _ = parse_function_ast(SIMPLE_C_FUNC)
        assert node_types[0] == "function_definition"

    def test_if_statement_present_in_branching_func(self):
        node_types, _ = parse_function_ast(IF_C_FUNC)
        assert "if_statement" in node_types

    def test_empty_code_returns_none(self):
        result = parse_function_ast(EMPTY_FUNC)
        assert result is None

    def test_no_anonymous_nodes(self):
        """Anonymous nodes like '{', ';', '(' should be excluded."""
        node_types, _ = parse_function_ast(SIMPLE_C_FUNC)
        anonymous = [nt for nt in node_types if nt in ("{", "}", ";", "(", ")", ",")]
        assert len(anonymous) == 0


class TestExtractTreesitterGraphs:
    """Integration tests for the full extraction pipeline."""

    def test_produces_pt_files(self, tmp_path):
        csv_path = tmp_path / "subset.csv"
        output_dir = tmp_path / "graph_artifacts"
        pd.DataFrame({
            "sample_id": ["s1", "s2"],
            "func_before": [SIMPLE_C_FUNC, IF_C_FUNC],
        }).to_csv(csv_path, index=False)

        extract_treesitter_graphs(
            input_csv=str(csv_path),
            output_dir=str(output_dir),
        )

        assert (output_dir / "s1.pt").exists()
        assert (output_dir / "s2.pt").exists()

    def test_pt_file_schema(self, tmp_path):
        csv_path = tmp_path / "subset.csv"
        output_dir = tmp_path / "graph_artifacts"
        pd.DataFrame({
            "sample_id": ["s1"],
            "func_before": [SIMPLE_C_FUNC],
        }).to_csv(csv_path, index=False)

        extract_treesitter_graphs(
            input_csv=str(csv_path),
            output_dir=str(output_dir),
        )

        data = torch.load(output_dir / "s1.pt", weights_only=True)
        assert "node_type_ids" in data
        assert "edge_index" in data
        assert "edge_type_ids" in data
        assert data["node_type_ids"].dtype == torch.long
        assert data["edge_index"].shape[0] == 2
        assert data["edge_type_ids"].dtype == torch.long
        # All edge types should be 0 (AST)
        assert (data["edge_type_ids"] == 0).all()

    def test_writes_vocab_files(self, tmp_path):
        csv_path = tmp_path / "subset.csv"
        output_dir = tmp_path / "graph_artifacts"
        pd.DataFrame({
            "sample_id": ["s1"],
            "func_before": [SIMPLE_C_FUNC],
        }).to_csv(csv_path, index=False)

        extract_treesitter_graphs(
            input_csv=str(csv_path),
            output_dir=str(output_dir),
        )

        node_vocab_path = output_dir / "node_type_vocab.json"
        edge_vocab_path = output_dir / "edge_type_vocab.json"
        assert node_vocab_path.exists()
        assert edge_vocab_path.exists()

        with open(node_vocab_path) as f:
            node_vocab = json.load(f)
        assert isinstance(node_vocab, dict)
        assert "function_definition" in node_vocab

        with open(edge_vocab_path) as f:
            edge_vocab = json.load(f)
        assert edge_vocab == {"AST": 0}

    def test_skips_existing_pt_files(self, tmp_path):
        csv_path = tmp_path / "subset.csv"
        output_dir = tmp_path / "graph_artifacts"
        output_dir.mkdir(parents=True)
        pd.DataFrame({
            "sample_id": ["s1"],
            "func_before": [SIMPLE_C_FUNC],
        }).to_csv(csv_path, index=False)

        # Pre-create a dummy .pt file
        dummy = {"node_type_ids": torch.tensor([99]), "edge_index": torch.zeros(2, 0, dtype=torch.long), "edge_type_ids": torch.zeros(0, dtype=torch.long)}
        torch.save(dummy, output_dir / "s1.pt")

        extract_treesitter_graphs(
            input_csv=str(csv_path),
            output_dir=str(output_dir),
        )

        # Should NOT have overwritten — node_type_ids should still be [99]
        data = torch.load(output_dir / "s1.pt", weights_only=True)
        assert data["node_type_ids"].tolist() == [99]

    def test_skips_empty_func_before(self, tmp_path):
        csv_path = tmp_path / "subset.csv"
        output_dir = tmp_path / "graph_artifacts"
        pd.DataFrame({
            "sample_id": ["s1", "s2"],
            "func_before": ["", SIMPLE_C_FUNC],
        }).to_csv(csv_path, index=False)

        extract_treesitter_graphs(
            input_csv=str(csv_path),
            output_dir=str(output_dir),
        )

        assert not (output_dir / "s1.pt").exists()
        assert (output_dir / "s2.pt").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_treesitter_graphs.py -v`
Expected: FAIL — `ImportError: cannot import name 'extract_treesitter_graphs' from 'graph_vul.pipeline.treesitter_graphs'`

- [ ] **Step 3: Commit**

```bash
git add tests/test_treesitter_graphs.py
git commit -m "Add failing tests for tree-sitter graph extraction"
```

---

### Task 3: Implement `parse_function_ast`

**Files:**
- Create: `graph_vul/pipeline/treesitter_graphs.py`

- [ ] **Step 1: Implement the AST parser function**

```python
"""Extract AST graph artifacts from C functions using tree-sitter.

Replaces the Joern-based pipeline (joern_export.py + extract_graphs.py)
with a fast, in-process alternative. Produces per-sample .pt files
compatible with func_dataset.py / func_finetune_dataset.py.
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch
import tree_sitter_c
from tree_sitter import Language, Parser
from tqdm.auto import tqdm

from graph_vul.config import (
    FUNC_SUBSET_CSV,
    GRAPH_ARTIFACTS_DIR,
)

logger = logging.getLogger(__name__)

C_LANGUAGE = Language(tree_sitter_c.language())


def _make_parser() -> Parser:
    parser = Parser(C_LANGUAGE)
    return parser


def parse_function_ast(
    func_code: str,
) -> tuple[list[str], list[tuple[int, int]]] | None:
    """Parse C function source into (node_types, edges).

    Returns None if the code is empty or produces no named nodes.
    Edges are directed parent -> child, using only named nodes
    (anonymous punctuation nodes are skipped).
    """
    if not isinstance(func_code, str) or not func_code.strip():
        return None

    parser = _make_parser()
    tree = parser.parse(func_code.encode("utf-8"))

    node_types: list[str] = []
    edges: list[tuple[int, int]] = []
    # Map tree-sitter node id -> index in our node_types list
    node_id_to_idx: dict[int, int] = {}

    def _walk(node, parent_idx: int | None) -> None:
        if not node.is_named:
            return
        idx = len(node_types)
        node_types.append(node.type)
        node_id_to_idx[node.id] = idx
        if parent_idx is not None:
            edges.append((parent_idx, idx))
        for child in node.children:
            _walk(child, idx)

    _walk(tree.root_node, None)

    if not node_types:
        return None

    return node_types, edges
```

- [ ] **Step 2: Run unit tests for `parse_function_ast`**

Run: `pytest tests/test_treesitter_graphs.py::TestParseFunctionAst -v`
Expected: All 7 tests PASS

- [ ] **Step 3: Commit**

```bash
git add graph_vul/pipeline/treesitter_graphs.py
git commit -m "Implement parse_function_ast with tree-sitter C parser"
```

---

### Task 4: Implement `extract_treesitter_graphs`

**Files:**
- Modify: `graph_vul/pipeline/treesitter_graphs.py`

- [ ] **Step 1: Add the extraction pipeline function**

Append after `parse_function_ast` in `treesitter_graphs.py`:

```python
def extract_treesitter_graphs(
    input_csv: str | None = None,
    output_dir: str | None = None,
) -> None:
    """Extract AST graphs from all functions in a subset CSV.

    Two-pass approach:
    1. Scan all samples to build node type vocabulary.
    2. Parse again and write per-sample .pt files.
    """
    csv_path = input_csv or str(FUNC_SUBSET_CSV)
    dst = Path(output_dir) if output_dir else GRAPH_ARTIFACTS_DIR
    dst.mkdir(parents=True, exist_ok=True)

    node_vocab_path = dst / "node_type_vocab.json"
    edge_vocab_path = dst / "edge_type_vocab.json"

    df = pd.read_csv(csv_path, usecols=["sample_id", "func_before"])
    items = [
        (str(row["sample_id"]), row.get("func_before", ""))
        for _, row in df.iterrows()
    ]

    # --- Pass 1: build vocab ---
    if not node_vocab_path.exists():
        logger.info("Building node type vocab from %d samples...", len(items))
        all_node_types: set[str] = set()
        for _sid, code in tqdm(items, desc="Scanning vocab", unit="sample"):
            result = parse_function_ast(code)
            if result is not None:
                node_types, _ = result
                all_node_types.update(node_types)

        node_vocab = {nt: i for i, nt in enumerate(sorted(all_node_types))}
        with open(node_vocab_path, "w") as f:
            json.dump(node_vocab, f, indent=2)
        logger.info("Wrote node vocab with %d entries", len(node_vocab))
    else:
        with open(node_vocab_path) as f:
            node_vocab = json.load(f)

    # Edge vocab is trivial — AST only
    edge_vocab = {"AST": 0}
    with open(edge_vocab_path, "w") as f:
        json.dump(edge_vocab, f, indent=2)

    # --- Pass 2: extract and write .pt files ---
    processed = 0
    skipped_exists = 0
    skipped_empty = 0

    progress = tqdm(items, desc="Extracting graphs", unit="sample")
    for sid, code in progress:
        out_path = dst / f"{sid}.pt"
        if out_path.exists():
            skipped_exists += 1
            progress.set_postfix(done=processed, skip=skipped_exists, empty=skipped_empty)
            continue

        result = parse_function_ast(code)
        if result is None:
            skipped_empty += 1
            progress.set_postfix(done=processed, skip=skipped_exists, empty=skipped_empty)
            continue

        node_types, edges = result

        node_type_ids = torch.tensor(
            [node_vocab.get(nt, 0) for nt in node_types], dtype=torch.long,
        )

        if edges:
            src_list = [e[0] for e in edges]
            dst_list = [e[1] for e in edges]
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_type_ids = torch.zeros(len(edges), dtype=torch.long)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_type_ids = torch.zeros(0, dtype=torch.long)

        torch.save(
            {
                "node_type_ids": node_type_ids,
                "edge_index": edge_index,
                "edge_type_ids": edge_type_ids,
            },
            out_path,
        )
        processed += 1
        progress.set_postfix(done=processed, skip=skipped_exists, empty=skipped_empty)

    logger.info(
        "Graph extraction complete: %d processed, %d skipped (existing), %d skipped (empty) out of %d total",
        processed, skipped_exists, skipped_empty, len(items),
    )
```

- [ ] **Step 2: Run all tests**

Run: `pytest tests/test_treesitter_graphs.py -v`
Expected: All 12 tests PASS

- [ ] **Step 3: Commit**

```bash
git add graph_vul/pipeline/treesitter_graphs.py
git commit -m "Implement extract_treesitter_graphs pipeline"
```

---

### Task 5: Add CLI entry point and DVC stage

**Files:**
- Modify: `graph_vul/pipeline/treesitter_graphs.py` (add `__main__` block)
- Modify: `dvc.yaml`

- [ ] **Step 1: Add CLI entry point**

Append at the bottom of `graph_vul/pipeline/treesitter_graphs.py`:

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract AST graph artifacts from C functions using tree-sitter",
    )
    parser.add_argument("--input", type=str, default=None, help="Path to subset CSV")
    parser.add_argument("--output", type=str, default=None, help="Output directory for graph artifacts")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    extract_treesitter_graphs(input_csv=args.input, output_dir=args.output)
```

- [ ] **Step 2: Add DVC stage**

Add a new `func_treesitter_graphs` stage in `dvc.yaml`. Place it after the commented-out `func_extract_graphs` block (around line 57). This is an active (uncommented) stage:

```yaml
  func_treesitter_graphs:
    cmd: python -m graph_vul.pipeline.treesitter_graphs --input data/func_subset.csv --output data/processed/graph_artifacts
    deps:
      - data/func_subset.csv
      - graph_vul/pipeline/treesitter_graphs.py
      - graph_vul/config.py
    outs:
      - data/processed/graph_artifacts
```

- [ ] **Step 3: Verify CLI runs**

Run: `python -m graph_vul.pipeline.treesitter_graphs --help`
Expected: Shows help text with `--input` and `--output` arguments

- [ ] **Step 4: Commit**

```bash
git add graph_vul/pipeline/treesitter_graphs.py dvc.yaml
git commit -m "Add CLI entry point and DVC stage for tree-sitter graph extraction"
```

---

### Task 6: Smoke test on real data

**Files:** None (manual verification)

- [ ] **Step 1: Run on a small slice to verify**

Run: `head -11 data/func_subset.csv > /tmp/test_slice.csv && python -m graph_vul.pipeline.treesitter_graphs --input /tmp/test_slice.csv --output /tmp/test_graphs`
Expected: Completes in seconds, produces `.pt` files and vocab JSON files in `/tmp/test_graphs`

- [ ] **Step 2: Inspect output**

Run: `python -c "import torch; d = torch.load('/tmp/test_graphs/$(ls /tmp/test_graphs | grep .pt | head -1)', weights_only=True); print({k: v.shape for k, v in d.items()})"`
Expected: Shows shapes like `{'node_type_ids': torch.Size([N]), 'edge_index': torch.Size([2, E]), 'edge_type_ids': torch.Size([E])}`

- [ ] **Step 3: Run full extraction**

Run: `python -m graph_vul.pipeline.treesitter_graphs`
Expected: Processes ~153K samples. Should complete in minutes with a tqdm progress bar.

- [ ] **Step 4: Commit any config tweaks if needed**
