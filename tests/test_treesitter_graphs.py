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
