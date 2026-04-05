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

    # tree-sitter wraps everything in a translation_unit root; skip it and
    # start from the first named child (the function_definition).
    start_node = tree.root_node
    named_children = [c for c in start_node.children if c.is_named]
    if named_children:
        start_node = named_children[0]

    _walk(start_node, None)

    if not node_types:
        return None

    return node_types, edges


def extract_treesitter_graphs(
    input_csv: str,
    output_dir: str,
) -> None:
    """Extract AST graph .pt files for all samples in input_csv.

    For each row, produces <output_dir>/<sample_id>.pt with keys:
      - node_type_ids: LongTensor of shape [N]
      - edge_index: LongTensor of shape [2, E]
      - edge_type_ids: LongTensor of shape [E] (all zeros = AST)

    Also writes node_type_vocab.json and edge_type_vocab.json to output_dir.
    Skips rows where func_before is empty or .pt already exists.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    # Build node type vocabulary incrementally
    node_type_to_id: dict[str, int] = {}
    edge_type_to_id: dict[str, int] = {"AST": 0}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting graphs"):
        sample_id = str(row["sample_id"])
        pt_path = output_path / f"{sample_id}.pt"

        if pt_path.exists():
            continue

        func_code = row["func_before"]
        if not isinstance(func_code, str) or not func_code.strip():
            continue

        result = parse_function_ast(func_code)
        if result is None:
            continue

        node_types, edges = result

        # Assign vocabulary IDs
        for nt in node_types:
            if nt not in node_type_to_id:
                node_type_to_id[nt] = len(node_type_to_id)

        node_type_ids = torch.tensor(
            [node_type_to_id[nt] for nt in node_types], dtype=torch.long
        )

        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)

        edge_type_ids = torch.zeros(edge_index.shape[1], dtype=torch.long)

        torch.save(
            {
                "node_type_ids": node_type_ids,
                "edge_index": edge_index,
                "edge_type_ids": edge_type_ids,
            },
            pt_path,
        )

    # Write vocabulary files
    node_vocab_path = output_path / "node_type_vocab.json"
    with open(node_vocab_path, "w") as f:
        json.dump(node_type_to_id, f, indent=2)

    edge_vocab_path = output_path / "edge_type_vocab.json"
    with open(edge_vocab_path, "w") as f:
        json.dump(edge_type_to_id, f, indent=2)


def main() -> None:
    """CLI entry point for tree-sitter graph extraction."""
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Extract AST graph artifacts from C functions using tree-sitter."
    )
    parser.add_argument(
        "--input-csv",
        default=str(FUNC_SUBSET_CSV),
        help="Path to CSV with sample_id and func_before columns.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(GRAPH_ARTIFACTS_DIR),
        help="Directory to write .pt files and vocab JSON.",
    )
    args = parser.parse_args()

    extract_treesitter_graphs(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
