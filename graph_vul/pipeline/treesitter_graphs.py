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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract AST graph artifacts from C functions using tree-sitter",
    )
    parser.add_argument("--input", type=str, default=None, help="Path to subset CSV")
    parser.add_argument("--output", type=str, default=None, help="Output directory for graph artifacts")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    extract_treesitter_graphs(input_csv=args.input, output_dir=args.output)
