"""Extract AST graph artifacts from C functions using tree-sitter.

Replaces the Joern-based pipeline (joern_export.py + extract_graphs.py)
with a fast, in-process alternative. Produces per-sample .pt files
consumed by func_finetune_dataset.py.
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data
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

    def _walk(node, parent_idx: int | None) -> None:
        if not node.is_named:
            return
        idx = len(node_types)
        node_types.append(node.type)
        if parent_idx is not None:
            edges.append((parent_idx, idx))
        for child in node.children:
            _walk(child, idx)

    # Include the whole snippet. Choosing the first named child can select a
    # leading comment or preprocessor directive and discard the function itself.
    _walk(tree.root_node, None)

    if not node_types:
        return None

    return node_types, edges


def function_graph(code, node_vocab, edge_vocab):
    """Use the training parser/vocabulary for preparation, external evaluation and demo."""
    result = parse_function_ast(code)
    if result is None:
        raise ValueError('Cannot extract AST from empty function')
    node_types, edges = result
    node_ids = torch.tensor([node_vocab.get(node, node_vocab['<UNK>']) for node in node_types], dtype=torch.long)
    edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous() if edges else torch.empty(2, 0, dtype=torch.long)
    return Data(node_type_ids=node_ids, edge_index=edge_index,
                edge_type_ids=torch.full((len(edges),), edge_vocab['AST'], dtype=torch.long))


def extract_treesitter_graphs(input_csv=None, output_dir=None, splits_dir=None):
    csv_path = input_csv or str(FUNC_SUBSET_CSV)
    dst = Path(output_dir) if output_dir else GRAPH_ARTIFACTS_DIR
    # A fresh directory prevents mixing graphs/vocabularies from earlier runs.
    dst.mkdir(parents=True, exist_ok=False)
    df = pd.read_csv(csv_path)
    training_df = df
    if splits_dir:
        assignments = pd.read_csv(Path(splits_dir) / 'assignments.csv')
        train_ids = set(assignments.loc[assignments['split'] == 'train', 'sample_id'])
        training_df = df[df['sample_id'].isin(train_ids)]
    node_types = set()
    for code in training_df['func_before']:
        result = parse_function_ast(code)
        if result is None:
            raise ValueError('Empty training function')
        node_types.update(result[0])
    # Unknown types have a dedicated ID; evaluation never adds vocabulary entries.
    node_vocab = {'<UNK>': 0, **{node: i + 1 for i, node in enumerate(sorted(node_types))}}
    edge_vocab = {'AST': 0}
    (dst / 'node_type_vocab.json').write_text(json.dumps(node_vocab, indent=2))
    (dst / 'edge_type_vocab.json').write_text(json.dumps(edge_vocab, indent=2))
    for row in tqdm(df.itertuples(), total=len(df), desc='Extracting AST graphs'):
        graph = function_graph(row.func_before, node_vocab, edge_vocab)
        torch.save(graph.to_dict(), dst / f'{row.sample_id}.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract AST graph artifacts from C functions using tree-sitter",
    )
    parser.add_argument("--input", type=str, default=None, help="Path to subset CSV")
    parser.add_argument("--output", type=str, default=None, help="Output directory for graph artifacts")
    parser.add_argument("--splits-dir", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    extract_treesitter_graphs(input_csv=args.input, output_dir=args.output, splits_dir=args.splits_dir)
