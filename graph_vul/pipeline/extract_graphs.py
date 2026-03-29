"""Extract function-local graph artifacts (AST+CFG) from Joern CPG exports.

Produces per-sample .pt files with node_type_ids, edge_index, edge_type_ids
and builds stable vocab JSON files for node and edge types.
"""

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from graph_vul.config import (
    FUNC_SUBSET_CSV,
    GRAPH_ARTIFACTS_DIR,
    GRAPH_EDGE_VOCAB_PATH,
    GRAPH_NODE_VOCAB_PATH,
    RAW_DIR,
)

logger = logging.getLogger(__name__)

_EDGE_LABEL_RE = re.compile(r'label\s*=\s*"(\w+):')
_ALLOWED_EDGE_TYPES = {"AST", "CFG"}
_EDGE_TYPE_MAP = {"AST": 0, "CFG": 1, "PDG": 2, "DDG": 2, "CDG": 2}


# ---------------------------------------------------------------------------
# Vocab helpers
# ---------------------------------------------------------------------------


def _load_or_build_vocab(
    vocab_path: Path,
    items: set[str],
) -> dict[str, int]:
    """Load an existing vocab or build one from *items*."""
    if vocab_path.exists():
        with open(vocab_path) as f:
            return json.load(f)
    vocab: dict[str, int] = {}
    for i, item in enumerate(sorted(items - {"<UNK>"})):
        vocab[item] = i
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)
    logger.info("Wrote vocab with %d entries to %s", len(vocab), vocab_path)
    return vocab


# ---------------------------------------------------------------------------
# CPG parsing — extracts node types + edge types (not text)
# ---------------------------------------------------------------------------


def _parse_cpg_structure(
    sample_dir: Path,
) -> tuple[list[str], list[tuple[int, int, str]]] | None:
    """Parse a CPG export and return (node_types, edges) with edges as (src_idx, dst_idx, type_str).

    Only AST and CFG edges are kept.
    """
    nodes_file = sample_dir / "nodes.csv"
    edges_file = sample_dir / "edges.csv"
    nodes_json = sample_dir / "nodes.json"
    edges_json = sample_dir / "edges.json"

    if nodes_file.exists() and edges_file.exists():
        return _parse_csv(nodes_file, edges_file)
    elif nodes_json.exists() and edges_json.exists():
        return _parse_json(nodes_json, edges_json)
    else:
        return _parse_dot(sample_dir)


def _parse_csv(
    nodes_file: Path, edges_file: Path,
) -> tuple[list[str], list[tuple[int, int, str]]] | None:
    try:
        nodes_df = pd.read_csv(nodes_file, sep="\t")
        edges_df = pd.read_csv(edges_file, sep="\t")
    except Exception:
        logger.exception("Failed to read CSV CPG files")
        return None
    if nodes_df.empty:
        return None

    node_ids = nodes_df.iloc[:, 0].tolist()
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    node_types: list[str] = []
    for _, node in nodes_df.iterrows():
        label = str(node.get("LABEL", ""))
        node_types.append(label if label != "nan" else "<UNK>")

    edges: list[tuple[int, int, str]] = []
    for _, edge in edges_df.iterrows():
        src_id = edge.iloc[0]
        dst_id = edge.iloc[1]
        if src_id in id_to_idx and dst_id in id_to_idx:
            etype = str(edge.get("type", edge.get("label", "AST"))).upper()
            if etype in _ALLOWED_EDGE_TYPES:
                edges.append((id_to_idx[src_id], id_to_idx[dst_id], etype))

    return node_types, edges


def _parse_json(
    nodes_file: Path, edges_file: Path,
) -> tuple[list[str], list[tuple[int, int, str]]] | None:
    try:
        with open(nodes_file) as f:
            nodes = json.load(f)
        with open(edges_file) as f:
            edges_raw = json.load(f)
    except Exception:
        logger.exception("Failed to read JSON CPG files")
        return None
    if not nodes:
        return None

    id_to_idx = {n.get("id", n.get("_id", i)): i for i, n in enumerate(nodes)}

    node_types: list[str] = []
    for n in nodes:
        label = n.get("LABEL", n.get("label", "<UNK>"))
        node_types.append(str(label) if label else "<UNK>")

    edges: list[tuple[int, int, str]] = []
    for e in edges_raw:
        src_id = e.get("src", e.get("inV", e.get("source")))
        dst_id = e.get("dst", e.get("outV", e.get("target")))
        if src_id in id_to_idx and dst_id in id_to_idx:
            etype = str(e.get("type", e.get("label", "AST"))).upper()
            if etype in _ALLOWED_EDGE_TYPES:
                edges.append((id_to_idx[src_id], id_to_idx[dst_id], etype))

    return node_types, edges


def _parse_dot(
    sample_dir: Path,
) -> tuple[list[str], list[tuple[int, int, str]]] | None:
    dot_files = list(sample_dir.glob("*.dot"))
    if not dot_files:
        return None

    nodes: dict[str, str] = {}  # node_id -> type string
    raw_edges: list[tuple[str, str, str]] = []

    for dot_file in dot_files:
        content = dot_file.read_text(errors="replace")
        # Skip stub DOT files where all nodes have empty labels (<>).
        # These are external callee signatures with no structural info;
        # call sites already appear as typed nodes in the main function DOT.
        node_lines = [
            l.strip() for l in content.splitlines()
            if l.strip() and "->" not in l
            and not l.strip().startswith(("digraph", "}", "//", "node"))
        ]
        if node_lines and all("label = <>" in l or "label=<>" in l for l in node_lines):
            continue
        for line in content.splitlines():
            line = line.strip()
            if "->" in line:
                parts = line.split("->")
                if len(parts) == 2:
                    src = parts[0].strip().strip('"')
                    rest = parts[1]
                    dst = rest.split("[")[0].strip().strip(";").strip('"')
                    m = _EDGE_LABEL_RE.search(rest)
                    etype = m.group(1).upper() if m else "AST"
                    if etype in _ALLOWED_EDGE_TYPES:
                        raw_edges.append((src, dst, etype))
            elif line and not line.startswith(("digraph", "}", "//", "node")):
                # Only parse lines that look like node definitions:
                # "node_id" [label = ...]. Skip continuation lines from
                # multi-line edge labels (DDG edges with long code snippets).
                if "[" not in line or "label" not in line:
                    continue
                # Skip nodes with empty labels — these are placeholder
                # nodes in secondary translation units with no type info.
                if "label = <>" in line or "label=<>" in line:
                    continue
                node_id = line.split("[")[0].strip().strip('"')
                # Extract node type: first part before comma in label
                node_type = "<UNK>"
                if "label" in line:
                    # Joern HTML labels: <TYPE, LINE<BR/>code text>
                    if "label = <" in line or "label=<" in line:
                        start = line.index("<") + 1
                        # Type is everything before the first comma
                        rest_label = line[start:]
                        comma_idx = rest_label.find(",")
                        if comma_idx > 0:
                            node_type = rest_label[:comma_idx].strip()
                    elif 'label="' in line:
                        start = line.index('label="') + 7
                        end = line.index('"', start)
                        label_text = line[start:end]
                        comma_idx = label_text.find(",")
                        if comma_idx > 0:
                            node_type = label_text[:comma_idx].strip()
                        else:
                            node_type = label_text.strip()
                if node_id:
                    nodes[node_id] = node_type

    if not nodes:
        return None

    node_id_list = list(nodes.keys())
    id_to_idx = {nid: i for i, nid in enumerate(node_id_list)}
    node_types = [nodes[nid] for nid in node_id_list]

    edges: list[tuple[int, int, str]] = []
    for src_id, dst_id, etype in raw_edges:
        if src_id in id_to_idx and dst_id in id_to_idx:
            edges.append((id_to_idx[src_id], id_to_idx[dst_id], etype))

    return node_types, edges


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------


def extract_graphs(
    input_dir: str | None = None,
    output_dir: str | None = None,
    subset_csv: str | None = None,
) -> None:
    src = Path(input_dir) if input_dir else RAW_DIR
    dst = Path(output_dir) if output_dir else GRAPH_ARTIFACTS_DIR
    csv_path = subset_csv or str(FUNC_SUBSET_CSV)
    dst.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(csv_path, usecols=["sample_id"])
    sample_ids = [str(sid) for sid in metadata["sample_id"].values]

    # --- Pass 1: build vocab (if not already present) ---
    need_vocab = not GRAPH_NODE_VOCAB_PATH.exists() or not GRAPH_EDGE_VOCAB_PATH.exists()
    all_node_types: set[str] = set()
    all_edge_types: set[str] = set()

    if need_vocab:
        logger.info("Building vocab from %d samples...", len(sample_ids))
        for sid in tqdm(sample_ids, desc="Scanning vocab", unit="sample"):
            sample_dir = src / sid
            if not sample_dir.exists():
                continue
            result = _parse_cpg_structure(sample_dir)
            if result is None:
                continue
            node_types, edges = result
            all_node_types.update(node_types)
            all_edge_types.update(etype for _, _, etype in edges)

    node_vocab = _load_or_build_vocab(GRAPH_NODE_VOCAB_PATH, all_node_types)
    edge_vocab = _load_or_build_vocab(GRAPH_EDGE_VOCAB_PATH, all_edge_types)

    # --- Pass 2: extract graph artifacts ---
    processed = 0
    skipped_exists = 0
    skipped_missing = 0
    total_nodes = 0
    total_edges = 0

    progress = tqdm(sample_ids, desc="Extracting graphs", unit="sample")
    for sid in progress:
        out_path = dst / f"{sid}.pt"
        if out_path.exists():
            skipped_exists += 1
            progress.set_postfix(done=processed, skip=skipped_exists, miss=skipped_missing)
            continue

        sample_dir = src / sid
        if not sample_dir.exists() or not any(sample_dir.iterdir()):
            skipped_missing += 1
            progress.set_postfix(done=processed, skip=skipped_exists, miss=skipped_missing)
            continue

        result = _parse_cpg_structure(sample_dir)
        if result is None:
            skipped_missing += 1
            progress.set_postfix(done=processed, skip=skipped_exists, miss=skipped_missing)
            continue

        node_types, edges = result

        node_type_ids = torch.tensor(
            [node_vocab.get(nt, 0) for nt in node_types], dtype=torch.long,
        )

        if edges:
            src_list = [e[0] for e in edges]
            dst_list = [e[1] for e in edges]
            edge_type_ids = torch.tensor(
                [edge_vocab.get(e[2], 0) for e in edges], dtype=torch.long,
            )
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
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

        total_nodes += len(node_types)
        total_edges += len(edges)
        processed += 1
        progress.set_postfix(done=processed, skip=skipped_exists, miss=skipped_missing)

    # --- Summary ---
    logger.info("Graph extraction complete:")
    logger.info("  Node types in vocab: %d", len(node_vocab))
    logger.info("  Edge types in vocab: %d", len(edge_vocab))
    logger.info("  Samples processed:   %d", processed)
    logger.info("  Skipped (existing):  %d", skipped_exists)
    logger.info("  Skipped (missing):   %d", skipped_missing)
    if processed > 0:
        logger.info("  Avg nodes/graph:     %.1f", total_nodes / processed)
        logger.info("  Avg edges/graph:     %.1f", total_edges / processed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract AST+CFG graph artifacts from Joern CPG exports")
    parser.add_argument("--input", type=str, default=None, help="CPG exports directory")
    parser.add_argument("--output", type=str, default=None, help="Output graph artifacts directory")
    parser.add_argument("--subset-csv", type=str, default=None, help="Subset CSV path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    extract_graphs(input_dir=args.input, output_dir=args.output, subset_csv=args.subset_csv)
