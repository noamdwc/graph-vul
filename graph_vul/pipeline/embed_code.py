"""Standalone script: compute DeBERTa embeddings for CPG node texts.

Reads Joern CPG exports from an input directory and writes .pt files containing
node features (x), edge_index, and edge_type to an output directory.
"""

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from graph_vul.config import DEBERTA_EMBEDDING_DIM, PROCESSED_DIR, RAW_DIR, SUBSET_CSV
from graph_vul.pipeline.embed_utils import embed_texts, load_deberta

logger = logging.getLogger(__name__)

EDGE_TYPE_MAP = {"AST": 0, "CFG": 1, "PDG": 2, "DDG": 2, "CDG": 2}
_EDGE_LABEL_RE = re.compile(r'label\s*=\s*"(\w+):')


def embed_code_nodes(
    input_dir: str | None = None,
    output_dir: str | None = None,
    subset_csv: str | None = None,
) -> None:
    src = Path(input_dir) if input_dir else RAW_DIR
    dst = Path(output_dir) if output_dir else PROCESSED_DIR / "code_embeddings"
    csv_path = subset_csv or str(SUBSET_CSV)
    dst.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(csv_path)
    tokenizer, model = load_deberta()

    total = len(metadata)
    processed, skipped = 0, 0
    progress = tqdm(
        metadata.iterrows(),
        total=total,
        desc="Embedding code",
        unit="sample",
    )

    for _, row in progress:
        sample_id = str(row["sample_id"])
        out_path = dst / f"{sample_id}.pt"

        if out_path.exists():
            skipped += 1
            progress.set_postfix(processed=processed, skipped=skipped)
            continue

        sample_dir = src / sample_id
        if not sample_dir.exists() or not any(sample_dir.iterdir()):
            logger.warning("Sample %s has no CPG export, skipping", sample_id)
            progress.set_postfix(processed=processed, skipped=skipped)
            continue

        edge_index, edge_type, node_texts = _parse_cpg(sample_dir)

        if edge_index is None:
            logger.warning("Failed to parse CPG for sample %s, writing minimal", sample_id)
            torch.save(
                {
                    "x": torch.zeros(1, DEBERTA_EMBEDDING_DIM),
                    "edge_index": torch.zeros(2, 0, dtype=torch.long),
                    "edge_type": torch.zeros(0, dtype=torch.long),
                },
                out_path,
            )
        else:
            x = embed_texts(node_texts, tokenizer, model)
            torch.save(
                {"x": x, "edge_index": edge_index, "edge_type": edge_type},
                out_path,
            )

        processed += 1
        progress.set_postfix(processed=processed, skipped=skipped)

    logger.info(
        "Code embedding complete: %d processed, %d skipped (existing)", processed, skipped
    )


# ---------------------------------------------------------------------------
# CPG parsing helpers (extracted from cpg_dataset.py)
# ---------------------------------------------------------------------------


def _parse_cpg(
    sample_dir: Path,
) -> tuple[torch.Tensor | None, torch.Tensor | None, list[str] | None]:
    nodes_file = sample_dir / "nodes.csv"
    edges_file = sample_dir / "edges.csv"
    nodes_json = sample_dir / "nodes.json"
    edges_json = sample_dir / "edges.json"

    if nodes_file.exists() and edges_file.exists():
        return _parse_csv_cpg(nodes_file, edges_file)
    elif nodes_json.exists() and edges_json.exists():
        return _parse_json_cpg(nodes_json, edges_json)
    else:
        return _parse_dot_cpg(sample_dir)


def _parse_csv_cpg(
    nodes_file: Path, edges_file: Path
) -> tuple[torch.Tensor | None, torch.Tensor | None, list[str] | None]:
    try:
        nodes_df = pd.read_csv(nodes_file, sep="\t")
        edges_df = pd.read_csv(edges_file, sep="\t")
    except Exception:
        logger.exception("Failed to read CSV CPG files")
        return None, None, None

    if nodes_df.empty:
        return None, None, None

    node_ids = nodes_df.iloc[:, 0].tolist()
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    node_texts = []
    for _, node in nodes_df.iterrows():
        text = str(node.get("CODE", node.get("NAME", node.get("LABEL", ""))))
        node_texts.append(text if text != "nan" else "")

    src_list, dst_list, type_list = [], [], []
    for _, edge in edges_df.iterrows():
        src_id = edge.iloc[0]
        dst_id = edge.iloc[1]
        if src_id in id_to_idx and dst_id in id_to_idx:
            src_list.append(id_to_idx[src_id])
            dst_list.append(id_to_idx[dst_id])
            etype = str(edge.get("type", edge.get("label", "AST"))).upper()
            type_list.append(EDGE_TYPE_MAP.get(etype, 0))

    if not src_list:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)
    else:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_type = torch.tensor(type_list, dtype=torch.long)

    return edge_index, edge_type, node_texts


def _parse_json_cpg(
    nodes_file: Path, edges_file: Path
) -> tuple[torch.Tensor | None, torch.Tensor | None, list[str] | None]:
    try:
        with open(nodes_file) as f:
            nodes = json.load(f)
        with open(edges_file) as f:
            edges = json.load(f)
    except Exception:
        logger.exception("Failed to read JSON CPG files")
        return None, None, None

    if not nodes:
        return None, None, None

    id_to_idx = {n.get("id", n.get("_id", i)): i for i, n in enumerate(nodes)}

    node_texts = []
    for n in nodes:
        text = n.get("CODE", n.get("NAME", n.get("LABEL", n.get("code", ""))))
        node_texts.append(str(text) if text else "")

    src_list, dst_list, type_list = [], [], []
    for e in edges:
        src_id = e.get("src", e.get("inV", e.get("source")))
        dst_id = e.get("dst", e.get("outV", e.get("target")))
        if src_id in id_to_idx and dst_id in id_to_idx:
            src_list.append(id_to_idx[src_id])
            dst_list.append(id_to_idx[dst_id])
            etype = str(e.get("type", e.get("label", "AST"))).upper()
            type_list.append(EDGE_TYPE_MAP.get(etype, 0))

    if not src_list:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)
    else:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_type = torch.tensor(type_list, dtype=torch.long)

    return edge_index, edge_type, node_texts


def _parse_dot_cpg(
    sample_dir: Path,
) -> tuple[torch.Tensor | None, torch.Tensor | None, list[str] | None]:
    dot_files = list(sample_dir.glob("*.dot"))
    if not dot_files:
        return None, None, None

    nodes: dict[str, str] = {}
    edges: list[tuple[str, str, str]] = []

    for dot_file in dot_files:
        content = dot_file.read_text(errors="replace")

        for line in content.splitlines():
            line = line.strip()
            if "->" in line:
                parts = line.split("->")
                if len(parts) == 2:
                    src = parts[0].strip().strip('"')
                    rest = parts[1]
                    dst = rest.split("[")[0].strip().strip(";").strip('"')
                    # Extract edge type from label attribute, e.g. [ label = "CFG: "]
                    m = _EDGE_LABEL_RE.search(rest)
                    etype = m.group(1).upper() if m else "AST"
                    edges.append((src, dst, etype))
            elif line and not line.startswith(("digraph", "}", "//")):
                node_id = line.split("[")[0].strip().strip('"')
                label = ""
                if "label" in line:
                    # Joern HTML labels: <TYPE, LINE<BR/>code text>
                    if "<BR/>" in line:
                        after_br = line.split("<BR/>", 1)[1]
                        label = after_br.split(">")[0].strip()
                    elif 'label="' in line:
                        start = line.index('label="') + 7
                        end = line.index('"', start)
                        label = line[start:end]
                    elif "label=<" in line or "label = <" in line:
                        start = line.index("<") + 1
                        end = line.rindex(">")
                        label = line[start:end]
                # Unescape HTML entities
                label = label.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
                if node_id:
                    nodes[node_id] = label

    if not nodes:
        return None, None, None

    node_ids = list(nodes.keys())
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    node_texts = [nodes[nid] for nid in node_ids]

    src_list, dst_list, type_list = [], [], []
    for src_id, dst_id, etype in edges:
        if src_id in id_to_idx and dst_id in id_to_idx:
            src_list.append(id_to_idx[src_id])
            dst_list.append(id_to_idx[dst_id])
            type_list.append(EDGE_TYPE_MAP.get(etype, 0))

    if not src_list:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)
    else:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_type = torch.tensor(type_list, dtype=torch.long)

    return edge_index, edge_type, node_texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute DeBERTa embeddings for CPG nodes")
    parser.add_argument("--input", type=str, default=None, help="CPG exports directory")
    parser.add_argument("--output", type=str, default=None, help="Output embeddings directory")
    parser.add_argument("--subset-csv", type=str, default=None, help="Subset CSV path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    embed_code_nodes(input_dir=args.input, output_dir=args.output, subset_csv=args.subset_csv)
