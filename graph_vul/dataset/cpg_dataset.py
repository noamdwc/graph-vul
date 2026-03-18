"""PyTorch Geometric Dataset for Joern CPG exports with DeBERTa embeddings."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from transformers import AutoModel, AutoTokenizer

from graph_vul.config import (
    DEBERTA_EMBEDDING_DIM,
    DEBERTA_MODEL_NAME,
    PROCESSED_DIR,
    RAW_DIR,
    SUBSET_CSV,
)

logger = logging.getLogger(__name__)

EDGE_TYPE_MAP = {"AST": 0, "CFG": 1, "PDG": 2}


class CPGDataset(Dataset):
    def __init__(
        self,
        root: str | None = None,
        subset_csv: str | None = None,
        raw_cpg_dir: str | None = None,
        transform=None,
        pre_transform=None,
    ):
        self._subset_csv = subset_csv or str(SUBSET_CSV)
        self._raw_cpg_dir = Path(raw_cpg_dir) if raw_cpg_dir else RAW_DIR
        self._cache_dir = Path(root or str(PROCESSED_DIR)) / "cached_embeddings"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._metadata = pd.read_csv(self._subset_csv)

        # Filter to samples that have valid CPG exports
        valid_ids = []
        for _, row in self._metadata.iterrows():
            sample_dir = self._raw_cpg_dir / str(row["sample_id"])
            if sample_dir.exists() and any(sample_dir.iterdir()):
                valid_ids.append(row["sample_id"])
            else:
                logger.warning("Sample %s has no valid CPG export, skipping", row["sample_id"])
        self._metadata = self._metadata[self._metadata["sample_id"].isin(valid_ids)].reset_index(
            drop=True
        )

        # Lazy-load DeBERTa
        self._tokenizer = None
        self._deberta = None

        super().__init__(root or str(PROCESSED_DIR), transform, pre_transform)

    def _load_deberta(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(DEBERTA_MODEL_NAME)
            self._deberta = AutoModel.from_pretrained(DEBERTA_MODEL_NAME)
            self._deberta.eval()

    @property
    def processed_file_names(self) -> list[str]:
        return [f"sample_{sid}.pt" for sid in self._metadata["sample_id"]]

    def len(self) -> int:
        return len(self._metadata)

    def get(self, idx: int) -> Data:
        row = self._metadata.iloc[idx]
        sample_id = str(row["sample_id"])
        cache_path = self._cache_dir / f"sample_{sample_id}.pt"

        # Load from cache if available
        if cache_path.exists():
            return torch.load(cache_path, weights_only=False)

        # Parse CPG
        sample_dir = self._raw_cpg_dir / sample_id
        edge_index, edge_type, node_texts = self._parse_cpg(sample_dir)

        if edge_index is None:
            logger.warning("Failed to parse CPG for sample %s", sample_id)
            # Return a minimal valid Data object
            data = Data(
                x=torch.zeros(1, DEBERTA_EMBEDDING_DIM),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                edge_type=torch.zeros(0, dtype=torch.long),
                text_vec=torch.zeros(DEBERTA_EMBEDDING_DIM),
                y=torch.tensor([float(row["vul"])]),
            )
            torch.save(data, cache_path)
            return data

        # Compute node embeddings via DeBERTa
        node_features = self._embed_texts(node_texts)

        # Compute CVE text vector
        cve_desc = row.get("cve_desc", "")
        if isinstance(cve_desc, str) and cve_desc.strip():
            text_vec = self._embed_texts([cve_desc])[0]
        else:
            text_vec = torch.zeros(DEBERTA_EMBEDDING_DIM)

        # Build Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_type=edge_type,
            text_vec=text_vec,
            y=torch.tensor([float(row["vul"])]),
        )

        # Cache to disk
        torch.save(data, cache_path)
        return data

    def _parse_cpg(
        self, sample_dir: Path
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, list[str] | None]:
        """Parse Joern CPG export files into edge_index, edge_type, and node texts."""
        nodes_file = sample_dir / "nodes.csv"
        edges_file = sample_dir / "edges.csv"

        # Also check for JSON format
        nodes_json = sample_dir / "nodes.json"
        edges_json = sample_dir / "edges.json"

        if nodes_file.exists() and edges_file.exists():
            return self._parse_csv_cpg(nodes_file, edges_file)
        elif nodes_json.exists() and edges_json.exists():
            return self._parse_json_cpg(nodes_json, edges_json)
        else:
            # Try to find any cpg14 export format
            return self._parse_dot_cpg(sample_dir)

    def _parse_csv_cpg(
        self, nodes_file: Path, edges_file: Path
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, list[str] | None]:
        try:
            nodes_df = pd.read_csv(nodes_file, sep="\t")
            edges_df = pd.read_csv(edges_file, sep="\t")
        except Exception:
            logger.exception("Failed to read CSV CPG files")
            return None, None, None

        if nodes_df.empty:
            return None, None, None

        # Build node ID to index mapping
        node_ids = nodes_df.iloc[:, 0].tolist()
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        # Extract node text from available columns
        node_texts = []
        for _, node in nodes_df.iterrows():
            text = str(node.get("CODE", node.get("NAME", node.get("LABEL", ""))))
            node_texts.append(text if text != "nan" else "")

        # Build edges
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
        self, nodes_file: Path, edges_file: Path
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
        self, sample_dir: Path
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, list[str] | None]:
        """Fallback: parse cpg14 dot format if present."""
        dot_files = list(sample_dir.glob("*.dot"))
        if not dot_files:
            return None, None, None

        nodes = {}  # id -> text
        edges = []  # (src_id, dst_id, type)

        for dot_file in dot_files:
            content = dot_file.read_text(errors="replace")
            # Determine edge type from filename
            fname = dot_file.stem.lower()
            if "cfg" in fname:
                etype = "CFG"
            elif "pdg" in fname or "ddg" in fname or "cdg" in fname:
                etype = "PDG"
            else:
                etype = "AST"

            for line in content.splitlines():
                line = line.strip()
                if "->" in line:
                    parts = line.split("->")
                    if len(parts) == 2:
                        src = parts[0].strip().strip('"')
                        dst = parts[1].split("[")[0].strip().strip(";").strip('"')
                        edges.append((src, dst, etype))
                elif line and not line.startswith(("digraph", "}", "//")):
                    # Node definition
                    node_id = line.split("[")[0].strip().strip('"')
                    label = ""
                    if "label=" in line:
                        start = line.index("label=") + 6
                        if line[start] == '"':
                            end = line.index('"', start + 1)
                            label = line[start + 1 : end]
                        else:
                            label = line[start:].split("]")[0].split(",")[0]
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

    @torch.no_grad()
    def _embed_texts(self, texts: list[str]) -> torch.Tensor:
        """Embed a list of text strings using DeBERTa, returning [len(texts), embedding_dim]."""
        self._load_deberta()

        embeddings = []
        for text in texts:
            if not text.strip():
                embeddings.append(torch.zeros(DEBERTA_EMBEDDING_DIM))
                continue

            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            )
            outputs = self._deberta(**inputs)
            # Mean pool over token embeddings
            emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
            embeddings.append(emb.cpu())

        return torch.stack(embeddings)
