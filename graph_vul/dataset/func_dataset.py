"""Simple torch Dataset for pre-computed function embeddings + CWE embeddings + labels.

Optionally loads graph artifacts (AST+CFG) when use_graph=True.
"""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from graph_vul.config import (
    DEBERTA_EMBEDDING_DIM,
    FUNC_EMBEDDINGS_DIR,
    FUNC_SPLITS_DIR,
    FUNC_SUBSET_CSV,
    GRAPH_ARTIFACTS_DIR,
    PROCESSED_DIR,
)


class FuncDataset(Dataset):
    """Loads pre-computed code + CWE embeddings and binary vulnerability labels.

    Filters samples by split (train/val/test) using the split assignments CSV.
    When use_graph=True, also loads graph artifacts for each sample.
    """

    def __init__(
        self,
        split: str,
        subset_csv: str | Path = FUNC_SUBSET_CSV,
        embeddings_dir: str | Path = FUNC_EMBEDDINGS_DIR,
        cwe_embeddings_dir: str | Path | None = None,
        splits_dir: str | Path = FUNC_SPLITS_DIR,
        use_graph: bool = False,
        graph_artifacts_dir: str | Path = GRAPH_ARTIFACTS_DIR,
    ) -> None:
        self.embeddings_dir = Path(embeddings_dir)
        self.cwe_dir = Path(cwe_embeddings_dir) if cwe_embeddings_dir else PROCESSED_DIR / "cwe_embeddings"
        self.use_graph = use_graph
        self.graph_dir = Path(graph_artifacts_dir)

        # Load subset metadata for labels and CWE IDs
        subset = pd.read_csv(subset_csv, usecols=["sample_id", "cwe_id", "vul"])

        # Load split assignments and filter to requested split
        assignments = pd.read_csv(Path(splits_dir) / "assignments.csv")
        split_ids = assignments[assignments["split"] == split]["sample_id"].values

        # Merge to get labels for this split
        split_df = subset[subset["sample_id"].isin(split_ids)]
        self.sample_ids = split_df["sample_id"].values
        self.cwe_ids = split_df["cwe_id"].values
        self.labels = split_df["vul"].values.astype(float)

        # Cache CWE embeddings (few unique CWEs)
        self._cwe_cache: dict[str, torch.Tensor] = {}

    def _get_cwe_embedding(self, cwe_id: str) -> torch.Tensor:
        if cwe_id not in self._cwe_cache:
            cwe_path = self.cwe_dir / f"{cwe_id}.pt"
            if cwe_path.exists():
                self._cwe_cache[cwe_id] = torch.load(cwe_path, weights_only=True)
            else:
                self._cwe_cache[cwe_id] = torch.zeros(DEBERTA_EMBEDDING_DIM)
        return self._cwe_cache[cwe_id]

    def _load_graph(self, sample_id: str) -> Data:
        graph_path = self.graph_dir / f"{sample_id}.pt"
        if graph_path.exists():
            data = torch.load(graph_path, weights_only=True)
            return Data(
                node_type_ids=data["node_type_ids"],
                edge_index=data["edge_index"],
                edge_type_ids=data["edge_type_ids"],
            )
        # Fallback: empty graph (no nodes, no edges) — encoder returns zero vector
        return Data(
            node_type_ids=torch.zeros(0, dtype=torch.long),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_type_ids=torch.zeros(0, dtype=torch.long),
        )

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> tuple:
        sample_id = self.sample_ids[idx]
        embedding = torch.load(
            self.embeddings_dir / f"{sample_id}.pt",
            weights_only=True,
        )
        cwe_embedding = self._get_cwe_embedding(str(self.cwe_ids[idx]))
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.use_graph:
            graph_data = self._load_graph(str(sample_id))
            return embedding, cwe_embedding, graph_data, label
        return embedding, cwe_embedding, label


def func_graph_collate(batch: list[tuple]) -> tuple:
    """Collate function for FuncDataset with use_graph=True.

    Stacks code/cwe embeddings and labels normally,
    batches variable-size graphs via PyG Batch.from_data_list.
    """
    code_embs, cwe_embs, graphs, labels = zip(*batch)
    return (
        torch.stack(code_embs),
        torch.stack(cwe_embs),
        Batch.from_data_list(list(graphs)),
        torch.stack(labels),
    )
