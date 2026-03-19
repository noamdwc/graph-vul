"""PyTorch Geometric Dataset that loads pre-computed embeddings from disk."""

import logging
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data, Dataset

from graph_vul.config import (
    DEBERTA_EMBEDDING_DIM,
    PROCESSED_DIR,
    SUBSET_CSV,
)

logger = logging.getLogger(__name__)


class CPGDataset(Dataset):
    def __init__(
        self,
        root: str | None = None,
        subset_csv: str | None = None,
        code_embeddings_dir: str | None = None,
        cve_embeddings_dir: str | None = None,
        transform=None,
        pre_transform=None,
    ):
        self._subset_csv = subset_csv or str(SUBSET_CSV)
        processed = Path(root or str(PROCESSED_DIR))
        self._code_dir = Path(code_embeddings_dir) if code_embeddings_dir else processed / "code_embeddings"
        self._cve_dir = Path(cve_embeddings_dir) if cve_embeddings_dir else processed / "cve_embeddings"

        self._metadata = pd.read_csv(self._subset_csv)

        # Filter to samples that have pre-computed code embeddings
        valid_ids = []
        for _, row in self._metadata.iterrows():
            sample_id = str(row["sample_id"])
            if (self._code_dir / f"{sample_id}.pt").exists():
                valid_ids.append(row["sample_id"])
            else:
                logger.warning("Sample %s has no code embedding, skipping", sample_id)
        self._metadata = self._metadata[self._metadata["sample_id"].isin(valid_ids)].reset_index(
            drop=True
        )

        super().__init__(root or str(PROCESSED_DIR), transform, pre_transform)

    @property
    def processed_file_names(self) -> list[str]:
        return [f"{sid}.pt" for sid in self._metadata["sample_id"]]

    def len(self) -> int:
        return len(self._metadata)

    def get(self, idx: int) -> Data:
        row = self._metadata.iloc[idx]
        sample_id = str(row["sample_id"])

        # Load pre-computed code embedding (x, edge_index, edge_type)
        code_data = torch.load(self._code_dir / f"{sample_id}.pt", weights_only=False)
        x = code_data["x"]
        edge_index = code_data["edge_index"]
        edge_type = code_data["edge_type"]

        # Load pre-computed CVE embedding
        cve_path = self._cve_dir / f"{sample_id}.pt"
        if cve_path.exists():
            text_vec = torch.load(cve_path, weights_only=False)
        else:
            text_vec = torch.zeros(DEBERTA_EMBEDDING_DIM)

        # Store graph-level text features as a single row so PyG batches them
        # into [batch_size, hidden_dim] instead of flattening 1D vectors.
        text_vec = text_vec.reshape(1, -1)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            text_vec=text_vec,
            y=torch.tensor(float(row["vul"])),
        )
