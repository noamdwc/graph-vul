"""Simple torch Dataset for pre-computed function embeddings + CWE embeddings + labels."""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from graph_vul.config import (
    DEBERTA_EMBEDDING_DIM,
    FUNC_EMBEDDINGS_DIR,
    FUNC_SPLITS_DIR,
    FUNC_SUBSET_CSV,
    PROCESSED_DIR,
)


class FuncDataset(Dataset):
    """Loads pre-computed code + CWE embeddings and binary vulnerability labels.

    Filters samples by split (train/val/test) using the split assignments CSV.
    """

    def __init__(
        self,
        split: str,
        subset_csv: str | Path = FUNC_SUBSET_CSV,
        embeddings_dir: str | Path = FUNC_EMBEDDINGS_DIR,
        cwe_embeddings_dir: str | Path | None = None,
        splits_dir: str | Path = FUNC_SPLITS_DIR,
    ) -> None:
        self.embeddings_dir = Path(embeddings_dir)
        self.cwe_dir = Path(cwe_embeddings_dir) if cwe_embeddings_dir else PROCESSED_DIR / "cwe_embeddings"

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

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_id = self.sample_ids[idx]
        embedding = torch.load(
            self.embeddings_dir / f"{sample_id}.pt",
            weights_only=True,
        )
        cwe_embedding = self._get_cwe_embedding(str(self.cwe_ids[idx]))
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return embedding, cwe_embedding, label
