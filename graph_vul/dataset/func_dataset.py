"""Simple torch Dataset for pre-computed function embeddings + labels."""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from graph_vul.config import FUNC_EMBEDDINGS_DIR, FUNC_SPLITS_DIR, FUNC_SUBSET_CSV


class FuncDataset(Dataset):
    """Loads pre-computed .pt embeddings and binary vulnerability labels.

    Filters samples by split (train/val/test) using the split assignments CSV.
    """

    def __init__(
        self,
        split: str,
        subset_csv: str | Path = FUNC_SUBSET_CSV,
        embeddings_dir: str | Path = FUNC_EMBEDDINGS_DIR,
        splits_dir: str | Path = FUNC_SPLITS_DIR,
    ) -> None:
        self.embeddings_dir = Path(embeddings_dir)

        # Load subset metadata for labels
        subset = pd.read_csv(subset_csv, usecols=["sample_id", "vul"])

        # Load split assignments and filter to requested split
        assignments = pd.read_csv(Path(splits_dir) / "assignments.csv")
        split_ids = assignments[assignments["split"] == split]["sample_id"].values

        # Merge to get labels for this split
        split_df = subset[subset["sample_id"].isin(split_ids)]
        self.sample_ids = split_df["sample_id"].values
        self.labels = split_df["vul"].values.astype(float)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_id = self.sample_ids[idx]
        embedding = torch.load(
            self.embeddings_dir / f"{sample_id}.pt",
            weights_only=True,
        )
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return embedding, label
