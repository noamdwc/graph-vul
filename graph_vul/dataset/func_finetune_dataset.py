"""Dataset for fine-tuning UniXcoder on function-level vulnerability detection.

Returns raw tokenized code (not pre-computed embeddings) so the encoder
can be trained end-to-end, along with pre-computed CWE embeddings and labels.
"""

import logging
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

from graph_vul.config import (
    DEBERTA_EMBEDDING_DIM,
    FUNC_SPLITS_DIR,
    FUNC_SUBSET_CSV,
    PROCESSED_DIR,
    UNIXCODER_MODEL_NAME,
)


class FuncFinetuneDataset(Dataset):
    """Tokenizes source code on-the-fly for end-to-end UniXcoder fine-tuning."""

    def __init__(
        self,
        split: str,
        tokenizer: AutoTokenizer | None = None,
        max_length: int = 512,
        subset_csv: str | Path = FUNC_SUBSET_CSV,
        cwe_embeddings_dir: str | Path | None = None,
        splits_dir: str | Path = FUNC_SPLITS_DIR,
    ) -> None:
        self.max_length = max_length
        self.cwe_dir = Path(cwe_embeddings_dir) if cwe_embeddings_dir else PROCESSED_DIR / "cwe_embeddings"

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(UNIXCODER_MODEL_NAME)

        # Load subset metadata
        subset = pd.read_csv(subset_csv, usecols=["sample_id", "func_before", "cwe_id", "vul"])

        # Filter to requested split
        assignments = pd.read_csv(Path(splits_dir) / "assignments.csv")
        split_ids = set(assignments[assignments["split"] == split]["sample_id"].values)
        subset = subset[subset["sample_id"].isin(split_ids)].reset_index(drop=True)

        self.cwe_ids = subset["cwe_id"].values
        self.labels = subset["vul"].values.astype(float)

        # Cache CWE embeddings
        self._cwe_cache: dict[str, torch.Tensor] = {}

        # Pre-tokenize all samples once to avoid repeated work each epoch
        codes = subset["func_before"].fillna("").tolist()
        logger.info("[%s] Pre-tokenizing %d samples (max_length=%d)...", split, len(codes), self.max_length)
        t0 = time.perf_counter()
        tokenized = self.tokenizer(
            codes,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        self.input_ids = tokenized["input_ids"]
        self.attention_masks = tokenized["attention_mask"]
        elapsed = time.perf_counter() - t0
        logger.info("[%s] Pre-tokenization complete in %.1fs — cached %d input_ids", split, elapsed, len(self.input_ids))

    def _get_cwe_embedding(self, cwe_id: str) -> torch.Tensor:
        if cwe_id not in self._cwe_cache:
            cwe_path = self.cwe_dir / f"{cwe_id}.pt"
            if cwe_path.exists():
                self._cwe_cache[cwe_id] = torch.load(cwe_path, weights_only=True)
            else:
                self._cwe_cache[cwe_id] = torch.zeros(DEBERTA_EMBEDDING_DIM)
        return self._cwe_cache[cwe_id]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "cwe_embedding": self._get_cwe_embedding(str(self.cwe_ids[idx])),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }
