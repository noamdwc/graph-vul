"""Adapters for external vulnerability datasets (HuggingFace) and eval dataset class."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from graph_vul.config import (
    DEBERTA_EMBEDDING_DIM,
    FUNC_MAX_WHITESPACE_TOKENS,
    PROCESSED_DIR,
    UNIXCODER_MODEL_NAME,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adapters — each returns a DataFrame with [sample_id, func_before, vul, cwe_id]
# ---------------------------------------------------------------------------


def _normalise_cwe(cwe_list) -> str:
    """Take the first CWE from a list/array, return 'CWE-XXX' or empty string."""
    if cwe_list is None or len(cwe_list) == 0:
        return ""
    cwe = str(cwe_list[0])
    if cwe.startswith("CWE-"):
        return cwe
    # Numeric-only → prefix
    if cwe.isdigit():
        return f"CWE-{cwe}"
    return cwe


def _token_count(text: str) -> int:
    return len(text.split())


def load_primevul() -> pd.DataFrame:
    """Load PrimeVul test split from HuggingFace and normalise."""
    logger.info("Loading PrimeVul from HuggingFace (test split)...")
    ds = load_dataset("colin/PrimeVul", "default", split="test")
    df = ds.to_pandas()

    df = df.rename(columns={"func": "func_before", "target": "vul"})
    df = df.dropna(subset=["vul", "func_before"]).reset_index(drop=True)
    df["vul"] = df["vul"].astype(int)
    df["cwe_id"] = df["cwe"].apply(_normalise_cwe)
    df["sample_id"] = "primevul_" + df.index.astype(str)

    # Filter by token length
    before = len(df)
    df["_ws_tokens"] = df["func_before"].fillna("").apply(_token_count)
    df = df[df["_ws_tokens"] <= FUNC_MAX_WHITESPACE_TOKENS].reset_index(drop=True)
    after = len(df)
    logger.info("PrimeVul: %d -> %d after %d-token filter (removed %d)", before, after, FUNC_MAX_WHITESPACE_TOKENS, before - after)

    return df[["sample_id", "func_before", "vul", "cwe_id"]].copy()


def load_diversevul() -> pd.DataFrame:
    """Load DiverseVul test split from HuggingFace and normalise."""
    logger.info("Loading DiverseVul from HuggingFace (test split)...")
    ds = load_dataset("bstee615/diversevul", split="test")
    df = ds.to_pandas()

    df = df.rename(columns={"func": "func_before", "target": "vul"})
    df["cwe_id"] = df["cwe"].apply(_normalise_cwe)
    df["sample_id"] = "diversevul_" + df.index.astype(str)

    # Filter by token length
    before = len(df)
    df["_ws_tokens"] = df["func_before"].fillna("").apply(_token_count)
    df = df[df["_ws_tokens"] <= FUNC_MAX_WHITESPACE_TOKENS].reset_index(drop=True)
    after = len(df)
    logger.info("DiverseVul: %d -> %d after %d-token filter (removed %d)", before, after, FUNC_MAX_WHITESPACE_TOKENS, before - after)

    return df[["sample_id", "func_before", "vul", "cwe_id"]].copy()


# Registry
EXTERNAL_DATASETS: dict[str, Callable[[], pd.DataFrame]] = {
    "primevul": load_primevul,
    "diversevul": load_diversevul,
}


# ---------------------------------------------------------------------------
# Dataset class for external evaluation
# ---------------------------------------------------------------------------


class ExternalEvalDataset(Dataset):
    """Tokenised dataset for external evaluation — accepts a DataFrame directly."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer | None = None,
        max_length: int = 512,
        cwe_embeddings_dir: str | Path | None = None,
    ) -> None:
        self.max_length = max_length
        self.cwe_dir = Path(cwe_embeddings_dir) if cwe_embeddings_dir else PROCESSED_DIR / "cwe_embeddings"

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(UNIXCODER_MODEL_NAME)

        self.sample_ids = df["sample_id"].values
        self.cwe_ids = df["cwe_id"].values
        self.labels = df["vul"].values.astype(float)

        self._cwe_cache: dict[str, torch.Tensor] = {}

        codes = df["func_before"].fillna("").tolist()
        logger.info("Pre-tokenizing %d external samples (max_length=%d)...", len(codes), self.max_length)
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
        logger.info("Pre-tokenization complete in %.1fs — cached %d input_ids", elapsed, len(self.input_ids))

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
            "token_count": self.attention_masks[idx].sum(),
            "sample_id": self.sample_ids[idx],
            "cwe_id": str(self.cwe_ids[idx]),
            "cwe_embedding": self._get_cwe_embedding(str(self.cwe_ids[idx])),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }
