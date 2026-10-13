"""Adapters for external vulnerability datasets (HuggingFace) and eval dataset class."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from graph_vul.config import (
    CODE_MODEL_NAME,
    CODE_MODEL_REVISION,
    FUNC_MAX_WHITESPACE_TOKENS,
)

from graph_vul.model.cwe_matching import parse_cwe_ids

logger = logging.getLogger(__name__)


EXTERNAL_DATASET_REVISIONS = {
    'primevul': '4fd7158322872d711e90f091dbd8673ef32cb1be',
    'diversevul': '3ed5dae8fdf3f6026f1e260f789656c4bb6d98e6',
}

# ---------------------------------------------------------------------------
# Adapters — each returns a DataFrame with [sample_id, func_before, vul, cwe_id]
# ---------------------------------------------------------------------------


def _normalise_cwe(values) -> str:
    return ";".join(parse_cwe_ids(values))


def _token_count(text: str) -> int:
    return len(text.split())


def load_primevul() -> pd.DataFrame:
    """Load PrimeVul test split from HuggingFace and normalise."""
    logger.info("Loading PrimeVul from HuggingFace (test split)...")
    ds = load_dataset("colin/PrimeVul", "default", split="test", revision=EXTERNAL_DATASET_REVISIONS["primevul"])
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
    ds = load_dataset("bstee615/diversevul", split="test", revision=EXTERNAL_DATASET_REVISIONS["diversevul"])
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
        node_vocab: dict | None = None,
        edge_vocab: dict | None = None,
    ) -> None:
        self.max_length = max_length
        self.graphs = None
        if node_vocab is not None:
            from graph_vul.pipeline.treesitter_graphs import function_graph
            self.graphs = [function_graph(code, node_vocab, edge_vocab) for code in df["func_before"]]

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                CODE_MODEL_NAME, revision=CODE_MODEL_REVISION
            )

        self.sample_ids = df["sample_id"].values
        self.cwe_ids = df["cwe_id"].values
        if not df["sample_id"].is_unique or not df["vul"].isin([0, 1]).all():
            raise ValueError("Expected unique external sample IDs and binary labels")
        self.labels = df["vul"].values.astype(float)

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

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "token_count": self.attention_masks[idx].sum(),
            "sample_id": self.sample_ids[idx],
            "cwe_ids": parse_cwe_ids(self.cwe_ids[idx]),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

        if self.graphs is not None:
            item["graph_data"] = self.graphs[idx]
        return item
