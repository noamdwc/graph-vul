"""Dataset for fine-tuning UniXcoder on function-level vulnerability detection.

Returns raw tokenized code (not pre-computed embeddings) so the encoder
can be trained end-to-end, along with pre-computed CWE embeddings and labels.
Optionally loads graph artifacts (AST+CFG) when use_graph=True.
"""

import logging
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from transformers import AutoTokenizer

from graph_vul.config import (
    CODE_MODEL_NAME,
    DEBERTA_EMBEDDING_DIM,
    FUNC_SPLITS_DIR,
    FUNC_SUBSET_CSV,
    GRAPH_ARTIFACTS_DIR,
    PROCESSED_DIR,
)

logger = logging.getLogger(__name__)


def load_split_assignments(
    split: str,
    splits_dir: str | Path = FUNC_SPLITS_DIR,
) -> pd.DataFrame:
    splits_dir = Path(splits_dir)
    assignments = pd.read_csv(splits_dir / "assignments.csv")
    if split in {"unseen_val", "unseen_test"}:
        derived_path = splits_dir / "unseen_assignments.csv"
        if derived_path.exists():
            derived = pd.read_csv(derived_path)
            return derived[derived["split"] == split][["sample_id", "split"]].copy()
    return assignments[assignments["split"] == split][["sample_id", "split"]].copy()


def available_splits(splits_dir: str | Path = FUNC_SPLITS_DIR) -> list[str]:
    assignments = pd.read_csv(Path(splits_dir) / "assignments.csv")
    derived_path = Path(splits_dir) / "unseen_assignments.csv"
    if derived_path.exists():
        assignments = pd.concat([assignments, pd.read_csv(derived_path)], ignore_index=True)
    ordered = []
    for split_name in ["train", "val", "test", "unseen_val", "unseen_test"]:
        if split_name in assignments["split"].values:
            ordered.append(split_name)
    return ordered


class FuncFinetuneDataset(Dataset):
    """Tokenizes source code on-the-fly for end-to-end UniXcoder fine-tuning."""

    def __init__(
        self,
        split: str,
        tokenizer: AutoTokenizer | None = None,
        max_length: int = 512,
        subset_csv: str | Path = FUNC_SUBSET_CSV,
        cwe_embeddings_dir: str | Path | None = None,
        embedding_dim: int = DEBERTA_EMBEDDING_DIM,
        splits_dir: str | Path = FUNC_SPLITS_DIR,
        use_graph: bool = False,
        graph_artifacts_dir: str | Path = GRAPH_ARTIFACTS_DIR,
    ) -> None:
        self.max_length = max_length
        self.cwe_dir = Path(cwe_embeddings_dir) if cwe_embeddings_dir else PROCESSED_DIR / "cwe_embeddings"
        self.embedding_dim = embedding_dim
        self.use_graph = use_graph
        self.graph_dir = Path(graph_artifacts_dir)

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(CODE_MODEL_NAME)

        # Qwen3 ships without a pad token; reuse EOS for right-padding.
        # attention_mask still distinguishes real vs pad, so last-non-pad-token
        # pooling and the loss mask stay correct. getattr-guarded so test fakes
        # without a pad_token attribute keep working.
        if getattr(self.tokenizer, "pad_token", None) is None and getattr(
            self.tokenizer, "eos_token", None
        ) is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load subset metadata
        subset = pd.read_csv(subset_csv, usecols=["sample_id", "func_before", "cwe_id", "vul"])

        # Filter to requested split
        split_assignments = load_split_assignments(split=split, splits_dir=splits_dir)
        split_ids = set(split_assignments["sample_id"].values)
        subset = subset[subset["sample_id"].isin(split_ids)].reset_index(drop=True)

        self.sample_ids = subset["sample_id"].values
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
                self._cwe_cache[cwe_id] = torch.load(cwe_path, weights_only=True).float()
            else:
                self._cwe_cache[cwe_id] = torch.zeros(self.embedding_dim, dtype=torch.float32)
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
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "token_count": self.attention_masks[idx].sum(),
            "sample_id": torch.tensor(int(self.sample_ids[idx]), dtype=torch.long),
            "cwe_id": str(self.cwe_ids[idx]),
            "cwe_embedding": self._get_cwe_embedding(str(self.cwe_ids[idx])),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }
        if self.use_graph:
            item["graph_data"] = self._load_graph(str(self.sample_ids[idx]))
        return item


def finetune_graph_collate(batch: list[dict]) -> dict:
    """Collate function for FuncFinetuneDataset with use_graph=True.

    Stacks tensor fields normally, batches graph_data via PyG Batch.from_data_list.
    """
    result: dict = {}
    keys = batch[0].keys()
    for key in keys:
        if key == "graph_data":
            result[key] = Batch.from_data_list([item[key] for item in batch])
        elif torch.is_tensor(batch[0][key]):
            result[key] = torch.stack([item[key] for item in batch])
        else:
            result[key] = [item[key] for item in batch]
    return result
