"""Dataset for fine-tuning UniXcoder on function-level vulnerability detection.

Returns raw tokenized code (not pre-computed embeddings) so the encoder
can be trained end-to-end, along with all CWE annotations as target metadata.
Optionally loads graph artifacts (AST) when use_graph=True.
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
    CODE_MODEL_REVISION,
    FUNC_SPLITS_DIR,
    FUNC_SUBSET_CSV,
    GRAPH_ARTIFACTS_DIR,
)

from graph_vul.model.cwe_matching import parse_cwe_ids

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
        splits_dir: str | Path = FUNC_SPLITS_DIR,
        use_graph: bool = False,
        graph_artifacts_dir: str | Path = GRAPH_ARTIFACTS_DIR,
    ) -> None:
        self.max_length = max_length
        self.use_graph = use_graph
        self.graph_dir = Path(graph_artifacts_dir)

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                CODE_MODEL_NAME, revision=CODE_MODEL_REVISION
            )

        # Load subset metadata
        subset = pd.read_csv(subset_csv, usecols=["sample_id", "func_before", "cwe_id", "vul"])

        # Filter to requested split
        split_assignments = load_split_assignments(split=split, splits_dir=splits_dir)
        split_ids = set(split_assignments["sample_id"].values)
        subset = subset[subset["sample_id"].isin(split_ids)].reset_index(drop=True)

        self.sample_ids = subset["sample_id"].values
        self.cwe_ids = subset["cwe_id"].values
        self.labels = subset["vul"].values.astype(float)
        if not subset["sample_id"].is_unique or not subset["vul"].isin([0, 1]).all():
            raise ValueError("Expected unique sample IDs and binary labels")
        if len(subset) != len(split_ids):
            raise ValueError("Split assignments reference missing samples")

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

    def _load_graph(self, sample_id: str) -> Data:
        graph_path = self.graph_dir / f"{sample_id}.pt"
        data = torch.load(graph_path, weights_only=True)
        return Data(**data)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "token_count": self.attention_masks[idx].sum(),
            "sample_id": torch.tensor(int(self.sample_ids[idx]), dtype=torch.long),
            "cwe_ids": parse_cwe_ids(self.cwe_ids[idx]),
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
