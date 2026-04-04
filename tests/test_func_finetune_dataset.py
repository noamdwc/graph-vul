"""Tests for the function-level fine-tuning dataset."""

import pandas as pd
import torch

from graph_vul.dataset.func_finetune_dataset import (
    FuncFinetuneDataset,
    available_splits,
    finetune_graph_collate,
    load_split_assignments,
)


class _FakeTokenizer:
    def __call__(self, codes, truncation, max_length, padding, return_tensors):
        assert truncation is True
        assert padding == "max_length"
        assert return_tensors == "pt"

        input_ids = []
        attention_masks = []
        for idx, code in enumerate(codes, start=1):
            token_count = min(len(code.split()), max_length)
            input_ids.append([idx] * token_count + [0] * (max_length - token_count))
            attention_masks.append([1] * token_count + [0] * (max_length - token_count))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }


def test_finetune_dataset_exposes_sample_id_and_token_count(tmp_path):
    subset_csv = tmp_path / "func_subset.csv"
    splits_dir = tmp_path / "splits"
    cwe_dir = tmp_path / "cwe_embeddings"
    splits_dir.mkdir()
    cwe_dir.mkdir()

    pd.DataFrame(
        [
            {"sample_id": 101, "func_before": "a b c", "cwe_id": "CWE-119", "vul": 1},
            {"sample_id": 202, "func_before": "d e", "cwe_id": "CWE-120", "vul": 0},
        ]
    ).to_csv(subset_csv, index=False)
    pd.DataFrame(
        [
            {"sample_id": 101, "split": "train"},
            {"sample_id": 202, "split": "train"},
        ]
    ).to_csv(splits_dir / "assignments.csv", index=False)

    torch.save(torch.ones(768), cwe_dir / "CWE-119.pt")
    torch.save(torch.zeros(768), cwe_dir / "CWE-120.pt")

    dataset = FuncFinetuneDataset(
        split="train",
        tokenizer=_FakeTokenizer(),
        max_length=8,
        subset_csv=subset_csv,
        cwe_embeddings_dir=cwe_dir,
        splits_dir=splits_dir,
    )

    first = dataset[0]
    second = dataset[1]

    assert first["sample_id"].item() == 101
    assert second["sample_id"].item() == 202
    assert first["token_count"].item() == 3
    assert second["token_count"].item() == 2

    batch = finetune_graph_collate([first, second])
    assert batch["sample_id"].tolist() == [101, 202]
    assert batch["token_count"].tolist() == [3, 2]


def test_available_splits_returns_present_splits_in_canonical_order(tmp_path):
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()

    pd.DataFrame(
        [
            {"sample_id": 1, "split": "test"},
            {"sample_id": 2, "split": "train"},
            {"sample_id": 3, "split": "val"},
        ]
    ).to_csv(splits_dir / "assignments.csv", index=False)

    assert available_splits(splits_dir) == ["train", "val", "test"]


def test_available_splits_includes_derived_unseen_splits(tmp_path):
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()

    pd.DataFrame(
        [
            {"sample_id": 1, "split": "train"},
            {"sample_id": 2, "split": "val"},
            {"sample_id": 3, "split": "test"},
        ]
    ).to_csv(splits_dir / "assignments.csv", index=False)
    pd.DataFrame(
        [
            {"sample_id": 2, "split": "unseen_val"},
            {"sample_id": 3, "split": "unseen_test"},
        ]
    ).to_csv(splits_dir / "unseen_assignments.csv", index=False)

    assert available_splits(splits_dir) == ["train", "val", "test", "unseen_val", "unseen_test"]
    unseen = load_split_assignments("unseen_test", splits_dir)
    assert unseen.to_dict("records") == [{"sample_id": 3, "split": "unseen_test"}]
