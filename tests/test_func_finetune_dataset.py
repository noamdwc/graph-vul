"""Tests for the function-level fine-tuning dataset."""

import sys
import types

import pandas as pd
import torch

torch_geometric_module = types.ModuleType("torch_geometric")
torch_geometric_data_module = types.ModuleType("torch_geometric.data")


class _FakeData:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeBatch:
    @staticmethod
    def from_data_list(data_list):
        return data_list


torch_geometric_data_module.Batch = _FakeBatch
torch_geometric_data_module.Data = _FakeData
torch_geometric_module.data = torch_geometric_data_module
sys.modules.setdefault("torch_geometric", torch_geometric_module)
sys.modules.setdefault("torch_geometric.data", torch_geometric_data_module)

from graph_vul.dataset.func_finetune_dataset import FuncFinetuneDataset, finetune_graph_collate


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
