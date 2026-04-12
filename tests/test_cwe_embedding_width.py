import pandas as pd
import torch

from graph_vul.dataset.external_datasets import ExternalEvalDataset
from graph_vul.dataset.func_finetune_dataset import FuncFinetuneDataset
from graph_vul.model.func_finetune_lit_module import FuncFinetuneLitModule


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


def test_external_eval_dataset_uses_configured_embedding_width_for_missing_cwe(tmp_path):
    df = pd.DataFrame(
        [
            {"sample_id": "row-1", "func_before": "int x = 0;", "vul": 1, "cwe_id": "CWE-999"},
        ]
    )

    dataset = ExternalEvalDataset(
        df=df,
        tokenizer=_FakeTokenizer(),
        cwe_embeddings_dir=tmp_path,
        embedding_dim=1600,
    )

    item = dataset[0]

    assert item["cwe_embedding"].shape == (1600,)
    assert item["cwe_embedding"].dtype == torch.float32


def test_func_finetune_dataset_uses_configured_embedding_width_for_missing_cwe(tmp_path):
    subset_csv = tmp_path / "func_subset.csv"
    splits_dir = tmp_path / "splits"
    cwe_dir = tmp_path / "cwe_embeddings"
    splits_dir.mkdir()
    cwe_dir.mkdir()

    pd.DataFrame(
        [
            {"sample_id": 101, "func_before": "a b c", "cwe_id": "CWE-999", "vul": 1},
        ]
    ).to_csv(subset_csv, index=False)
    pd.DataFrame(
        [
            {"sample_id": 101, "split": "train"},
        ]
    ).to_csv(splits_dir / "assignments.csv", index=False)

    dataset = FuncFinetuneDataset(
        split="train",
        tokenizer=_FakeTokenizer(),
        max_length=8,
        subset_csv=subset_csv,
        cwe_embeddings_dir=cwe_dir,
        embedding_dim=1600,
        splits_dir=splits_dir,
    )

    item = dataset[0]

    assert item["cwe_embedding"].shape == (1600,)
    assert item["cwe_embedding"].dtype == torch.float32


def test_lit_module_uses_model_cwe_width_for_embedding_store(monkeypatch, tmp_path):
    class DummyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {"hidden_size": 768})()
            self.emb = torch.nn.Embedding(1024, 768)
            self.encoder = type(
                "Encoder",
                (),
                {"layer": torch.nn.ModuleList([torch.nn.Identity() for _ in range(12)])},
            )()

        def forward(self, input_ids=None, attention_mask=None):
            hidden = self.emb(input_ids)
            return type("Output", (), {"last_hidden_state": hidden})()

    monkeypatch.setattr(
        "graph_vul.model.func_finetune_model.AutoModel.from_pretrained",
        lambda *args, **kwargs: DummyEncoder(),
    )

    module = FuncFinetuneLitModule(
        cwe_embeddings_dir=tmp_path / "cwe_embeddings",
        subset_csv=tmp_path / "missing_subset.csv",
    )

    assert module.cwe_store.embedding_dim == module.model.cwe_projection.in_features
