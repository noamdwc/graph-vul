from pathlib import Path

import torch

from graph_vul.model.func_finetune_lit_module import FuncFinetuneLitModule


def test_checkpoint_hparams_are_safe_for_weights_only_load(monkeypatch, tmp_path):
    class DummyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {"hidden_size": 768})()
            self.emb = torch.nn.Embedding(1024, 768)
            self.encoder = type("Encoder", (), {"layer": torch.nn.ModuleList([torch.nn.Identity() for _ in range(12)])})()

        def forward(self, input_ids=None, attention_mask=None):
            hidden = self.emb(input_ids)
            return type("Output", (), {"last_hidden_state": hidden})()

    monkeypatch.setattr(
        "graph_vul.model.func_finetune_model.AutoModel.from_pretrained",
        lambda *args, **kwargs: DummyEncoder(),
    )

    module = FuncFinetuneLitModule(
        cwe_embeddings_dir=Path("/tmp/cwe_embeddings"),
        subset_csv=Path("/tmp/func_subset.csv"),
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"hyper_parameters": dict(module.hparams)}, checkpoint_path)

    loaded = torch.load(checkpoint_path, map_location="cpu")
    hparams = loaded["hyper_parameters"]
    assert isinstance(hparams["cwe_embeddings_dir"], str)
    assert isinstance(hparams["subset_csv"], str)
