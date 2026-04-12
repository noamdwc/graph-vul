import torch

from graph_vul.model.func_finetune_model import FuncFinetuneModel


def test_matching_projection_accepts_half_precision_cwe_embeddings(monkeypatch):
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

    model = FuncFinetuneModel()
    input_ids = torch.ones(2, 4, dtype=torch.long)
    attention_mask = torch.ones(2, 4, dtype=torch.long)
    cwe_embedding = torch.randn(2, model.cwe_projection.in_features, dtype=torch.float16)

    outputs = model(
        input_ids,
        attention_mask,
        cwe_embedding,
        return_matching_outputs=True,
    )

    assert outputs["cwe_match"].dtype == model.cwe_projection.weight.dtype
    assert outputs["function_match"].dtype == model.function_projection.weight.dtype
    assert outputs["cwe_match"].shape == (2, model.cwe_projection.out_features)
    assert outputs["function_match"].shape == (2, model.function_projection.out_features)


def test_project_cwe_embeddings_casts_to_projection_dtype(monkeypatch):
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

    model = FuncFinetuneModel()
    cwe_embedding = torch.randn(3, model.cwe_projection.in_features, dtype=torch.float16)

    projected = model.project_cwe_embeddings(cwe_embedding)

    assert projected.dtype == model.cwe_projection.weight.dtype
    assert projected.shape == (3, model.cwe_projection.out_features)
