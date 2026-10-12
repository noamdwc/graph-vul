from pathlib import Path
import torch
from graph_vul.model.func_finetune_lit_module import FuncFinetuneLitModule
from tests.test_matching_contract import tiny_backbone, prepared


def test_checkpoint_hparams_are_safe_and_self_contained(
    tiny_backbone, prepared, tmp_path
):
    import json

    catalog = json.loads((prepared / "cwe_embeddings/catalog.json").read_text())
    module = FuncFinetuneLitModule(
        cwe_embeddings_dir=prepared / "cwe_embeddings", catalog=catalog
    )
    path = tmp_path / "hyperparameters.pt"
    torch.save(dict(module.hparams), path)
    loaded = torch.load(path, weights_only=True)
    assert loaded["cwe_embeddings_dir"] is None
    assert loaded["catalog"] == catalog
