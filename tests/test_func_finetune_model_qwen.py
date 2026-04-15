"""Smoke tests for FuncFinetuneModel with the Qwen3-0.6B backbone."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from graph_vul.config import CODE_EMBEDDING_DIM, CODE_MODEL_NAME, DEBERTA_EMBEDDING_DIM
from graph_vul.model.func_finetune_model import FuncFinetuneModel


@pytest.fixture(scope="module")
def tiny_model():
    model = FuncFinetuneModel(
        model_name=CODE_MODEL_NAME,
        cwe_dim=DEBERTA_EMBEDDING_DIM,
        graph_dim=64,
        num_node_types=8,
        num_relations=4,
        fusion_layers=[0, 13, 27],
        fusion_method="scalar_gated_residual",
    )
    model.eval()
    return model


def _dummy_graph_batch(batch_size: int) -> Batch:
    data_list = []
    for _ in range(batch_size):
        data_list.append(
            Data(
                node_type_ids=torch.tensor([0, 1, 2], dtype=torch.long),
                edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                edge_type_ids=torch.tensor([0, 1], dtype=torch.long),
            )
        )
    return Batch.from_data_list(data_list)


def test_forward_shapes(tiny_model):
    B, S = 2, 16
    input_ids = torch.randint(0, 1000, (B, S), dtype=torch.long)
    attention_mask = torch.ones(B, S, dtype=torch.long)
    attention_mask[1, 10:] = 0
    cwe = torch.randn(B, DEBERTA_EMBEDDING_DIM)
    graph_batch = _dummy_graph_batch(B)

    with torch.no_grad():
        out = tiny_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cwe_embedding=cwe,
            graph_batch=graph_batch,
            return_matching_outputs=True,
        )

    assert out["logits"].shape == (B, 1)
    assert out["pooled"].shape == (B, CODE_EMBEDDING_DIM)
    assert out["function_match"].shape[0] == B
    assert out["cwe_match"].shape[0] == B


def test_last_token_pooling_uses_last_non_pad(tiny_model):
    B, S = 1, 16
    input_ids = torch.randint(0, 1000, (B, S), dtype=torch.long)

    mask_full = torch.ones(B, S, dtype=torch.long)
    mask_padded = torch.zeros(B, S, dtype=torch.long)
    mask_padded[0, :5] = 1
    cwe = torch.randn(B, DEBERTA_EMBEDDING_DIM)
    graph_batch = _dummy_graph_batch(B)

    with torch.no_grad():
        out_full = tiny_model(
            input_ids=input_ids, attention_mask=mask_full,
            cwe_embedding=cwe, graph_batch=graph_batch,
            return_matching_outputs=True,
        )
        out_padded = tiny_model(
            input_ids=input_ids, attention_mask=mask_padded,
            cwe_embedding=cwe, graph_batch=graph_batch,
            return_matching_outputs=True,
        )

    assert not torch.allclose(out_full["pooled"], out_padded["pooled"]), (
        "Pooled vectors must differ when attention_mask selects a different last token"
    )


def test_fusion_hook_receives_gradient(tiny_model):
    B, S = 2, 16
    input_ids = torch.randint(0, 1000, (B, S), dtype=torch.long)
    attention_mask = torch.ones(B, S, dtype=torch.long)
    cwe = torch.randn(B, DEBERTA_EMBEDDING_DIM)
    graph_batch = _dummy_graph_batch(B)

    tiny_model.train()
    logits = tiny_model(
        input_ids=input_ids, attention_mask=attention_mask,
        cwe_embedding=cwe, graph_batch=graph_batch,
    )
    logits.sum().backward()

    fused_layer_key = next(iter(tiny_model.fusion_modules))
    fusion_mod = tiny_model.fusion_modules[fused_layer_key]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in fusion_mod.parameters()), (
        "Fusion module received no gradient — hook did not fire"
    )
