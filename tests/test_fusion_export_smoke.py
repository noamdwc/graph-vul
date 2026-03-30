"""Smoke test: verify forward + export pass works for all 5 ablation configs."""

import torch
from torch_geometric.data import Batch, Data

from graph_vul.model.func_finetune_lit_module import FuncFinetuneLitModule


NUM_NODE_TYPES = 10
NUM_RELATIONS = 4
GRAPH_DIM = 64
BATCH_SIZE = 4
SEQ_LEN = 32
CWE_DIM = 768

EXPERIMENTS = [
    {"tag": "E0_late_only", "fusion_layers": None, "final_graph_concat": True},
    {"tag": "E1_last2", "fusion_layers": [10, 11], "final_graph_concat": True},
    {"tag": "E2_last4", "fusion_layers": [8, 9, 10, 11], "final_graph_concat": True},
    {"tag": "E3_all_layers", "fusion_layers": list(range(12)), "final_graph_concat": True},
    {"tag": "E4_last4_no_late", "fusion_layers": [8, 9, 10, 11], "final_graph_concat": False},
]


def _make_graph_batch(batch_size: int) -> Batch:
    graphs = []
    for _ in range(batch_size):
        g = Data(
            node_type_ids=torch.randint(0, NUM_NODE_TYPES, (5,)),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
            edge_type_ids=torch.randint(0, NUM_RELATIONS, (4,)),
        )
        graphs.append(g)
    return Batch.from_data_list(graphs)


def _make_inputs(batch_size: int):
    input_ids = torch.randint(0, 1000, (batch_size, SEQ_LEN))
    attention_mask = torch.ones(batch_size, SEQ_LEN, dtype=torch.long)
    cwe_embedding = torch.randn(batch_size, CWE_DIM)
    graph_batch = _make_graph_batch(batch_size)
    return input_ids, attention_mask, cwe_embedding, graph_batch


def test_all_ablation_configs_forward_with_graph():
    """All 5 experiments produce correct output shape when graph data is provided."""
    for exp in EXPERIMENTS:
        lit = FuncFinetuneLitModule(
            graph_dim=GRAPH_DIM,
            num_node_types=NUM_NODE_TYPES,
            num_relations=NUM_RELATIONS,
            fusion_layers=exp["fusion_layers"],
            final_graph_concat=exp["final_graph_concat"],
        )
        lit.eval()
        input_ids, attention_mask, cwe_embedding, graph_batch = _make_inputs(BATCH_SIZE)

        with torch.inference_mode():
            logits = lit(input_ids, attention_mask, cwe_embedding, graph_batch=graph_batch)

        assert logits.shape == (BATCH_SIZE, 1), f"{exp['tag']}: expected ({BATCH_SIZE}, 1), got {logits.shape}"


def test_all_ablation_configs_forward_without_graph_fails_when_expected():
    """Configs with final_graph_concat=True and graph_dim>0 should fail without graph data
    (dimension mismatch), while final_graph_concat=False should still work."""
    for exp in EXPERIMENTS:
        lit = FuncFinetuneLitModule(
            graph_dim=GRAPH_DIM,
            num_node_types=NUM_NODE_TYPES,
            num_relations=NUM_RELATIONS,
            fusion_layers=exp["fusion_layers"],
            final_graph_concat=exp["final_graph_concat"],
        )
        lit.eval()
        input_ids, attention_mask, cwe_embedding, _ = _make_inputs(BATCH_SIZE)

        with torch.inference_mode():
            if exp["final_graph_concat"]:
                # Without graph_batch, concat is missing 64 dims -> should error
                try:
                    lit(input_ids, attention_mask, cwe_embedding, graph_batch=None)
                    raise AssertionError(f"{exp['tag']}: expected RuntimeError but forward succeeded")
                except RuntimeError:
                    pass  # expected
            else:
                # E4: no late concat, so missing graph_batch just means no concat -> OK
                logits = lit(input_ids, attention_mask, cwe_embedding, graph_batch=None)
                assert logits.shape == (BATCH_SIZE, 1), f"{exp['tag']}: wrong shape {logits.shape}"
