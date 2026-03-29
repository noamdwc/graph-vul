"""2-layer MLP baseline classifier for function-level vulnerability detection.

Concatenates UniXcoder code embeddings with DeBERTa CWE description embeddings.
Optionally fuses a graph embedding from FunctionGraphEncoder when graph_dim > 0.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from graph_vul.config import (
    DEBERTA_EMBEDDING_DIM,
    FUNC_MLP_DROPOUT,
    FUNC_MLP_HIDDEN_DIM,
    GRAPH_DROPOUT,
    GRAPH_HIDDEN_DIM,
    GRAPH_NODE_EMBED_DIM,
    UNIXCODER_EMBEDDING_DIM,
)
from graph_vul.model.graph_encoder import FunctionGraphEncoder


class FuncBaselineModel(nn.Module):
    """Linear(code_dim + [graph_dim] + cwe_dim, hidden) -> ReLU -> Dropout -> Linear(hidden, 1)."""

    def __init__(
        self,
        code_dim: int = UNIXCODER_EMBEDDING_DIM,
        cwe_dim: int = DEBERTA_EMBEDDING_DIM,
        hidden_dim: int = FUNC_MLP_HIDDEN_DIM,
        dropout: float = FUNC_MLP_DROPOUT,
        graph_dim: int = 0,
        num_node_types: int = 0,
        num_relations: int = 0,
        graph_node_embed_dim: int = GRAPH_NODE_EMBED_DIM,
        graph_hidden_dim: int = GRAPH_HIDDEN_DIM,
        graph_dropout: float = GRAPH_DROPOUT,
    ) -> None:
        super().__init__()
        self.graph_dim = graph_dim

        if graph_dim > 0:
            self.graph_encoder = FunctionGraphEncoder(
                num_node_types=num_node_types,
                num_relations=num_relations,
                node_embed_dim=graph_node_embed_dim,
                hidden_dim=graph_hidden_dim,
                dropout=graph_dropout,
            )

        input_dim = code_dim + graph_dim + cwe_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        code_emb: torch.Tensor,
        cwe_emb: torch.Tensor,
        graph_batch: Batch | None = None,
    ) -> torch.Tensor:
        parts = [code_emb]

        if self.graph_dim > 0 and graph_batch is not None:
            graph_emb = self.graph_encoder(
                graph_batch.node_type_ids,
                graph_batch.edge_index,
                graph_batch.edge_type_ids,
                graph_batch.batch,
                num_graphs=graph_batch.num_graphs,
            )
            parts.append(graph_emb)

        parts.append(cwe_emb)
        x = torch.cat(parts, dim=-1)
        return self.mlp(x)  # [N, 1]
