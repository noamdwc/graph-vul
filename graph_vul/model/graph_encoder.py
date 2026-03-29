"""Lightweight graph encoder for function-level AST+CFG structure.

Maps node type IDs through an embedding table, applies 2-layer RGCN
with relation-aware convolution, and produces a fixed-size graph vector
via global mean pooling.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv, global_mean_pool

from graph_vul.config import GRAPH_DROPOUT, GRAPH_HIDDEN_DIM, GRAPH_NODE_EMBED_DIM


class FunctionGraphEncoder(nn.Module):
    """nn.Embedding → 2-layer RGCN → global_mean_pool → [B, hidden_dim]."""

    def __init__(
        self,
        num_node_types: int,
        num_relations: int,
        node_embed_dim: int = GRAPH_NODE_EMBED_DIM,
        hidden_dim: int = GRAPH_HIDDEN_DIM,
        num_layers: int = 2,
        dropout: float = GRAPH_DROPOUT,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_embedding = nn.Embedding(num_node_types, node_embed_dim)

        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(node_embed_dim, hidden_dim, num_relations))
        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_type_ids: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int | None = None,
    ) -> torch.Tensor:
        if num_graphs is None:
            num_graphs = batch.max().item() + 1 if batch.numel() > 0 else 1

        # All graphs in batch are empty — return zero vectors
        if node_type_ids.shape[0] == 0:
            return torch.zeros(num_graphs, self.hidden_dim, device=node_type_ids.device)

        x = self.node_embedding(node_type_ids)

        for conv in self.convs:
            x = conv(x, edge_index, edge_type_ids)
            x = torch.relu(x)
            x = self.dropout(x)

        # size=num_graphs ensures empty graphs get zero vectors
        return global_mean_pool(x, batch, size=num_graphs)
