"""GAT encoder for Code Property Graphs."""

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool

from graph_vul.config import (
    DEBERTA_EMBEDDING_DIM,
    GAT_HEADS,
    GAT_HIDDEN_DIM,
    GAT_LAYERS,
    NUM_EDGE_TYPES,
)


class GATEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = DEBERTA_EMBEDDING_DIM,
        hidden_dim: int = GAT_HIDDEN_DIM,
        heads: int = GAT_HEADS,
        num_layers: int = GAT_LAYERS,
        num_edge_types: int = NUM_EDGE_TYPES,
    ):
        super().__init__()
        self.edge_embedding = nn.Embedding(num_edge_types, hidden_dim * heads)

        self.convs = nn.ModuleList()
        # First layer: in_dim -> hidden_dim (per head)
        self.convs.append(
            GATv2Conv(in_dim, hidden_dim, heads=heads, edge_dim=hidden_dim * heads, concat=True)
        )
        # Subsequent layers: hidden_dim * heads -> hidden_dim (per head)
        for _ in range(num_layers - 1):
            self.convs.append(
                GATv2Conv(
                    hidden_dim * heads,
                    hidden_dim,
                    heads=heads,
                    edge_dim=hidden_dim * heads,
                    concat=False,
                )
            )

        self.activation = nn.LeakyReLU()
        self.output_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        edge_attr = self.edge_embedding(edge_type)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = self.activation(x)

        # Global mean pooling: one vector per graph
        return global_mean_pool(x, batch)
