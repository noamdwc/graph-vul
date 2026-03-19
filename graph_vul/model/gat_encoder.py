"""GAT encoder for Code Property Graphs."""

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

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

        # First layer: in_dim -> hidden_dim (per head), concat=True → output dim = hidden_dim * heads
        self.conv1 = GATv2Conv(
            in_dim, hidden_dim, heads=heads, edge_dim=hidden_dim * heads, concat=True
        )
        # Second layer: hidden_dim * heads -> hidden_dim, concat=False → output dim = hidden_dim
        self.conv2 = GATv2Conv(
            hidden_dim * heads,
            hidden_dim,
            heads=heads,
            edge_dim=hidden_dim * heads,
            concat=False,
        )

        self.activation = nn.LeakyReLU()
        self.layer1_dim = hidden_dim * heads
        self.output_dim = hidden_dim

    def forward_layer1(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        edge_attr = self.edge_embedding(edge_type)
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        return self.activation(x)

    def forward_layer2(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        edge_attr = self.edge_embedding(edge_type)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        return self.activation(x)
