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

class GATLayer(nn.Module):
    def __init__(self,
                in_dim: int,
                edge_embedding: nn.Embedding,
                activation: nn.Module = nn.LeakyReLU(),
                hidden_dim: int = GAT_HIDDEN_DIM,
                heads: int = GAT_HEADS,
                concat: bool = True):
        super().__init__()
        self.edge_embedding = edge_embedding
        self.conv = GATv2Conv(
            in_dim, hidden_dim, heads=heads, edge_dim=hidden_dim * heads, concat=concat
        )
        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        edge_attr = self.edge_embedding(edge_type)
        return self.activation(self.conv(x, edge_index, edge_attr=edge_attr))


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
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.activation = nn.LeakyReLU()
        self.layer_dim = hidden_dim * heads
        self.output_dim = hidden_dim * heads
        self.num_layers = num_layers
        self.num_edge_types = num_edge_types
        self.layers = nn.ModuleList([self._create_layer(is_first=True if i == 0 else False) for i in range(num_layers)])

    def _create_layer(self, is_first: bool = True):
        return GATLayer(in_dim=self.in_dim if is_first else self.hidden_dim * self.heads,
                        edge_embedding=self.edge_embedding,
                        activation=self.activation,
                        hidden_dim=self.hidden_dim,
                        heads=self.heads)
