"""Graph-level classifier with early CWE fusion and top-level GraphVul model."""

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from graph_vul.config import (
    GAT_HEADS,
    GAT_HIDDEN_DIM,
    MLP_DROPOUT,
    NODE_CLASSIFIER_HIDDEN_DIM,
    FUSION_LAYER,
)
from graph_vul.model.gat_encoder import GATEncoder
from graph_vul.model.text_encoder import TextProjection


class FusionProjection(nn.Module):
    """Projects concatenated [node_feats || text_broadcast] back to GAT layer 2 input dim."""

    def __init__(
        self,
        node_dim: int = GAT_HIDDEN_DIM * GAT_HEADS,
        text_dim: int = GAT_HIDDEN_DIM,
    ):
        super().__init__()
        self.proj = nn.Linear(node_dim + text_dim, node_dim)
        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.proj(x))


class NodeClassifier(nn.Module):
    """Per-node MLP classifier."""

    def __init__(
        self,
        in_dim: int = GAT_HIDDEN_DIM,
        hidden_dim: int = NODE_CLASSIFIER_HIDDEN_DIM,
        dropout: float = MLP_DROPOUT,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)


class GraphVulModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gat_encoder = GATEncoder()
        self.text_projection = TextProjection()
        self.fusion_proj = FusionProjection(
            node_dim=self.gat_encoder.layer_dim,
            text_dim=self.text_projection.output_dim,
        )
        self.node_classifier = NodeClassifier(
            in_dim=self.gat_encoder.output_dim,
        )
        self.fusion_layer: int = FUSION_LAYER

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch: torch.Tensor,
        text_vec: torch.Tensor,
    ) -> torch.Tensor:
        for i, layer in enumerate(self.gat_encoder.layers) : # i is the index of the layer
            x = layer(x, edge_index, edge_type)
            if i == self.fusion_layer - 1:
                text_proj = self.text_projection(text_vec)
                text_broadcast = text_proj[batch]
                x = torch.cat([x, text_broadcast], dim=-1)
                x = self.fusion_proj(x) # project back to layer_dim

        # Per-node logits, then max-pool to graph level
        node_logits = self.node_classifier(x)
        graph_logits = scatter(node_logits, batch, dim=0, reduce="max")
        return graph_logits
