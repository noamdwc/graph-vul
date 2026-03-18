"""Fusion MLP and top-level GraphVul model."""

import torch
import torch.nn as nn

from graph_vul.config import GAT_HIDDEN_DIM, MLP_DROPOUT, MLP_HIDDEN_DIM
from graph_vul.model.gat_encoder import GATEncoder
from graph_vul.model.text_encoder import TextProjection


class FusionMLP(nn.Module):
    def __init__(
        self,
        code_dim: int = GAT_HIDDEN_DIM,
        text_dim: int = GAT_HIDDEN_DIM,
        hidden_dim: int = MLP_HIDDEN_DIM,
        dropout: float = MLP_DROPOUT,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(code_dim + text_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, code_vec: torch.Tensor, text_vec: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([code_vec, text_vec], dim=-1)
        return self.mlp(fused)


class GraphVulModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gat_encoder = GATEncoder()
        self.text_projection = TextProjection()
        self.fusion = FusionMLP(
            code_dim=self.gat_encoder.output_dim,
            text_dim=self.text_projection.output_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch: torch.Tensor,
        text_vec: torch.Tensor,
    ) -> torch.Tensor:
        code_vec = self.gat_encoder(x, edge_index, edge_type, batch)
        text_proj = self.text_projection(text_vec)
        logits = self.fusion(code_vec, text_proj)
        return logits.squeeze(-1)
