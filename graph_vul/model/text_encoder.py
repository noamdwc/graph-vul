"""Linear projection for pre-computed DeBERTa text vectors."""

import torch
import torch.nn as nn

from graph_vul.config import DEBERTA_EMBEDDING_DIM, GAT_HIDDEN_DIM


class TextProjection(nn.Module):
    def __init__(
        self,
        in_dim: int = DEBERTA_EMBEDDING_DIM,
        out_dim: int = GAT_HIDDEN_DIM,
    ):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.activation = nn.LeakyReLU()
        self.output_dim = out_dim

    def forward(self, text_vec: torch.Tensor) -> torch.Tensor:
        return self.activation(self.proj(text_vec))
