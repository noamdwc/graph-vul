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
        self.input_dim = in_dim

    def forward(self, text_vec: torch.Tensor) -> torch.Tensor:
        if text_vec.dim() == 1:
            text_vec = text_vec.reshape(-1, self.input_dim)
        elif text_vec.dim() > 2:
            text_vec = text_vec.reshape(text_vec.size(0), -1)
        text_vec = text_vec.to(dtype=self.proj.weight.dtype)
        return self.activation(self.proj(text_vec))
