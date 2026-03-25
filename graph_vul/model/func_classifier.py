"""2-layer MLP baseline classifier for function-level vulnerability detection.

Concatenates UniXcoder code embeddings with DeBERTa CWE description embeddings.
"""

import torch
import torch.nn as nn

from graph_vul.config import DEBERTA_EMBEDDING_DIM, FUNC_MLP_DROPOUT, FUNC_MLP_HIDDEN_DIM, UNIXCODER_EMBEDDING_DIM


class FuncBaselineModel(nn.Module):
    """Linear(code_dim + cwe_dim, hidden) -> ReLU -> Dropout -> Linear(hidden, 1)."""

    def __init__(
        self,
        code_dim: int = UNIXCODER_EMBEDDING_DIM,
        cwe_dim: int = DEBERTA_EMBEDDING_DIM,
        hidden_dim: int = FUNC_MLP_HIDDEN_DIM,
        dropout: float = FUNC_MLP_DROPOUT,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(code_dim + cwe_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, code_emb: torch.Tensor, cwe_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([code_emb, cwe_emb], dim=-1)
        return self.mlp(x)  # [N, 1]
