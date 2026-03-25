"""2-layer MLP baseline classifier for function-level vulnerability detection."""

import torch
import torch.nn as nn

from graph_vul.config import FUNC_MLP_DROPOUT, FUNC_MLP_HIDDEN_DIM, UNIXCODER_EMBEDDING_DIM


class FuncBaselineModel(nn.Module):
    """Linear(768, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 1)."""

    def __init__(
        self,
        input_dim: int = UNIXCODER_EMBEDDING_DIM,
        hidden_dim: int = FUNC_MLP_HIDDEN_DIM,
        dropout: float = FUNC_MLP_DROPOUT,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)  # [N, 1]
