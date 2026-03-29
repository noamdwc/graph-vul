"""Gated residual fusion for injecting graph embeddings into transformer layers."""

import torch
import torch.nn as nn


class GatedResidualFusion(nn.Module):
    """Fuse a pooled graph embedding into a transformer hidden state.

    output = hidden_state + gate(hidden_state) * proj(graph_emb).unsqueeze(1)

    The gate is a per-token scalar sigmoid, initialized with bias=-2.0
    so fusion starts weak (~0.12) and must be learned.
    """

    def __init__(self, graph_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(graph_dim, hidden_size)
        self.gate = nn.Linear(hidden_size, 1)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(
        self,
        hidden_state: torch.Tensor,
        graph_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Apply gated residual fusion.

        Args:
            hidden_state: [B, seq_len, hidden_size] from transformer layer.
            graph_emb: [B, graph_dim] pooled graph embedding.

        Returns:
            Fused hidden state of same shape as input.
        """
        projected = self.proj(graph_emb).unsqueeze(1)  # [B, 1, hidden_size]
        gate_val = torch.sigmoid(self.gate(hidden_state))  # [B, seq_len, 1]
        return hidden_state + gate_val * projected
