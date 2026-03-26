"""UniXcoder fine-tuning model for function-level vulnerability detection.

Unfreezes the UniXcoder encoder and trains it end-to-end with a classification
head that fuses mean-pooled code representations with CWE description embeddings.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

from graph_vul.config import (
    DEBERTA_EMBEDDING_DIM,
    FUNC_MLP_DROPOUT,
    FUNC_MLP_HIDDEN_DIM,
    UNIXCODER_EMBEDDING_DIM,
    UNIXCODER_MODEL_NAME,
)


class FuncFinetuneModel(nn.Module):
    """UniXcoder encoder + mean pooling + CWE fusion + MLP head."""

    def __init__(
        self,
        model_name: str = UNIXCODER_MODEL_NAME,
        cwe_dim: int = DEBERTA_EMBEDDING_DIM,
        hidden_dim: int = FUNC_MLP_HIDDEN_DIM,
        dropout: float = FUNC_MLP_DROPOUT,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        encoder_dim = self.encoder.config.hidden_size  # 768

        self.head = nn.Sequential(
            nn.Linear(encoder_dim + cwe_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cwe_embedding: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [B, seq_len, 768]
        # Mean pool over non-padding tokens
        mask = attention_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # [B, 768]

        fused = torch.cat([pooled, cwe_embedding], dim=-1)  # [B, 768 + cwe_dim]
        return self.head(fused)  # [B, 1]
