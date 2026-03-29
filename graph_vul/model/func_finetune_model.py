"""UniXcoder fine-tuning model for function-level vulnerability detection.

Unfreezes the UniXcoder encoder and trains it end-to-end with a classification
head that fuses mean-pooled code representations with CWE description embeddings.
Optionally fuses a graph embedding from FunctionGraphEncoder when graph_dim > 0.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from transformers import AutoModel

from graph_vul.config import (
    DEBERTA_EMBEDDING_DIM,
    FUNC_MLP_DROPOUT,
    FUNC_MLP_HIDDEN_DIM,
    GRAPH_DROPOUT,
    GRAPH_HIDDEN_DIM,
    GRAPH_NODE_EMBED_DIM,
    UNIXCODER_MODEL_NAME,
)
from graph_vul.model.graph_encoder import FunctionGraphEncoder


class FuncFinetuneModel(nn.Module):
    """UniXcoder encoder + mean pooling + CWE fusion + [optional graph] + MLP head."""

    def __init__(
        self,
        model_name: str = UNIXCODER_MODEL_NAME,
        cwe_dim: int = DEBERTA_EMBEDDING_DIM,
        hidden_dim: int = FUNC_MLP_HIDDEN_DIM,
        dropout: float = FUNC_MLP_DROPOUT,
        graph_dim: int = 0,
        num_node_types: int = 0,
        num_relations: int = 0,
        graph_node_embed_dim: int = GRAPH_NODE_EMBED_DIM,
        graph_hidden_dim: int = GRAPH_HIDDEN_DIM,
        graph_dropout: float = GRAPH_DROPOUT,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        encoder_dim = self.encoder.config.hidden_size  # 768
        self.graph_dim = graph_dim

        if graph_dim > 0:
            self.graph_encoder = FunctionGraphEncoder(
                num_node_types=num_node_types,
                num_relations=num_relations,
                node_embed_dim=graph_node_embed_dim,
                hidden_dim=graph_hidden_dim,
                dropout=graph_dropout,
            )

        input_dim = encoder_dim + graph_dim + cwe_dim
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cwe_embedding: torch.Tensor,
        graph_batch: Batch | None = None,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [B, seq_len, 768]
        # Mean pool over non-padding tokens
        mask = attention_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # [B, 768]

        parts = [pooled]

        if self.graph_dim > 0 and graph_batch is not None:
            graph_emb = self.graph_encoder(
                graph_batch.node_type_ids,
                graph_batch.edge_index,
                graph_batch.edge_type_ids,
                graph_batch.batch,
                num_graphs=graph_batch.num_graphs,
            )
            parts.append(graph_emb)

        parts.append(cwe_embedding)
        fused = torch.cat(parts, dim=-1)
        return self.head(fused)  # [B, 1]
