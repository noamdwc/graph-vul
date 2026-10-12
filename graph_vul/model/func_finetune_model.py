"""UniXcoder function encoding, optional AST RGCN, and CWE pair scoring."""
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

from graph_vul.config import (
    CODE_MODEL_NAME,
    DEBERTA_EMBEDDING_DIM,
    FUNC_MLP_HIDDEN_DIM,
    FUNC_MLP_DROPOUT,
)
from graph_vul.model.graph_encoder import FunctionGraphEncoder


class FuncFinetuneModel(nn.Module):
    def __init__(
        self,
        cwe_dim=DEBERTA_EMBEDDING_DIM,
        hidden_dim=FUNC_MLP_HIDDEN_DIM,
        dropout=FUNC_MLP_DROPOUT,
        graph_dim=0,
        num_node_types=0,
        num_relations=0,
        matching_projection_dim=128,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(CODE_MODEL_NAME)
        self.graph_dim = graph_dim
        encoder_dim = self.encoder.config.hidden_size
        if graph_dim:
            self.graph_encoder = FunctionGraphEncoder(
                num_node_types=num_node_types,
                num_relations=num_relations,
                hidden_dim=graph_dim,
            )
        feature_dim = encoder_dim + graph_dim
        self.head = nn.Sequential(
            nn.Linear(feature_dim + cwe_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.function_projection = nn.Linear(feature_dim, matching_projection_dim)
        self.cwe_projection = nn.Linear(cwe_dim, matching_projection_dim)
        # Learned benign candidate: explicit parameter, optimized and saved in state_dict.
        self.none_embedding = nn.Parameter(torch.empty(cwe_dim))
        nn.init.normal_(self.none_embedding, std=0.02)

    def encode_functions(self, input_ids, attention_mask, graph_batch=None):
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        if self.graph_dim:
            if graph_batch is None:
                raise ValueError("Graph-enabled inference requires AST graphs")
            graph = self.graph_encoder(
                graph_batch.node_type_ids,
                graph_batch.edge_index,
                graph_batch.edge_type_ids,
                graph_batch.batch,
                num_graphs=graph_batch.num_graphs,
            )
            pooled = torch.cat([pooled, graph], dim=-1)
        return pooled

    def project_cwe_embeddings(self, embeddings):
        return F.normalize(
            self.cwe_projection(embeddings.to(self.cwe_projection.weight.dtype)), dim=-1
        )

    def score_candidates(self, features, candidates, chunk_size=64):
        """Score the full catalog from cached features; never rerun either encoder."""
        parts = []
        for chunk in candidates.split(chunk_size):
            code = features[:, None, :].expand(-1, len(chunk), -1)
            cwe = chunk[None, :, :].expand(len(features), -1, -1)
            pairs = torch.cat([code, cwe], dim=-1).to(self.head[0].weight.dtype)
            parts.append(self.head(pairs).squeeze(-1))
        return torch.cat(parts, dim=1)
