"""UniXcoder fine-tuning model for function-level vulnerability detection.

Unfreezes the UniXcoder encoder and trains it end-to-end with a classification
head that fuses mean-pooled code representations with CWE description embeddings.
Optionally fuses a graph embedding from FunctionGraphEncoder when graph_dim > 0.

When ``fusion_layers`` is set, gated residual fusion injects the pooled graph
embedding into selected encoder layers via learned gates.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from transformers import AutoModel

from graph_vul.config import (
    CODE_MODEL_NAME,
    DEBERTA_EMBEDDING_DIM,
    FUNC_MLP_DROPOUT,
    FUNC_MLP_HIDDEN_DIM,
    GRAPH_DROPOUT,
    GRAPH_HIDDEN_DIM,
    GRAPH_NODE_EMBED_DIM,
    MATCHING_PROJECTION_DIM,
)
from graph_vul.model.gated_residual_fusion import FUSION_REGISTRY
from graph_vul.model.cwe_matching import MatchingConfig, build_matching_config
from graph_vul.model.graph_encoder import FunctionGraphEncoder


def _encoder_layers(encoder):
    """Return the list of transformer blocks regardless of backbone family.

    - AutoModel on Qwen3 returns ``Qwen3Model`` directly, which exposes
      ``.layers`` at the top level.
    - A CausalLM wrapper would expose them one level deeper at ``.model.layers``.
    - RoBERTa-style models such as UniXcoder expose them at ``encoder.layer``.

    The fusion-hook contract (tuple or tensor output) is the same for all.
    """
    if hasattr(encoder, "layers"):
        return encoder.layers
    if hasattr(encoder, "model") and hasattr(encoder.model, "layers"):
        return encoder.model.layers
    return encoder.encoder.layer


class FuncFinetuneModel(nn.Module):
    """UniXcoder encoder + mean pooling + CWE fusion + [optional graph] + MLP head."""

    def __init__(
        self,
        model_name: str = CODE_MODEL_NAME,
        cwe_dim: int = DEBERTA_EMBEDDING_DIM,
        hidden_dim: int = FUNC_MLP_HIDDEN_DIM,
        dropout: float | None = FUNC_MLP_DROPOUT,
        graph_dim: int = 0,
        num_node_types: int = 0,
        num_relations: int = 0,
        graph_node_embed_dim: int = GRAPH_NODE_EMBED_DIM,
        graph_hidden_dim: int = GRAPH_HIDDEN_DIM,
        graph_dropout: float = GRAPH_DROPOUT,
        fusion_layers: list[int] | None = None,
        fusion_method: str = "scalar_gated_residual",
        final_graph_concat: bool = True,
        matching_projection_dim: int = MATCHING_PROJECTION_DIM,
        matching_config: MatchingConfig | None = None,
    ) -> None:
        super().__init__()
        # Force fp32 on load: Qwen3 ships as bf16, but graph encoder, fusion
        # modules, and head are fp32; a mixed dtype at the fusion residual
        # breaks downstream Linear ops. Mixed precision at train time is
        # handled by Lightning's ``precision=bf16-mixed`` instead.
        self.encoder = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        encoder_dim = self.encoder.config.hidden_size
        self.graph_dim = graph_dim
        self.matching_config = matching_config or build_matching_config()
        if dropout is None:
            dropout = FUNC_MLP_DROPOUT
        self.final_graph_concat = final_graph_concat

        if graph_dim > 0:
            self.graph_encoder = FunctionGraphEncoder(
                num_node_types=num_node_types,
                num_relations=num_relations,
                node_embed_dim=graph_node_embed_dim,
                hidden_dim=graph_hidden_dim,
                dropout=graph_dropout,
            )

        # Internal fusion via gated residual on selected encoder layers
        self.fusion_layers = fusion_layers
        self._fusion_hooks: list[torch.utils.hooks.RemovableHook] = []
        self._current_graph_emb: torch.Tensor | None = None

        if fusion_layers and graph_dim > 0:
            FusionClass = FUSION_REGISTRY[fusion_method]
            self.fusion_modules = nn.ModuleDict(
                {str(i): FusionClass(graph_dim, encoder_dim) for i in fusion_layers}
            )
            self._register_fusion_hooks()

        # Late concat input dim
        late_graph_dim = graph_dim if (graph_dim > 0 and final_graph_concat) else 0
        input_dim = encoder_dim + late_graph_dim + cwe_dim
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.function_projection = nn.Linear(encoder_dim, matching_projection_dim)
        self.cwe_projection = nn.Linear(cwe_dim, matching_projection_dim)

    def _project_function_match(self, pooled: torch.Tensor) -> torch.Tensor:
        projection_dtype = self.function_projection.weight.dtype
        projected = self.function_projection(pooled.to(dtype=projection_dtype))
        return torch.nn.functional.normalize(projected, dim=-1)

    def _project_cwe_match(self, cwe_embedding: torch.Tensor) -> torch.Tensor:
        projection_dtype = self.cwe_projection.weight.dtype
        projected = self.cwe_projection(cwe_embedding.to(dtype=projection_dtype))
        return torch.nn.functional.normalize(projected, dim=-1)

    def _register_fusion_hooks(self) -> None:
        """Register forward hooks on selected encoder layers for fusion."""
        for handle in self._fusion_hooks:
            handle.remove()
        self._fusion_hooks.clear()

        # Backbone-agnostic layer access: Qwen3 uses encoder.model.layers,
        # RoBERTa-style models use encoder.encoder.layer.
        encoder_layers = _encoder_layers(self.encoder)
        for layer_idx in self.fusion_layers:
            fusion_mod = self.fusion_modules[str(layer_idx)]

            def make_hook(mod: nn.Module):
                def hook_fn(layer_module, input, output):
                    graph_emb = self._current_graph_emb
                    if graph_emb is None:
                        return output
                    # RoBERTa layer output is a tuple (hidden_state, ...) or just hidden_state
                    if isinstance(output, tuple):
                        hidden_state = output[0]
                        fused = mod(hidden_state, graph_emb)
                        return (fused,) + output[1:]
                    else:
                        return mod(output, graph_emb)
                return hook_fn

            handle = encoder_layers[layer_idx].register_forward_hook(make_hook(fusion_mod))
            self._fusion_hooks.append(handle)

    def _encode_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cwe_embedding: torch.Tensor,
        graph_batch: Batch | None = None,
    ) -> dict[str, torch.Tensor]:
        # Compute graph embedding early so fusion hooks can access it
        graph_emb: torch.Tensor | None = None
        if self.graph_dim > 0 and graph_batch is not None:
            graph_emb = self.graph_encoder(
                graph_batch.node_type_ids,
                graph_batch.edge_index,
                graph_batch.edge_type_ids,
                graph_batch.batch,
                num_graphs=graph_batch.num_graphs,
            )

        # Store for fusion hooks (zero vector if no graph data)
        if self.fusion_layers and self.graph_dim > 0:
            if graph_emb is not None:
                self._current_graph_emb = graph_emb
            else:
                self._current_graph_emb = torch.zeros(
                    input_ids.shape[0], self.graph_dim, device=input_ids.device
                )

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [B, seq_len, 768]

        # Clear hook state
        self._current_graph_emb = None

        # Last-non-pad-token pooling.
        # Qwen3 is a causal (left-to-right) LM, so only the final non-pad token
        # has attended to the entire function. Mean-over-non-pad would mix
        # causally-truncated representations and is not idiomatic for causal LMs.
        last_idx = attention_mask.sum(dim=1) - 1  # [B]
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]  # [B, hidden]

        parts = [pooled]

        if self.graph_dim > 0 and self.final_graph_concat and graph_emb is not None:
            parts.append(graph_emb)

        parts.append(cwe_embedding)
        fused = torch.cat(parts, dim=-1)
        logits = self.head(fused)
        function_match = self._project_function_match(pooled)
        cwe_match = self._project_cwe_match(cwe_embedding)
        return {
            "logits": logits,
            "pooled": pooled,
            "function_match": function_match,
            "cwe_match": cwe_match,
        }

    def project_cwe_embeddings(self, cwe_embedding: torch.Tensor) -> torch.Tensor:
        return self._project_cwe_match(cwe_embedding)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cwe_embedding: torch.Tensor,
        graph_batch: Batch | None = None,
        return_matching_outputs: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        outputs = self._encode_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cwe_embedding=cwe_embedding,
            graph_batch=graph_batch,
        )
        if return_matching_outputs:
            return outputs
        return outputs["logits"]  # [B, 1]
