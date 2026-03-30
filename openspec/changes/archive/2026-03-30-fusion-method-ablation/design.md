## Context

The current codebase has a single fusion mechanism (`GatedResidualFusion`) that injects pooled graph embeddings into UniXcoder transformer layers via forward hooks. The depth ablation proved all-layer injection is best. Now we need to compare *how* the injection is done.

The existing ablation notebook (`colab_fusion_ablation.ipynb`) provides a proven pattern: define experiments as dicts, loop over them, train each model, export predictions, compute per-bucket metrics, display comparison table. The new notebook follows the same pattern but varies `fusion_method` instead of `fusion_layers`.

## Goals / Non-Goals

**Goals:**
- Compare 7 fusion methods + late-only baseline under identical conditions
- Identify which method maximizes unseen short-function (0–64 token) F1
- Self-contained notebook — no changes to `graph_vul/` package needed to run
- Reuse existing training, export, and analysis infrastructure

**Non-Goals:**
- Hyperparameter tuning per method (all share the same training config)
- Combining multiple fusion methods in one model
- Modifying the main pipeline (that happens after we pick the winner)

## Decisions

### 1. Fusion modules defined inline in the notebook

**Choice:** Define all 7 fusion classes in a single notebook cell, register them in a `FUSION_REGISTRY` dict.

**Why:** Keeps the experiment self-contained. No package changes means no risk of breaking the main pipeline. The winning method gets promoted to `graph_vul/model/` later.

**Alternative considered:** Adding all modules to `graph_vul/model/fusion_methods.py` first. Rejected because it pollutes the package with experimental code before we know which method works.

### 2. Monkey-patch `FuncFinetuneModel` to accept fusion_method

**Choice:** After importing the model, patch `__init__` and `_register_fusion_hooks` to look up the fusion class from the registry instead of hardcoding `GatedResidualFusion`. The patch is minimal — only the module instantiation line changes.

**Why:** Avoids forking the entire model class into the notebook. The training loop, export pipeline, and analysis code all work unchanged.

**Alternative considered:** Subclassing `FuncFinetuneModel`. Rejected because the LitModule instantiates the model internally, so we'd also need to subclass the LitModule, leading to more code duplication.

### 3. Cross-attention (E7) uses learned graph tokens from attention pooling

**Choice:** The graph encoder produces node embeddings → an attention-pooling layer compresses them to K=8 fixed graph tokens → cross-attention in each layer attends from code tokens to these K graph tokens.

**Why:** Using all graph nodes would be variable-length and expensive. A small fixed set of graph tokens keeps the cross-attention cost constant and comparable to the other methods. 8 tokens captures more structure than the single pooled vector used by other methods.

**Implementation:** `CrossAttentionGraphTokensFusion` wraps `nn.MultiheadAttention(embed_dim=768, num_heads=8)`. Graph tokens are projected from graph_dim to 768. The module is called in the same forward-hook pattern as other methods, but the hook passes the K graph tokens as key/value.

**Alternative considered:** K=4 tokens. Will use K=8 as default but this is configurable in the experiment dict.

### 4. All methods use the same forward-hook injection pattern

**Choice:** Every fusion method implements the same interface: `forward(hidden_state, graph_emb) -> fused_hidden_state`. The hook machinery in `_register_fusion_hooks` is unchanged — only the module class instantiated per layer changes.

**Why:** Ensures a fair comparison. All methods see the same hidden states at the same layers with the same graph input.

**Exception:** E7 (cross-attention) needs K graph tokens instead of a single pooled vector. The hook for E7 stores `graph_tokens` (shape [B, K, graph_dim]) instead of `graph_emb` (shape [B, graph_dim]). The cross-attention module handles the different input shape internally.

### 5. Late concat kept for all methods except E0

**Choice:** `final_graph_concat=True` for E1–E7, `False` only for E0 which has no internal fusion either (pure late-only baseline uses concat).

Wait — actually E0 is the late-only baseline, so it needs `final_graph_concat=True`. All experiments use `final_graph_concat=True`. This matches the user's spec.

### 6. Graph token pooling for E7 uses a small wrapper around the existing graph encoder

**Choice:** Add a `GraphTokenPooling` module that takes the existing `FunctionGraphEncoder`'s per-node embeddings (before global mean pooling) and applies multi-head attention pooling to produce K learned query tokens. This requires accessing intermediate graph encoder outputs.

**Implementation:** Modify the forward hook to call the graph encoder in a mode that returns per-node embeddings, then apply the pooling. The pooled vector for late concat is still computed via mean pooling as usual.

## Risks / Trade-offs

- **E7 complexity** → E7 (cross-attention) is architecturally more complex than the other methods and may need more training time or be harder to optimize. Mitigation: same hyperparameters, same epochs — if it doesn't converge in 10 epochs, that's a valid result.
- **Monkey-patching fragility** → The patch assumes specific internal structure of `FuncFinetuneModel`. Mitigation: the patch is thin (only module instantiation), and we pin the git revision in the notebook.
- **Colab GPU memory** → Cross-attention adds more parameters per layer. Mitigation: K=8 tokens with 768-dim attention adds ~2.4M params per layer (28.8M total for 12 layers). Should fit in a T4/A100 with batch_size=64. If OOM, reduce batch size.
- **Runtime** → 8 experiments × ~30 min each = ~4 hours on A100. Mitigation: save checkpoints to Drive so interrupted runs can be resumed.
