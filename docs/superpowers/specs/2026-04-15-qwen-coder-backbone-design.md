# Swap Code Backbone: UniXcoder → Qwen3-0.6B

**Date:** 2026-04-15
**Status:** Approved for implementation
**Scope:** Minimum clean changes to replace the code backbone while preserving the current fusion design, training losses, and interfaces.

## Goal

Replace UniXcoder-base with Qwen3-0.6B as the code encoder. Preserve:

1. Graph embedding fused into the code model between attention residual blocks (gated residual fusion hooks).
2. CWE enters late — after the function-representation stage, concatenated before the head MLP.
3. Training still uses triplet + hard-negative mining with wrong CWEs for the same function.

Keep the current data flow and training logic as close as possible. Do not redesign. Add a comment at every architectural change.

## Non-goals

- Extending sequence length (stays at 512).
- Changing the baseline frozen-embedding stage (`graph_vul/pipeline/embed_functions.py`) — remains UniXcoder; it is orthogonal to fine-tuning.
- Supporting both backbones in the training path (single-backbone after the swap; old UniXcoder checkpoints cannot be loaded into the new model).
- Changing training loop, losses, matching head, graph encoder, hparam logging, export/eval stages.

## Changes

### 1. `graph_vul/config.py`

- Rename `UNIXCODER_MODEL_NAME` → `CODE_MODEL_NAME = "Qwen/Qwen3-0.6B"`.
- Rename `UNIXCODER_EMBEDDING_DIM` → `CODE_EMBEDDING_DIM = 1024`.
- Leave `max_length = 512`, `FUNC_MLP_*`, `MATCHING_*`, `GRAPH_*` untouched.
- Add a single anchor comment: `# Backbone: was UniXcoder-base (768d, 12L), now Qwen3-0.6B (1024d, 28L).`

Training-notebook change (separate file): in `notebooks/colab_finetune.ipynb` (and any active training notebook), update the `fusion_layers` default from `list(range(12))` → `list(range(28))`. The model accepts `fusion_layers` as a constructor arg; there is no config constant to change.

### 2. `graph_vul/model/func_finetune_model.py`

Three targeted edits; nothing else in this file changes.

1. **Encoder load (unchanged API).** `self.encoder = AutoModel.from_pretrained(model_name)`. `encoder_dim = self.encoder.config.hidden_size` already drives head and fusion sizing — 768 → 1024 propagates automatically.

2. **Layer-path abstraction.** Add a module-level helper:
   ```python
   def _encoder_layers(encoder):
       # Qwen3 (decoder-only, Llama-style) exposes layers at encoder.model.layers;
       # RoBERTa-style models expose them at encoder.encoder.layer. Detect once.
       if hasattr(encoder, "model") and hasattr(encoder.model, "layers"):
           return encoder.model.layers
       return encoder.encoder.layer
   ```
   Use it in `_register_fusion_hooks`. Llama-style decoder layers return a `(hidden_state, ...)` tuple from the forward hook — same contract as RoBERTa — so the existing hook body is unchanged.

3. **Pooling swap.** In `_encode_features`, replace mean-over-non-pad pooling with last-non-pad-token pooling:
   ```python
   # Qwen is causal: only the last non-pad token has attended to the full function.
   last_idx = attention_mask.sum(dim=1) - 1
   pooled = hidden[torch.arange(hidden.size(0)), last_idx]  # [B, hidden]
   ```

Matching-head projections, CWE late-concat, graph late-concat, triplet + wrong-CWE losses, and head MLP depend only on `pooled` and `encoder_dim` — all unchanged.

Fusion modules (`GatedResidualFusion`, `ScalarGatedResidualFusion`) are hidden-size-parameterised and need no edits.

### 3. `graph_vul/dataset/func_finetune_dataset.py`

- Default tokenizer name → `CODE_MODEL_NAME`.
- After loading the tokenizer, add:
  ```python
  # Qwen has no pad token by default; reuse eos for right-padding.
  # attention_mask distinguishes real vs pad so loss/pooling stay correct.
  if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token
  ```
- Keep right padding, `max_length = 512`, `truncation = True`, `padding = "max_length"`. Last-non-pad-token pooling derives its index from `attention_mask.sum(dim=1) - 1`, so right padding is correct.

### 4. Files intentionally untouched

- `graph_vul/model/gated_residual_fusion.py` — hidden-size-agnostic.
- `graph_vul/model/graph_encoder.py`, `graph_vul/model/cwe_matching.py`.
- `graph_vul/model/func_finetune_lit_module.py` — triplet loss, wrong-CWE loss, hard-negative mining preserved.
- `graph_vul/train_finetune.py` — CLI surface and hparam logging preserved.
- `graph_vul/export_finetune_predictions.py`, `graph_vul/eval_external.py`.
- `graph_vul/pipeline/embed_functions.py` — frozen baseline path stays on UniXcoder.

Every file above reads `encoder_dim` or `CODE_EMBEDDING_DIM` via the model or config, so the hidden-dim change propagates without code edits.

## Risks & tradeoffs

- **Checkpoint incompatibility.** Old UniXcoder checkpoints cannot be loaded into the new model (hidden size differs). `LightningModule.load_from_checkpoint` will error clearly.
- **Not apples-to-apples.** Swapping to a causal backbone with last-token pooling is a behaviour change, not just a model swap. Comparisons against UniXcoder runs should carry that caveat.
- **Tokenizer divergence.** Qwen's byte-level BPE yields different per-sample token counts; stratification analyses keyed on token length will shift — re-derive bucket edges before comparing.
- **W&B views.** `model_name` now logs `"Qwen/Qwen3-0.6B"`; existing filters need manual updates.

## Testing

- Smoke test for `FuncFinetuneModel`: construct with graph + matching enabled, run forward on a dummy batch (B=2, seq=16); assert shapes of `logits`, `pooled`, `function_match`, `cwe_match`.
- Fusion hook test: after one backward pass, assert a fusion-module parameter in `fusion_modules` has a non-None `.grad`.
- End-to-end smoke: one training step on a tiny subset, assert loss is finite.

## Acceptance criteria

- `dvc repro train` (or the equivalent fine-tune entrypoint) starts and completes one epoch on L4 without OOM.
- No reference to `UNIXCODER_MODEL_NAME` or `UNIXCODER_EMBEDDING_DIM` remains in `graph_vul/model/` or `graph_vul/dataset/`.
- Every architectural change in the three edited files carries a short comment explaining *why* (causal pooling, Qwen layer path, pad-token fallback).
- Training logs show `model_name = "Qwen/Qwen3-0.6B"` and fusion modules attached to 28 layers.
