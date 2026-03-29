## Context

The current `FuncFinetuneModel` uses late fusion: UniXcoder encodes code into a 768-dim pooled vector, then graph_emb (64-dim from RGCN) and cwe_embedding (768-dim) are concatenated before the MLP head. The graph signal never influences transformer attention — it arrives too late to help the encoder focus on structurally important tokens.

Short functions (0–64 tokens) have minimal textual context but often retain meaningful AST+CFG structure. Injecting graph information inside the encoder could help the transformer leverage structural cues during self-attention, potentially improving short-function detection without hurting longer functions.

The existing training pipeline (`train_finetune.py`), Lightning module, dataset, and post-analysis code are stable and well-tested. The ablation notebook must reuse them with minimal additions.

## Goals / Non-Goals

**Goals:**
- Determine whether gated residual fusion of graph_emb inside UniXcoder layers improves vulnerability detection, especially for 0–64 token functions
- Compare layer placements: none, last-2, last-4, all-12, and last-4-without-late-concat
- Deliver a single Colab notebook that runs all 5 experiments and produces one comparison table
- Keep all changes backward-compatible; existing default behavior unchanged

**Non-Goals:**
- Cross-attention fusion, FiLM conditioning, or other fusion methods (future work)
- Hyperparameter sweeps over fusion gate init, learning rate, etc.
- Changes to the graph encoder architecture (RGCN stays as-is)
- Changes to the dataset, splits, or evaluation methodology
- A general sweep framework or experiment orchestrator

## Decisions

### D1: Gated residual fusion mechanism

**Choice:** A small `GatedResidualFusion` module that projects pooled graph_emb to match hidden_size (768), applies a learned sigmoid gate, and adds the result as a residual to the transformer hidden state after the selected layer's output.

```
h_out = h + gate(h) * proj(graph_emb_expanded)
```

Where `gate` is `sigmoid(linear(h))` (scalar per token) and `proj` is a `Linear(graph_dim, hidden_size)`.

**Why over alternatives:**
- Simpler than cross-attention (no extra KV heads, no quadratic cost)
- Learned gate lets the model ignore graph signal when unhelpful (gate → 0)
- Residual connection preserves pretrained representations
- One module per fused layer, sharing the same projected graph_emb

**Alternative considered:** Additive bias (just add projected graph_emb). Rejected because it has no mechanism to attenuate the signal — could hurt layers where graph info is irrelevant.

### D2: Hook into UniXcoder via encoder output_hidden_states

**Choice:** Run the encoder with `output_hidden_states=True`, then apply fusion modules to selected layer outputs, and recompute the final hidden state by running remaining layers (or directly use the fused final hidden state for mean-pooling).

**Revised approach:** Actually, modifying the forward pass of HuggingFace RoBERTa internals is fragile. Instead, wrap the encoder and register forward hooks on the selected `encoder.layer[i]` modules. The hook applies `GatedResidualFusion` in-place on the layer output. This is cleaner and doesn't require copying/modifying transformers code.

**Why hooks:** Non-invasive, works with any RoBERTa-based model, easily toggled on/off per experiment.

### D3: Config plumbing — minimal new parameters

**Choice:** Add to `FuncFinetuneModel.__init__`:
- `fusion_layers: list[int] | None = None` — layer indices where fusion is applied (e.g., `[10, 11]` for last-2). `None` = no internal fusion (current behavior).
- `final_graph_concat: bool = True` — whether to include graph_emb in the late concat before MLP head. `True` = current behavior.

These flow through `FuncFinetuneLitModule` (pass-through) and `train()` (new kwargs).

**Why not a FusionConfig dataclass:** Only 2 fields. A dataclass adds indirection for no benefit at this scale.

### D4: Notebook structure mirrors colab_finetune.ipynb

**Choice:** Same cell sequence: GPU check → Drive mount → directory creation → data validation → graph validation → HF cache → clone + install → symlink → WandB → config cell → experiment loop → summary table.

The config cell defines all 5 experiments as a list of dicts. The runner cell loops over them, calling `train()` with per-experiment config, then running prediction export + post-analysis to extract length-bucket metrics. Results accumulate in a list of dicts and render as a DataFrame at the end.

### D5: Per-experiment output isolation

**Choice:** Each experiment writes to `{DRIVE_DATA}/fusion_ablation/{run_tag}/`:
- `func_finetune_model.ckpt`
- `func_finetune_metrics.json`
- `predictions/` (from export)

The `train()` function needs configurable `checkpoint_path` and `metrics_path`. Rather than modifying `train()` signature extensively, the notebook will temporarily patch `graph_vul.train_finetune.CHECKPOINT_PATH` and `METRICS_PATH` module-level variables before each call.

### D6: Length-bucket metric extraction

**Choice:** After each experiment's training completes, the notebook calls `export_split_predictions()` for test and unseen_test splits, then uses `stratified_metrics()` from `post_experiment_analysis.py` to compute per-bucket F1. This reuses the existing analysis pipeline entirely.

A small inline helper in the notebook collects the specific metrics (test_f1, unseen_test_f1, bucket F1s, precision, recall for 0–64) into one summary row.

## Risks / Trade-offs

- **[Forward hooks on HF internals]** → Hooks depend on `model.encoder.encoder.layer[i]` structure. Mitigation: assert layer count at init; this is standard RoBERTa and won't change for unixcoder-base.
- **[5 experiments × ~1h20m = ~6.5h total runtime]** → Long Colab session. Mitigation: each experiment saves independently, so partial runs are recoverable. Notebook documents this.
- **[Module-level path patching for output isolation]** → Slightly hacky. Mitigation: restore original paths after each run; clearly documented in notebook. Keeps `train()` signature changes minimal.
- **[Gate initialization]** → If gate bias starts too high, fusion signal could dominate early training and destabilize. Mitigation: initialize gate bias to -2.0 (sigmoid ≈ 0.12) so fusion starts weak and must be learned.
