## Why

The current graph fusion approach concatenates pooled graph embeddings **after** the UniXcoder encoder (late fusion). We don't know whether injecting graph structure **inside** the encoder layers — so the transformer can attend to graph-informed representations — improves detection, especially for short functions (0–64 tokens) where code context is sparse but graph structure is rich. A small, fixed ablation suite answers this before committing to a larger sweep.

## What Changes

- **New Colab notebook** (`notebooks/colab_fusion_ablation.ipynb`) that runs 5 fixed experiments (E0–E4) varying where graph embeddings are fused into UniXcoder, and whether late graph concat is kept.
- **Gated residual fusion module** — a small `nn.Module` that injects pooled `graph_emb` into selected UniXcoder transformer layers via a learned gate + residual connection. One fusion method only; no cross-attention or sweep framework.
- **Minimal config plumbing** — new parameters (`fusion_layers`, `final_graph_concat`) threaded through `FuncFinetuneModel` and `FuncFinetuneLitModule` to `train()`, with zero changes to existing default behavior.
- **Ablation summary helper** — a thin function that runs the existing prediction-export + post-analysis pipeline per experiment and collects length-bucketed metrics into one comparison DataFrame.
- Each experiment writes to its own output directory; a final notebook cell displays one compact comparison table (test_f1, unseen_test_f1, 0–64 / 64–128 / 128–256 / 256–512 bucket F1s, train time).

## Capabilities

### New Capabilities
- `gated-residual-fusion`: Gated residual fusion module that injects pooled graph embeddings into selected UniXcoder encoder layers
- `fusion-ablation-notebook`: Colab notebook that runs a fixed 5-experiment ablation suite comparing internal fusion layer placements and reports a compact comparison table with length-bucket metrics

### Modified Capabilities
- `model-architecture`: Add optional `fusion_layers` and `final_graph_concat` parameters to `FuncFinetuneModel` so it can apply gated residual fusion at specified layers while preserving current default behavior (no fusion)

## Impact

- **New files**: `notebooks/colab_fusion_ablation.ipynb`, `graph_vul/model/gated_residual_fusion.py`, small helper in `graph_vul/analysis/` or inline in notebook
- **Modified files**: `graph_vul/model/func_finetune_model.py` (new optional params, hook fusion into encoder forward), `graph_vul/model/func_finetune_lit_module.py` (pass-through new params), `graph_vul/train_finetune.py` (accept + forward new params, configurable output paths)
- **No breaking changes**: all new parameters default to current behavior; existing notebook, CLI, and DVC pipeline are unaffected
- **Dependencies**: no new packages required (uses existing PyTorch, transformers, torch-geometric)
