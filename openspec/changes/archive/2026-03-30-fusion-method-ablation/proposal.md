## Why

The depth ablation (E0–E4) showed that all-layer graph fusion dramatically improves unseen short-function F1 (0.647 vs 0.412). But only one fusion mechanism was tested — gated residual. Different injection methods (FiLM, cross-attention, concat adapters) may extract more from the graph signal, especially for the 0–64 token bucket on unseen CWEs where the model struggles most.

## What Changes

- Create a new Colab ablation notebook (`notebooks/colab_fusion_method_ablation.ipynb`) that:
  - Defines 6 new fusion module classes inline (no changes to `graph_vul/` package)
  - Monkey-patches `FuncFinetuneModel` to swap the fusion module per experiment
  - Runs 8 experiments (E0–E7) with identical data/splits/seed/hyperparameters/GNN/head
  - Exports predictions and computes per-bucket metrics using existing analysis utilities
  - Displays a final comparison table ranked by unseen 0–64 F1
- All fusion modules defined in the notebook — the main codebase is untouched until we decide which method wins

## Capabilities

### New Capabilities
- `fusion-method-ablation-notebook`: Self-contained Colab notebook running 8 fusion-method experiments (late-only, gated residual, ungated residual, scalar-gated residual, FiLM, concat adapter, gated concat adapter, cross-attention graph tokens) with comparison table and decision rule

### Modified Capabilities

_(none — notebook is self-contained, main pipeline unchanged)_

## Impact

- **New files**: `notebooks/colab_fusion_method_ablation.ipynb`
- **Modified files**: none
- **Dependencies**: no new packages (cross-attention uses `nn.MultiheadAttention` from PyTorch; all others use standard `nn.Linear`/`nn.LayerNorm`)
- **Runtime**: ~8x single-experiment training time on Colab GPU
