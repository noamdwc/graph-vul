## Context

The main pipeline hardcodes `GatedResidualFusion` in `FuncFinetuneModel.__init__` (line 69 of `func_finetune_model.py`). The fusion method ablation notebook used monkey-patching with a `FUSION_REGISTRY` dict to swap in alternative fusion modules. We now need to bring E3's `ScalarGatedResidualFusion` into the pipeline properly and make the fusion method configurable.

Current flow: `train_finetune.py` → `FuncFinetuneLitModule(fusion_layers=...)` → `FuncFinetuneModel(fusion_layers=...)` → hardcoded `GatedResidualFusion`.

## Goals / Non-Goals

**Goals:**
- Make fusion method configurable via a `fusion_method: str` parameter
- Add `ScalarGatedResidualFusion` to the codebase
- Default to `"scalar_gated_residual"` in the training pipeline
- Preserve backward compatibility with existing `"gated_residual"` checkpoints

**Non-Goals:**
- Porting all 7 fusion methods from the notebook (only scalar_gated_residual needed now; others can be added later)
- Changing the fusion hook mechanism (hooks remain the same, only the module class changes)
- Re-training or migrating existing checkpoints

## Decisions

### 1. Fusion registry as a module-level dict in `gated_residual_fusion.py`

Add `ScalarGatedResidualFusion` to the same file as `GatedResidualFusion` and expose a `FUSION_REGISTRY: dict[str, type[nn.Module]]` mapping string keys to classes.

**Rationale:** Both classes are small (~15 lines), share the same interface `(graph_dim, hidden_size) → Module`, and belong to the same conceptual unit. A separate file or factory class would be over-engineering for 2 classes. The registry dict is the same pattern the notebook used successfully.

**Alternative considered:** Separate file per fusion class — rejected because they're trivially small and share the same `__init__` signature.

### 2. `fusion_method` parameter on `FuncFinetuneModel`

Add `fusion_method: str = "scalar_gated_residual"` alongside the existing `fusion_layers` parameter. In `__init__`, look up the class from `FUSION_REGISTRY` and use it instead of hardcoded `GatedResidualFusion`.

The change is minimal: line 69 changes from `GatedResidualFusion(graph_dim, encoder_dim)` to `FUSION_REGISTRY[fusion_method](graph_dim, encoder_dim)`.

### 3. Thread through LitModule and train_finetune.py

`FuncFinetuneLitModule.__init__` and `train_finetune.train()` each get a `fusion_method` parameter that passes through to the model. Default: `"scalar_gated_residual"`.

## Risks / Trade-offs

- **[Checkpoint incompatibility]** → State dict keys include `fusion_modules.{layer_idx}.{param}`. Both `GatedResidualFusion` and `ScalarGatedResidualFusion` have a `proj` linear layer, but the gate differs (`gate` Linear vs `alpha` Parameter). Loading old checkpoints with the new default will fail. → **Mitigation:** Pass `fusion_method="gated_residual"` when loading old checkpoints. Document this in the parameter docstring.

- **[Rename of module file]** → The file is called `gated_residual_fusion.py` but now contains multiple fusion classes. → **Mitigation:** Acceptable for now; rename to `fusion.py` would touch many imports for marginal benefit.
