## Why

The fusion method ablation study (E0–E7) tested different mechanisms for injecting graph embeddings into UniXcoder. The main pipeline currently hardcodes `GatedResidualFusion` as the only fusion module. To use E3's scalar gated residual fusion — or any other method — the pipeline needs a configurable fusion method parameter backed by a registry of implementations.

## What Changes

- Add `ScalarGatedResidualFusion` module to `graph_vul/model/` (currently only exists in the notebook)
- Introduce a `fusion_method` parameter and fusion registry so `FuncFinetuneModel` can instantiate any registered fusion module
- Default `fusion_method` to `"scalar_gated_residual"` (E3) in the training pipeline
- Thread the `fusion_method` parameter through `FuncFinetuneLitModule` and `train_finetune.py`

## Capabilities

### New Capabilities
- `configurable-fusion-method`: A fusion method registry and `fusion_method` string parameter on `FuncFinetuneModel` that selects which fusion module to instantiate (replacing the hardcoded `GatedResidualFusion`). Includes the `ScalarGatedResidualFusion` implementation.

### Modified Capabilities
- `gated-residual-fusion`: The fusion hooks requirement is unchanged, but the module instantiated per layer is now determined by `fusion_method` rather than hardcoded to `GatedResidualFusion`. The existing `GatedResidualFusion` remains available as `"gated_residual"` in the registry.

## Impact

- **Code:** `graph_vul/model/func_finetune_model.py` (add `fusion_method` param + registry lookup), `graph_vul/model/gated_residual_fusion.py` (add `ScalarGatedResidualFusion` class), `graph_vul/model/func_finetune_lit_module.py` (thread param), `graph_vul/train_finetune.py` (thread param, change default)
- **Backward compatibility:** Existing checkpoints trained with `GatedResidualFusion` can still be loaded by passing `fusion_method="gated_residual"`. No breaking changes.
- **Dependencies:** None — uses only `torch.nn` primitives already in the project.
