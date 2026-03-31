## 1. Fusion Module and Registry

- [x] 1.1 Add `ScalarGatedResidualFusion` class to `graph_vul/model/gated_residual_fusion.py`
- [x] 1.2 Add `FUSION_REGISTRY` dict mapping `"gated_residual"` and `"scalar_gated_residual"` to their classes

## 2. Model Integration

- [x] 2.1 Add `fusion_method: str = "scalar_gated_residual"` parameter to `FuncFinetuneModel.__init__` and use `FUSION_REGISTRY[fusion_method]` instead of hardcoded `GatedResidualFusion`
- [x] 2.2 Add `fusion_method` parameter to `FuncFinetuneLitModule.__init__` and pass through to model
- [x] 2.3 Add `fusion_method` parameter to `train_finetune.train()` and pass through to LitModule

## 3. Verification

- [x] 3.1 Run `python -c "from graph_vul.model.gated_residual_fusion import FUSION_REGISTRY; print(FUSION_REGISTRY)"` to verify registry
- [x] 3.2 Run existing tests to confirm no regressions
