## 1. Notebook scaffold

- [x] 1.1 Create `notebooks/colab_fusion_method_ablation.ipynb` with boilerplate cells (GPU check, Drive mount, repo clone, pip install, symlink, wandb login) copied from `colab_fusion_ablation.ipynb`

## 2. Fusion module implementations

- [x] 2.1 Implement `GatedResidualFusion` (re-export from existing module for registry consistency)
- [x] 2.2 Implement `UngatedResidualFusion`: `hidden + proj(graph_emb).unsqueeze(1)`
- [x] 2.3 Implement `ScalarGatedResidualFusion`: `hidden + sigmoid(alpha) * proj(graph_emb).unsqueeze(1)`, alpha scalar init -2.0
- [x] 2.4 Implement `FiLMFusion`: `gamma(g) * hidden + beta(g)`, gamma bias=1.0, beta bias=0.0
- [x] 2.5 Implement `ConcatAdapterFusion`: concat + bottleneck + LayerNorm + residual
- [x] 2.6 Implement `GatedConcatAdapterFusion`: gated variant of concat adapter, gate bias=-2.0
- [x] 2.7 Implement `CrossAttentionGraphTokensFusion`: attention pooling to K=8 graph tokens + `nn.MultiheadAttention` cross-attention + residual
- [x] 2.8 Build `FUSION_REGISTRY` dict mapping method names to classes

## 3. Model patching

- [x] 3.1 Write monkey-patch cell that modifies `FuncFinetuneModel.__init__` to accept `fusion_method` string and look up fusion class from `FUSION_REGISTRY`
- [x] 3.2 Handle E7 special case: graph token pooling wrapper around `FunctionGraphEncoder` and hook that passes graph tokens instead of pooled vector
- [x] 3.3 Verify patch preserves existing behavior when `fusion_method="gated_residual"`

## 4. Experiment configuration

- [x] 4.1 Define shared hyperparameters cell (epochs=10, batch_size=64, grad_accum=8, encoder_lr=2e-5, head_lr=1e-3, seed=42)
- [x] 4.2 Define `EXPERIMENTS` list with 8 experiment dicts (E0–E7), each with run_tag, fusion_method, fusion_layers, final_graph_concat

## 5. Training loop

- [x] 5.1 Implement experiment loop with skip-if-checkpoint-exists logic
- [x] 5.2 Train each experiment, export test + unseen_test predictions
- [x] 5.3 Compute per-bucket metrics using existing analysis utilities

## 6. Results display

- [x] 6.1 Build comparison DataFrame with columns: run_tag, unseen_test_f1, unseen_0_64_f1, unseen_64_128_f1, test_f1, 0_64_f1, train_minutes
- [x] 6.2 Sort by unseen_0_64_f1 descending, highlight best per column
- [x] 6.3 Add unseen_test bucket detail table (all buckets, precision/recall for 0_64)
