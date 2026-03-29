## 1. Gated Residual Fusion Module

- [x] 1.1 Create `graph_vul/model/gated_residual_fusion.py` with `GatedResidualFusion(nn.Module)` — accepts `graph_dim` and `hidden_size`, projects graph_emb, computes learned sigmoid gate on hidden state, returns `hidden_state + gate * proj(graph_emb).unsqueeze(1)`. Initialize gate bias to -2.0.

## 2. Model Architecture Changes

- [x] 2.1 Add `fusion_layers: list[int] | None = None` and `final_graph_concat: bool = True` parameters to `FuncFinetuneModel.__init__`. When `fusion_layers` is set and `graph_dim > 0`, create one `GatedResidualFusion` per layer and register forward hooks on `self.encoder.encoder.layer[i]`.
- [x] 2.2 Update `FuncFinetuneModel.forward()`: before calling the encoder, store `graph_emb` on the instance for hook access. Compute `graph_emb` early (from `graph_batch`) so hooks can use it. When `final_graph_concat=False`, exclude `graph_emb` from the late concat parts list. Adjust `input_dim` for the MLP head accordingly.
- [x] 2.3 Update `FuncFinetuneLitModule.__init__` to accept and forward `fusion_layers` and `final_graph_concat` to `FuncFinetuneModel`. Ensure both are captured by `save_hyperparameters()`.

## 3. Train Function Changes

- [x] 3.1 Add `fusion_layers`, `final_graph_concat`, `checkpoint_path`, and `metrics_path` optional parameters to `train()` in `train_finetune.py`. Forward `fusion_layers` and `final_graph_concat` to `FuncFinetuneLitModule`. Use `checkpoint_path` and `metrics_path` instead of module-level constants for file I/O. Default values preserve current behavior.
- [x] 3.2 Include `fusion_layers`, `final_graph_concat`, and `run_tag` in the WandB config dict logged at training start.

## 4. Colab Notebook

- [x] 4.1 Create `notebooks/colab_fusion_ablation.ipynb` with setup cells (GPU check, Drive mount, directory creation, data validation, graph validation, HF cache, repo clone + deps, symlink, WandB) matching `colab_finetune.ipynb`.
- [x] 4.2 Add configuration cell defining the 5 experiments as a list of dicts: `[{run_tag, fusion_layers, final_graph_concat}, ...]` for E0–E4, plus shared hyperparams (NUM_EPOCHS, BATCH_SIZE, ENCODER_LR, HEAD_LR, GRAD_ACCUM, SEED).
- [x] 4.3 Add main runner cell that loops over experiments: creates per-experiment output dir, calls `train()` with experiment-specific config, records wall-clock time, then calls `export_split_predictions()` for test/unseen_test and `stratified_metrics()` to extract length-bucket metrics. Appends one summary row per experiment.
- [x] 4.4 Add final summary cell that constructs a pandas DataFrame from collected rows (columns: run_tag, test_f1, unseen_test_f1, 0_64_f1, 0_64_precision, 0_64_recall, 64_128_f1, 128_256_f1, 256_512_f1, train_minutes) and displays it.

## 5. Verification

- [x] 5.1 Verify that importing `FuncFinetuneModel` with default parameters (no fusion_layers, final_graph_concat=True) produces an identical model to the current implementation — same parameter count and forward output shape.
- [x] 5.2 Verify that `GatedResidualFusion` forward pass produces correct output shape and gate values near 0.12 at initialization.
