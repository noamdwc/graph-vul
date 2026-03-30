## ADDED Requirements

### Requirement: Fusion module registry
The notebook SHALL define a `FUSION_REGISTRY` dict mapping string method names to fusion module classes. All 7 fusion methods MUST be registered: `gated_residual`, `ungated_residual`, `scalar_gated_residual`, `film`, `concat_adapter`, `gated_concat_adapter`, `cross_attention_graph_tokens`.

#### Scenario: Registry contains all methods
- **WHEN** the fusion registry cell is executed
- **THEN** `FUSION_REGISTRY` contains exactly 7 entries with the keys listed above, each mapping to an `nn.Module` subclass

### Requirement: Uniform fusion interface
Every fusion module (except `cross_attention_graph_tokens`) SHALL implement `forward(hidden_state: [B, seq_len, 768], graph_emb: [B, graph_dim]) -> [B, seq_len, 768]`. The `cross_attention_graph_tokens` module SHALL implement `forward(hidden_state: [B, seq_len, 768], graph_tokens: [B, K, graph_dim]) -> [B, seq_len, 768]`.

#### Scenario: Gated residual forward pass
- **WHEN** `GatedResidualFusion.forward` is called with hidden_state [2, 128, 768] and graph_emb [2, 64]
- **THEN** output shape is [2, 128, 768]

#### Scenario: Cross-attention forward pass
- **WHEN** `CrossAttentionGraphTokensFusion.forward` is called with hidden_state [2, 128, 768] and graph_tokens [2, 8, 64]
- **THEN** output shape is [2, 128, 768]

### Requirement: Ungated residual fusion (E2)
The module SHALL compute `hidden_state + proj(graph_emb).unsqueeze(1)` — a simple additive residual with no gating.

#### Scenario: Ungated residual adds projected graph
- **WHEN** forward is called
- **THEN** output equals hidden_state plus the projected graph embedding broadcast across the sequence dimension

### Requirement: Scalar-gated residual fusion (E3)
The module SHALL compute `hidden_state + sigmoid(alpha) * proj(graph_emb).unsqueeze(1)` where `alpha` is a single learned scalar parameter (not per-token), initialized to -2.0.

#### Scenario: Scalar gate is a single parameter
- **WHEN** the module is instantiated
- **THEN** the gate parameter has shape [] (scalar) and value -2.0

### Requirement: FiLM fusion (E4)
The module SHALL compute Feature-wise Linear Modulation: `gamma(graph_emb).unsqueeze(1) * hidden_state + beta(graph_emb).unsqueeze(1)` where gamma and beta are learned linear projections from graph_dim to hidden_size. Gamma bias SHALL be initialized to 1.0 and beta bias to 0.0 so the initial transform is identity.

#### Scenario: FiLM initial output approximates identity
- **WHEN** the module is freshly initialized and forward is called
- **THEN** output approximately equals hidden_state (gamma ≈ 1, beta ≈ 0)

### Requirement: Concat adapter fusion (E5)
The module SHALL concatenate `hidden_state` with `proj(graph_emb).unsqueeze(1)` along the feature dimension, then reduce back to hidden_size via a bottleneck linear layer followed by LayerNorm. Output is a residual: `hidden_state + adapter_output`.

#### Scenario: Concat adapter output shape
- **WHEN** forward is called with hidden_state [B, seq_len, 768] and graph_emb [B, 64]
- **THEN** output shape is [B, seq_len, 768]

### Requirement: Gated concat adapter fusion (E6)
The module SHALL be identical to concat adapter but with a learned sigmoid gate on the adapter output: `hidden_state + sigmoid(gate(hidden_state)) * adapter_output`. Gate bias initialized to -2.0.

#### Scenario: Gated concat adapter initial gate is weak
- **WHEN** the module is freshly initialized
- **THEN** the gate bias is -2.0, so initial sigmoid ≈ 0.12

### Requirement: Cross-attention graph tokens fusion (E7)
The module SHALL use `nn.MultiheadAttention` where code tokens are queries and graph tokens are keys/values. Graph tokens (K=8) are derived from graph encoder node embeddings via a learned attention-pooling layer. Output is a residual: `hidden_state + cross_attn_output`.

#### Scenario: Graph token count
- **WHEN** graph tokens are produced from the graph encoder
- **THEN** exactly K=8 graph tokens are generated regardless of input graph size

#### Scenario: Cross-attention residual
- **WHEN** forward is called
- **THEN** output equals hidden_state plus the cross-attention result

### Requirement: Experiment configuration
The notebook SHALL define 8 experiments with these exact run tags and methods:

| Tag | fusion_method | fusion_layers | final_graph_concat |
|-----|--------------|---------------|--------------------|
| E0_late_only | None | None | True |
| E1_gated_residual | gated_residual | list(range(12)) | True |
| E2_ungated_residual | ungated_residual | list(range(12)) | True |
| E3_scalar_gated_residual | scalar_gated_residual | list(range(12)) | True |
| E4_film | film | list(range(12)) | True |
| E5_concat_adapter | concat_adapter | list(range(12)) | True |
| E6_gated_concat_adapter | gated_concat_adapter | list(range(12)) | True |
| E7_cross_attention_graph_tokens | cross_attention_graph_tokens | list(range(12)) | True |

#### Scenario: All experiments use same hyperparameters
- **WHEN** the experiment loop runs
- **THEN** all 8 experiments use identical seed=42, epochs=10, batch_size=64, grad_accum=8, encoder_lr=2e-5, head_lr=1e-3

### Requirement: Fixed experimental controls
All experiments SHALL use the same data splits, seed, training hyperparameters, GNN/graph artifacts, classifier head architecture, and evaluation pipeline.

#### Scenario: Only fusion method varies
- **WHEN** comparing any two experiments E1–E7
- **THEN** the only difference is the fusion module class used; all other model components are identical

### Requirement: Prediction export and per-bucket metrics
The notebook SHALL export predictions for `test` and `unseen_test` splits for every experiment, and compute token-length bucket metrics using `canonicalize_prediction_frame`, `stratified_metrics`, and `DEFAULT_LENGTH_BUCKETS` from `graph_vul.analysis.post_experiment_analysis`.

#### Scenario: Predictions saved to Drive
- **WHEN** an experiment completes
- **THEN** `{exp_dir}/predictions/test_predictions.csv` and `{exp_dir}/predictions/unseen_test_predictions.csv` exist

### Requirement: Comparison table
The notebook SHALL display a final comparison table with columns: run_tag, unseen_test_f1, unseen_0_64_f1, unseen_64_128_f1, test_f1, 0_64_f1, train_minutes. The table SHALL be sorted by unseen_0_64_f1 descending and highlight the best value in each metric column.

#### Scenario: Table sorted by primary metric
- **WHEN** the results table is displayed
- **THEN** rows are sorted by unseen_0_64_f1 in descending order

### Requirement: Skip completed experiments
The notebook SHALL check if a checkpoint already exists for an experiment and skip training if so, loading metrics from the saved JSON instead. This allows resuming after Colab disconnects.

#### Scenario: Experiment already completed
- **WHEN** `{exp_dir}/func_finetune_metrics.json` and `{exp_dir}/func_finetune_model.ckpt` both exist
- **THEN** training is skipped and saved metrics are loaded directly
