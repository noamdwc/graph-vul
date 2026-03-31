## ADDED Requirements

### Requirement: ScalarGatedResidualFusion module
The system SHALL provide a `ScalarGatedResidualFusion` class in `graph_vul/model/gated_residual_fusion.py` that fuses a pooled graph embedding into a transformer hidden state via a single learned scalar gate and residual connection.

The module SHALL:
- Accept `graph_dim` (int) and `hidden_size` (int) at construction (same interface as `GatedResidualFusion`)
- Project `graph_emb` from `graph_dim` to `hidden_size` via a linear layer
- Compute a single scalar gate via `sigmoid(alpha)` where `alpha` is an `nn.Parameter`
- Output: `hidden_state + sigmoid(alpha) * proj(graph_emb).unsqueeze(1)`
- Initialize `alpha` to -2.0

#### Scenario: Forward pass with valid inputs
- **WHEN** called with `hidden_state` of shape `[B, seq_len, 768]` and `graph_emb` of shape `[B, graph_dim]`
- **THEN** returns a tensor of shape `[B, seq_len, 768]` equal to `hidden_state + sigmoid(alpha) * proj(graph_emb).unsqueeze(1)`

#### Scenario: Gate starts near zero
- **WHEN** module is freshly initialized
- **THEN** the gate value SHALL be approximately 0.12 (sigmoid(-2.0))

### Requirement: Fusion method registry
The system SHALL provide a `FUSION_REGISTRY` dict in `graph_vul/model/gated_residual_fusion.py` mapping string keys to fusion module classes.

The registry SHALL contain at minimum:
- `"gated_residual"` → `GatedResidualFusion`
- `"scalar_gated_residual"` → `ScalarGatedResidualFusion`

#### Scenario: Registry lookup
- **WHEN** code accesses `FUSION_REGISTRY["scalar_gated_residual"]`
- **THEN** it SHALL return the `ScalarGatedResidualFusion` class

#### Scenario: Unknown fusion method
- **WHEN** code accesses `FUSION_REGISTRY` with an unregistered key
- **THEN** a `KeyError` SHALL be raised

### Requirement: Configurable fusion_method parameter on FuncFinetuneModel
`FuncFinetuneModel.__init__` SHALL accept a `fusion_method: str` parameter (default `"scalar_gated_residual"`) that determines which fusion module class is instantiated for internal fusion layers.

The model SHALL look up the class from `FUSION_REGISTRY[fusion_method]` and instantiate one module per layer in `fusion_layers`.

#### Scenario: Default fusion method
- **WHEN** `FuncFinetuneModel` is constructed with `graph_dim > 0` and `fusion_layers` set, without specifying `fusion_method`
- **THEN** the model SHALL use `ScalarGatedResidualFusion` modules for internal fusion

#### Scenario: Explicit gated_residual method
- **WHEN** `FuncFinetuneModel` is constructed with `fusion_method="gated_residual"`
- **THEN** the model SHALL use `GatedResidualFusion` modules (backward compatible behavior)

### Requirement: fusion_method threaded through LitModule and training
`FuncFinetuneLitModule` and `train_finetune.train()` SHALL accept a `fusion_method: str` parameter and pass it through to `FuncFinetuneModel`.

#### Scenario: Training with scalar_gated_residual
- **WHEN** `train()` is called with `fusion_method="scalar_gated_residual"` and `use_graph=True`
- **THEN** the trained model SHALL use `ScalarGatedResidualFusion` modules in its fusion layers
