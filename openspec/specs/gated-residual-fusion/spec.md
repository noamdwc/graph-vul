## ADDED Requirements

### Requirement: Gated residual fusion module
The system SHALL provide a `GatedResidualFusion` module (`graph_vul/model/gated_residual_fusion.py`) that fuses a pooled graph embedding into a transformer hidden state via a learned gate and residual connection.

The module SHALL:
- Accept `graph_dim` (int) and `hidden_size` (int) at construction
- Project `graph_emb` from `graph_dim` to `hidden_size` via a linear layer
- Compute a per-token scalar gate via `sigmoid(Linear(hidden_size, 1))` applied to the transformer hidden state
- Output: `hidden_state + gate * projected_graph_emb` (broadcast over sequence length)
- Initialize the gate bias to -2.0 so fusion starts weak (~0.12 contribution)

#### Scenario: Forward pass with valid inputs
- **WHEN** called with `hidden_state` of shape `[B, seq_len, 768]` and `graph_emb` of shape `[B, graph_dim]`
- **THEN** returns a tensor of shape `[B, seq_len, 768]` equal to `hidden_state + gate * proj(graph_emb).unsqueeze(1)`

#### Scenario: Gate starts near zero
- **WHEN** module is freshly initialized (no training)
- **THEN** the gate output SHALL be approximately 0.12 (sigmoid(-2.0)), so the fusion contribution is small relative to the original hidden state

### Requirement: Fusion hooks on encoder layers
The system SHALL register PyTorch forward hooks on specified UniXcoder encoder layers to apply `GatedResidualFusion` in-place. The hooks SHALL:
- Be registered only for layer indices specified in `fusion_layers`
- Receive the pooled `graph_emb` for the current batch (stored on the model instance before the forward pass)
- Apply `GatedResidualFusion` to the layer's output hidden state
- Be removable (hooks stored as handles for cleanup)

#### Scenario: Fusion applied to last 2 layers
- **WHEN** `fusion_layers=[10, 11]` and the model runs a forward pass with graph data
- **THEN** layers 10 and 11 of the UniXcoder encoder SHALL have their output hidden states modified by the gated residual fusion module

#### Scenario: No fusion when fusion_layers is None
- **WHEN** `fusion_layers=None` (default)
- **THEN** no hooks are registered and the encoder forward pass is identical to the current baseline behavior

#### Scenario: No graph data at inference
- **WHEN** `graph_batch` is None during forward pass but fusion_layers is configured
- **THEN** the fusion hooks SHALL use a zero vector for graph_emb, producing no fusion effect (gate * zeros = zeros)
