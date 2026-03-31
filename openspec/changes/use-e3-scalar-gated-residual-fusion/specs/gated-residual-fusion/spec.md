## MODIFIED Requirements

### Requirement: Fusion hooks on encoder layers
The system SHALL register PyTorch forward hooks on specified UniXcoder encoder layers to apply the configured fusion module in-place. The hooks SHALL:
- Be registered only for layer indices specified in `fusion_layers`
- Use the fusion module class determined by `fusion_method` (looked up from `FUSION_REGISTRY`)
- Receive the pooled `graph_emb` for the current batch (stored on the model instance before the forward pass)
- Apply the fusion module to the layer's output hidden state
- Be removable (hooks stored as handles for cleanup)

#### Scenario: Fusion applied to last 2 layers
- **WHEN** `fusion_layers=[10, 11]` and the model runs a forward pass with graph data
- **THEN** layers 10 and 11 of the UniXcoder encoder SHALL have their output hidden states modified by the configured fusion module

#### Scenario: No fusion when fusion_layers is None
- **WHEN** `fusion_layers=None` (default)
- **THEN** no hooks are registered and the encoder forward pass is identical to the current baseline behavior

#### Scenario: No graph data at inference
- **WHEN** `graph_batch` is None during forward pass but fusion_layers is configured
- **THEN** the fusion hooks SHALL use a zero vector for graph_emb, producing no fusion effect (gate * zeros = zeros)
