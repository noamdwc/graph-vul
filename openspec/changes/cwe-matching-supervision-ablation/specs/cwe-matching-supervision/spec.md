## ADDED Requirements

### Requirement: Configurable CWE-matching ablation modes
The function-level fine-tuning pipeline SHALL support five named ablation modes for CWE-matching supervision:
- `E0`: baseline with all auxiliary matching supervision disabled
- `E1`: wrong-CWE negatives enabled and triplet loss disabled
- `E2`: triplet loss enabled and wrong-CWE negatives disabled
- `E3`: wrong-CWE negatives and triplet loss both enabled
- `E4`: same as `E3`, but with hard-negative preference for similar or same-family CWEs

The ablation mode SHALL be configurable from the training entrypoint without editing code, and `E0` SHALL preserve the current baseline behavior.

#### Scenario: Baseline mode preserves current training
- **WHEN** training runs with ablation mode `E0`
- **THEN** no wrong-CWE negatives are sampled, no triplet loss is computed, and the training path matches the current baseline objective

#### Scenario: Combined supervision mode enabled
- **WHEN** training runs with ablation mode `E3`
- **THEN** the run computes the existing classifier loss plus wrong-CWE negative supervision and vulnerable-only triplet loss

### Requirement: Shared matching-space projections
The model SHALL project the pooled function representation and the input CWE embedding into a shared matching space using dedicated projection heads, while keeping the current classifier head intact. The matching-space outputs SHALL be available to loss computation and evaluation code without replacing the main binary vulnerability head.

#### Scenario: Matching heads enabled
- **WHEN** auxiliary supervision is enabled
- **THEN** the model returns or exposes projected function and CWE representations for the current batch in addition to the main classification logits

#### Scenario: Classifier head preserved
- **WHEN** the model computes vulnerability logits
- **THEN** it still uses the existing classifier head rather than replacing it with a similarity-only score

### Requirement: Wrong-CWE negative sampling
Wrong-CWE negative supervision SHALL sample wrong CWE candidates for vulnerable positive samples using the existing subset metadata and CWE embedding inventory. The default configuration SHALL sample one wrong-CWE negative per positive pair, and the feature SHALL be configurable and easy to disable.

#### Scenario: Default wrong-negative count
- **WHEN** wrong-CWE negatives are enabled with default settings
- **THEN** each vulnerable positive sample is paired with exactly one wrong-CWE negative candidate

#### Scenario: No negative sampling for disabled mode
- **WHEN** wrong-CWE negatives are disabled
- **THEN** the training batch contains no auxiliary wrong-CWE supervision targets

### Requirement: Hard-negative preference for E4
When ablation mode `E4` is enabled, wrong-CWE sampling SHALL prefer similar or same-family CWE negatives when such candidates are available for the current positive CWE. If no preferred hard negative is available, the sampler SHALL fall back to any valid wrong CWE rather than failing.

#### Scenario: Hard negative available
- **WHEN** a vulnerable sample with CWE `CWE-X` has an available similar or same-family wrong CWE candidate
- **THEN** the sampler chooses that preferred candidate before considering unrelated CWEs

#### Scenario: Hard negative unavailable
- **WHEN** no similar or same-family wrong CWE is available for a vulnerable sample
- **THEN** the sampler falls back to a standard wrong-CWE negative and training continues

### Requirement: Vulnerable-only triplet loss
Triplet loss SHALL be computed only for vulnerable samples using the projected function representation as anchor, the true CWE projection as positive, and a wrong-CWE projection as negative. The default triplet margin SHALL be `0.2`, and the triplet contribution SHALL be weighted by a configurable `lambda` with default `0.1`.

#### Scenario: Vulnerable sample contributes triplet loss
- **WHEN** a vulnerable sample is present and triplet loss is enabled
- **THEN** the run computes a triplet term using that sample's true CWE and a sampled wrong CWE

#### Scenario: Benign sample excluded from triplet loss
- **WHEN** a benign sample is present in the batch
- **THEN** it contributes to the main classifier loss only and not to the triplet term

### Requirement: Separate auxiliary loss logging
Training SHALL log the main classification loss and each enabled auxiliary matching loss as separate metrics so ablation runs remain comparable and debuggable.

#### Scenario: Combined run loss logging
- **WHEN** ablation mode `E3` or `E4` is used
- **THEN** training logs include at minimum the BCE loss, wrong-negative loss component, triplet loss component, and total loss
