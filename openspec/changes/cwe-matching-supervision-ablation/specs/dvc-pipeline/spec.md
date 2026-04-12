## ADDED Requirements

### Requirement: Function-level ablation runs are configurable in the existing DVC flow
The function-level DVC pipeline SHALL support running CWE-matching ablations through configurable parameters or stage commands without requiring a separate pipeline definition for each variant. The current baseline path SHALL remain runnable as the default configuration.

#### Scenario: Baseline DVC run remains unchanged
- **WHEN** the function-level pipeline is run without ablation overrides
- **THEN** it executes the baseline configuration equivalent to `E0`

#### Scenario: Named ablation run configured
- **WHEN** the function-level pipeline is run with ablation mode `E4`
- **THEN** the fine-tune and downstream export/eval stages receive the matching-supervision settings for `E4`

### Requirement: Per-variant metrics and reports
The function-level export and evaluation stages SHALL write metrics and summary artifacts that identify the ablation variant so E0-E4 outputs can coexist and be compared without manual renaming.

#### Scenario: Variant-specific outputs preserved
- **WHEN** `E1` and `E3` are run sequentially
- **THEN** each run writes metrics and reports that remain distinguishable by variant name or manifest metadata rather than overwriting each other ambiguously
