## MODIFIED Requirements

### Requirement: Per-dataset inference
The external evaluation script SHALL run inference on each registered external dataset independently using the PrimeVul-trained checkpoint and tokenizer. For the PrimeVul-first baseline, the external benchmark set SHALL exclude PrimeVul itself and SHALL remain compatible with BigVul and DiverseVul comparison.

#### Scenario: Inference on DiverseVul
- **WHEN** the evaluation script processes DiverseVul for the PrimeVul-first baseline
- **THEN** it loads the DiverseVul adapter output, tokenizes with UniXcoder (`max_length=512`), runs the model in eval mode with `torch.inference_mode()`, and collects logits, probabilities, and binary predictions

#### Scenario: BigVul comparison remains available
- **WHEN** BigVul test predictions from the same trained checkpoint are available in the exported prediction artifacts
- **THEN** the external evaluation summary includes them as a comparison row without treating PrimeVul as an external dataset

### Requirement: Markdown summary report
The script SHALL generate `data/analysis/external_eval_tier1/summary.md` containing a comparison table for the PrimeVul-first baseline with rows for BigVul test and each active external dataset such as DiverseVul. PrimeVul SHALL NOT be listed as an external dataset in this report once it becomes the training source.

#### Scenario: Summary excludes PrimeVul-as-external
- **WHEN** the PrimeVul-first external evaluation summary is generated
- **THEN** the table compares the PrimeVul-trained checkpoint against BigVul test and DiverseVul
- **AND** PrimeVul is not duplicated as an external benchmark row
