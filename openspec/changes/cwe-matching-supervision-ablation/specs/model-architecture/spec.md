## ADDED Requirements

### Requirement: Optional CWE-matching projection heads
`FuncFinetuneModel` SHALL accept optional configuration for a shared CWE-matching space, including whether matching supervision is enabled and the matching projection dimension. When enabled, the model SHALL create projection heads for the pooled function representation and the input CWE embedding while preserving the existing classifier head.

#### Scenario: Matching heads disabled by default
- **WHEN** `FuncFinetuneModel` is constructed with matching supervision disabled
- **THEN** the model behaves as the current baseline and does not require auxiliary matching tensors to train or infer

#### Scenario: Matching heads enabled
- **WHEN** `FuncFinetuneModel` is constructed with matching supervision enabled
- **THEN** it instantiates both function and CWE projection heads for the shared matching space in addition to the existing classifier head

### Requirement: Lightning and train entrypoints expose matching config
`FuncFinetuneLitModule` and `train_finetune.train()` SHALL accept and persist the matching-supervision configuration needed to reproduce ablation runs, including ablation mode, negative-sampling settings, triplet margin, and auxiliary-loss weight.

#### Scenario: Checkpoint captures matching configuration
- **WHEN** a model is trained with ablation mode `E3`
- **THEN** the Lightning checkpoint hyperparameters record the enabled matching-supervision settings needed to reload that run faithfully
