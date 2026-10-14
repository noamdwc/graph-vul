> Retired historical specification. Per-function embedding caches and their generator have been removed. UniXcoder now encodes tokenized functions inside the model; see the [matching contract](../../../docs/MATCHING_CONTRACT.md).

## ADDED Requirements

### Requirement: UniXcoder mean-pooled embedding extraction
The `embed_functions` pipeline stage SHALL encode each function's `func_before` text using a frozen `microsoft/unixcoder-base` model and save the mean-pooled hidden-state vector as a `.pt` file.

#### Scenario: Embedding shape
- **WHEN** a function is embedded
- **THEN** the saved tensor SHALL have shape `[768]` (UNIXCODER_EMBEDDING_DIM)

#### Scenario: Mean pooling over non-padding tokens
- **WHEN** the encoder produces hidden states for a function
- **THEN** the embedding SHALL be the mean of all non-padding token hidden states, excluding padding positions from the average

### Requirement: Tokenizer truncation
The tokenizer SHALL truncate inputs to 512 tokens (`truncation=True`, `max_length=512`). Functions exceeding this limit SHALL be truncated, not skipped.

#### Scenario: Function exceeding token limit
- **WHEN** a function's subword tokenization exceeds 512 tokens
- **THEN** the tokenizer SHALL truncate to 512 tokens and the function SHALL still produce a valid embedding

### Requirement: Skip existing embeddings
The embedding stage SHALL skip functions whose `.pt` file already exists in FUNC_EMBEDDINGS_DIR, enabling incremental runs.

#### Scenario: Re-running embed stage
- **WHEN** `embed_functions` is run and some embeddings already exist
- **THEN** only missing embeddings SHALL be computed; existing files SHALL NOT be overwritten

### Requirement: Embedding storage layout
Each embedding SHALL be saved to `FUNC_EMBEDDINGS_DIR/{sample_id}.pt` where `sample_id` is the function's unique identifier from the subset CSV.

#### Scenario: File naming
- **WHEN** a function with sample_id 12345 is embedded
- **THEN** the embedding SHALL be saved at `data/processed/func_embeddings/12345.pt`

### Requirement: Frozen encoder
The UniXcoder model SHALL run in eval mode with `torch.no_grad()`. No gradients SHALL be computed during embedding extraction.

#### Scenario: No gradient computation
- **WHEN** the embedding stage runs
- **THEN** model parameters SHALL NOT be updated and no gradient graph SHALL be constructed
