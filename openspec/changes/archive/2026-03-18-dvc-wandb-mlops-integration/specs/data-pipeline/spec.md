## ADDED Requirements

### Requirement: Standalone code embedding script
The system SHALL provide a standalone script (`graph_vul/pipeline/embed_code.py`) that reads CPG exports from a specified input directory, computes DeBERTa embeddings for each node's text, and writes the resulting tensors to an output directory. The script SHALL accept input and output paths as CLI arguments.

#### Scenario: Code embedding generation
- **WHEN** `embed_code.py` is run with `--input data/raw/ --output data/processed/code_embeddings/`
- **THEN** for each sample's CPG, a `.pt` file SHALL be written containing the node feature tensor, edge index, and edge type tensor

#### Scenario: Idempotent re-run
- **WHEN** `embed_code.py` is run and some output files already exist
- **THEN** existing outputs SHALL be skipped and only missing samples SHALL be processed

### Requirement: Standalone CVE embedding script
The system SHALL provide a standalone script (`graph_vul/pipeline/embed_cve.py`) that reads CVE descriptions from the subset CSV and computes DeBERTa embeddings, writing each result to an output directory. The script SHALL accept input CSV path and output directory as CLI arguments.

#### Scenario: CVE embedding generation
- **WHEN** `embed_cve.py` is run with `--input data/subset_cwe119.csv --output data/processed/cve_embeddings/`
- **THEN** for each sample, a `.pt` file SHALL be written containing the 768-dimensional CVE text embedding vector

#### Scenario: Missing CVE description
- **WHEN** a sample has no CVE description text
- **THEN** the script SHALL write a zero vector of dimension 768 as the embedding

## MODIFIED Requirements

### Requirement: Pipeline idempotency
The system SHALL skip CPG generation for samples whose export directory already exists and is non-empty, allowing the pipeline to be re-run without redundant work. The embedding scripts SHALL similarly skip samples whose output files already exist.

#### Scenario: Re-run with existing exports
- **WHEN** the pipeline is run and some sample directories already contain CPG exports
- **THEN** those samples SHALL be skipped and only missing samples SHALL be processed

#### Scenario: Re-run with existing embeddings
- **WHEN** an embedding script is run and some output `.pt` files already exist
- **THEN** those samples SHALL be skipped and only missing embeddings SHALL be computed
