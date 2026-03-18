## ADDED Requirements

### Requirement: Extract CWE-119 subset from Big-Vul
The system SHALL read a local Big-Vul CSV file and extract a balanced subset of approximately 1,000 samples filtered to CWE-119 (buffer overflow). The subset SHALL contain roughly equal numbers of vulnerable and benign function samples. Each record SHALL retain the function source code, the CVE ID, the CVE description text, and the vulnerability label.

#### Scenario: Successful subset extraction
- **WHEN** the extraction script is run with a path to the Big-Vul CSV
- **THEN** a filtered CSV is written containing ~500 vulnerable and ~500 benign CWE-119 samples with columns for function code, CVE ID, CVE description, and binary label

#### Scenario: Insufficient CWE-119 samples
- **WHEN** the Big-Vul CSV contains fewer than 500 vulnerable CWE-119 samples
- **THEN** the script SHALL use all available vulnerable samples and match the count with an equal number of benign samples, logging a warning about the reduced dataset size

### Requirement: Generate CPGs via Joern CLI
The system SHALL invoke the Joern CLI to parse each extracted C/C++ function and export its Code Property Graph (containing AST, CFG, and PDG edges) as structured files to a local output directory. Each function's CPG SHALL be stored in its own subdirectory named by a unique sample identifier.

#### Scenario: Successful CPG generation
- **WHEN** Joern is invoked on a valid C/C++ function
- **THEN** a CPG export directory is created containing node and edge files that represent the function's AST, CFG, and PDG structures

#### Scenario: Joern parse failure
- **WHEN** Joern fails to parse a malformed or incomplete C/C++ function
- **THEN** the system SHALL log the failure with the sample identifier and continue processing the remaining functions without terminating

### Requirement: Pipeline idempotency
The system SHALL skip CPG generation for samples whose export directory already exists and is non-empty, allowing the pipeline to be re-run without redundant work. The embedding scripts SHALL similarly skip samples whose output files already exist.

#### Scenario: Re-run with existing exports
- **WHEN** the pipeline is run and some sample directories already contain CPG exports
- **THEN** those samples SHALL be skipped and only missing samples SHALL be processed

#### Scenario: Re-run with existing embeddings
- **WHEN** an embedding script is run and some output `.pt` files already exist
- **THEN** those samples SHALL be skipped and only missing embeddings SHALL be computed

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
