## ADDED Requirements

### Requirement: Subset extraction stage
The extract_subset stage SHALL read the Big-Vul CSV, filter to CWE-119, and produce a structured subset using the `structured-subset-sampling` capability. The output CSV SHALL include columns: `sample_id`, `project`, `func_before`, `cve_id`, `cve_desc`, `vul`. The stage SHALL also produce a manifest JSON and report MD.

#### Scenario: Subset produced with project column
- **WHEN** the extract_subset stage runs
- **THEN** the output CSV contains a `project` column and all rows have non-null project values

#### Scenario: Subset artifacts produced
- **WHEN** the extract_subset stage completes
- **THEN** three files exist: subset CSV, subset manifest JSON, and subset report MD

### Requirement: Generate CPGs via Joern CLI
The system SHALL invoke the Joern CLI to parse each extracted C/C++ function and export its Code Property Graph (containing AST, CFG, and PDG edges) as structured files to a local output directory. Each function's CPG SHALL be stored in its own subdirectory named by a unique sample identifier. Before invoking Joern, the system SHALL check a local content-addressed cache and use cached results when available. Exports SHALL be processed in parallel across multiple worker processes.

#### Scenario: Successful CPG generation
- **WHEN** Joern is invoked on a valid C/C++ function
- **THEN** a CPG export directory is created containing node and edge files that represent the function's AST, CFG, and PDG structures

#### Scenario: Joern parse failure
- **WHEN** Joern fails to parse a malformed or incomplete C/C++ function
- **THEN** the system SHALL log the failure with the sample identifier and continue processing the remaining functions without terminating

#### Scenario: Cached CPG available
- **WHEN** a function's source code hash matches an existing cache entry
- **THEN** the system SHALL use the cached CPG export without invoking Joern

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

### Requirement: Split stage position in pipeline
The split_data stage SHALL run after the embedding stages. It SHALL depend on the subset CSV for metadata. The split SHALL use the `leakage-aware-split` capability producing project-disjoint splits.

#### Scenario: Split depends on subset
- **WHEN** the split stage runs
- **THEN** it reads the subset CSV for project and CVE metadata to inform split decisions

#### Scenario: Split produces assignments
- **WHEN** the split stage completes
- **THEN** it produces an assignments CSV and a diagnostics report
