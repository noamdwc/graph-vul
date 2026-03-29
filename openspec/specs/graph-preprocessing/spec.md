## ADDED Requirements

### Requirement: Extract function-local graph artifacts from Joern CPG exports
The system SHALL provide a preprocessing script that reads existing Joern CPG exports from `data/raw/{sample_id}/` and produces per-sample `.pt` files in `data/processed/graph_artifacts/{sample_id}.pt`. Each artifact SHALL contain `node_type_ids` (LongTensor), `edge_index` (LongTensor [2, E]), and `edge_type_ids` (LongTensor [E]).

#### Scenario: Successful extraction from CSV-format CPG export
- **WHEN** a sample directory contains `nodes.csv` and `edges.csv` with valid Joern CPG data
- **THEN** the script SHALL parse node LABEL fields to determine node types, filter edges to AST and CFG only (dropping PDG/DDG/CDG), map types to integer IDs via the vocab, and write a `.pt` file with `node_type_ids`, `edge_index`, and `edge_type_ids`

#### Scenario: Successful extraction from JSON-format CPG export
- **WHEN** a sample directory contains `nodes.json` and `edges.json`
- **THEN** the script SHALL parse the JSON structure, extract node types and edge types, filter to AST+CFG, and produce the same `.pt` output format

#### Scenario: Successful extraction from DOT-format CPG export
- **WHEN** a sample directory contains `.dot` files
- **THEN** the script SHALL parse DOT node labels to extract node types, parse edge labels for edge types, filter to AST+CFG, and produce the same `.pt` output format

#### Scenario: Missing or empty CPG export
- **WHEN** a sample has no CPG export directory or the directory is empty
- **THEN** the script SHALL skip the sample, log a warning, and increment a missing-graph counter

#### Scenario: Skip already-processed samples
- **WHEN** the output `.pt` file already exists for a sample
- **THEN** the script SHALL skip that sample without re-processing

### Requirement: Build and persist stable vocabulary mappings
The system SHALL produce two vocabulary files: `node_type_vocab.json` (mapping Joern node type strings to integer IDs) and `edge_type_vocab.json` (mapping edge type strings to integer IDs), stored in `data/processed/graph_artifacts/`. Index 0 SHALL be reserved for `<UNK>` in both vocabs.

#### Scenario: First run builds vocab from corpus
- **WHEN** the vocab files do not exist and the preprocessing script runs
- **THEN** the script SHALL scan all available CPG exports, collect unique node type strings and edge type strings, assign sequential integer IDs starting from 1 (0 = UNK), and write the vocab JSON files

#### Scenario: Subsequent runs reuse existing vocab
- **WHEN** the vocab files already exist
- **THEN** the script SHALL load them and use the existing mappings, assigning UNK (0) to any type strings not in the vocab

### Requirement: Log graph preprocessing statistics
The system SHALL log summary statistics after preprocessing completes.

#### Scenario: Statistics logged on completion
- **WHEN** preprocessing finishes
- **THEN** the script SHALL log: number of node types in vocab, number of edge types in vocab, total samples processed, total samples skipped (missing CPG), total samples skipped (already processed), average nodes per graph, and average edges per graph
