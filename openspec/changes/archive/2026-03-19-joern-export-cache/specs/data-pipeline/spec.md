## MODIFIED Requirements

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
