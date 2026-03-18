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
The system SHALL skip CPG generation for samples whose export directory already exists and is non-empty, allowing the pipeline to be re-run without redundant work.

#### Scenario: Re-run with existing exports
- **WHEN** the pipeline is run and some sample directories already contain CPG exports
- **THEN** those samples SHALL be skipped and only missing samples SHALL be processed
