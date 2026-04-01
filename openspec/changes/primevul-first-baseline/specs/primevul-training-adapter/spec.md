## ADDED Requirements

### Requirement: PrimeVul training adapter loads official Hugging Face splits
The system SHALL provide a PrimeVul training adapter that loads PrimeVul from Hugging Face using the `datasets` library and reads the dataset's official training, validation, and test splits when they are available.

#### Scenario: Official splits loaded
- **WHEN** the PrimeVul training adapter is invoked for the function-level baseline
- **THEN** it loads PrimeVul directly from Hugging Face
- **AND** it materializes rows from the dataset's official train, validation, and test partitions instead of deriving new random splits

### Requirement: PrimeVul rows are normalized to the function-level training schema
The PrimeVul training adapter SHALL normalize each row into the function-level schema used by the current baseline pipeline. At minimum, the normalized dataset SHALL contain `sample_id`, `func_before`, `vul`, `cwe_id`, and `source_split`, and it SHALL preserve or synthesize any additional fields needed by downstream reporting.

#### Scenario: PrimeVul field mapping
- **WHEN** a PrimeVul sample is normalized
- **THEN** the function source is stored in `func_before`
- **AND** the binary vulnerable/benign label is stored in `vul`
- **AND** the CWE label is normalized into `cwe_id` using the `CWE-XXX` format or an empty string when missing
- **AND** the official split label is stored in `source_split`

### Requirement: PrimeVul training sample ids remain stable and numeric
The normalized PrimeVul training dataset SHALL assign a stable numeric `sample_id` to every row so existing training, split, and prediction-export code can join on `sample_id` without requiring string-id refactors.

#### Scenario: Stable numeric ids
- **WHEN** the PrimeVul normalization stage runs twice on the same source data
- **THEN** each normalized row receives the same numeric `sample_id`
- **AND** all `sample_id` values remain unique across train, validation, and test rows

### Requirement: PrimeVul normalization preserves baseline preprocessing assumptions
The PrimeVul training adapter SHALL preserve the existing function-level baseline assumptions for binary classification and tokenization compatibility. It MUST filter out functions above `FUNC_MAX_WHITESPACE_TOKENS`, and it MUST emit source code suitable for the current tokenizer/truncation flow without changing the tokenizer or max token length.

#### Scenario: Long function filtered
- **WHEN** a PrimeVul function exceeds the configured whitespace token limit
- **THEN** it is excluded from the normalized training dataset

#### Scenario: Binary labels preserved
- **WHEN** the adapter finishes normalization
- **THEN** the `vul` column contains only `0` and `1`

### Requirement: PrimeVul normalization writes the canonical function-level training artifacts
The PrimeVul-backed extraction stage SHALL write the same canonical function-level artifacts expected by the current pipeline: `data/func_subset.csv`, `data/func_subset_manifest.json`, and `data/func_subset_report.md`.

#### Scenario: Canonical artifacts produced
- **WHEN** the PrimeVul extraction stage completes successfully
- **THEN** the canonical function-level subset CSV, manifest, and report files exist
- **AND** the manifest identifies PrimeVul as the source dataset
