### Requirement: Partition func_subset by token length
The partition stage SHALL read `func_subset.csv`, compute `code_token_count` as the whitespace-split token count of `func_before`, and write two disjoint CSVs: one containing rows where `code_token_count <= SHORT_FUNCTION_MAX_TOKENS` (short partition) and one containing all remaining rows (other partition). Every row in the input SHALL appear in exactly one output partition.

#### Scenario: Short function routed to short partition
- **WHEN** a row has `func_before` with 42 whitespace-delimited tokens
- **THEN** it SHALL appear in the short partition CSV and NOT in the other partition CSV

#### Scenario: Long function routed to other partition
- **WHEN** a row has `func_before` with 100 whitespace-delimited tokens
- **THEN** it SHALL appear in the other partition CSV and NOT in the short partition CSV

#### Scenario: Boundary value (exactly 64 tokens)
- **WHEN** a row has `func_before` with exactly 64 whitespace-delimited tokens
- **THEN** it SHALL appear in the short partition CSV (boundary is inclusive: `<= 64`)

#### Scenario: Row count preservation
- **WHEN** the partition stage completes
- **THEN** the sum of rows in the short and other partition CSVs SHALL equal the row count of the input `func_subset.csv`

---

### Requirement: Partition output paths
The short partition SHALL be written to `data/partitions/short/func_subset.csv` and the other partition SHALL be written to `data/partitions/other/func_subset.csv`.

#### Scenario: Output file locations
- **WHEN** the partition stage completes
- **THEN** both `data/partitions/short/func_subset.csv` and `data/partitions/other/func_subset.csv` SHALL exist

---

### Requirement: Configurable token threshold
The partition threshold SHALL be defined as `SHORT_FUNCTION_MAX_TOKENS` in `graph_vul.config` with a default value of 64. The partition script SHALL import and use this constant.

#### Scenario: Default threshold
- **WHEN** the partition stage runs with default config
- **THEN** the short partition SHALL contain rows with `code_token_count <= 64`

---

### Requirement: CLI interface
The partition script SHALL be runnable as `python -m graph_vul.pipeline.partition_by_length` and SHALL accept `--input` (path to func_subset CSV, default from config) and `--output-dir` (parent directory for partition outputs, default `data/partitions`).

#### Scenario: Custom input and output
- **WHEN** the script is invoked with `--input /tmp/test.csv --output-dir /tmp/parts`
- **THEN** outputs SHALL be written to `/tmp/parts/short/func_subset.csv` and `/tmp/parts/other/func_subset.csv`

---

### Requirement: Partition summary logging
The partition stage SHALL log the row count, vulnerable count, project count, and CWE count for each partition upon completion.

#### Scenario: Summary logged
- **WHEN** the partition stage completes
- **THEN** it SHALL print a summary showing at minimum: partition name, total rows, vulnerable rows, number of distinct projects, and number of distinct CWEs for each partition

---

### Requirement: Column preservation
Both partition CSVs SHALL contain all columns from the input `func_subset.csv` without modification, plus the `code_token_count` column if it was not already present.

#### Scenario: Columns preserved
- **WHEN** the input has columns `[sample_id, func_before, vul, cwe_id, project]`
- **THEN** both output CSVs SHALL have at least those same columns
