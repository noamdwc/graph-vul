### Requirement: Project-size filtering
The `func_extract_subset` pipeline stage SHALL filter BigVul to exclude only `(project, cwe_id)` groups whose row count exceeds `FUNC_MAX_PROJECT_ROWS` (2000). This filter SHALL be applied independently per CWE within each project, and the stage SHALL NOT apply a minimum `(project, cwe_id)` row threshold.

#### Scenario: Small project retained for a CWE
- **WHEN** a project has fewer than 50 rows for a specific CWE
- **THEN** those rows SHALL remain eligible for inclusion in the subset for that CWE, subject to other filters

#### Scenario: Large project excluded for a CWE
- **WHEN** a project has more than 2000 rows for a specific CWE
- **THEN** all that project's functions for that CWE SHALL be excluded from the subset

#### Scenario: Mid-size project included for a CWE
- **WHEN** a project has between 1 and 2000 rows for a specific CWE, inclusive
- **THEN** it SHALL be eligible for inclusion in the subset for that CWE

### Requirement: Token-length filtering
Functions with more than `FUNC_MAX_WHITESPACE_TOKENS` whitespace-split tokens SHALL be excluded from the subset. With the current configuration value of 512, functions with 513 or more whitespace tokens SHALL be excluded and functions with 512 or fewer SHALL remain eligible.

#### Scenario: Long function excluded
- **WHEN** a function's `func_before` text splits into 513 or more whitespace tokens
- **THEN** it SHALL be excluded from the subset

#### Scenario: Threshold-length function included
- **WHEN** a function's `func_before` text splits into exactly 512 whitespace tokens
- **THEN** it SHALL remain eligible for inclusion

### Requirement: Exact code deduplication
The `func_extract_subset` pipeline stage SHALL preserve the current exact-code deduplication behavior based on the `func_before` field. When duplicates exist, it SHALL keep the first occurrence under the existing row ordering and remove later exact matches before subsequent filters run.

#### Scenario: Duplicate function text appears multiple times
- **WHEN** two or more rows have identical `func_before` content
- **THEN** the subset stage SHALL keep the first such row and exclude later exact duplicates

### Requirement: Metadata derivation before filtering
The `func_extract_subset` pipeline stage SHALL derive metadata from `func_before` before deduplication and filtering. It SHALL compute `code_char_len`, `code_line_count`, and `code_token_count`, plus bucketed variants for these fields when the derivation succeeds.

#### Scenario: Metadata fields available for reporting
- **WHEN** the subset stage completes successfully
- **THEN** the manifest and report inputs SHALL include derived code-length metadata for downstream analysis

### Requirement: Filter ordering
The `func_extract_subset` pipeline stage SHALL preserve the cleanup order of reading raw labeled data, deriving metadata, deduplicating exact code, filtering long functions, filtering oversized `(project, cwe_id)` groups, and then writing outputs.

#### Scenario: Group counts reflect the post-cleanup pool
- **WHEN** `(project, cwe_id)` group counts are computed for the max-row filter
- **THEN** they SHALL be computed after metadata derivation, exact-code deduplication, and token-length filtering have already run

### Requirement: Separate output from graph pipeline
The subset SHALL be written to `FUNC_SUBSET_CSV` (`data/func_subset.csv`), with a manifest at `FUNC_SUBSET_MANIFEST` and a report at `FUNC_SUBSET_REPORT`. These outputs SHALL remain independent of the existing graph-pipeline `data/subset.csv`, while preserving the existing machine-readable manifest plus markdown report pattern.

#### Scenario: Graph pipeline subset unchanged
- **WHEN** `func_extract_subset` runs
- **THEN** `data/subset.csv` SHALL NOT be modified

#### Scenario: Subset outputs generated
- **WHEN** `func_extract_subset` completes
- **THEN** it SHALL write `data/func_subset.csv`, `data/func_subset_manifest.json`, and `data/func_subset_report.md`
