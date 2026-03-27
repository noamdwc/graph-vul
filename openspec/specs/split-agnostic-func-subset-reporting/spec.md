### Requirement: Manifest records filter-stage survival counts
The `func_extract_subset` pipeline stage SHALL write a machine-readable manifest that records, for each major filter stage, the total row count before the stage, the total row count after the stage, and the number of surviving unique CWEs after the stage.

#### Scenario: Stage counts captured in manifest
- **WHEN** the manifest is written
- **THEN** it SHALL include stage-level before/after row counts and surviving CWE counts for raw labeled input, deduplication, token-length filtering, and max-row `(project, cwe_id)` filtering

### Requirement: Report summarizes final CWE distribution
The markdown report SHALL include a section showing the top CWEs by row count after final filtering.

#### Scenario: Top final CWEs shown
- **WHEN** the report is generated
- **THEN** it SHALL list the highest-row-count CWEs in the final filtered subset

### Requirement: Report and manifest identify oversized group removals
The subset outputs SHALL report how many `(project, cwe_id)` groups were removed by the max-row filter.

#### Scenario: Oversized groups counted
- **WHEN** one or more `(project, cwe_id)` groups exceed `FUNC_MAX_PROJECT_ROWS`
- **THEN** the manifest and report SHALL state how many such groups were removed

### Requirement: Report summarizes post-filter token-length distribution
The markdown report SHALL summarize the token-length distribution of the final filtered subset using the derived `code_token_count` metadata.

#### Scenario: Token summary present after filtering
- **WHEN** the final subset report is generated
- **THEN** it SHALL include a token-length summary derived from the surviving rows
