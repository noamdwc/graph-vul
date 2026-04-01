## MODIFIED Requirements

### Requirement: Project-size filtering
The `func_extract_subset` pipeline stage SHALL support adapter-backed function datasets in addition to BigVul. When the source dataset does not provide project metadata suitable for `(project, cwe_id)` filtering, the stage SHALL skip that filter and record the skip in the manifest/report instead of failing.

#### Scenario: PrimeVul lacks usable project metadata
- **WHEN** the PrimeVul-backed extraction stage cannot form reliable `(project, cwe_id)` groups
- **THEN** it skips the max-project-row filter
- **AND** the manifest/report state that the filter was not applied for this dataset

#### Scenario: PrimeVul provides usable project metadata
- **WHEN** the PrimeVul-backed extraction stage can form `(project, cwe_id)` groups
- **THEN** it applies the existing `FUNC_MAX_PROJECT_ROWS` filter using those normalized fields

### Requirement: Filter ordering
The `func_extract_subset` pipeline stage SHALL preserve the cleanup order of loading source rows, normalizing them into the function-level schema, deriving metadata, deduplicating exact code, filtering long functions, conditionally filtering oversized `(project, cwe_id)` groups, and then writing outputs.

#### Scenario: Adapter normalization happens before cleanup
- **WHEN** the PrimeVul-backed extraction stage runs
- **THEN** PrimeVul rows are first normalized into the function-level schema
- **AND** deduplication and token filtering operate on the normalized `func_before` field

### Requirement: Separate output from graph pipeline
The function-level subset SHALL continue to be written to `FUNC_SUBSET_CSV` (`data/func_subset.csv`), with a manifest at `FUNC_SUBSET_MANIFEST` and a report at `FUNC_SUBSET_REPORT`, even when PrimeVul is the backing dataset.

#### Scenario: Canonical subset path retained
- **WHEN** the PrimeVul-backed `func_extract_subset` stage completes
- **THEN** `data/func_subset.csv`, `data/func_subset_manifest.json`, and `data/func_subset_report.md` are written
- **AND** downstream function-level stages can consume them without path changes
