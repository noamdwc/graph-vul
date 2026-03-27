## MODIFIED Requirements

### Requirement: Multi-CWE configuration
The config MAY continue to define `CWE_IDS` and `HELDOUT_CWE_IDS` for later split or evaluation stages, but `func_extract_subset` SHALL treat those lists as split-policy configuration rather than subset-construction filters. The old `CWE_ID` scalar SHALL remain removed.

#### Scenario: Split config exists without constraining subset extraction
- **WHEN** the config module defines `CWE_IDS` and `HELDOUT_CWE_IDS`
- **THEN** `func_extract_subset` SHALL NOT limit subset rows to the union of those lists

#### Scenario: Old CWE_ID removed
- **WHEN** code references `CWE_ID` from config
- **THEN** it SHALL fail with an ImportError because the name no longer exists

### Requirement: All-CWE filtering
The subset extraction stage SHALL keep all labeled BigVul rows with a non-null `cwe_id`, rather than filtering rows to the configured union of `CWE_IDS` and `HELDOUT_CWE_IDS`.

#### Scenario: Row with rare labeled CWE included
- **WHEN** a BigVul row has a labeled `cwe_id` that is not present in `CWE_IDS` or `HELDOUT_CWE_IDS`
- **THEN** it SHALL remain eligible for inclusion, subject to deduplication, token-length filtering, and max-row `(project, cwe_id)` filtering

#### Scenario: Row with missing CWE excluded from labeled subset
- **WHEN** a BigVul row has a null or missing `cwe_id`
- **THEN** it SHALL be excluded from the labeled function subset

### Requirement: Per-CWE project-size filtering
The project-size filter SHALL be applied per CWE within each project using only the maximum threshold `FUNC_MAX_PROJECT_ROWS`. A project's rows for one CWE are evaluated independently of its rows for other CWEs, and no minimum row threshold SHALL be applied.

#### Scenario: Project with mixed CWE sizes
- **WHEN** project "linux" has 2200 rows for `CWE-119` and 12 rows for `CWE-476`
- **THEN** the `CWE-119` rows SHALL be excluded and the `CWE-476` rows SHALL remain eligible

#### Scenario: Same project eligible for multiple CWEs
- **WHEN** project "FFmpeg" has 100 rows for `CWE-119` and 8 rows for `CWE-125`
- **THEN** both sets of rows SHALL remain eligible because both are at or below `FUNC_MAX_PROJECT_ROWS`

### Requirement: Output CSV includes cwe_id
The output `func_subset.csv` SHALL contain a non-null `cwe_id` column identifying which labeled CWE each row belongs to. The output SHALL not be limited to a pre-curated set of 8 configured CWEs.

#### Scenario: CWE column present for the full labeled universe
- **WHEN** the subset CSV is written
- **THEN** each row SHALL have a non-null `cwe_id` value from the labeled BigVul data that survived the cleanup filters
