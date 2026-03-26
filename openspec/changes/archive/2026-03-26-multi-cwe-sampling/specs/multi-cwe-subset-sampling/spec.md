## ADDED Requirements

### Requirement: Multi-CWE configuration
The config SHALL define `CWE_IDS` as a list of 6 CWE identifiers used for train/val/test, and `HELDOUT_CWE_IDS` as a list of 2 CWE identifiers reserved exclusively for test evaluation. The old `CWE_ID` scalar SHALL be removed.

#### Scenario: Config lists defined
- **WHEN** the config module is loaded
- **THEN** `CWE_IDS` SHALL equal `["CWE-119", "CWE-125", "CWE-20", "CWE-476", "CWE-190", "CWE-787"]` and `HELDOUT_CWE_IDS` SHALL equal `["CWE-200", "CWE-416"]`

#### Scenario: Old CWE_ID removed
- **WHEN** code references `CWE_ID` from config
- **THEN** it SHALL fail with an ImportError (the name no longer exists)

### Requirement: All-CWE filtering
The subset extraction stage SHALL filter BigVul to include only rows whose `cwe_id` is in the union of `CWE_IDS` and `HELDOUT_CWE_IDS`.

#### Scenario: Row with unlisted CWE excluded
- **WHEN** a BigVul row has `cwe_id = "CWE-78"`
- **THEN** it SHALL be excluded from the subset

#### Scenario: Row with training CWE included
- **WHEN** a BigVul row has `cwe_id = "CWE-119"`
- **THEN** it SHALL be eligible for inclusion (subject to other filters)

#### Scenario: Row with held-out CWE included
- **WHEN** a BigVul row has `cwe_id = "CWE-416"`
- **THEN** it SHALL be eligible for inclusion (subject to other filters)

### Requirement: Per-CWE project-size filtering
The project-size filter (FUNC_MIN_PROJECT_ROWS to FUNC_MAX_PROJECT_ROWS) SHALL be applied per CWE within each project. A project's rows for one CWE are evaluated independently of its rows for other CWEs.

#### Scenario: Project with mixed CWE eligibility
- **WHEN** project "linux" has 200 CWE-119 rows and 30 CWE-476 rows
- **THEN** the 200 CWE-119 rows SHALL be eligible and the 30 CWE-476 rows SHALL be excluded (below minimum of 50)

#### Scenario: Same project eligible for multiple CWEs
- **WHEN** project "FFmpeg" has 100 CWE-119 rows and 80 CWE-125 rows
- **THEN** both sets of rows SHALL be eligible (both within 50-2000 range)

### Requirement: Output CSV includes cwe_id
The output `func_subset.csv` SHALL contain a `cwe_id` column identifying which CWE each row belongs to.

#### Scenario: CWE column present
- **WHEN** the subset CSV is written
- **THEN** each row SHALL have a non-null `cwe_id` value from the set of 8 configured CWEs
