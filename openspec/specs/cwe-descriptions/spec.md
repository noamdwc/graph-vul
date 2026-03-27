### Requirement: CWE descriptions for all labeled dataset CWEs
The file `data/cwe_descriptions.json` SHALL contain description entries for every labeled CWE used by the function-subset pipeline. For the current dataset snapshot, this set contains 91 CWE IDs from the raw labeled BigVul input, including CWEs that are filtered out later in subset construction.

#### Scenario: All labeled dataset CWEs have descriptions
- **WHEN** the CWE embedding pipeline reads `data/cwe_descriptions.json`
- **THEN** it SHALL find a non-empty description string for each labeled dataset CWE ID

#### Scenario: Descriptions cover current function subset and pre-filter extras
- **WHEN** the current `data/func_subset.csv` and `data/func_subset_manifest.json` are present
- **THEN** the descriptions file SHALL cover all 89 CWEs in `func_subset.csv`
- **AND** it SHALL also include the 2 labeled CWEs that exist before token-length filtering but not in the final subset

#### Scenario: Descriptions sourced from MITRE wording
- **WHEN** a CWE description is added
- **THEN** it SHALL reflect the corresponding MITRE CWE entry's meaning using close manual wording
