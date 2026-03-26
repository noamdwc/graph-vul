## ADDED Requirements

### Requirement: CWE descriptions for all target CWEs
The file `data/cwe_descriptions.json` SHALL contain description entries for all 8 CWEs: CWE-119, CWE-125, CWE-20, CWE-476, CWE-190, CWE-787, CWE-200, CWE-416.

#### Scenario: All CWEs have descriptions
- **WHEN** the CWE embedding pipeline reads `data/cwe_descriptions.json`
- **THEN** it SHALL find a non-empty description string for each of the 8 target CWE IDs

#### Scenario: Descriptions sourced from MITRE
- **WHEN** a CWE description is added
- **THEN** it SHALL be the official MITRE CWE "Description" field for that CWE identifier

### Requirement: Existing CWE-119 description preserved
The existing CWE-119 entry SHALL NOT be modified.

#### Scenario: CWE-119 unchanged
- **WHEN** the descriptions file is updated with new CWEs
- **THEN** the CWE-119 entry SHALL remain identical to its current value
