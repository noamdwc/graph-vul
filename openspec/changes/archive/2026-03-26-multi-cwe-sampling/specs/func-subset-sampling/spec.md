## MODIFIED Requirements

### Requirement: Project-size filtering
The `func_extract_subset` pipeline stage SHALL filter BigVul to include only projects whose row count **for a given CWE** is between FUNC_MIN_PROJECT_ROWS (50) and FUNC_MAX_PROJECT_ROWS (2000), inclusive. This filter is applied independently per CWE within each project.

#### Scenario: Small project excluded for a CWE
- **WHEN** a project has fewer than 50 rows for a specific CWE
- **THEN** all that project's functions for that CWE SHALL be excluded from the subset

#### Scenario: Large project excluded for a CWE
- **WHEN** a project has more than 2000 rows for a specific CWE
- **THEN** all that project's functions for that CWE SHALL be excluded from the subset

#### Scenario: Mid-size project included for a CWE
- **WHEN** a project has between 50 and 2000 rows for a specific CWE (inclusive)
- **THEN** it SHALL be eligible for inclusion in the subset for that CWE

## REMOVED Requirements

### Requirement: CVE-based sampling parameters
**Reason:** The multi-CWE pipeline no longer selects a bounded number of CVEs. All CVEs for eligible project-CWE pairs are included.
**Migration:** Remove `FUNC_MIN_CVES` and `FUNC_MAX_CVES` references. CVE selection is replaced by per-CWE project-size filtering.
