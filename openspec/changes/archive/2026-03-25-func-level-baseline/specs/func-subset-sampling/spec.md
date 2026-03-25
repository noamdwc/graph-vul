## ADDED Requirements

### Requirement: Project-size filtering
The `func_extract_subset` pipeline stage SHALL filter BigVul to include only projects whose total CWE-119 row count is between FUNC_MIN_PROJECT_ROWS (50) and FUNC_MAX_PROJECT_ROWS (2000), inclusive.

#### Scenario: Small project excluded
- **WHEN** a project has fewer than 50 CWE-119 rows
- **THEN** all its functions SHALL be excluded from the subset

#### Scenario: Large project excluded
- **WHEN** a project has more than 2000 CWE-119 rows
- **THEN** all its functions SHALL be excluded from the subset

#### Scenario: Mid-size project included
- **WHEN** a project has between 50 and 2000 CWE-119 rows (inclusive)
- **THEN** it SHALL be eligible for inclusion in the subset

### Requirement: Minimum vulnerable functions per project
The subset SHALL include only projects with at least FUNC_MIN_VUL_PER_PROJECT (3) vulnerable functions (target=1).

#### Scenario: Project with insufficient vulnerable functions
- **WHEN** a project passes the size filter but has fewer than 3 vulnerable functions
- **THEN** it SHALL be excluded from the subset

### Requirement: Token-length filtering
Functions with FUNC_MAX_WHITESPACE_TOKENS (512) or more whitespace-split tokens SHALL be excluded from the subset.

#### Scenario: Long function excluded
- **WHEN** a function's `func_before` text splits into 513 or more whitespace tokens
- **THEN** it SHALL be excluded from the subset

#### Scenario: Short function included
- **WHEN** a function's `func_before` text splits into fewer than 513 whitespace tokens
- **THEN** it SHALL remain eligible for inclusion

### Requirement: CVE-based sampling parameters
The sampling stage SHALL use FUNC_MIN_CVES (15) and FUNC_MAX_CVES (30) as the CVE count bounds for selecting the subset. The top-2 project penalty in `_score_cve` SHALL be removed (the hard project-size filter makes it redundant).

#### Scenario: CVE selection range
- **WHEN** sampling CVEs from eligible projects
- **THEN** at least 15 and at most 30 CVEs SHALL be selected

### Requirement: Separate output from graph pipeline
The subset SHALL be written to FUNC_SUBSET_CSV (`data/func_subset.csv`), with a manifest at FUNC_SUBSET_MANIFEST and a report at FUNC_SUBSET_REPORT. These SHALL be independent of the existing `data/subset.csv`.

#### Scenario: Graph pipeline subset unchanged
- **WHEN** `func_extract_subset` runs
- **THEN** `data/subset.csv` SHALL NOT be modified

#### Scenario: Subset report generated
- **WHEN** `func_extract_subset` completes
- **THEN** `data/func_subset_report.md` SHALL contain project counts, function counts, and class balance statistics
