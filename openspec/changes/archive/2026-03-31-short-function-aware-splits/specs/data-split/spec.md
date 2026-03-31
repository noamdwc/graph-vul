## MODIFIED Requirements

### Requirement: Split diagnostics report
The split stage SHALL produce a Markdown report containing: split sizes with vulnerable/benign counts and ratios for all five splits, per-CWE breakdown per split, CVE coverage per split, project coverage per split, project overlap verification for all 10 pairs, per-split project details, metadata bucket distributions, project exclusivity section, train balancing section, fixed token-count bucket distributions per split with vulnerable/benign breakdown, short-function evaluation viability per eval split, per-project short-bucket contributions in eval splits, and warnings.

#### Scenario: Report covers all five splits
- **WHEN** the report is generated
- **THEN** it SHALL include statistics for train, val, test, unseen_val, and unseen_test

#### Scenario: Warning on skewed split
- **WHEN** a split has fewer than 5% of total rows or has only one class
- **THEN** the report includes a warning

#### Scenario: Report includes fixed token bucket distributions
- **WHEN** the report is generated
- **THEN** it SHALL include a table showing per-split row counts for each fixed token bucket (0–64, 64–128, 128–256, 256–512, 512+) with vulnerable and benign sub-counts

#### Scenario: Report includes short-function viability assessment
- **WHEN** the report is generated
- **THEN** it SHALL include a section assessing whether each eval split has enough short-function samples for reliable evaluation, with pass/warn indicators
