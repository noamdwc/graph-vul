## ADDED Requirements

### Requirement: Unseen audit CSV output
The split stage SHALL produce an `unseen_audit.csv` file alongside other split outputs containing one row per (split, project) combination for unseen splits only. Columns SHALL be: `split`, `project`, `total_rows`, `vulnerable_rows`, `benign_rows`, `unique_cwes`.

#### Scenario: Audit CSV written on successful split
- **WHEN** the split completes with unseen splits present
- **THEN** `unseen_audit.csv` SHALL exist in the output directory with one row per project per unseen split

#### Scenario: Audit CSV content accuracy
- **WHEN** `unseen_audit.csv` is loaded
- **THEN** for each row, `total_rows` SHALL equal `vulnerable_rows + benign_rows`
- **AND** `unique_cwes` SHALL equal the number of distinct CWE IDs for that project in that split

#### Scenario: No unseen pool
- **WHEN** no held-out CWEs are configured and the unseen pool is empty
- **THEN** `unseen_audit.csv` SHALL either not be written or be empty

---

### Requirement: Extended unseen reporting in split report
The split report SHALL include an unseen-specific diagnostics section showing for each of `unseen_val` and `unseen_test`: project count, rows per project, vulnerable and benign counts per project, and the largest-project share (fraction of split rows from the single largest project).

#### Scenario: Report includes unseen project balance
- **WHEN** the split report is generated with unseen splits present
- **THEN** the report SHALL contain a section titled "Unseen Split Balance" or equivalent
- **AND** it SHALL list per-project row counts for both `unseen_val` and `unseen_test`
- **AND** it SHALL show the largest-project share as a percentage for each unseen split

---

### Requirement: Dominance warning
The split report SHALL emit a warning if any single project accounts for more than 50% of rows in an unseen split.

#### Scenario: Dominant project triggers warning
- **WHEN** one project has > 50% of `unseen_val` rows
- **THEN** the split report warnings section SHALL include a warning naming the dominant project and its share

#### Scenario: No dominant project
- **WHEN** no project exceeds 50% in any unseen split
- **THEN** no dominance warning SHALL be emitted
