### Requirement: Configurable sampling targets
The subset sampler SHALL accept `target_vulnerable_samples` and `target_benign_samples` as configuration parameters. The sampler SHALL support benign counts larger than vulnerable counts for training-oriented oversampling. Default values SHALL be defined in `config.py`.

#### Scenario: Benign oversampling with 2:1 ratio
- **WHEN** target_vulnerable_samples=1000 and target_benign_samples=2000
- **THEN** the sampler produces approximately 1000 vulnerable and 2000 benign rows, totaling ~3000

#### Scenario: Infeasible benign target
- **WHEN** the requested benign count exceeds available benign rows for selected CVEs
- **THEN** the sampler samples all available benign rows, logs a warning with requested vs achieved counts, and completes without error

---

### Requirement: CVE selection with diversity scoring
The sampler SHALL select between `MIN_CVES` and `MAX_CVES` CVEs from the CWE-119 pool. Selection SHALL use a greedy algorithm that scores candidates on: available vulnerable row count, available benign row count, project diversity relative to already-selected CVEs, and metadata spread (code length distribution).

#### Scenario: CVE count within configured range
- **WHEN** the sampler runs with MIN_CVES=7 and MAX_CVES=12
- **THEN** the output subset contains rows from at least 7 and at most 12 distinct CVEs

#### Scenario: Project diversity in CVE selection
- **WHEN** multiple candidate CVEs exist from the same project (e.g., Chrome)
- **THEN** the selection algorithm penalizes additional CVEs from already-represented projects, preferring CVEs from new projects

#### Scenario: Held-out candidate inclusion
- **WHEN** the sampler selects CVEs
- **THEN** at least 2 selected CVEs SHALL come from smaller/mid-size projects suitable for held-out evaluation (not from the top-2 largest projects by row count)

---

### Requirement: Per-group caps
The sampler SHALL enforce configurable caps: `MAX_ROWS_PER_PROJECT`, `MAX_ROWS_PER_CVE`, and `MAX_ROWS_PER_GROUP` (project+CVE combination). No single project, CVE, or group SHALL exceed its cap in the output subset.

#### Scenario: Project cap enforced
- **WHEN** a project has 5000 available rows and MAX_ROWS_PER_PROJECT=500
- **THEN** at most 500 rows from that project appear in the subset

#### Scenario: CVE cap enforced
- **WHEN** a CVE has 670 available rows and MAX_ROWS_PER_CVE=500
- **THEN** at most 500 rows from that CVE appear in the subset

#### Scenario: Group cap enforced
- **WHEN** a project+CVE group has 400 rows and MAX_ROWS_PER_GROUP=300
- **THEN** at most 300 rows from that group appear in the subset

---

### Requirement: Metadata-aware sampling within groups
The sampler SHALL derive metadata columns from `func_before`: `code_char_len`, `code_line_count`, `code_token_count`. It SHALL bucket these into quartile-based categories. Within each project+CVE group, sampling SHALL draw from across metadata buckets to avoid length bias.

#### Scenario: Length buckets used in sampling
- **WHEN** sampling within a CVE group
- **THEN** the sampled rows span multiple length buckets rather than concentrating in one

#### Scenario: Missing metadata columns handled gracefully
- **WHEN** `func_before` is empty or null for some rows
- **THEN** those rows are assigned a default metadata bucket and sampling proceeds without error

---

### Requirement: Exact code deduplication
When `DEDUPLICATE_CODE` is True, the sampler SHALL remove rows with identical `func_before` text before sampling, keeping one representative per unique code string.

#### Scenario: Duplicates removed
- **WHEN** DEDUPLICATE_CODE=True and two rows have identical func_before
- **THEN** only one of the rows is eligible for sampling

#### Scenario: Deduplication disabled
- **WHEN** DEDUPLICATE_CODE=False
- **THEN** duplicate code strings are allowed in the output

---

### Requirement: Reproducible sampling with seed
The sampler SHALL use the configured `SEED` value for all random operations. Given identical input data and configuration, the sampler SHALL produce identical output.

#### Scenario: Deterministic output
- **WHEN** the sampler runs twice with the same seed and input
- **THEN** the output subset is identical

---

### Requirement: Project column preserved in output
The subset CSV SHALL include a `project` column containing the source project for each row, in addition to existing columns: `sample_id`, `func_before`, `cve_id`, `cve_desc`, `vul`.

#### Scenario: Project column present
- **WHEN** the subset CSV is written
- **THEN** it contains columns: sample_id, project, func_before, cve_id, cve_desc, vul

---

### Requirement: Subset manifest output
The sampler SHALL produce a JSON manifest file containing: selected CVEs with per-CVE metadata (project, row counts by class), selection rationale scores, configuration used, achieved vs requested sample counts, per-project totals, per-group totals, and metadata bucket distributions.

#### Scenario: Manifest written
- **WHEN** the subset stage completes
- **THEN** a JSON manifest file exists at the configured path with all required fields

---

### Requirement: Subset report output
The sampler SHALL produce a Markdown report summarizing: chosen CVEs and projects, class counts, per-CVE counts, per-project counts, metadata bucket distributions, benign oversampling details (requested vs achieved), and selection rationale.

#### Scenario: Report written
- **WHEN** the subset stage completes
- **THEN** a Markdown report file exists at the configured path with human-readable summary tables
