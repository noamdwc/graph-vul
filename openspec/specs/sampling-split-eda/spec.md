## ADDED Requirements

### Requirement: Section 7 header and helper functions cell
The notebook SHALL contain a markdown cell titled `## 7. Sampling and Split Design EDA` followed by a code cell defining all helper functions used by subsequent subsections. Helper functions SHALL include: `summarize_group_sizes`, `compute_group_label_balance`, `sample_random`, `sample_stratified_vul`, `sample_capped_per_group`, `sample_hierarchical`, `evaluate_sample_diversity`, and `simulate_group_split`.

#### Scenario: Helper functions cell executes without error
- **WHEN** the helper functions cell is executed after the existing Section 6 cells
- **THEN** all helper functions are defined and available for use by subsequent cells

#### Scenario: Missing columns do not cause errors
- **WHEN** a required column is absent from the dataframe
- **THEN** helper functions that depend on it SHALL print a clear skip message and return None or an empty result

---

### Requirement: 7.1 Group structure and potential leakage sources
The notebook SHALL inspect which grouping columns exist (`cve_id`, `cwe_id`, `project`, `commit_id`, `file_name`, `func_before`, `func_after`) and for each existing grouping column compute unique group count, samples-per-group summary stats, and top-20 largest groups. The notebook SHALL plot group-size distributions for `cve_id`, `cwe_id`, and `project` (log scale if needed). If `func_before` exists, it SHALL compute exact duplicate rates. If both `func_before` and `vul` exist, it SHALL detect conflicting labels on identical code.

#### Scenario: All grouping columns present
- **WHEN** all candidate columns exist in the dataframe
- **THEN** the cell outputs summary stats tables for each column, distribution plots, duplicate analysis, and a conclusion about leakage risk

#### Scenario: Some grouping columns missing
- **WHEN** one or more candidate columns are absent
- **THEN** the cell skips analysis for those columns with a printed message and completes without error

---

### Requirement: 7.2 Label balance by group
The notebook SHALL compute per-group label balance (total, vulnerable count, benign count, vulnerable ratio) for each existing group column among `cve_id`, `cwe_id`, `project` when `vul` exists. It SHALL produce summary stats of vulnerable ratio, histograms, top-20 tables, and counts of all-benign / all-vulnerable / mixed groups. For `cwe_id`, it SHALL also rank by vulnerable count.

#### Scenario: Label balance computed
- **WHEN** `vul` column exists along with at least one group column
- **THEN** the cell outputs histograms, tables, group purity counts, and an interpretation

#### Scenario: No vul column
- **WHEN** `vul` column is absent
- **THEN** the entire subsection is skipped with a printed message

---

### Requirement: 7.3 Diversity diagnostics for candidate sampling strategies
The notebook SHALL implement 6 sampling strategies (random, stratified-vul, capped-per-CVE, capped-per-CWE, hierarchical, balanced+diversity-cap) as functions returning at most 1000 rows with seed 42. For each strategy, it SHALL compute a comparison table with: total rows, vulnerable ratio, unique CVEs/CWEs/projects, median and max rows per CVE/CWE, and code length stats if available. It SHALL produce bar charts comparing unique CVEs/CWEs/projects, vulnerable ratio, and max CVE concentration across strategies.

#### Scenario: All strategies run successfully
- **WHEN** `cve_id`, `cwe_id`, and `vul` columns exist
- **THEN** all 6 strategies produce samples, comparison table is displayed, bar charts are rendered, and a recommendation is printed

#### Scenario: Missing columns reduce available strategies
- **WHEN** `cwe_id` is absent
- **THEN** CWE-dependent strategies are skipped and the comparison table includes only feasible strategies

---

### Requirement: 7.4 Concentration and dominance analysis
The notebook SHALL compute the share of rows covered by top-1, top-5, and top-10 CVEs (and CWEs if available) for the full dataset and each candidate sampling strategy. It SHALL optionally compute HHI for CVE and CWE concentration. Results SHALL be presented in a compact comparison table.

#### Scenario: Concentration table produced
- **WHEN** at least one sampling strategy has been computed
- **THEN** a comparison table shows concentration metrics for each strategy plus the full dataset, with an interpretation of which strategies reduce dominance

---

### Requirement: 7.5 Split design diagnostics
The notebook SHALL simulate train/val/test splits (80/10/10) for each feasible split unit (row-random, group-by-CVE, group-by-project, group-by-commit) with seed 42. For each split, it SHALL report split sizes, vulnerable ratio per split, unique CVEs/CWEs/projects per split, and verify no group overlap across splits. It SHALL report whether rare CWEs disappear from val/test.

#### Scenario: Split diagnostics completed
- **WHEN** at least `cve_id` and `vul` exist
- **THEN** the cell outputs a comparison of split strategies with leakage verification and label coverage analysis, plus a summary answering which split best prevents leakage and preserves coverage

#### Scenario: Only row-random possible
- **WHEN** no grouping columns exist
- **THEN** only row-random split is simulated, with a warning that leakage cannot be assessed

---

### Requirement: 7.6 Outlier and truncation-relevant diagnostics
The notebook SHALL compute character length percentiles (50, 75, 90, 95, 99, 99.5) for code text if available, plus line-count and token-count percentiles from existing derived columns. It SHALL show clipped histograms up to the 95th percentile, top-20 longest samples (metadata only), and length stats broken out by `vul` if available.

#### Scenario: Code length analysis produced
- **WHEN** `func_before` or `code_char_len` exists
- **THEN** percentile tables, histograms, and top-20 metadata table are displayed with a note on outlier impact

---

### Requirement: 7.7 Final recommendation cell
The notebook SHALL end with a markdown cell and a code/print cell summarizing: (1) main leakage risk, (2) why random sampling is bad, (3) recommended sampling strategy, (4) recommended split strategy, (5) follow-up checks. It SHALL print a concrete "Recommended next step" paragraph in plain English.

#### Scenario: Recommendations produced
- **WHEN** all prior subsections have executed (possibly with some skipped analyses)
- **THEN** the final cell produces a coherent summary referencing the computed evidence, even if some analyses were skipped
