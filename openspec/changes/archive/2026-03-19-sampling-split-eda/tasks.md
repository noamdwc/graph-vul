## 1. Setup

- [x] 1.1 Add markdown cell `## 7. Sampling and Split Design EDA` after existing Section 6 in `notebooks/full_dataset_eda.ipynb`
- [x] 1.2 Add code cell defining all helper functions: `summarize_group_sizes`, `compute_group_label_balance`, `sample_random`, `sample_stratified_vul`, `sample_capped_per_group`, `sample_hierarchical`, `sample_balanced_diversity`, `evaluate_sample_diversity`, `simulate_group_split`. Each function guards on column existence and uses seed 42.

## 2. Group Structure and Leakage (7.1)

- [x] 2.1 Add markdown cell for subsection 7.1 explaining group structure and leakage risks
- [x] 2.2 Add code cell: inspect which grouping columns exist (`cve_id`, `cwe_id`, `project`, `commit_id`, `file_name`, `func_before`, `func_after`), compute per-column unique counts, samples-per-group summary stats, and top-20 largest groups
- [x] 2.3 Add code cell: plot group-size distributions for `cve_id`, `cwe_id`, `project` with log scale
- [x] 2.4 Add code cell: duplicate analysis on `func_before` (duplicate rates, top repeated snippets by count) and conflicting-label detection if both `func_before` and `vul` exist
- [x] 2.5 Add markdown/print conclusion on leakage risk and best grouping candidates

## 3. Label Balance by Group (7.2)

- [x] 3.1 Add markdown cell for subsection 7.2 explaining label balance analysis
- [x] 3.2 Add code cell: compute per-group label balance for `cve_id`, `cwe_id`, `project` — summary stats of vul ratio, histograms, top-20 tables, all-benign/all-vul/mixed counts. Extra CWE table ranked by vulnerable count.
- [x] 3.3 Add markdown/print interpretation of which grouping levels are too skewed vs have mixed structure

## 4. Sampling Strategy Comparison (7.3)

- [x] 4.1 Add markdown cell for subsection 7.3 explaining sampling diversity diagnostics
- [x] 4.2 Add code cell: run all 6 sampling strategies (random, stratified-vul, capped-per-CVE, capped-per-CWE, hierarchical, balanced+diversity-cap), compute comparison table (total rows, vul ratio, unique CVEs/CWEs/projects, median/max rows per CVE/CWE, code length stats)
- [x] 4.3 Add code cell: bar charts comparing unique CVEs/CWEs/projects, vul ratio, and max CVE concentration across strategies
- [x] 4.4 Add markdown/print recommendation of best 1–2 strategies

## 5. Concentration and Dominance (7.4)

- [x] 5.1 Add markdown cell for subsection 7.4 explaining concentration analysis
- [x] 5.2 Add code cell: compute top-1/5/10 CVE and CWE row shares plus HHI for full dataset and each sampling strategy, display compact comparison table with interpretation

## 6. Split Design Diagnostics (7.5)

- [x] 6.1 Add markdown cell for subsection 7.5 explaining split design goals
- [x] 6.2 Add code cell: simulate 80/10/10 train/val/test splits for row-random, group-by-CVE, group-by-project, group-by-commit (where columns exist). Report split sizes, vul ratio, unique CVEs/CWEs/projects, leakage verification, rare CWE disappearance.
- [x] 6.3 Add markdown/print comparison summary: best split for leakage prevention vs label coverage, trade-offs

## 7. Outlier and Truncation Diagnostics (7.6)

- [x] 7.1 Add markdown cell for subsection 7.6 explaining outlier/truncation relevance
- [x] 7.2 Add code cell: char/line/token length percentiles (50–99.5), clipped histograms to 95th pctile, top-20 longest samples (metadata only), length stats by `vul`
- [x] 7.3 Add markdown/print note on outlier distortion and whether length-aware sampling is needed

## 8. Final Recommendation (7.7)

- [x] 8.1 Add final markdown cell summarizing the 5 key questions: main leakage risk, why random sampling fails, recommended sampling strategy, recommended split strategy, follow-up checks
- [x] 8.2 Add final code/print cell outputting a concrete "Recommended next step" paragraph in plain English
