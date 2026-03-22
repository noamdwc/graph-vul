## 1. Config Updates

- [x] 1.1 Add subset sampling constants to `config.py`: TARGET_VULNERABLE_SAMPLES, TARGET_BENIGN_SAMPLES, MIN_CVES, MAX_CVES, MAX_ROWS_PER_PROJECT, MAX_ROWS_PER_CVE, MAX_ROWS_PER_GROUP, DEDUPLICATE_CODE
- [x] 1.2 Add split constants to `config.py`: HELD_OUT_PROJECT_COUNT
- [x] 1.3 Add new path constants to `config.py`: SUBSET_CSV (renamed), SUBSET_MANIFEST, SUBSET_REPORT, SPLIT_ASSIGNMENTS, SPLIT_REPORT
- [x] 1.4 Remove old TARGET_SAMPLES constant, update SUBSET_CSV path from `subset_cwe119.csv` to `subset.csv`

## 2. Subset Sampling Implementation

- [x] 2.1 Implement metadata derivation: compute code_char_len, code_line_count, code_token_count from func_before; bucket into quartile-based categories
- [x] 2.2 Implement exact code deduplication (controlled by DEDUPLICATE_CODE flag)
- [x] 2.3 Implement CVE diversity scoring: score candidates on vul/benign counts, project diversity vs already-selected CVEs, metadata spread; filter CVEs with too few vulnerable samples
- [x] 2.4 Implement greedy CVE selection loop (7–12 CVEs) with project diversity bonus and held-out candidate inclusion (at least 2 CVEs from smaller projects)
- [x] 2.5 Implement budget allocation across selected CVEs with per-CVE and per-project caps
- [x] 2.6 Implement class-aware, metadata-aware row sampling within each CVE group, respecting per-group caps and benign oversampling ratio
- [x] 2.7 Implement graceful degradation when requested benign/vulnerable targets are infeasible (log warnings, sample what's available)
- [x] 2.8 Output subset CSV with columns: sample_id, project, func_before, cve_id, cve_desc, vul
- [x] 2.9 Generate and write subset manifest JSON (selected CVEs, scores, config, per-project/CVE/group counts, metadata distributions)
- [x] 2.10 Generate and write subset report MD (human-readable summary tables of CVEs, projects, class counts, metadata buckets, oversampling details)

## 3. Split Implementation

- [x] 3.1 Implement held-out project selection: choose HELD_OUT_PROJECT_COUNT projects with both classes and sufficient size, preferring smaller/mid-size projects
- [x] 3.2 Implement project-disjoint split assignment: all held-out project rows → val/test, remaining → train
- [x] 3.3 Implement val/test distribution of held-out projects (split roughly equally, different projects to different splits when feasible)
- [x] 3.4 Implement split validation: verify zero project overlap, all rows assigned, warn on skewed splits
- [x] 3.5 Output split assignments CSV (sample_id, split)
- [x] 3.6 Generate and write split diagnostics report MD (sizes, class ratios, CVE/project counts per split, overlap verification, held-out summary, metadata distributions, warnings)

## 4. DVC Pipeline Updates

- [x] 4.1 Update extract_subset stage in dvc.yaml: new command args, add manifest JSON and report MD as outputs
- [x] 4.2 Update split_data stage in dvc.yaml: new outputs (assignments CSV, report), dependencies on subset CSV
- [x] 4.3 Update all downstream stages (joern_export, embed_code, embed_cve) to reference `data/subset.csv` instead of `data/subset_cwe119.csv`
- [x] 4.4 Update train stage if it references old split JSON files

## 5. Dataset/Downstream Updates

- [x] 5.1 Update `cpg_dataset.py` to read split assignments CSV instead of per-split JSON index files
- [x] 5.2 Update `train.py` if it directly references split JSON files or old subset path
- [x] 5.3 Update any other pipeline modules referencing old SUBSET_CSV path

## 6. Tests

- [x] 6.1 Create `tests/` directory with `__init__.py` and `conftest.py` (shared fixtures for synthetic BigVul-like DataFrames)
- [x] 6.2 Write subset sampling tests: reproducibility with seed, class-aware sampling, benign oversampling, CVE count within range, per-project/CVE/group caps respected, metadata buckets used, deduplication, graceful degradation on infeasible targets
- [x] 6.3 Write split tests: train/val/test all exist, held-out projects absent from train, zero project overlap, all rows assigned, deterministic with seed, warning on skewed splits

## 7. Documentation

- [x] 7.1 Update README.md: describe leakage-aware pipeline, explain why project is a leakage source, document new stages and artifacts, update usage examples
