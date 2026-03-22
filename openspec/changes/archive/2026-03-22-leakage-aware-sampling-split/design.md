## Context

The graph-vul project detects CWE-119 vulnerabilities using GATv2 over Code Property Graphs fused with CVE text embeddings. The current pipeline samples 1500 vulnerable + 1500 benign rows randomly from Big-Vul CWE-119 data (26,497 rows, 637 CVEs, 124 projects), then splits via naive row-level stratified sampling. EDA showed this creates leakage: same project's coding style appears in both train and eval sets. The `project` column isn't even preserved in the subset CSV.

Key dataset fact: in CWE-119, each CVE maps to exactly one project. Chrome dominates with 8,967 rows across many CVEs. Top CVEs have 400–670 rows each.

## Goals / Non-Goals

**Goals:**
- Produce a ~3,000-row subset using 7–12 CVEs with project and metadata diversity
- Support configurable benign oversampling (e.g., 2:1 benign:vulnerable)
- Guarantee zero project overlap between train and val/test splits
- Emit diagnostic artifacts (manifest, reports) proving split properties
- Add test coverage for sampling and splitting logic
- Integrate cleanly into DVC

**Non-Goals:**
- Changing the model architecture or training loop
- Supporting CWEs other than CWE-119 (future work)
- Automated hyperparameter tuning of sampling parameters
- Real-time or online sampling — this is a batch pipeline stage

## Decisions

### 1. Subset sampling: greedy CVE selection with diversity scoring

**Choice:** Score candidate CVEs on available vulnerable rows, benign rows, project diversity relative to already-selected CVEs, and metadata spread. Greedily select 7–12 CVEs.

**Why not random CVE selection?** Random selection risks picking 10 Chrome CVEs, giving no project diversity. Greedy selection with project-diversity bonus ensures we cover multiple projects.

**Why not exhaustive optimization?** With 637 candidate CVEs and a target of ~10, greedy with diversity scoring is fast and good enough. ILP or combinatorial search adds complexity for marginal gain.

### 2. Split unit: entire projects, not CVE groups

**Choice:** Splits are project-disjoint. All rows from a held-out project go to val/test; no project appears in both train and eval.

**Why not CVE-level split?** Since each CVE maps to one project, CVE-level splits could still leak project signals (e.g., two Chrome CVEs in different splits leak Chrome coding style). Project-level is the correct leakage boundary.

**Alternative considered:** CVE-level split with project overlap allowed. Rejected because project is the primary leakage vector (shared style, patterns, build system).

### 3. Held-out project selection: prefer smaller/mid-size projects

**Choice:** Select 2–4 held-out projects from smaller projects (not Chrome/linux) that have both vulnerable and benign samples and reasonable size for evaluation.

**Why?** Holding out Chrome would remove the largest data source from training. Holding out small projects with 1–2 CVEs gives us unseen-project evaluation without gutting training data.

### 4. Benign oversampling at subset stage, not split stage

**Choice:** The subset itself is constructed with more benign than vulnerable samples (configurable ratio). The split preserves whatever ratio the subset has.

**Why?** The training objective benefits from seeing more benign examples (the real-world ratio is ~16:1 benign:vulnerable). Applying oversampling at subset stage means the entire pipeline after subset (embedding, caching) works on the actual training data. No runtime resampling needed.

### 5. Metadata buckets for sampling diversity

**Choice:** Derive code_char_len, code_line_count, code_token_count from func_before. Bucket into quartiles (short/medium/long/very_long). Sample across buckets within each CVE to avoid length bias.

**Why not ignore length?** EDA showed heavy-tailed length distributions. Without length awareness, random sampling over-represents short functions (more numerous) or accidentally concentrates on one length range.

### 6. Output format: CSV with project column + JSON manifest + MD report

**Choice:** Subset CSV adds `project` column. Manifest JSON captures selection metadata. Report MD summarizes decisions for human review.

**Why JSON manifest?** Machine-readable record of which CVEs were selected, why, and what caps were applied. Useful for reproducibility auditing beyond the seed.

### 7. Split assignments as CSV instead of per-split JSON index files

**Choice:** Single `assignments.csv` with columns `sample_id, split` replaces three JSON files (`train.json`, `val.json`, `test.json`).

**Why?** Easier to join with subset metadata for diagnostics. Single file is simpler to manage in DVC. The dataset class can filter by split column.

## Risks / Trade-offs

**[Small eval sets from strict project disjointness]** → If held-out projects are small, val/test may have few samples. Mitigation: subset stage deliberately selects CVEs from projects with enough data for evaluation (minimum thresholds in config).

**[Project diversity limited by CWE-119]** → CWE-119 is dominated by Chrome and linux. With only 124 projects, strict project disjointness limits diversity. Mitigation: cap per-project rows so no single project dominates; accept that some project concentration is inherent in the data.

**[Benign oversampling may not match desired ratio exactly]** → Some CVEs have few benign rows. Mitigation: graceful degradation — sample what's available, warn if target ratio is infeasible, report achieved vs. requested ratio.

**[Downstream path changes]** → Renaming `subset_cwe119.csv` → `subset.csv` requires updating joern_export, embed_code, embed_cve, and the dataset class. Mitigation: update all references in config.py and DVC; small mechanical change.
