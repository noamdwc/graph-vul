## Why

The current subset sampling (random per-class) and split (naive row-level stratified) pipeline ignores project and CVE structure, creating data leakage. EDA confirmed that project is a leakage source (shared coding style, repository signals) and CVE groups cluster related vulnerability instances. Random row splits allow the same project+CVE group to appear in both train and evaluation, inflating metrics. We need a leakage-aware, metadata-aware pipeline that produces trustworthy evaluation results.

## What Changes

- **Replace simple random subset sampling** with a structured sampler that selects 7–12 CVEs, uses project+CVE as grouping key, respects per-project/CVE/group caps, and supports configurable benign oversampling for training
- **Replace naive row-level split** with a project-disjoint split where held-out projects are fully excluded from training — zero project overlap between train and val/test
- **Add metadata awareness** to both subset and split stages: code length, line count, and token count buckets inform sampling diversity and split stratification
- **Add the `project` column** to the subset CSV output (currently missing)
- **Add subset manifest and reports** (JSON manifest, markdown report) as pipeline artifacts
- **Add split diagnostics** (assignments CSV, markdown report with overlap checks and distribution summaries)
- **Expose all sampling/split controls** as config constants (target counts, caps, held-out project count, ratios, seed)
- **Add tests** for subset and split logic (none exist currently)
- **Update DVC pipeline** to wire the new stages and artifacts
- **Update README** documenting the leakage-aware design

## Capabilities

### New Capabilities
- `structured-subset-sampling`: Metadata-aware, CVE-diverse, project-capped subset sampling with configurable benign oversampling. Produces subset CSV, manifest JSON, and report MD.
- `leakage-aware-split`: Project-disjoint train/val/test split with held-out projects for evaluation. Produces split assignments CSV and diagnostics report.

### Modified Capabilities
- `data-pipeline`: Subset stage output changes (new columns, new artifacts), split stage moves after embedding and uses new logic
- `dvc-pipeline`: New stage outputs and dependencies for subset manifest/report and split assignments/report
- `data-split`: Entirely replaced by `leakage-aware-split` — old naive row-level split is removed

## Impact

- `graph_vul/config.py` — new constants for sampling targets, caps, held-out config, paths
- `graph_vul/pipeline/extract_subset.py` — full rewrite
- `graph_vul/pipeline/split_data.py` — full rewrite
- `dvc.yaml` — updated stage definitions, new outputs
- `data/subset_cwe119.csv` — replaced by `data/subset.csv` (new schema with `project` column)
- `data/splits/` — new `assignments.csv` replaces per-split JSON index files
- `tests/` — new directory with test modules
- `README.md` — updated pipeline documentation
- Downstream stages (joern_export, embed_code, embed_cve, train) need minor path updates for renamed subset CSV
