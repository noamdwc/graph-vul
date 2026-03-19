## Context

The project uses Big-Vul (`data/bigvul.csv`) as its source dataset. Currently, `extract_subset.py` filters this to 1,000 CWE-119 samples (`data/subset_cwe119.csv`), and all EDA in `notebooks/eda.ipynb` operates on that subset. The full Big-Vul CSV contains ~188K rows spanning many CWEs and CVEs. Before expanding the POC beyond CWE-119, we need to understand the full dataset's structure.

## Goals / Non-Goals

**Goals:**
- Provide a standalone notebook that answers: how many CVEs, how many CWEs, how many samples per CVE/CWE, and what the class balance looks like across the full dataset
- Identify which CVEs/CWEs have sufficient data for future modeling
- Keep the notebook lightweight — raw CSV analysis only, no embedding or graph computation

**Non-Goals:**
- Modifying any existing pipeline code or config
- Processing CPGs or embeddings for the full dataset
- Making modeling decisions (the notebook informs decisions, doesn't make them)

## Decisions

### 1. Separate notebook rather than extending existing EDA
The existing `notebooks/eda.ipynb` is focused on the CWE-119 subset and processed PyG data. The full-dataset EDA has a different scope (raw CSV, cross-CWE analysis) so a separate notebook avoids cluttering the existing one.

**Alternative considered**: Adding sections to `eda.ipynb`. Rejected because the existing notebook already has 30+ cells and mixes raw and processed data analysis.

### 2. Raw CSV only — no graph or embedding analysis
The full Big-Vul CSV has ~188K rows. Running Joern CPG extraction or DeBERTa embedding on all of them is out of scope. The notebook focuses on metadata-level analysis (counts, distributions, text lengths) which can run in seconds on the CSV alone.

### 3. Use existing config paths
The notebook imports `BIGVUL_CSV` from `graph_vul.config` to stay consistent with the project's path conventions.

## Risks / Trade-offs

- **[Big-Vul CSV not downloaded]** → The notebook depends on `data/bigvul.csv` existing. A markdown cell at the top will note the prerequisite (`dvc repro data_ingestion`).
- **[Column name drift]** → Big-Vul CSV columns may differ across versions. The notebook will inspect columns first and fail early with a clear message if expected columns are missing.
