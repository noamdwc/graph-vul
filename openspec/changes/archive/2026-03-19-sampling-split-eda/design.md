## Context

The `notebooks/full_dataset_eda.ipynb` notebook loads the full BigVul CSV (`data/bigvul.csv`, ~188K rows) into a pandas DataFrame `df` and currently covers sections 1–6 (overview, CVE/CWE distributions, code/description length stats, expansion candidates). The notebook uses pandas, numpy, matplotlib, and seaborn.

The downstream pipeline (`graph_vul/pipeline/extract_subset.py`) currently draws 1000 rows via simple random sampling and splits them row-level random into train/val/test. This risks both unrepresentative sampling and data leakage.

Key available columns: `func_before`, `cve_id`, `cwe_id`, `vul`, `project`, `commit_id`, `file_name`, `func_after`, plus derived columns (`code_char_len`, `code_line_count`, `code_token_count`).

## Goals / Non-Goals

**Goals:**
- Add Section 7 (subsections 7.1–7.7) to the existing notebook with no disruption to existing cells.
- Quantify group structure, label balance, leakage risks, and sampling diversity using only existing data and libraries.
- Compare 6 candidate sampling strategies and 4 candidate split strategies side-by-side.
- Produce a concrete, evidence-based recommendation for sampling and splitting.

**Non-Goals:**
- Modifying `extract_subset.py` or any pipeline code (that's a follow-up change).
- Training or evaluating any model.
- Adding new Python dependencies.
- Changing the data itself.

## Decisions

### 1. Append to `full_dataset_eda.ipynb`, not `eda.ipynb`
**Rationale:** `full_dataset_eda.ipynb` already operates on the full 188K-row BigVul dataset and covers the distributions we need to build on. `eda.ipynb` works on the 1000-row subset, which is what we're trying to improve.

### 2. Column-existence guards in every analysis block
**Rationale:** BigVul schema can vary across versions. Each code cell checks for required columns before using them and prints a skip message if missing, keeping the notebook runnable regardless of CSV version.

### 3. Helper functions defined in a single code cell at the start of Section 7
**Rationale:** Centralizing helpers (`summarize_group_sizes`, `compute_group_label_balance`, `evaluate_sample_diversity`, `simulate_group_split`, and the 6 sampler functions) in one cell avoids repetition and makes it easy to reuse or extract them later. Each subsequent subsection cell calls these helpers.

### 4. Fixed seed (42) for all random operations
**Rationale:** Reproducibility. Every sampler and split simulation uses `random_state=42` or `np.random.seed(42)`.

### 5. Compact output policy
**Rationale:** Notebook outputs must be decision-oriented. Tables are capped at top-20 rows. Plots use log scale where distributions are heavy-tailed. No raw code bodies are printed — only metadata and counts.

### 6. Sampling strategies chosen for complementary coverage
The 6 strategies (random, stratified-vul, capped-per-CVE, capped-per-CWE, hierarchical CWE→CVE→row, balanced+diversity-cap) are chosen to span the trade-off space between class balance, CVE/CWE diversity, and simplicity. The comparison table and bar charts make it easy to pick winners.

### 7. Split strategies cover the realistic options
Row-level random, group-by-CVE, group-by-project, and group-by-commit cover the practical split granularities. Each is simulated with an 80/10/10 ratio and checked for leakage and label coverage.

## Risks / Trade-offs

- **[Risk] Some columns may be absent in certain BigVul versions** → Mitigation: Every block guards on column existence and degrades gracefully.
- **[Risk] Helper functions cell becomes large** → Mitigation: Functions are small and focused; each has a clear docstring. Can be extracted to a module later.
- **[Risk] Hierarchical sampler may produce fewer than 1000 rows if CWEs have very few CVEs** → Mitigation: The sampler fills remaining budget with random rows and documents shortfall.
- **[Trade-off] Group-by-CVE split may leave some CWEs absent from val/test** → The diagnostic explicitly reports this so the user can decide whether to accept or mitigate.
