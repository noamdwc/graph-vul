## Context

The pipeline currently hard-codes `CWE_ID = "CWE-119"` in `config.py` and filters all data to this single CWE in `func_extract_subset.py`. The split stage (`split_data.py`) uses `StratifiedGroupKFold` stratified on the binary `vul` label and grouped by `project`.

We now want to support 8 CWEs, with 2 held out for zero-shot CWE generalization testing.

Selected CWEs (from `docs/cwe_selection_analysis.md`):
- Train/val/test: CWE-119, CWE-125, CWE-20, CWE-476, CWE-190, CWE-787
- Held-out (test only): CWE-200, CWE-416

## Goals / Non-Goals

**Goals:**
- Extract a multi-CWE subset using the same filter chain (dedup, token length, project size)
- Held-out CWEs appear exclusively in the test split, enabling zero-shot CWE evaluation
- Add CWE descriptions for all 8 CWEs to the descriptions JSON
- Maintain backward compatibility of the output CSV schema (same columns, `cwe_id` already present)

**Non-Goals:**
- Changing the model architecture (stays binary vul/benign for now)
- Changing the embedding pipeline (UniXcoder embedding stage is CWE-agnostic)
- Changing training loss or evaluation metrics
- Balancing row counts across CWEs (natural distribution is fine)

## Decisions

### D1: Per-CWE project filtering

Apply the project-size filter (50–2000 rows) **per CWE within each project**, not across all CWEs combined. A project with 30 CWE-119 rows and 80 CWE-125 rows would exclude its CWE-119 rows but keep CWE-125.

**Rationale:** The filter exists to remove projects that are too small (noisy) or too large (dominating). These properties are CWE-specific — a project's buffer-overflow profile is independent of its null-deref profile.

**Alternative considered:** Global row count across all CWEs per project. Rejected because it conflates unrelated vulnerability classes and would admit small-CWE contributions that are individually too sparse.

### D2: Held-out CWE selection — CWE-200 and CWE-416

CWE-200 (Information Exposure) and CWE-416 (Use After Free) are held out because:
- **CWE-200** is semantically distinct from all training CWEs (information flow vs. memory/type/input bugs). Tests whether the model generalizes beyond memory-safety.
- **CWE-416** is memory-safety related but mechanistically different from CWE-119/125/787 (temporal vs. spatial). Tests within-domain generalization.
- Both have reasonable post-filter data (2,403 and 1,628 rows) for reliable evaluation.

**Alternative considered:** Holding out CWE-476 (null deref). Rejected because it has the most data of the mid-tier CWEs (3,960 rows) — more valuable in training.

### D3: Split strategy for held-out CWEs

Held-out CWE rows are appended directly to the test split after the normal `StratifiedGroupKFold` split runs on the 6 training CWEs. They do not participate in the stratification or group-balancing of train/val/test.

**Rationale:** Keeps the existing split logic unchanged for training CWEs. Held-out rows are purely additive to test. The split report will clearly mark which test rows are from held-out CWEs.

### D4: Config representation

Replace `CWE_ID: str` with two lists:
```python
CWE_IDS = ["CWE-119", "CWE-125", "CWE-20", "CWE-476", "CWE-190", "CWE-787"]
HELDOUT_CWE_IDS = ["CWE-200", "CWE-416"]
```

The old `CWE_ID` is removed entirely (not deprecated). All references updated.

### D5: CWE descriptions source

CWE descriptions are sourced from the official MITRE CWE database. Each entry in `cwe_descriptions.json` is the CWE's official "Description" field — a 1–3 sentence summary of the weakness.

## Risks / Trade-offs

- **Class imbalance across CWEs** — CWE-119 has 5x more rows than CWE-416. The model sees an uneven CWE distribution. → Acceptable for now; the model predicts vul/benign not CWE class. Future work can add CWE-weighted sampling if needed.
- **Project overlap across CWEs** — A single project (e.g., linux) may contribute rows to multiple CWEs, which is correct and expected. The split ensures project-disjointness within the training CWEs. → Held-out CWE rows may share projects with training CWEs, but since the CWE itself is unseen, this is acceptable for generalization testing.
- **Held-out CWE evaluation is test-only** — No validation set for held-out CWEs, so no hyperparameter tuning on them. → This is by design; held-out means truly unseen.
