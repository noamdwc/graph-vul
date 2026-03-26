# Multi-CWE Selection Analysis

**Date:** 2026-03-26
**Dataset:** Big-Vul (188,636 rows, 91 unique CWEs, 10,900 vulnerable functions)

## Objective

Select CWE classes for extending the binary vulnerability classifier to multi-CWE
classification. Candidates must survive the existing function-pipeline filters and
provide enough data for robust training and evaluation.

## Pipeline Filter Criteria

The function-level pipeline (`func_extract_subset.py`) applies the following filters
sequentially. Each candidate CWE is evaluated under the same chain:

| Filter | Criterion |
|--------|-----------|
| Deduplication | Remove exact-match `func_before` duplicates (SHA-256) |
| Token length | Drop functions with > 512 whitespace tokens |
| Project size | Keep only projects contributing 50-2000 rows for the given CWE |

## Raw CWE Distribution (Top 30)

Before any filtering, sorted by total rows:

| CWE | Total | Vul | Benign | Vul % | Projects | CVEs |
|---------|-------|------|--------|-------|----------|------|
| CWE-119 | 26,497 | 2,127 | 24,370 | 8.0 | 124 | 637 |
| CWE-20 | 20,501 | 1,142 | 19,359 | 5.6 | 68 | 392 |
| CWE-399 | 14,806 | 736 | 14,070 | 5.0 | 27 | 220 |
| CWE-264 | 12,487 | 508 | 11,979 | 4.1 | 29 | 185 |
| CWE-416 | 9,780 | 330 | 9,450 | 3.4 | 27 | 144 |
| CWE-200 | 8,636 | 503 | 8,133 | 5.8 | 37 | 218 |
| CWE-125 | 7,717 | 625 | 7,092 | 8.1 | 67 | 318 |
| CWE-189 | 6,964 | 337 | 6,627 | 4.8 | 40 | 137 |
| CWE-362 | 6,073 | 278 | 5,795 | 4.6 | 11 | 82 |
| CWE-476 | 4,572 | 215 | 4,357 | 4.7 | 46 | 128 |
| CWE-190 | 3,671 | 307 | 3,364 | 8.4 | 43 | 87 |
| CWE-254 | 3,000 | 123 | 2,877 | 4.1 | 12 | 31 |
| CWE-787 | 2,918 | 198 | 2,720 | 6.8 | 32 | 64 |
| CWE-284 | 2,425 | 177 | 2,248 | 7.3 | 18 | 45 |
| CWE-732 | 1,480 | 66 | 1,414 | 4.5 | 4 | 8 |
| CWE-310 | 1,478 | 94 | 1,384 | 6.4 | 12 | 35 |
| CWE-400 | 1,282 | 48 | 1,234 | 3.7 | 12 | 25 |
| CWE-772 | 1,205 | 46 | 1,159 | 3.8 | 7 | 39 |
| CWE-59 | 1,049 | 52 | 997 | 5.0 | 12 | 17 |
| CWE-617 | 1,026 | 22 | 1,004 | 2.1 | 8 | 14 |
| CWE-415 | 993 | 81 | 912 | 8.2 | 16 | 29 |
| CWE-269 | 930 | 38 | 892 | 4.1 | 6 | 7 |
| CWE-404 | 916 | 62 | 854 | 6.8 | 2 | 5 |
| CWE-704 | 908 | 24 | 884 | 2.6 | 6 | 14 |
| CWE-79 | 877 | 55 | 822 | 6.3 | 3 | 24 |
| CWE-835 | 864 | 34 | 830 | 3.9 | 14 | 24 |
| CWE-369 | 740 | 34 | 706 | 4.6 | 12 | 22 |
| CWE-19 | 723 | 55 | 668 | 7.6 | 4 | 14 |
| CWE-22 | 690 | 35 | 655 | 5.1 | 12 | 19 |
| CWE-17 | 684 | 51 | 633 | 7.5 | 7 | 21 |

## After Pipeline Filters

CWEs with >= 1,000 rows surviving dedup + token + project filters:

| CWE | Raw | After Dedup | After Token | Final | Vul | Benign | Vul % | Projects | CVEs |
|---------|--------|-------------|-------------|-------|-----|--------|-------|----------|------|
| CWE-119 | 26,497 | 26,497 | 25,678 | 8,433 | 526 | 7,907 | 6.2 | 51 | 249 |
| CWE-125 | 7,717 | 7,717 | 7,298 | 6,610 | 395 | 6,215 | 6.0 | 30 | 232 |
| CWE-20 | 20,501 | 20,501 | 20,088 | 5,944 | 321 | 5,623 | 5.4 | 28 | 141 |
| CWE-476 | 4,572 | 4,572 | 4,449 | 3,960 | 160 | 3,800 | 4.0 | 20 | 94 |
| CWE-189 | 6,964 | 6,964 | 6,795 | 3,633 | 183 | 3,450 | 5.0 | 13 | 62 |
| CWE-190 | 3,671 | 3,671 | 3,545 | 2,965 | 238 | 2,727 | 8.0 | 14 | 52 |
| CWE-254 | 3,000 | 3,000 | 2,964 | 2,839 | 95 | 2,744 | 3.3 | 6 | 23 |
| CWE-264 | 12,487 | 12,487 | 12,344 | 2,715 | 138 | 2,577 | 5.1 | 13 | 56 |
| CWE-200 | 8,636 | 8,636 | 8,420 | 2,403 | 119 | 2,284 | 5.0 | 12 | 49 |
| CWE-787 | 2,918 | 2,918 | 2,846 | 2,366 | 114 | 2,252 | 4.8 | 9 | 28 |
| CWE-362 | 6,073 | 6,073 | 5,973 | 2,212 | 105 | 2,107 | 4.7 | 5 | 19 |
| CWE-284 | 2,425 | 2,425 | 2,394 | 2,148 | 148 | 2,000 | 6.9 | 5 | 31 |
| CWE-416 | 9,780 | 9,780 | 9,673 | 1,628 | 44 | 1,584 | 2.7 | 11 | 27 |
| CWE-399 | 14,806 | 14,806 | 14,600 | 1,448 | 88 | 1,360 | 6.1 | 11 | 49 |
| CWE-732 | 1,480 | 1,480 | 1,478 | 1,409 | 57 | 1,352 | 4.0 | 2 | 6 |
| CWE-310 | 1,478 | 1,478 | 1,414 | 1,288 | 69 | 1,219 | 5.4 | 6 | 26 |
| CWE-400 | 1,282 | 1,282 | 1,255 | 1,154 | 35 | 1,119 | 3.0 | 5 | 14 |
| CWE-772 | 1,205 | 1,205 | 1,159 | 1,143 | 28 | 1,115 | 2.4 | 6 | 36 |

**Combined total: 54,298 rows, 2,863 vulnerable.**

## Cross-CWE Overlap

Zero functions appear under multiple CWE labels in the dataset. Each CWE's sample
pool is fully disjoint — no deduplication across CWEs is needed.

## Exclusion Rationale

Several high-volume CWEs were excluded from the final selection despite surviving
the pipeline filters:

| CWE | Reason |
|---------|--------|
| CWE-399 | Deprecated catch-all ("Resource Management Errors"). Vague semantics — would add label noise. |
| CWE-264 | Deprecated catch-all ("Permissions, Privileges, and Access Controls"). Superseded by CWE-284 and children. |
| CWE-189 | Deprecated catch-all ("Numeric Errors"). Superseded by specific CWEs like CWE-190, CWE-369. |
| CWE-254 | Deprecated category ("7PK - Security Features"). Not a concrete vulnerability type. |
| CWE-362 | Race condition — only 5 projects. Risk of project-level leakage across train/val/test splits. |
| CWE-284 | "Improper Access Control" — only 5 projects. Same leakage concern. Overlaps semantically with CWE-264. |
| CWE-732 | Only 2 projects, 6 CVEs. Far too concentrated for reliable cross-project evaluation. |
| CWE-310 | Deprecated ("Cryptographic Issues"). Only 6 projects. |
| CWE-400 | "Uncontrolled Resource Consumption" — only 5 projects, 3.0% vul rate. |
| CWE-772 | "Missing Release of Resource after Effective Lifetime" — only 6 projects, 2.4% vul rate. |

## Selected CWEs

The following 8 CWEs are recommended for multi-class classification:

| CWE | Name | Final Rows | Vul | Vul % | Projects | CVEs |
|---------|-------------------------------|-----------|-----|-------|----------|------|
| CWE-119 | Buffer Overflow | 8,433 | 526 | 6.2 | 51 | 249 |
| CWE-125 | Out-of-bounds Read | 6,610 | 395 | 6.0 | 30 | 232 |
| CWE-20 | Improper Input Validation | 5,944 | 321 | 5.4 | 28 | 141 |
| CWE-476 | NULL Pointer Dereference | 3,960 | 160 | 4.0 | 20 | 94 |
| CWE-190 | Integer Overflow/Wraparound | 2,965 | 238 | 8.0 | 14 | 52 |
| CWE-787 | Out-of-bounds Write | 2,366 | 114 | 4.8 | 9 | 28 |
| CWE-200 | Information Exposure | 2,403 | 119 | 5.0 | 12 | 49 |
| CWE-416 | Use After Free | 1,628 | 44 | 2.7 | 11 | 27 |

**Combined total: 34,309 rows, 1,917 vulnerable.**

### Selection Criteria

1. **Data volume** — all selected CWEs have >= 1,600 rows after pipeline filtering,
   sufficient for train/val/test splits.
2. **Project diversity** — all have >= 9 distinct projects, reducing the risk of
   data leakage in project-grouped splits.
3. **Semantic distinctness** — each CWE represents a genuinely different vulnerability
   class. No deprecated catch-all categories.
   - Memory safety: CWE-119 (buffer overflow), CWE-125 (OOB read), CWE-787 (OOB write)
   - Lifetime: CWE-416 (use-after-free)
   - Type safety: CWE-190 (integer overflow)
   - Null safety: CWE-476 (null deref)
   - Input validation: CWE-20
   - Information flow: CWE-200
4. **No cross-CWE sample overlap** — verified that zero functions appear under
   multiple CWE labels in the dataset.
5. **CWE-416 included despite lower vul rate (2.7%)** — use-after-free is a critical,
   well-defined vulnerability class that is important for real-world applicability.

### Notes

- CWE descriptions for the 7 new CWEs (all except CWE-119) need to be added to
  `data/cwe_descriptions.json` before the CWE embedding pipeline stage can run.
- The project-size filter thresholds (50-2000) may need revisiting when filtering
  across all 8 CWEs simultaneously, since project row counts will change when
  pooling multiple CWEs together.
