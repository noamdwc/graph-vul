# Project-Disjoint Split Mode for PrimeVul

## Problem

PrimeVul's official train/val/test splits share 213 projects across splits (69% of all rows). This allows models to exploit project-level coding patterns (naming conventions, style, build structure) rather than learning actual vulnerability detection.

Key numbers:
- 719 total projects, 152,688 rows
- train/val overlap: 145 projects
- train/test overlap: 130 projects  
- 67 projects appear in all 3 splits (49% of data)
- Top offenders: linux (13.8K rows), qemu (4.9K), ImageMagick (4K), php-src (3.9K)

## Solution

Add a `--split-mode` CLI argument to `split_data.py`:
- **`project_disjoint`** (default): enforce that no project appears in more than one split
- **`official`**: current passthrough of PrimeVul source splits (for A/B comparison)

## Design: `project_disjoint` mode

### Algorithm

1. Load PrimeVul data with `source_split` column
2. Classify projects:
   - **Single-split projects** (506 projects, ~48K rows): appear in exactly one official split. These are anchored to their current split.
   - **Multi-split projects** (213 projects, ~105K rows): appear in 2+ official splits. These are pooled for reallocation.
3. Compute per-split row counts from anchored projects
4. Compute deficit targets: what each split still needs to reach the configured `VAL_SPLIT` (10%) and `TEST_SPLIT` (10%) fractions
5. Run `_greedy_project_split` on the pooled multi-split projects, using deficit-adjusted targets. The greedy allocator already optimizes for:
   - Row count targets (with overshoot/undershoot penalties)
   - Vulnerability ratio preservation
   - CWE coverage distribution
   - Token bucket distribution
   - Short-function (0-64 token) minimum counts in eval splits
6. Merge anchored + reallocated assignments into final `assignments.csv`
7. Proceed with existing downstream logic: unseen CWE derivation, train balancing, manifest generation, validation, report

### Why reuse the greedy allocator

The existing `_greedy_project_split` in `split_data.py` already handles all the constraints we care about. By feeding it only the conflicting projects with adjusted targets, we get project disjointness while preserving the distribution-matching properties.

## File changes

### `graph_vul/pipeline/split_data.py`

**New function: `_dejoin_official_splits(df, seed, val_frac, test_frac)`**
- Identifies single-split vs multi-split projects from `source_split`
- Anchors single-split projects
- Computes deficit targets for the greedy allocator
- Runs `_greedy_project_split` on multi-split projects
- Returns combined row-level assignments (same schema as `_materialize_official_split_assignments`)

**Modified: `split_data(input_csv, output_dir, seed, split_mode)`**
- New `split_mode` parameter (default `"project_disjoint"`)
- When `split_mode == "project_disjoint"` and `source_split` column is present:
  - Call `_dejoin_official_splits` instead of `_materialize_official_split_assignments`
- When `split_mode == "official"`:
  - Current behavior unchanged
- Pass `split_mode` to report generator

**Modified: `__main__` block**
- Add `--split-mode` argument: `choices=["project_disjoint", "official"], default="project_disjoint"`
- Pass to `split_data()`

### `dvc.yaml`

- Update `func_split` stage command to include `--split-mode project_disjoint` (explicit default for clarity)

### No changes needed

- `config.py`: reuses existing `VAL_SPLIT`, `TEST_SPLIT`, `SEED`, allocator weights
- `primevul_adapter.py`: unchanged, still loads `source_split`
- `func_extract_subset.py`: unchanged
- All validation functions: unchanged (project overlap check will now pass)
- Report generation: unchanged structure, `mode` label updates automatically
- Train balancing and manifest: unchanged
- Downstream training/eval pipeline: unchanged (reads same `assignments.csv`)

## Benchmarking

To compare project-disjoint vs official splits:

```bash
# A: project-disjoint (default)
python -m graph_vul.pipeline.split_data --input data/func_subset.csv --output data/func_splits --split-mode project_disjoint

# B: official (baseline for comparison)
python -m graph_vul.pipeline.split_data --input data/func_subset.csv --output data/func_splits --split-mode official
```

The A vs B performance gap quantifies how much project leakage inflated evaluation metrics.

## Risks and mitigations

- **Val/test shrinkage**: Naive majority assignment would gut val (-57%). The greedy allocator with explicit targets mitigates this by steering large projects toward eval splits when they're undersized.
- **Short-function sparsity**: The allocator's `SHORT_BUCKET_DEFICIT_WEIGHT` penalty ensures eval splits maintain minimum short-function vulnerable/benign counts.
- **CWE coverage loss**: The allocator tracks top-K CWE distribution per split, preventing any CWE from being entirely absent in eval.
