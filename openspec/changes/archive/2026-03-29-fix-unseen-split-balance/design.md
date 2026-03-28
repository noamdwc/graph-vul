## Context

The unseen split pipeline in `split_data.py` uses `_greedy_project_split()` to allocate whole projects to `unseen_val` and `unseen_test`. The scoring function heavily weights row-count targets (weight 100.0), with vulnerability ratio (8.0), CWE distribution (3.0), and code buckets (1.0) as secondary objectives. There is no project-count term. Combined with largest-first ordering, this causes a few large projects to fill `unseen_val`'s row quota quickly, leaving it with only ~7 projects while `unseen_test` gets many more.

The existing `_validate_splits()` checks row counts and project overlap but does not validate project-count balance or per-project dominance within unseen splits.

## Goals / Non-Goals

**Goals:**
- Add a project-count balance penalty to the greedy allocator's scoring function, active for unseen splits
- Add hard validation for minimum project counts and vulnerable rows in unseen splits
- Extend the split report with unseen-specific balance diagnostics
- Save an audit CSV with per-project unseen split statistics
- Keep the change surgical — only modify what's needed in the existing pipeline

**Non-Goals:**
- Redesign the greedy allocator or the overall split strategy
- Change how the seen pool (train/val/test) is split
- Change how the unseen pool is formed (held-out CWE routing)
- Guarantee exact 50/50 project counts (the allocator should balance, not force equality)

## Decisions

### 1. Add project-count penalty to `score_state()` — not a separate pass

**Choice:** Add a `project_count_penalty` term inside the existing `score_state()` function, weighted by a named constant `UNSEEN_PROJECT_COUNT_PENALTY = 20.0` (between row-count 100.0 and vulnerability 8.0).

**Why:** A single-pass approach keeps the allocator simple and deterministic. The penalty computes `(actual_project_count / target_project_count - 1.0)^2` per split. Weight 20.0 is high enough to prevent extreme imbalance but won't override row-count optimization for most inputs.

**Alternative considered:** Post-hoc rebalancing pass that moves projects until counts are balanced. Rejected because the local refinement loop already does this — adding the penalty to the score function lets refinement naturally optimize for project counts too.

### 2. Pass project-count targets via a new parameter

**Choice:** Add an optional `project_count_targets: dict[str, int] | None` parameter to `_greedy_project_split()`. When provided, the scoring function includes the project-count penalty. When `None`, behavior is unchanged (backwards compatible for seen splits).

**Why:** The seen-pool split doesn't need project-count balancing (it uses StratifiedGroupKFold). Making the parameter optional keeps seen-split behavior identical.

### 3. Compute targets from total project count × split fractions

**Choice:** `_assign_unseen_splits()` computes targets as:
- `target_unseen_val = max(1, round(n_projects * UNSEEN_VAL_FRAC))`
- `target_unseen_test = n_projects - target_unseen_val`

This ensures the two targets always sum to `n_projects` exactly, and `unseen_val` always gets at least 1 project.

**Why:** Simple, proportional, consistent with how row targets are already computed. The `max(1, ...)` guard and remainder-based `unseen_test` target avoid rounding inconsistencies.

### 4. Hard validation thresholds as module-level constants

**Choice:** Add `MIN_UNSEEN_PROJECTS = 8` and `MIN_UNSEEN_VUL_ROWS = 10` as module-level constants. Validation is relaxed when the total unseen pool is too small (< 2 × threshold for projects).

**Why:** Module-level constants are consistent with existing config pattern (`UNSEEN_VAL_FRAC`, `VAL_SPLIT`, etc.). Relaxation prevents false failures on small datasets.

### 5. Audit CSV as a separate output file

**Choice:** Write `unseen_audit.csv` from `split_data()` after assignments are complete, using a simple groupby on the merged DataFrame.

**Why:** Keeps the audit artifact independent of the report generator. Easy to load in notebooks for post-hoc analysis.

## Risks / Trade-offs

- **[Existing split results change]** → Expected and desired. The DVC pipeline will produce different assignments on re-run. Mitigated by the deterministic seed — results are still reproducible.
- **[Project-count penalty conflicts with row-count penalty]** → On highly skewed project-size distributions, the allocator may compromise row balance to improve project count. Mitigated by keeping project-count weight (20.0) well below row-count weight (100.0), so row balance remains dominant.
- **[Hard validation may fail on small datasets]** → Mitigated by proportional relaxation when the unseen pool is small. Warnings are emitted instead of hard failures when relaxation kicks in.
