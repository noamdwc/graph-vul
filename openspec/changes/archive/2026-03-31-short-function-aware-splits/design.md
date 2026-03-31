## Context

The split pipeline (`graph_vul/pipeline/split_data.py`) uses a greedy project-level allocator to assign projects to train/val/test/unseen_val/unseen_test. It currently derives length buckets via `pd.qcut` with quartile-based labels (short/medium/long/very_long), and the greedy scorer treats bucket balance as a soft, low-weight objective (weight 1.0 vs 100.0 for row targets). This means there is no guarantee that val/test contain enough short functions (0–64 tokens) for stable evaluation — a bucket that matters most for vulnerability detection.

Key constraints:
- Projects are atomic: all rows in a project go to the same split. We cannot cherry-pick individual short functions.
- The held-out CWE / unseen pool logic must remain unchanged.
- Downstream training code must not change.

## Goals / Non-Goals

**Goals:**
- Replace qcut-based token bucketing with fixed-boundary buckets (0–64, 64–128, 128–256, 256–512, 512+) so bucket definitions are stable across datasets.
- Ensure val and test each contain configurable minimum counts of short-function (0–64 token) vulnerable and benign samples.
- Make the greedy allocator actively prefer assignments that satisfy short-bucket minimums, not just match proportions.
- Add post-split validation and reporting that proves short-function evaluation viability.

**Non-Goals:**
- Changing the overall split architecture (pools, project-disjointness, held-out CWE logic).
- Adding new output files beyond extending the existing report.
- Modifying training code or model inputs.
- Guaranteeing exact minimum counts — project-level grouping means some datasets may not have enough short-function projects to satisfy the minimums, in which case we warn rather than fail.

## Decisions

### D1: Replace qcut entirely for `code_token_count_bucket`; keep qcut for char/line buckets

**Decision:** Replace `pd.qcut` for `code_token_count_bucket` with fixed bins using `pd.cut`. Keep qcut for `code_char_len_bucket` and `code_line_count_bucket` as secondary diagnostics.

**Rationale:** The token-count bucket is the primary evaluation axis. Fixed boundaries (0–64, 64–128, 128–256, 256–512, 512+) are interpretable and stable. Char-len and line-count buckets are only used in the report, not the allocator, so qcut is fine for them.

**Alternative considered:** Replace all three. Rejected because fixed char/line boundaries would require additional calibration with no clear benefit.

### D2: Extend `_build_project_stats` to track short-bucket vulnerable/benign counts per project

**Decision:** Add two new stat columns per project: `short_vul` (vulnerable rows in 0–64 token bucket) and `short_benign` (benign rows in 0–64 bucket). These are computed alongside the existing bucket counts.

**Rationale:** The greedy allocator needs per-project short-bucket composition to evaluate whether assigning a project to val/test moves toward the minimum targets.

**Implementation:** In `_build_project_stats`, after computing bucket columns, add:
```python
short_mask = df["code_token_count_bucket"] == "0-64"
agg["short_vul"] = df[short_mask & (df["vul"] == 1)].groupby("project").size().reindex(agg["project"], fill_value=0).to_numpy()
agg["short_benign"] = df[short_mask & (df["vul"] == 0)].groupby("project").size().reindex(agg["project"], fill_value=0).to_numpy()
```

### D3: Add a short-bucket deficit penalty to the greedy scorer

**Decision:** Add a new penalty term to `score_state` and the per-project scoring loop that penalizes val/test when their short-bucket counts are below the configured minimums.

**Penalty design:**
```
for eval_split in [val, test]:
    short_vul_deficit = max(0, min_short_vul - state[split]["short_vul"])
    short_benign_deficit = max(0, min_short_benign - state[split]["short_benign"])
    short_penalty += (short_vul_deficit / min_short_vul)^2 + (short_benign_deficit / min_short_benign)^2
```

**Weight:** 15.0 — higher than bucket balance (1.0) and CWE distribution (3.0), lower than row targets (100.0). This makes it a strong preference but not an override of the fundamental row-balance constraint.

**Rationale:** A penalty-based approach integrates naturally into the existing multi-objective scorer. Hard constraints (reject assignments that don't help) would require complex feasibility checks and could prevent the allocator from finding any valid assignment. The penalty weight of 15.0 ensures the allocator actively pursues short-bucket targets without breaking row balance.

**Alternative considered:** Two-phase allocation (first satisfy short-bucket targets, then allocate remaining projects). Rejected because it fragments the optimization and can produce worse overall splits.

### D4: Track short-bucket state in the allocator

**Decision:** Add `short_vul` and `short_benign` to the allocator's state dict alongside `rows` and `vul_rows`. Pass per-project `short_vul` and `short_benign` through the same mechanism as CWE and bucket columns.

**Implementation:** These are just additional tracked columns (`short_vul::eval`, `short_benign::eval`) that follow the same pattern as `cwe::*` and `bucket::*` columns, but with a custom penalty function (D3) rather than the generic proportional penalty.

### D5: New config constants

**Decision:** Add to `config.py`:
```python
# Fixed token-count bucket boundaries
TOKEN_BUCKET_EDGES = [0, 64, 128, 256, 512]
TOKEN_BUCKET_LABELS = ["0-64", "64-128", "128-256", "256-512", "512+"]

# Minimum short-function (0-64 token) counts for evaluation splits
MIN_SHORT_VUL_PER_EVAL_SPLIT = 15
MIN_SHORT_BENIGN_PER_EVAL_SPLIT = 30

# Greedy allocator penalty weight for short-bucket deficit
SHORT_BUCKET_DEFICIT_WEIGHT = 15.0
```

**Rationale:** 15 vulnerable and 30 benign is a reasonable floor for computing metrics like precision/recall in the 0–64 bucket. These are configurable so users can adjust based on dataset size.

### D6: Post-split validation checks

**Decision:** Add a new validation function `_validate_short_bucket_coverage` called after `_validate_splits`. It:
1. Computes per-split × per-bucket × per-label counts.
2. Checks val, test, unseen_val, unseen_test against the configured minimums for the 0–64 bucket.
3. Emits warnings (not errors) when minimums are not met — hard failure would be inappropriate since project-level grouping may make it impossible.
4. Emits a separate warning when short-bucket positives are below 10 (too few for reliable recall/precision).

### D7: Report additions

**Decision:** Extend `_generate_split_report` with:
1. A "Token Length Buckets" section showing fixed-bucket distribution per split (rows × bucket × label).
2. A "Short-Function Evaluation Viability" section highlighting 0–64 bucket vul/benign counts per eval split with pass/warn status.
3. Per-project contribution to short buckets in eval splits.

These replace the current generic bucket distribution section for `code_token_count_bucket`.

### D8: Fallback behavior

**Decision:** When the allocator cannot satisfy short-bucket minimums (because the dataset simply doesn't have enough short-function projects, or they all end up in train due to row-balance pressure):
1. The penalty term naturally degrades — it pulls toward the target but doesn't force infeasible assignments.
2. Post-split validation emits explicit warnings naming the deficit.
3. The split report's viability section shows red/yellow/green status per eval split.
4. The split still completes — we never fail the pipeline for short-bucket shortfall.

## Risks / Trade-offs

**[Risk: Short-bucket penalty distorts row balance]** → The weight of 15.0 is calibrated to be meaningful but not dominant. Row-balance weight remains 100.0. In practice, moving one project from train to val to gain short functions only costs ~1-2% of row balance. Mitigation: the local refinement phase will undo any assignment that hurts overall score.

**[Risk: Very few short-function projects in dataset]** → If the dataset has fewer than MIN_SHORT_VUL_PER_EVAL_SPLIT short vulnerable functions across all projects, no allocator change can help. Mitigation: clear warnings in validation and report. The user must either lower the threshold or accept limited short-function evaluation.

**[Risk: Unseen pool has too few short functions]** → The unseen pool is small by design and constrained by held-out CWE membership. We apply the same penalty but with relaxed expectations — unseen splits get warnings, not failures. The report clearly distinguishes seen vs unseen viability.

**[Trade-off: Fixed buckets may have empty bins]** → If the dataset has no functions in the 0–64 range, the bucket exists but is empty. This is correct behavior — it surfaces the gap rather than hiding it behind qcut relabeling.
