## Context

`graph_vul/pipeline/func_extract_subset.py` currently combines three concerns:

1. Building a clean function-level subset from `data/bigvul.csv`
2. Encoding a chosen train-vs-held-out CWE policy through `CWE_IDS` and `HELDOUT_CWE_IDS`
3. Enforcing both minimum and maximum `(project, cwe_id)` group sizes

That coupling made sense for the first multi-CWE experiment, but it is now in the way. The next project step is dataset analysis followed by split design, so the subset stage needs to produce the full labeled, post-cleanup universe rather than a pre-curated training universe.

The current implementation already has useful structure worth preserving:
- metadata derivation from `func_before`
- exact-code deduplication with stable first-row retention
- token-length filtering at `FUNC_MAX_WHITESPACE_TOKENS`
- max-size protection on `(project, cwe_id)` buckets
- manifest and markdown report generation

The design therefore targets a focused refactor: remove split policy and minimum-support filtering from subset extraction while keeping the cleanup flow, output locations, and reporting style stable.

## Goals / Non-Goals

**Goals:**
- Make `func_extract_subset.py` split-agnostic by keeping all labeled CWEs after cleanup, not only CWEs named in config.
- Simplify `(project, cwe_id)` filtering from min/max support to max-only support.
- Preserve current deduplication semantics, token filtering, metadata derivation, and output generation.
- Improve manifest/report outputs so each filtering stage has explicit before/after visibility, including surviving CWE counts.
- Clarify configuration so subset extraction keeps `FUNC_MAX_WHITESPACE_TOKENS` and `FUNC_MAX_PROJECT_ROWS` while removing or deprecating the minimum-support setting cleanly.
- Avoid breaking the DVC stage topology and output file locations.

**Non-Goals:**
- Designing the eventual train/val/test split or selecting held-out CWEs.
- Adding balancing, resampling, or replacement support thresholds at the CWE level.
- Changing downstream model architecture, embedding generation, or training behavior.
- Renaming output files or moving the subset stage to a different DVC stage.

## Decisions

### D1: Subset extraction will use all labeled BigVul rows, not configured CWE lists

`func_extract_subset.py` will stop importing and using `CWE_IDS` and `HELDOUT_CWE_IDS` for row filtering. Instead, it will read BigVul, keep rows with non-null labeled `cwe_id` values, then continue through metadata derivation and cleanup.

This makes the stage split-agnostic and aligns its responsibility with the requested next steps: analysis first, split design second.

Alternative considered:
- Keep the current lists but add a flag to disable filtering. Rejected because it preserves policy coupling in the wrong stage and creates two subset semantics behind one entry point.

### D2: `(project, cwe_id)` filtering becomes max-only and keeps small groups

The existing group-count filter will be simplified from:
- keep `FUNC_MIN_PROJECT_ROWS <= row_count <= FUNC_MAX_PROJECT_ROWS`

to:
- keep `row_count <= FUNC_MAX_PROJECT_ROWS`

Rows from small `(project, cwe_id)` groups will remain in the subset. Oversized groups will still be removed to prevent domination by very large project-CWE pockets.

Alternative considered:
- Replace the removed minimum with a different hidden threshold, such as per-CWE minimum total rows. Rejected because it would violate the requested behavior and continue to discard rare CWEs before the analysis stage.

### D3: Preserve current cleanup order and stable dedup semantics

The implementation will keep the existing high-level order:
1. read raw data
2. derive metadata
3. deduplicate exact code
4. filter long functions
5. filter oversized `(project, cwe_id)` groups
6. write subset, manifest, and report

Exact-code deduplication will keep using the current `func_before` field and the current first-row retention semantics. The token-length threshold will remain configurable through `FUNC_MAX_WHITESPACE_TOKENS`, with the effective inclusion rule staying `<= 512` under the current constant value.

Alternative considered:
- Reorder metadata derivation after deduplication or token filtering to save work. Rejected because the current ordering is already the basis for downstream counts and reporting, and the user explicitly wants to preserve it.

### D4: Separate subset-extraction config from split-policy config

`CWE_IDS` and `HELDOUT_CWE_IDS` stay in config because later split stages already depend on them, but they will no longer be part of the subset-extraction API contract. The subset stage’s relevant config surface becomes:
- `DEDUPLICATE_CODE`
- `FUNC_MAX_WHITESPACE_TOKENS`
- `FUNC_MAX_PROJECT_ROWS`

`FUNC_MIN_PROJECT_ROWS` will be removed if no remaining runtime caller needs it. If compatibility pressure exists in pipeline docs or stale configs, it can be left temporarily as deprecated and unused, but the preferred outcome is removal to avoid implying active behavior that no longer exists.

For naming clarity, `FUNC_MAX_PROJECT_ROWS` may be retained as-is for minimal churn, though the code and report text should describe it explicitly as a max `(project, cwe_id)` bucket size.

Alternative considered:
- Rename `FUNC_MAX_PROJECT_ROWS` immediately to `FUNC_MAX_PROJECT_CWE_ROWS`. Rejected for this change because it adds avoidable DVC/config churn without changing behavior. The design should note the clearer name as future cleanup, not as a requirement here.

### D5: Manifest/report contract expands around filter-stage observability

The current manifest/report pattern is useful and should remain machine-readable plus human-readable. The change will extend that structure rather than replace it.

The manifest/report will include, at minimum:
- total rows before and after each filter stage
- unique surviving CWE counts after each filter stage
- number of `(project, cwe_id)` groups removed by the max-row filter
- top CWEs by final row count
- post-filter token-length summary, using existing derived token counts

This will likely be represented as a stage summary section in the manifest plus matching report sections/tables. Existing summary and per-CWE sections can remain, but their configuration text must stop presenting subset extraction as train-vs-held-out CWE selection.

Alternative considered:
- Replace the manifest schema entirely with a new report format. Rejected because the current structure already works for the pipeline and only needs targeted additions.

### D6: Backward compatibility will be handled at the data-contract level, not by preserving old filtering behavior

The DVC stage name, command, output paths, and primary CSV shape should remain stable. Backward compatibility work therefore focuses on:
- manifest/report schema additions being additive where possible
- updating tests that assumed curated CWE filtering
- reviewing downstream code and docs that may assume `func_subset.csv` only contains `CWE_IDS + HELDOUT_CWE_IDS`

If any downstream logic truly requires the curated subset, that logic belongs in split design or a later analysis/sampling stage, not in `func_extract_subset.py`.

## Risks / Trade-offs

- Larger subset size may increase downstream runtime and storage costs -> Mitigation: keep existing deduplication, token filtering, and oversized-group removal; surface row-count growth explicitly in the report.
- Rare CWEs may introduce long-tail label noise into downstream consumers that were implicitly tuned to 8 CWEs -> Mitigation: treat this as an expected consequence, update assumptions in downstream code/tests, and keep split policy outside subset extraction.
- Manifest/report readers may assume old config keys like `min_project_rows` or training/held-out CWE lists are always present -> Mitigation: prefer additive schema changes and update readers/tests in the same implementation change where necessary.
- The current spec set contains stale requirements, including minimum vulnerable-function filtering, that do not match the intended future behavior -> Mitigation: remove or modify those requirements explicitly in this change’s spec deltas.

## Migration Plan

1. Update OpenSpec requirements for function subset sampling, multi-CWE subset sampling, and the stronger reporting contract.
2. Refactor `func_extract_subset.py` to remove curated-CWE filtering and minimum project-row filtering, while preserving filter order and output file locations.
3. Update config constants and imports to remove or deprecate `FUNC_MIN_PROJECT_ROWS`.
4. Expand manifest/report generation and adjust tests to validate the new stage summaries and max-only bucket filtering.
5. Re-run the `func_extract_subset` DVC stage and verify output shape, stage counts, and report contents.

Rollback strategy:
- Revert the implementation and spec delta together. Because stage names and output paths remain unchanged, rollback is straightforward and does not require DVC graph changes.

## Open Questions

- Whether any downstream notebooks or scripts currently parse `func_subset_manifest.json` with a fixed schema needs verification during implementation.
- Whether the report should show token-length distribution as percentiles, buckets, or both can be decided during implementation; the requirement is that post-filter token-length summary be present, not the exact formatting.
