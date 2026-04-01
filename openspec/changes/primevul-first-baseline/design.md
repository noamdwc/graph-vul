## Context

The current fine-tuning path is already close to what we need: `FuncFinetuneDataset` expects a normalized function-level CSV plus `assignments.csv`, `train_finetune.py` reuses the tokenizer and training loop, and `export_finetune_predictions.py` plus `eval_external.py` consume the same checkpoint and prediction artifacts. The main friction is that the data preparation side is BigVul-shaped:

- `dvc.yaml` points `func_extract_subset` at `data/bigvul.csv`
- `func_extract_subset.py` assumes BigVul column names and BigVul-specific filtering inputs
- `split_data.py` assumes project-disjoint seen/unseen allocation driven by held-out CWEs
- training and export code assume unseen splits always exist

PrimeVul changes the source and split semantics, but not the model contract we want to preserve.

## Goals / Non-Goals

**Goals:**
- Train the existing function-level fine-tuning baseline on PrimeVul with minimal code churn.
- Reuse PrimeVul's official split if the Hugging Face dataset exposes train/validation/test directly.
- Keep the existing function-level schema and canonical downstream artifact contracts stable enough that training, prediction export, post-analysis, and later external evaluation still work.
- Keep the change easy to revert with git and easy to swap again later.

**Non-Goals:**
- Refactoring the whole dataset stack into a generalized plugin architecture.
- Changing model architecture, tokenizer, truncation length, loss, metrics, or reporting methodology.
- Reworking the graph pipeline or BigVul graph artifacts.
- Introducing multi-class/CWE prediction; this stays binary vulnerable vs benign.

## Decisions

### 1. Add a narrow PrimeVul adapter and keep the rest of the pipeline schema-stable

The least disruptive approach is to normalize PrimeVul into the same row shape already consumed by the function-level baseline instead of teaching every downstream component about PrimeVul directly.

Recommended normalized training columns:

- `sample_id`: stable integer id across concatenated official splits
- `func_before`: PrimeVul function source
- `vul`: binary integer label
- `cwe_id`: normalized `CWE-XXX` string or empty string
- `source_split`: `train`, `val`, or `test`
- `project`: pass through PrimeVul project/repo field if available, otherwise a placeholder
- `cve_id`: pass through if available, otherwise empty string
- derived metadata fields already produced by `func_extract_subset.py`

This keeps `FuncFinetuneDataset`, weighted sampling, export joins, and post-analysis close to their current shape.

Alternative considered: refactor `FuncFinetuneDataset` and all downstream consumers to accept arbitrary DataFrames and string ids. Rejected because it spreads dataset-awareness into more modules than necessary.

### 2. Keep `func_extract_subset` and `func_split` stage names, but change what they do

Keeping the stage names stable minimizes DVC churn and makes the new baseline easy to compare against the old one in git history.

- `func_extract_subset` should become the PrimeVul normalization stage:
  - load PrimeVul from Hugging Face
  - concatenate official splits
  - map fields into the normalized schema
  - apply existing token-length filtering and exact-code deduplication
  - write `data/func_subset.csv`, manifest, and report
- `func_split` should materialize `assignments.csv` from PrimeVul official splits instead of running the BigVul seen/unseen allocator

Alternative considered: add an entirely new `primevul_*` stage family. Rejected for the minimal path because it would force more DVC edits and downstream path changes than the user asked for.

### 3. Extend `split_data.py` with an official-split mode instead of replacing it

`split_data.py` already owns split artifact creation, so the lowest-risk change is to add a mode that detects or accepts `source_split` input and emits `assignments.csv` directly from it.

Expected behavior in official-split mode:

- map PrimeVul split labels to canonical pipeline values: `train`, `val`, `test`
- do not fabricate `unseen_val` / `unseen_test`
- still write the standard split artifacts and a report explaining that unseen splits are absent because the source dataset provided official splits

Alternative considered: bypass `func_split` entirely and let `func_extract_subset` write assignments. Rejected because that couples two responsibilities and makes the pipeline less swappable later.

### 4. Make unseen split handling optional in training/export

The current fine-tune flow assumes BigVul-style unseen splits exist. PrimeVul official splits will likely not provide them.

Minimal compatibility changes:

- `train_finetune.py` should only construct/evaluate `unseen_val` and `unseen_test` loaders if those split names exist in `assignments.csv`
- `export_finetune_predictions.py` should export only available splits by default, or skip missing unseen splits without treating that as exceptional
- downstream metrics/report code should tolerate a train/val/test-only regime

This keeps the baseline simple and avoids inventing pseudo-unseen splits for PrimeVul.

### 5. Keep canonical training artifact paths, but add dataset provenance and update ambiguous names only where needed

For the minimal reversible change, the safest default is to keep:

- `data/func_subset.csv`
- `data/func_splits/`
- `data/func_finetune_model.ckpt`
- `data/func_finetune_metrics.json`
- `data/analysis/func_finetune_predictions/`

This avoids a broad path refactor across DVC, training, export, notebook analysis, and external-eval code. To prevent confusion:

- add `"dataset_name": "primevul"` and split provenance to manifests/reports/metrics
- update human-readable report titles that still say BigVul-first
- only introduce PrimeVul-prefixed raw/cache paths if a stage would otherwise collide with another dataset-specific artifact

Alternative considered: rename every function-level artifact to `primevul_*`. Rejected for now because it multiplies code churn and breaks more references than necessary.

### 6. External evaluation should stop treating PrimeVul as an external benchmark for this baseline

Once PrimeVul is the training dataset, external evaluation should compare the PrimeVul-trained checkpoint against:

- BigVul test predictions exported from the same run, if available
- DiverseVul via the existing Hugging Face adapter path

That means `eval_external.py` and the external registry/reporting should no longer assume PrimeVul is part of the external benchmark set for this baseline.

## Risks / Trade-offs

**[PrimeVul schema mismatch]** PrimeVul may expose slightly different split names or field names than the current external adapter assumes.
→ Mitigation: keep the adapter logic isolated in one module and normalize split names (`validation` vs `val`) before writing assignments.

**[Missing project metadata]** PrimeVul may not provide a robust `project` field, which weakens some existing report sections.
→ Mitigation: populate `project` from the best available field; if absent, use a dataset-level placeholder and clearly annotate reduced project-level report fidelity.

**[Sample id type mismatch]** Current training code expects numeric `sample_id` values.
→ Mitigation: emit stable integer `sample_id` values in the normalized training CSV instead of propagating raw string ids.

**[Unseen split assumptions]** Parts of the existing evaluation/reporting stack may quietly assume unseen splits exist.
→ Mitigation: make split availability explicit in `assignments.csv` consumers and keep missing unseen splits as a supported state, not a corner case.

**[Dedup/filter behavior changes class balance]** Exact-code deduplication and the 512-token filter may remove different proportions of PrimeVul positives/negatives than they did in BigVul.
→ Mitigation: record pre/post counts by split and label in the PrimeVul manifest/report so any distribution shift is visible.

## Migration Plan

1. Add the PrimeVul adapter and switch `func_extract_subset` to use it.
2. Extend `func_split` / `split_data.py` to materialize official split assignments.
3. Make training/export tolerant of train/val/test-only assignments.
4. Re-enable or update the DVC fine-tune stage so the function-level pipeline runs end-to-end again.
5. Update external evaluation to benchmark BigVul and DiverseVul against the PrimeVul-trained checkpoint.
6. Validate the full DVC path and confirm manifests/metrics clearly identify PrimeVul as the source dataset.

Rollback is straightforward: revert the change in git, restoring the previous BigVul-backed extraction and split logic.

## Open Questions

- Which exact PrimeVul Hugging Face fields should be treated as `project` and `cve_id`, if they exist?
- Does the dataset expose `validation` or `val` as its official development split name?
- Should `func_extract_subset` keep per-project/CWE max-row filtering as a no-op for PrimeVul when project metadata is absent, or remove that step entirely in PrimeVul mode?
