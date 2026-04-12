## ADDED Requirements

### Requirement: Comparable E0-E4 experiment manifests
Each CWE-matching ablation run SHALL record its ablation mode and relevant matching-supervision settings in a machine-readable manifest or metrics payload. At minimum this SHALL include whether wrong-CWE negatives are enabled, whether triplet loss is enabled, whether hard negatives are preferred, the wrong-negative count, the triplet margin, and the auxiliary-loss weight.

#### Scenario: Run metadata recorded
- **WHEN** an `E4` experiment completes
- **THEN** its outputs record that wrong-CWE negatives and triplet loss were enabled, hard-negative preference was enabled, `negative_count=1`, `margin=0.2`, and `lambda=0.1`

### Requirement: Retrieval metrics from shared matching space
Evaluation SHALL compute CWE retrieval metrics from the shared matching space by ranking candidate CWE embeddings for each function. At minimum the outputs SHALL include Top-1 and Top-3 retrieval accuracy.

#### Scenario: Retrieval metrics available
- **WHEN** prediction export or evaluation runs for a checkpoint with matching heads
- **THEN** the outputs include Top-1 and Top-3 CWE retrieval metrics derived from ranked similarity scores

#### Scenario: Baseline retrieval still reported
- **WHEN** evaluation runs for ablation mode `E0`
- **THEN** the reporting path still emits retrieval metrics if the run exposes the matching-space outputs needed to compute them

### Requirement: Standard ablation summary outputs
For each evaluation split or dataset, the reporting layer SHALL emit:
- overall classification metrics
- per-CWE metrics
- Top-1 and Top-3 CWE retrieval metrics
- unseen or external dataset results when those datasets are evaluated
- fixed function-length slice metrics for `0-64`, `64-128`, `128-256`, and `256+`

#### Scenario: In-distribution split report
- **WHEN** the test split is evaluated
- **THEN** the summary includes aggregate metrics, per-CWE metrics, retrieval metrics, and the four required function-length slices

#### Scenario: External dataset report
- **WHEN** an external dataset is evaluated
- **THEN** the summary includes the same aggregate, retrieval, and fixed length-slice outputs for that dataset

### Requirement: Fixed function-length slices for ablation comparison
Length-slice reporting for this ablation SHALL use the fixed buckets `0-64`, `64-128`, `128-256`, and `256+`. These slices SHALL be used consistently across baseline, unseen, and external evaluation outputs so short-function comparisons are directly comparable across ablations.

#### Scenario: Boundary value slice
- **WHEN** a function has token length `64`
- **THEN** it is reported in the `64-128` slice for ablation summaries

#### Scenario: Long function slice
- **WHEN** a function has token length `400`
- **THEN** it is reported in the `256+` slice for ablation summaries

### Requirement: Per-CWE export artifacts
Per-CWE metrics for each evaluated split or dataset SHALL be written to stable output files so ablation runs can be compared offline without re-running notebooks.

#### Scenario: Per-CWE metrics persisted
- **WHEN** unseen_test evaluation completes
- **THEN** a per-CWE metrics artifact exists for `unseen_test` alongside the aggregate summary outputs
