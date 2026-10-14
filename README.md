# Graph-Vul

Graph-Vul matches a C function against a fixed CWE catalog and an explicit **NONE** candidate. It tests whether AST structure and triplet supervision improve vulnerability detection and CWE retrieval beyond a code–CWE pair classifier.

A function is encoded once by **UniXcoder**. An optional **RGCN** encodes its tree-sitter **AST**. The pair head scores those features against every versioned CWE description embedding and a learned NONE representation. Ground-truth annotations supply training targets and evaluation metadata; they never select inference inputs.

```
C function → UniXcoder mean pooling ─────┐
C function → tree-sitter AST → RGCN ────┤→ cached function features
                                      ↓
Versioned CWE catalog + learned NONE → pair head → all candidate logits
                                      ↓
                     best real-CWE logit − NONE logit
                                      ↓
                 validation-calibrated threshold → CWE or NONE
```

The strict threshold comparison resolves ties to NONE. All valid annotated CWEs are positives; benign functions match NONE. Triplet experiments add random or model-scored incorrect candidates while excluding every positive label.

**Results status:** historical experiments are exploratory and superseded. No headline performance is claimed for this implementation. CPU contract tests include training, checkpoint selection, calibration, graph-enabled external scoring, exports and report generation with a small synthetic encoder. A bounded real UniXcoder pilot also completed all four configurations on CPU, including calibration, internal/external exports and a diagnostic report. The full four-configuration matrix with seeds **42, 43, 44** remains pending GPU execution. See [rerun status](docs/RERUN_STATUS.md) for completed preparation stages and exact pending work.

The controlled comparison is:

| Experiment | Function features | Pair supervision | Triplet supervision |
|---|---|---|---|
| `pair` | UniXcoder | Correct CWE / NONE targets | — |
| `graph_pair` | UniXcoder + AST | Same | — |
| `triplet_random` | UniXcoder + AST | Same, random negatives | Random incorrect candidates |
| `triplet_hard` | UniXcoder + AST | Same, highest-scoring negatives | Highest-scoring incorrect candidates |

All runs share preparation, catalog, project-disjoint splits, NONE supervision, calibration rule and training budget. Their thresholds are fitted separately on validation data and then fixed for final tests. The canonical report includes PR-AUC, precision, recall, F1, benign false-positive rate, CWE top-1/top-3 retrieval, support, short-function slices, per-CWE results and external stress tests.

Start with [operating instructions](docs/PIPELINE.md) for environment setup, local preparation, Colab training, artifact handoff and the CLI demo. [Scientific contract](docs/MATCHING_CONTRACT.md) explains targets, unknown annotations, checkpoint selection and experiment provenance.
