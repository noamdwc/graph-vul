### Requirement: Fixed token-count buckets
The split stage SHALL assign each row a `code_token_count_bucket` value using fixed boundaries based on `code_token_count`: 0–64, 64–128, 128–256, 256–512, 512+. Bucket assignment SHALL NOT use quantile-based binning (pd.qcut) for `code_token_count_bucket`. The bucket boundaries SHALL be configurable via `TOKEN_BUCKET_EDGES` and `TOKEN_BUCKET_LABELS` in config.

#### Scenario: Bucket assignment with fixed boundaries
- **WHEN** a row has `code_token_count = 42`
- **THEN** its `code_token_count_bucket` SHALL be `"0-64"`

#### Scenario: Boundary value assignment
- **WHEN** a row has `code_token_count = 64`
- **THEN** its `code_token_count_bucket` SHALL be `"64-128"` (left-exclusive except for the first bucket)

#### Scenario: Large function bucketing
- **WHEN** a row has `code_token_count = 800`
- **THEN** its `code_token_count_bucket` SHALL be `"512+"`

#### Scenario: Stable across datasets
- **WHEN** the same function appears in two different dataset versions with different overall distributions
- **THEN** its `code_token_count_bucket` SHALL be identical (because boundaries are fixed, not data-dependent)

---

### Requirement: Per-project short-bucket statistics
The `_build_project_stats` function SHALL compute per-project counts of short-bucket (0–64 token) vulnerable rows and short-bucket benign rows, available as `short_vul` and `short_benign` columns in the project stats DataFrame.

#### Scenario: Project with mixed short functions
- **WHEN** a project has 5 rows in the 0–64 bucket, 3 vulnerable and 2 benign
- **THEN** its project stats SHALL show `short_vul = 3` and `short_benign = 2`

#### Scenario: Project with no short functions
- **WHEN** a project has zero rows in the 0–64 bucket
- **THEN** its project stats SHALL show `short_vul = 0` and `short_benign = 0`

---

### Requirement: Short-bucket deficit penalty in greedy allocator
The greedy allocator scoring function SHALL include a penalty term that penalizes eval splits (val, test) when their accumulated short-bucket vulnerable or benign counts are below the configured minimums (`MIN_SHORT_VUL_PER_EVAL_SPLIT`, `MIN_SHORT_BENIGN_PER_EVAL_SPLIT`). The penalty SHALL be weighted by `SHORT_BUCKET_DEFICIT_WEIGHT` (default 15.0). The penalty SHALL be zero when minimums are met.

#### Scenario: Val below short-vul minimum
- **WHEN** val has 5 short vulnerable rows and the minimum is 15
- **THEN** the scoring function SHALL add a positive penalty proportional to the deficit

#### Scenario: Val meets short-vul minimum
- **WHEN** val has 20 short vulnerable rows and the minimum is 15
- **THEN** the scoring function SHALL add zero short-bucket deficit penalty for val's short_vul component

#### Scenario: Train excluded from penalty
- **WHEN** train has 2 short vulnerable rows and the minimum is 15
- **THEN** no short-bucket deficit penalty SHALL be applied to train (only eval splits are penalized)

#### Scenario: Penalty applies to unseen eval splits
- **WHEN** unseen_val has fewer short vulnerable rows than the minimum
- **THEN** the scoring function SHALL add a short-bucket deficit penalty for unseen_val

---

### Requirement: Configurable short-function evaluation minimums
The config SHALL define `MIN_SHORT_VUL_PER_EVAL_SPLIT` (default 15) and `MIN_SHORT_BENIGN_PER_EVAL_SPLIT` (default 30) as the minimum short-bucket counts required for each eval split. These SHALL be importable from `graph_vul.config`.

#### Scenario: Defaults applied
- **WHEN** config is loaded without overrides
- **THEN** `MIN_SHORT_VUL_PER_EVAL_SPLIT` SHALL be 15 and `MIN_SHORT_BENIGN_PER_EVAL_SPLIT` SHALL be 30

---

### Requirement: Post-split short-bucket validation
After split creation, a validation step SHALL check each eval split (val, test, unseen_val, unseen_test) for short-bucket coverage. It SHALL emit warnings when:
- The 0–64 bucket in an eval split has fewer than `MIN_SHORT_VUL_PER_EVAL_SPLIT` vulnerable rows.
- The 0–64 bucket in an eval split has fewer than `MIN_SHORT_BENIGN_PER_EVAL_SPLIT` benign rows.
- The 0–64 bucket in an eval split has fewer than 10 vulnerable rows (too few for reliable metrics regardless of configuration).

#### Scenario: Val short-bucket below configured minimum
- **WHEN** val has 8 short vulnerable rows and `MIN_SHORT_VUL_PER_EVAL_SPLIT = 15`
- **THEN** a warning SHALL be emitted naming val, the bucket, and the deficit

#### Scenario: Val short-bucket meets minimum
- **WHEN** val has 20 short vulnerable rows and `MIN_SHORT_VUL_PER_EVAL_SPLIT = 15`
- **THEN** no warning SHALL be emitted for val's short vulnerable count

#### Scenario: Unseen split too small for reliable metrics
- **WHEN** unseen_test has 3 short vulnerable rows
- **THEN** a warning SHALL be emitted that short-function metrics for unseen_test are unreliable

#### Scenario: Validation does not fail the pipeline
- **WHEN** short-bucket validation detects deficits
- **THEN** the pipeline SHALL continue (warnings, not errors) and the split SHALL still be written

---

### Requirement: Split report shows fixed token buckets
The split report SHALL include a "Token Length Buckets" section showing per-split counts for each fixed token bucket (0–64, 64–128, 128–256, 256–512, 512+), broken down by vulnerable and benign rows.

#### Scenario: Report contains fixed bucket table
- **WHEN** the split report is generated
- **THEN** it SHALL contain a table with columns for each fixed token bucket and rows for each split, with vulnerable and benign sub-counts

#### Scenario: Report shows short-function evaluation viability
- **WHEN** the split report is generated
- **THEN** it SHALL contain a "Short-Function Evaluation Viability" section showing, for each eval split: 0–64 bucket vul count, 0–64 bucket benign count, and a pass/warn status against the configured minimums

#### Scenario: Report shows per-project short-bucket contributions
- **WHEN** the split report is generated
- **THEN** it SHALL list, for each eval split, which projects contribute short-bucket rows and how many (vulnerable and benign)
