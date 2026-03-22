### Requirement: Deterministic split output
The split SHALL be seeded with a configurable random seed (default 42) so that identical inputs produce identical splits across runs.

#### Scenario: Reproducible splits
- **WHEN** `split_data` runs twice with the same seed and input CSV
- **THEN** the resulting train, val, and test index lists SHALL be identical

### Requirement: CLI interface
The script SHALL accept `--input` (path to subset CSV), `--output` (output directory), and `--seed` (random seed) as command-line arguments and SHALL be runnable as `python -m graph_vul.pipeline.split_data`.

#### Scenario: Custom output directory
- **WHEN** `split_data` is invoked with `--output /tmp/splits`
- **THEN** the split output files SHALL be written to `/tmp/splits/`
