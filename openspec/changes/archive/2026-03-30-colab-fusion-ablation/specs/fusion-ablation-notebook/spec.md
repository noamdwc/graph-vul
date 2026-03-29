## ADDED Requirements

### Requirement: Notebook runs fixed 5-experiment ablation suite
The notebook `notebooks/colab_fusion_ablation.ipynb` SHALL define and run exactly 5 experiments:

| Tag | fusion_layers | final_graph_concat | Description |
|-----|--------------|-------------------|-------------|
| E0_late_only | None | True | Baseline: no internal fusion |
| E1_last2 | [10, 11] | True | Fuse in last 2 layers |
| E2_last4 | [8, 9, 10, 11] | True | Fuse in last 4 layers |
| E3_all_layers | [0..11] | True | Fuse in all 12 layers |
| E4_last4_no_late_graph | [8, 9, 10, 11] | False | Internal fusion only, no late graph concat |

All experiments SHALL use `USE_GRAPH=True` and identical hyperparameters (epochs, batch size, LR, seed).

#### Scenario: All 5 experiments complete
- **WHEN** the notebook is run end-to-end
- **THEN** all 5 experiments complete, each saving outputs to `{DRIVE_DATA}/fusion_ablation/{run_tag}/`

#### Scenario: Partial run recovery
- **WHEN** the notebook session is interrupted after N experiments
- **THEN** the first N experiments' checkpoints and metrics are preserved on Drive, and the notebook can be re-run starting from experiment N+1 (by commenting out completed experiments)

### Requirement: Notebook structure mirrors colab_finetune.ipynb
The notebook SHALL follow the same cell sequence as `colab_finetune.ipynb`:
1. GPU check
2. Drive mount
3. Directory creation
4. Data validation
5. Graph artifact validation
6. HF cache setup
7. Repo clone + dependency install
8. Data symlink
9. Optional WandB authentication
10. Configuration cell (experiment definitions)
11. Main experiment runner loop
12. Final summary table

#### Scenario: Setup cells match existing notebook
- **WHEN** comparing cells 1–9 of this notebook with colab_finetune.ipynb
- **THEN** the setup flow SHALL be functionally identical (same Drive paths, same validation, same dependency list)

### Requirement: Per-experiment output isolation
Each experiment SHALL write all outputs to `{DRIVE_DATA}/fusion_ablation/{run_tag}/`:
- `func_finetune_model.ckpt` — best checkpoint by val_f1
- `func_finetune_metrics.json` — standard training metrics
- `predictions/` — prediction CSVs for test and unseen_test splits

#### Scenario: No cross-contamination between experiments
- **WHEN** experiments E0 and E1 both complete
- **THEN** E0's checkpoint and metrics are in `fusion_ablation/E0_late_only/` and E1's are in `fusion_ablation/E1_last2/`, with no shared files

### Requirement: Compact comparison table with length-bucket metrics
The final notebook cell SHALL display a single pandas DataFrame comparing all completed experiments. The table SHALL include at minimum:

- `run_tag` (row index)
- `test_f1`
- `unseen_test_f1`
- `0_64_f1`, `0_64_precision`, `0_64_recall`
- `64_128_f1`
- `128_256_f1`
- `256_512_f1`
- `train_minutes`

#### Scenario: Summary table after full run
- **WHEN** all 5 experiments have completed
- **THEN** the summary table has 5 rows and all required columns populated with numeric values

#### Scenario: Length-bucket metrics use existing analysis pipeline
- **WHEN** extracting bucket metrics for an experiment
- **THEN** the notebook SHALL call `export_split_predictions()` and `stratified_metrics()` from the existing codebase rather than reimplementing evaluation logic
