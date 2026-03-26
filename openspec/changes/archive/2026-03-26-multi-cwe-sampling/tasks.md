## 1. Config

- [x] 1.1 Replace `CWE_ID = "CWE-119"` with `CWE_IDS` list and `HELDOUT_CWE_IDS` list in `graph_vul/config.py`
- [x] 1.2 Update all imports of `CWE_ID` across the codebase to use `CWE_IDS` / `HELDOUT_CWE_IDS`

## 2. CWE Descriptions

- [x] 2.1 Add descriptions for CWE-125, CWE-20, CWE-476, CWE-190, CWE-787, CWE-200, CWE-416 to `data/cwe_descriptions.json` (sourced from MITRE)

## 3. Subset Extraction

- [x] 3.1 Update `func_extract_subset.py` to filter for all 8 CWEs (union of `CWE_IDS` + `HELDOUT_CWE_IDS`) instead of single `CWE_ID`
- [x] 3.2 Change `_filter_projects` to apply the project-size filter per CWE within each project (not globally)
- [x] 3.3 Update logging and report generation to reflect multi-CWE output (per-CWE stats in manifest/report)

## 4. Split Data

- [x] 4.1 Update `split_data.py` to separate held-out CWE rows before running `StratifiedGroupKFold`, then append them to the test split
- [x] 4.2 Update split report to show per-CWE breakdown in test split, distinguishing held-out from training CWEs

## 5. Pipeline Integration

- [x] 5.1 Update `dvc.yaml` stage dependencies if needed (e.g., `cwe_descriptions.json` dependency for `func_extract_subset`) — no changes needed, config.py already a dep
- [x] 5.2 Run `func_extract_subset` and verify output: 8 CWEs present, per-CWE project filtering correct
- [ ] 5.3 Run `split_data` on the new subset and verify held-out CWEs appear only in test (deferred)
