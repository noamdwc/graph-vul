## REMOVED Requirements

### Requirement: Row-level stratified split
**Reason**: Replaced by project-disjoint leakage-aware split in the `leakage-aware-split` capability. Row-level stratified splitting ignores project structure and creates data leakage.
**Migration**: Use the new `leakage-aware-split` split stage which produces `data/splits/assignments.csv` instead of per-split JSON index files. Update dataset loading code to read assignments CSV and filter by split column.

### Requirement: Per-split JSON index files
**Reason**: Replaced by single assignments CSV with `sample_id, split` columns. Simpler to manage and join with metadata.
**Migration**: Update `cpg_dataset.py` and any code reading `train.json`/`val.json`/`test.json` to read `assignments.csv` instead.
