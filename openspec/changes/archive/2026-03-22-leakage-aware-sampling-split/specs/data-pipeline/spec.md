## MODIFIED Requirements

### Requirement: Subset extraction stage
The extract_subset stage SHALL read the Big-Vul CSV, filter to CWE-119, and produce a structured subset using the `structured-subset-sampling` capability. The output CSV SHALL include columns: `sample_id`, `project`, `func_before`, `cve_id`, `cve_desc`, `vul`. The stage SHALL also produce a manifest JSON and report MD.

#### Scenario: Subset produced with project column
- **WHEN** the extract_subset stage runs
- **THEN** the output CSV contains a `project` column and all rows have non-null project values

#### Scenario: Subset artifacts produced
- **WHEN** the extract_subset stage completes
- **THEN** three files exist: subset CSV, subset manifest JSON, and subset report MD

---

## MODIFIED Requirements

### Requirement: Split stage position in pipeline
The split_data stage SHALL run after the embedding stages. It SHALL depend on the subset CSV for metadata. The split SHALL use the `leakage-aware-split` capability producing project-disjoint splits.

#### Scenario: Split depends on subset
- **WHEN** the split stage runs
- **THEN** it reads the subset CSV for project and CVE metadata to inform split decisions

#### Scenario: Split produces assignments
- **WHEN** the split stage completes
- **THEN** it produces an assignments CSV and a diagnostics report
