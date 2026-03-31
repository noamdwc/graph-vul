## ADDED Requirements

### Requirement: Adapter returns normalised DataFrame
Each external dataset adapter SHALL return a `pd.DataFrame` with columns `[sample_id, func_before, vul, cwe_id]` where `sample_id` is a unique string identifier, `func_before` is the raw function source code, `vul` is an integer binary label (1 = vulnerable, 0 = benign), and `cwe_id` is a string in `CWE-XXX` format (or empty string if unavailable).

#### Scenario: PrimeVul adapter returns correct schema
- **WHEN** `load_primevul()` is called
- **THEN** it returns a DataFrame with exactly the columns `[sample_id, func_before, vul, cwe_id]`, all `sample_id` values are unique, `vul` contains only 0 and 1, and `func_before` contains non-empty strings

#### Scenario: DiverseVul adapter returns correct schema
- **WHEN** `load_diversevul()` is called
- **THEN** it returns a DataFrame with exactly the columns `[sample_id, func_before, vul, cwe_id]`, all `sample_id` values are unique, `vul` contains only 0 and 1, and `func_before` contains non-empty strings

### Requirement: Adapters load from HuggingFace
Each adapter SHALL load its dataset using the `datasets` library from HuggingFace Hub. The adapter MUST NOT require any local pre-downloaded files beyond the HuggingFace cache.

#### Scenario: First-time download
- **WHEN** the adapter is called and no local cache exists
- **THEN** the dataset is downloaded from HuggingFace Hub and cached locally

#### Scenario: Cached dataset
- **WHEN** the adapter is called and the dataset is already cached
- **THEN** the dataset is loaded from local cache without network access

### Requirement: Token length filtering
Each adapter SHALL filter out functions exceeding the configured maximum whitespace-token count (512 tokens, matching `FUNC_MAX_WHITESPACE_TOKENS`). The adapter MUST log the number of samples removed by this filter.

#### Scenario: Long function filtered
- **WHEN** a function in the external dataset has more than 512 whitespace-delimited tokens
- **THEN** it is excluded from the returned DataFrame

#### Scenario: Short function retained
- **WHEN** a function has 512 or fewer whitespace-delimited tokens
- **THEN** it is included in the returned DataFrame

### Requirement: CWE ID normalisation
Each adapter SHALL normalise CWE identifiers to the `CWE-XXX` string format used by the existing pipeline. If a dataset does not provide CWE IDs, the adapter SHALL use an empty string.

#### Scenario: Numeric CWE ID
- **WHEN** the source dataset provides a CWE ID as an integer (e.g., `119`)
- **THEN** the adapter converts it to `CWE-119`

#### Scenario: Missing CWE ID
- **WHEN** a sample has no CWE information
- **THEN** `cwe_id` is set to an empty string

### Requirement: Adapter registry
A module-level dictionary SHALL map dataset names (lowercase strings) to their adapter functions. Adding a new external dataset MUST only require adding one function and one registry entry.

#### Scenario: Registry lookup
- **WHEN** the evaluation script looks up `"primevul"` in the registry
- **THEN** it receives the `load_primevul` function

#### Scenario: Unknown dataset
- **WHEN** a dataset name is not in the registry
- **THEN** a `KeyError` is raised
