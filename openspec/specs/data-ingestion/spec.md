### Requirement: Download Big-Vul dataset
The system SHALL provide a Python script (`scripts/download_bigvul.py`) that downloads the Big-Vul CSV dataset from Google Drive and saves it to `data/bigvul.csv`.

#### Scenario: Successful download
- **WHEN** the script is executed and the Google Drive source is reachable
- **THEN** the file `data/bigvul.csv` SHALL be created with the complete dataset content

#### Scenario: Output directory does not exist
- **WHEN** the script is executed and the `data/` directory does not exist
- **THEN** the script SHALL create the `data/` directory before downloading

#### Scenario: Download failure
- **WHEN** the Google Drive source is unreachable or the download fails
- **THEN** the script SHALL exit with a non-zero exit code and print an error message with manual download instructions

### Requirement: DVC pipeline data_ingestion stage
The `dvc.yaml` file SHALL contain a `data_ingestion` stage that runs the download script and produces `data/bigvul.csv` as a DVC-tracked output.

#### Scenario: Stage is first in pipeline
- **WHEN** `dvc repro` is executed from a clean state
- **THEN** the `data_ingestion` stage SHALL run before `extract_subset` and produce `data/bigvul.csv`

#### Scenario: Cached output skips re-download
- **WHEN** `dvc repro` is executed and `data/bigvul.csv` is already cached by DVC
- **THEN** the `data_ingestion` stage SHALL be skipped (standard DVC caching behavior)

### Requirement: gdown dependency
The project SHALL include `gdown` as a Python dependency for Google Drive downloads.

#### Scenario: Dependency available
- **WHEN** the project's dependencies are installed
- **THEN** `gdown` SHALL be importable and usable by the download script
