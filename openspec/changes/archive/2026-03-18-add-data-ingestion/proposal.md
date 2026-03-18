## Why

The pipeline currently assumes `data/bigvul.csv` already exists locally, with no automated way to obtain it. This makes onboarding fragile and the pipeline non-reproducible from a clean checkout. Adding a data-ingestion stage ensures the full DVC pipeline is self-contained and can run end-to-end without manual data preparation.

## What Changes

- Add a new script (`scripts/download_bigvul.py`) that downloads the Big-Vul CSV dataset from its official source.
- Add a new `data_ingestion` stage in `dvc.yaml` that runs this script and produces `data/bigvul.csv` as a DVC-tracked output.
- The existing `extract_subset` stage will depend on this new stage (via its existing `data/bigvul.csv` dependency), requiring no changes to downstream stages.

## Capabilities

### New Capabilities
- `data-ingestion`: Automated download of the Big-Vul CSV dataset, wired as the first DVC pipeline stage.

### Modified Capabilities

## Impact

- **dvc.yaml**: New `data_ingestion` stage added before `extract_subset`.
- **scripts/**: New `download_bigvul.py` script.
- **Dependencies**: May require `requests` or `gdown` (for Google Drive downloads) as a Python dependency.
- No changes to existing pipeline stages or model code.
