## 1. Dependencies

- [x] 1.1 Add `gdown` to project dependencies (pyproject.toml or requirements.txt)

## 2. Download Script

- [x] 2.1 Create `scripts/download_bigvul.py` that uses `gdown` to download the Big-Vul CSV from Google Drive to `data/bigvul.csv`
- [x] 2.2 Ensure the script creates the `data/` directory if it doesn't exist
- [x] 2.3 Add error handling with a clear message and manual download instructions on failure
- [x] 2.4 Add download progress output

## 3. DVC Pipeline

- [x] 3.1 Add `data_ingestion` stage to `dvc.yaml` with `scripts/download_bigvul.py` as a dependency and `data/bigvul.csv` as an output
- [x] 3.2 Verify the stage is ordered before `extract_subset` in `dvc.yaml`
