## Context

The Graph-Vul pipeline currently starts at the `extract_subset` stage which expects `data/bigvul.csv` to already exist. The Big-Vul dataset is hosted on Google Drive (the standard distribution channel used by the paper authors). There is no automated mechanism to fetch it, so new contributors must manually download and place the file.

## Goals / Non-Goals

**Goals:**
- Automate downloading of the Big-Vul CSV so `dvc repro` works from a clean checkout.
- Wire the download as the first DVC pipeline stage (`data_ingestion`), producing `data/bigvul.csv`.

**Non-Goals:**
- Mirroring or re-hosting the dataset.
- Downloading alternative vulnerability datasets.
- Validating or transforming the CSV content (that's the job of downstream stages).

## Decisions

### 1. Python script over shell script
**Choice**: `scripts/download_bigvul.py` (Python) rather than a bash script.
**Rationale**: The dataset is hosted on Google Drive, which requires handling redirect cookies and confirmation tokens. `gdown` (a Python library purpose-built for Google Drive downloads) handles this reliably. A shell script with `curl`/`wget` would need fragile cookie handling as Google Drive changes its flow.
**Alternative considered**: Shell script with `wget` — rejected due to Google Drive redirect complexity.

### 2. Use `gdown` for Google Drive downloads
**Choice**: Use the `gdown` Python package.
**Rationale**: `gdown` is the de-facto standard for programmatic Google Drive downloads, handles large-file confirmation prompts automatically, and is already widely used in ML research pipelines.
**Alternative considered**: `requests` with manual cookie handling — more code, more fragile.

### 3. DVC stage with `data/bigvul.csv` as output
**Choice**: Add a `data_ingestion` stage to `dvc.yaml` with `data/bigvul.csv` as an output.
**Rationale**: The existing `extract_subset` stage already lists `data/bigvul.csv` as a dependency, so adding it as an output of the new stage automatically chains the pipeline. No changes to downstream stages are needed.

## Risks / Trade-offs

- **Google Drive rate-limiting or link rot** → The script should print a clear error message with manual download instructions if the download fails. The Google Drive file ID can be configured via an environment variable or constant for easy updates.
- **Large file size** → Big-Vul CSV is ~200 MB. The script should show download progress. DVC caching prevents re-downloading on subsequent runs.
- **Network dependency** → The `data_ingestion` stage requires internet access. This is acceptable since it only runs once (DVC caches the output).
