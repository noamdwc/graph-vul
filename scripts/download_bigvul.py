#!/usr/bin/env python3
"""Download the Big-Vul CSV dataset from Google Drive."""

import os
import sys
from pathlib import Path

GDRIVE_FILE_ID = os.environ.get(
    "BIGVUL_GDRIVE_ID", "1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X"
)
OUTPUT_PATH = Path("data/bigvul.csv")


def main() -> None:
    try:
        import gdown
    except ImportError:
        print(
            "ERROR: gdown is not installed. Install it with:\n"
            "  pip install gdown>=5.1.0",
            file=sys.stderr,
        )
        sys.exit(1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    print(f"Downloading Big-Vul dataset to {OUTPUT_PATH} ...")

    try:
        gdown.download(url, str(OUTPUT_PATH), quiet=False)
    except Exception as exc:
        print(
            f"\nERROR: Download failed: {exc}\n\n"
            "Manual download instructions:\n"
            f"  1. Open https://drive.google.com/uc?id={GDRIVE_FILE_ID} in a browser\n"
            f"  2. Download the file and save it as {OUTPUT_PATH}\n",
            file=sys.stderr,
        )
        sys.exit(1)

    if not OUTPUT_PATH.exists() or OUTPUT_PATH.stat().st_size == 0:
        print(
            f"\nERROR: Download produced an empty or missing file at {OUTPUT_PATH}.\n\n"
            "Manual download instructions:\n"
            f"  1. Open https://drive.google.com/uc?id={GDRIVE_FILE_ID} in a browser\n"
            f"  2. Download the file and save it as {OUTPUT_PATH}\n",
            file=sys.stderr,
        )
        sys.exit(1)

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Done. Downloaded {size_mb:.1f} MB to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
