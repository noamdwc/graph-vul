"""Export Code Property Graphs for each sample using the Joern CLI."""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from graph_vul.config import SUBSET_CSV, RAW_DIR

logger = logging.getLogger(__name__)


def export_cpgs(
    subset_csv: str | None = None,
    output_dir: str | None = None,
) -> None:
    src = subset_csv or str(SUBSET_CSV)
    dst = Path(output_dir) if output_dir else RAW_DIR
    dst.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    total = len(df)
    skipped = 0
    failed = 0

    for idx, row in df.iterrows():
        sample_id = str(row["sample_id"])
        sample_dir = dst / sample_id

        # Idempotency: skip if export already exists and is non-empty
        if sample_dir.exists() and any(sample_dir.iterdir()):
            skipped += 1
            continue

        sample_dir.mkdir(parents=True, exist_ok=True)

        func_code = row.get("func_before", "")
        if not isinstance(func_code, str) or not func_code.strip():
            logger.warning("Sample %s has empty function code, skipping", sample_id)
            failed += 1
            continue

        try:
            _run_joern(func_code, sample_dir)
        except Exception:
            logger.exception("Joern failed for sample %s", sample_id)
            failed += 1
            # Clean up partial output
            if sample_dir.exists():
                shutil.rmtree(sample_dir)
                sample_dir.mkdir(parents=True, exist_ok=True)
            continue

    processed = total - skipped - failed
    logger.info(
        "CPG export complete: %d processed, %d skipped (existing), %d failed out of %d total",
        processed,
        skipped,
        failed,
        total,
    )


def _run_joern(func_code: str, output_dir: Path) -> None:
    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
        f.write(func_code)
        tmp_path = f.name

    try:
        # Step 1: Parse the file into a CPG
        cpg_path = str(output_dir / "cpg.bin")
        subprocess.run(
            ["joern-parse", tmp_path, "-o", cpg_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Step 2: Export the CPG
        subprocess.run(
            ["joern-export", cpg_path, "--repr", "cpg14", "--out", str(output_dir)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Clean up the binary CPG file
        bin_path = Path(cpg_path)
        if bin_path.exists():
            bin_path.unlink()

    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    export_cpgs()
