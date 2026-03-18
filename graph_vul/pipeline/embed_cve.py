"""Standalone script: compute DeBERTa embeddings for CVE description texts.

Reads the subset CSV and writes per-sample .pt files containing
768-dimensional CVE text embedding vectors.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch

from graph_vul.config import DEBERTA_EMBEDDING_DIM, PROCESSED_DIR, SUBSET_CSV
from graph_vul.pipeline.embed_utils import embed_texts, load_deberta

logger = logging.getLogger(__name__)


def embed_cve_texts(
    input_csv: str | None = None,
    output_dir: str | None = None,
) -> None:
    src = input_csv or str(SUBSET_CSV)
    dst = Path(output_dir) if output_dir else PROCESSED_DIR / "cve_embeddings"
    dst.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(src)
    tokenizer, model = load_deberta()

    processed, skipped = 0, 0
    for _, row in metadata.iterrows():
        sample_id = str(row["sample_id"])
        out_path = dst / f"{sample_id}.pt"

        if out_path.exists():
            skipped += 1
            continue

        cve_desc = row.get("cve_desc", "")
        if isinstance(cve_desc, str) and cve_desc.strip():
            text_vec = embed_texts([cve_desc], tokenizer, model)[0]
        else:
            text_vec = torch.zeros(DEBERTA_EMBEDDING_DIM)

        torch.save(text_vec, out_path)
        processed += 1

    logger.info(
        "CVE embedding complete: %d processed, %d skipped (existing)", processed, skipped
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute DeBERTa embeddings for CVE texts")
    parser.add_argument("--input", type=str, default=None, help="Subset CSV path")
    parser.add_argument("--output", type=str, default=None, help="Output embeddings directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    embed_cve_texts(input_csv=args.input, output_dir=args.output)
