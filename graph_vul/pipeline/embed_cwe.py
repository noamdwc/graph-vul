"""Compute DeBERTa embeddings for CWE description texts.

Reads a CWE descriptions JSON file and writes one .pt file per CWE ID
containing a 768-dimensional embedding vector.
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from graph_vul.config import DATA_DIR, DEBERTA_EMBEDDING_DIM, PROCESSED_DIR
from graph_vul.pipeline.embed_utils import embed_texts, load_deberta

logger = logging.getLogger(__name__)

CWE_DESCRIPTIONS_PATH = DATA_DIR / "cwe_descriptions.json"


def embed_cwe_texts(
    descriptions_json: str | None = None,
    output_dir: str | None = None,
) -> None:
    src = Path(descriptions_json) if descriptions_json else CWE_DESCRIPTIONS_PATH
    dst = Path(output_dir) if output_dir else PROCESSED_DIR / "cwe_embeddings"
    dst.mkdir(parents=True, exist_ok=True)

    with open(src) as f:
        cwe_descriptions: dict[str, str] = json.load(f)

    if not cwe_descriptions:
        logger.warning("No CWE descriptions found in %s", src)
        return

    tokenizer, model = load_deberta()

    processed, skipped = 0, 0
    for cwe_id, description in cwe_descriptions.items():
        out_path = dst / f"{cwe_id}.pt"

        if out_path.exists():
            skipped += 1
            continue

        if description.strip():
            text_vec = embed_texts([description], tokenizer, model)[0]
        else:
            text_vec = torch.zeros(DEBERTA_EMBEDDING_DIM)

        torch.save(text_vec, out_path)
        processed += 1

    logger.info(
        "CWE embedding complete: %d processed, %d skipped (existing)", processed, skipped
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute DeBERTa embeddings for CWE texts")
    parser.add_argument("--input", type=str, default=None, help="CWE descriptions JSON path")
    parser.add_argument("--output", type=str, default=None, help="Output embeddings directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    embed_cwe_texts(descriptions_json=args.input, output_dir=args.output)
