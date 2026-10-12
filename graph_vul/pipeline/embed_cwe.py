"""Compute DeBERTa embeddings for CWE description texts.

Reads a CWE descriptions JSON file and writes one .pt file per CWE ID
containing a 768-dimensional embedding vector.
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from graph_vul.model.cwe_matching import load_catalog, CATALOG_PATH
from graph_vul.config import PROCESSED_DIR, DEBERTA_MODEL_NAME
from graph_vul.artifacts import code_identity, write_json
from graph_vul.pipeline.embed_utils import embed_texts, load_deberta

logger = logging.getLogger(__name__)

CWE_DESCRIPTIONS_PATH = CATALOG_PATH


def embed_cwe_texts(
    descriptions_json: str | None = None,
    output_dir: str | None = None,
) -> None:
    src = Path(descriptions_json) if descriptions_json else CWE_DESCRIPTIONS_PATH
    dst = Path(output_dir) if output_dir else PROCESSED_DIR / "cwe_embeddings"
    dst.mkdir(parents=True, exist_ok=False)
    catalog = load_catalog(src)
    cwe_descriptions = catalog["descriptions"]
    (dst / "catalog.json").write_text(json.dumps(catalog, indent=2))

    tokenizer, model = load_deberta()

    vectors = embed_texts(list(cwe_descriptions.values()), tokenizer, model)
    for cwe_id, text_vec in zip(cwe_descriptions, vectors):
        torch.save(text_vec, dst / f"{cwe_id}.pt")
    write_json(
        dst / "embedding_manifest.json",
        {
            "code": code_identity(),
            "catalog_version": catalog["version"],
            "model": DEBERTA_MODEL_NAME,
            "model_revision": getattr(model.config, "_commit_hash", None),
            "max_length": 128,
            "batch_size": 16,
            "rows": len(vectors),
        },
    )
    logger.info("CWE embedding complete: %d processed", len(vectors))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute DeBERTa embeddings for CWE texts"
    )
    parser.add_argument(
        "--input", type=str, default=None, help="CWE descriptions JSON path"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output embeddings directory"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    embed_cwe_texts(descriptions_json=args.input, output_dir=args.output)
