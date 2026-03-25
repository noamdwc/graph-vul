"""Compute frozen UniXcoder mean-pooled embeddings for function-level source code.

Reads the function subset CSV and writes per-sample .pt files containing
768-dimensional mean-pooled embedding vectors.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from graph_vul.config import FUNC_EMBEDDINGS_DIR, FUNC_SUBSET_CSV, UNIXCODER_EMBEDDING_DIM
from graph_vul.pipeline.embed_utils import embed_mean_pool, load_unixcoder

logger = logging.getLogger(__name__)


def embed_functions(
    input_csv: str | None = None,
    output_dir: str | None = None,
) -> None:
    src = input_csv or str(FUNC_SUBSET_CSV)
    dst = Path(output_dir) if output_dir else FUNC_EMBEDDINGS_DIR
    dst.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(src)
    tokenizer, model = load_unixcoder()

    total = len(metadata)
    processed, skipped = 0, 0
    progress = tqdm(
        metadata.iterrows(),
        total=total,
        desc="Embedding functions",
        unit="func",
    )

    for _, row in progress:
        sample_id = str(row["sample_id"])
        out_path = dst / f"{sample_id}.pt"

        if out_path.exists():
            skipped += 1
            progress.set_postfix(processed=processed, skipped=skipped)
            continue

        func_text = row.get("func_before", "")
        if isinstance(func_text, str) and func_text.strip():
            vec = embed_mean_pool(func_text, tokenizer, model)
        else:
            vec = torch.zeros(UNIXCODER_EMBEDDING_DIM)

        torch.save(vec, out_path)
        processed += 1
        progress.set_postfix(processed=processed, skipped=skipped)

    logger.info(
        "Function embedding complete: %d processed, %d skipped (existing)",
        processed, skipped,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute UniXcoder embeddings for functions")
    parser.add_argument("--input", type=str, default=None, help="Subset CSV path")
    parser.add_argument("--output", type=str, default=None, help="Output embeddings directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    embed_functions(input_csv=args.input, output_dir=args.output)
