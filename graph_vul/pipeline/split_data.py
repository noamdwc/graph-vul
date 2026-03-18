"""Split dataset indices into train/val/test sets and write JSON index files."""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from graph_vul.config import SEED, SPLITS_DIR, SUBSET_CSV, TEST_SPLIT, VAL_SPLIT

logger = logging.getLogger(__name__)


def split_data(
    input_csv: str = str(SUBSET_CSV),
    output_dir: str = str(SPLITS_DIR),
    seed: int = SEED,
) -> None:
    df = pd.read_csv(input_csv)
    indices = list(range(len(df)))
    labels = df["vul"].tolist()

    # Two-step stratified split: train (70%) | temp (30%) -> val (15%) | test (15%)
    temp_size = VAL_SPLIT + TEST_SPLIT
    train_idx, temp_idx = train_test_split(
        indices, test_size=temp_size, stratify=labels, random_state=seed
    )

    temp_labels = [labels[i] for i in temp_idx]
    val_ratio = VAL_SPLIT / temp_size
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(1 - val_ratio), stratify=temp_labels, random_state=seed
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, idx_list in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        path = out / f"{name}.json"
        with open(path, "w") as f:
            json.dump(sorted(idx_list), f)
        logger.info("Wrote %d indices to %s", len(idx_list), path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data into train/val/test")
    parser.add_argument("--input", type=str, default=str(SUBSET_CSV))
    parser.add_argument("--output", type=str, default=str(SPLITS_DIR))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    split_data(input_csv=args.input, output_dir=args.output, seed=args.seed)
