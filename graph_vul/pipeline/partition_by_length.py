"""Partition func_subset into short (<=N tokens) and other (>N tokens) subsets."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from graph_vul.config import (
    DATA_DIR,
    FUNC_SUBSET_CSV,
    SHORT_FUNCTION_MAX_TOKENS,
)

logger = logging.getLogger(__name__)


def compute_token_count(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``code_token_count`` column if not already present."""
    if "code_token_count" not in df.columns:
        code = df["func_before"].fillna("")
        df["code_token_count"] = code.str.split().str.len().fillna(0).astype(int)
    return df


def partition(
    df: pd.DataFrame,
    threshold: int = SHORT_FUNCTION_MAX_TOKENS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* into (short, other) based on ``code_token_count``."""
    df = compute_token_count(df)
    short_mask = df["code_token_count"] <= threshold
    return df[short_mask].copy(), df[~short_mask].copy()


def _log_partition_summary(name: str, df: pd.DataFrame) -> None:
    vul_count = int(df["vul"].sum()) if "vul" in df.columns else 0
    projects = df["project"].nunique() if "project" in df.columns else 0
    cwes = df["cwe_id"].nunique() if "cwe_id" in df.columns else 0
    logger.info(
        "Partition %-6s: %d rows, %d vulnerable, %d projects, %d CWEs",
        name, len(df), vul_count, projects, cwes,
    )


def main(input_path: Path, output_dir: Path) -> None:
    logger.info("Reading %s", input_path)
    df = pd.read_csv(input_path)

    short_df, other_df = partition(df)

    short_dir = output_dir / "short"
    other_dir = output_dir / "other"
    short_dir.mkdir(parents=True, exist_ok=True)
    other_dir.mkdir(parents=True, exist_ok=True)

    short_path = short_dir / "func_subset.csv"
    other_path = other_dir / "func_subset.csv"

    short_df.to_csv(short_path, index=False)
    other_df.to_csv(other_path, index=False)

    _log_partition_summary("short", short_df)
    _log_partition_summary("other", other_df)

    logger.info("Wrote %s (%d rows)", short_path, len(short_df))
    logger.info("Wrote %s (%d rows)", other_path, len(other_df))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Partition func_subset by token length")
    parser.add_argument(
        "--input", type=Path, default=FUNC_SUBSET_CSV,
        help="Path to func_subset CSV",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DATA_DIR / "partitions",
        help="Parent directory for partition outputs",
    )
    args = parser.parse_args()
    main(args.input, args.output_dir)
