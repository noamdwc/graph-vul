"""Extract a balanced CWE-119 subset from the Big-Vul dataset."""

import logging
import pandas as pd

from graph_vul.config import BIGVUL_CSV, SUBSET_CSV, CWE_ID, TARGET_SAMPLES

logger = logging.getLogger(__name__)


def extract_subset(
    input_csv: str | None = None,
    output_csv: str | None = None,
    target_per_class: int = TARGET_SAMPLES,
) -> pd.DataFrame:
    src = input_csv or str(BIGVUL_CSV)
    dst = output_csv or str(SUBSET_CSV)

    logger.info("Reading Big-Vul CSV from %s", src)
    df = pd.read_csv(src)

    # Filter to CWE-119
    vul = df[(df["cwe_id"] == CWE_ID) & (df["vul"] == 1)]
    benign = df[df["vul"] == 0]

    n_vul = min(len(vul), target_per_class)
    if n_vul < target_per_class:
        logger.warning(
            "Only %d vulnerable CWE-119 samples available (target: %d)",
            n_vul,
            target_per_class,
        )

    vul_sample = vul.sample(n=n_vul, random_state=42)
    benign_sample = benign.sample(n=n_vul, random_state=42)

    subset = pd.concat([vul_sample, benign_sample], ignore_index=True)
    subset = subset.sample(frac=1, random_state=42).reset_index(drop=True)

    # Keep only the columns we need
    keep_cols = []
    for col in ["func_before", "cve_id", "cve_desc", "vul"]:
        if col in subset.columns:
            keep_cols.append(col)
    # Add an explicit sample_id
    subset = subset[keep_cols].copy()
    subset.insert(0, "sample_id", range(len(subset)))

    subset.to_csv(dst, index=False)
    logger.info("Wrote %d samples (%d vul, %d benign) to %s", len(subset), n_vul, n_vul, dst)

    return subset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract CWE-119 subset from Big-Vul")
    parser.add_argument("--input", type=str, default=None, help="Path to Big-Vul CSV")
    parser.add_argument("--output", type=str, default=None, help="Path to output subset CSV")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    extract_subset(input_csv=args.input, output_csv=args.output)
