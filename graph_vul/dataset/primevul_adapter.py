"""PrimeVul adapter for the function-level training pipeline."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd
from datasets import DatasetDict, load_dataset

from graph_vul.config import PRIMEVUL_DATASET_CONFIG, PRIMEVUL_DATASET_NAME

logger = logging.getLogger(__name__)

OFFICIAL_SPLIT_MAP = {
    "train": "train",
    "validation": "val",
    "val": "val",
    "test": "test",
}


def _normalise_cwe(cwe_values: object) -> str:
    if cwe_values is None:
        return ""
    if isinstance(cwe_values, str):
        values: Iterable[object] = [cwe_values]
    elif isinstance(cwe_values, Iterable):
        values = cwe_values
    else:
        values = [cwe_values]

    values = list(values)
    if not values:
        return ""

    first = str(values[0]).strip()
    if not first:
        return ""
    if first.startswith("CWE-"):
        return first
    if first.isdigit():
        return f"CWE-{first}"
    return first


def _normalize_split_name(split_name: str) -> str:
    normalized = OFFICIAL_SPLIT_MAP.get(split_name.lower())
    if normalized is None:
        raise ValueError(f"Unsupported PrimeVul split name: {split_name}")
    return normalized


def load_primevul_training_data(
    dataset_name: str = PRIMEVUL_DATASET_NAME,
    dataset_config: str = PRIMEVUL_DATASET_CONFIG,
) -> pd.DataFrame:
    """Load PrimeVul official splits and normalize to the function training schema."""
    logger.info("Loading PrimeVul training data from Hugging Face: %s (%s)", dataset_name, dataset_config)
    dataset = load_dataset(dataset_name, dataset_config)
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected DatasetDict from load_dataset, got {type(dataset)!r}")

    normalized_parts: list[pd.DataFrame] = []
    next_sample_id = 0
    for raw_split_name in ("train", "validation", "test"):
        if raw_split_name not in dataset:
            raise ValueError(f"PrimeVul split '{raw_split_name}' not found in dataset")

        split_df = dataset[raw_split_name].to_pandas()
        split_df = split_df.rename(columns={"func": "func_before", "target": "vul"})
        split_df["source_split"] = _normalize_split_name(raw_split_name)
        split_df["cwe_id"] = split_df["cwe"].apply(_normalise_cwe)
        split_df["project"] = split_df.get("project", pd.Series(["primevul"] * len(split_df))).fillna("primevul")
        split_df["commit_id"] = split_df.get("commit_id", pd.Series([""] * len(split_df))).fillna("")
        split_df["cve_id"] = split_df.get("cve", pd.Series([""] * len(split_df))).fillna("")
        split_df["cve_desc"] = split_df.get("cve_desc", pd.Series([""] * len(split_df))).fillna("")
        split_df["sample_id"] = range(next_sample_id, next_sample_id + len(split_df))
        next_sample_id += len(split_df)

        normalized_parts.append(
            split_df[
                [
                    "sample_id",
                    "project",
                    "commit_id",
                    "func_before",
                    "cve_id",
                    "cve_desc",
                    "cwe_id",
                    "vul",
                    "source_split",
                ]
            ].copy()
        )

    combined = pd.concat(normalized_parts, ignore_index=True)
    combined["vul"] = combined["vul"].astype(int)
    combined["func_before"] = combined["func_before"].fillna("").astype(str)
    logger.info(
        "Loaded PrimeVul official splits: %d rows (%d train / %d val / %d test)",
        len(combined),
        int((combined["source_split"] == "train").sum()),
        int((combined["source_split"] == "val").sum()),
        int((combined["source_split"] == "test").sum()),
    )
    return combined
