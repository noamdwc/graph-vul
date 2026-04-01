"""Regression checks for CWE description coverage."""

import csv
import json
from pathlib import Path


def _load_func_subset_cwes() -> set[str]:
    func_subset_path = Path("data/func_subset.csv")
    with func_subset_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            row["cwe_id"].strip()
            for row in reader
            if row.get("cwe_id") and row["cwe_id"].strip()
        }


def test_cwe_descriptions_cover_all_expected_dataset_cwes():
    descriptions_path = Path("data/cwe_descriptions.json")
    descriptions = json.loads(descriptions_path.read_text())
    func_subset_cwes = _load_func_subset_cwes()

    assert func_subset_cwes.issubset(descriptions.keys())
    assert all(isinstance(value, str) and value.strip() for value in descriptions.values())


def test_cwe_descriptions_include_primevul_special_labels():
    descriptions = json.loads(Path("data/cwe_descriptions.json").read_text())
    assert "NVD-CWE-Other" in descriptions
    assert "NVD-CWE-noinfo" in descriptions
