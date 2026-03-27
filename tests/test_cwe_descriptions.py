"""Regression checks for CWE description coverage."""

import csv
import json
from pathlib import Path


EXPECTED_DATASET_CWES = {
    "CWE-16",
    "CWE-17",
    "CWE-18",
    "CWE-19",
    "CWE-20",
    "CWE-22",
    "CWE-59",
    "CWE-74",
    "CWE-77",
    "CWE-78",
    "CWE-79",
    "CWE-89",
    "CWE-90",
    "CWE-93",
    "CWE-94",
    "CWE-119",
    "CWE-120",
    "CWE-125",
    "CWE-129",
    "CWE-134",
    "CWE-172",
    "CWE-189",
    "CWE-190",
    "CWE-191",
    "CWE-200",
    "CWE-209",
    "CWE-252",
    "CWE-254",
    "CWE-255",
    "CWE-264",
    "CWE-269",
    "CWE-281",
    "CWE-284",
    "CWE-285",
    "CWE-287",
    "CWE-290",
    "CWE-295",
    "CWE-310",
    "CWE-311",
    "CWE-320",
    "CWE-327",
    "CWE-330",
    "CWE-331",
    "CWE-345",
    "CWE-346",
    "CWE-347",
    "CWE-352",
    "CWE-354",
    "CWE-358",
    "CWE-361",
    "CWE-362",
    "CWE-369",
    "CWE-388",
    "CWE-399",
    "CWE-400",
    "CWE-404",
    "CWE-415",
    "CWE-416",
    "CWE-426",
    "CWE-436",
    "CWE-476",
    "CWE-494",
    "CWE-502",
    "CWE-522",
    "CWE-532",
    "CWE-601",
    "CWE-611",
    "CWE-617",
    "CWE-664",
    "CWE-665",
    "CWE-668",
    "CWE-674",
    "CWE-682",
    "CWE-693",
    "CWE-704",
    "CWE-706",
    "CWE-732",
    "CWE-754",
    "CWE-755",
    "CWE-763",
    "CWE-769",
    "CWE-770",
    "CWE-772",
    "CWE-787",
    "CWE-824",
    "CWE-834",
    "CWE-835",
    "CWE-862",
    "CWE-909",
    "CWE-918",
    "CWE-1021",
}


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

    assert set(descriptions) == EXPECTED_DATASET_CWES
    assert len(descriptions) == 91
    assert all(isinstance(value, str) and value.strip() for value in descriptions.values())


def test_cwe_descriptions_cover_func_subset_and_pre_filter_extras():
    descriptions = json.loads(Path("data/cwe_descriptions.json").read_text())
    func_subset_cwes = _load_func_subset_cwes()

    assert func_subset_cwes.issubset(descriptions.keys())
    assert len(func_subset_cwes) == 89
    assert EXPECTED_DATASET_CWES - func_subset_cwes == {"CWE-209", "CWE-252"}
