"""Tests for the PrimeVul training adapter."""

import pandas as pd
from datasets import Dataset, DatasetDict

from graph_vul.dataset.primevul_adapter import load_primevul_training_data


def test_load_primevul_training_data_normalizes_schema(monkeypatch):
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "project": ["proj-a"],
                    "commit_id": ["abc123"],
                    "target": [1],
                    "func": ["int f() { return 0; }"],
                    "cwe": [["416"]],
                    "cve": ["CVE-1"],
                    "cve_desc": ["desc"],
                }
            ),
            "validation": Dataset.from_dict(
                {
                    "project": ["proj-b"],
                    "commit_id": ["def456"],
                    "target": [0],
                    "func": ["int g() { return 1; }"],
                    "cwe": [["CWE-119"]],
                    "cve": ["CVE-2"],
                    "cve_desc": ["desc2"],
                }
            ),
            "test": Dataset.from_dict(
                {
                    "project": ["proj-c"],
                    "commit_id": ["ghi789"],
                    "target": [1],
                    "func": ["int h() { return 2; }"],
                    "cwe": [[]],
                    "cve": [""],
                    "cve_desc": [""],
                }
            ),
        }
    )

    monkeypatch.setattr("graph_vul.dataset.primevul_adapter.load_dataset", lambda *args, **kwargs: dataset)

    df = load_primevul_training_data()

    assert list(df.columns) == [
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
    assert df["sample_id"].tolist() == [0, 1, 2]
    assert df["source_split"].tolist() == ["train", "val", "test"]
    assert df["cwe_id"].tolist() == ["CWE-416", "CWE-119", ""]
    assert df["vul"].tolist() == [1, 0, 1]
