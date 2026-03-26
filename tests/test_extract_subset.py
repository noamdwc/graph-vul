"""Tests for the subset extraction pipeline."""

import json
from pathlib import Path

import pandas as pd
import pytest

from graph_vul.pipeline.extract_subset import (
    _deduplicate,
    _derive_metadata,
    extract_subset,
)


class TestDeriveMetadata:
    def test_adds_metadata_columns(self, bigvul_df):
        result = _derive_metadata(bigvul_df)
        assert "code_char_len" in result.columns
        assert "code_line_count" in result.columns
        assert "code_token_count" in result.columns
        assert "code_char_len_bucket" in result.columns

    def test_handles_empty_func_before(self):
        df = pd.DataFrame({"func_before": [None, "", "int x = 1;"]})
        result = _derive_metadata(df)
        assert len(result) == 3
        assert result["code_char_len"].iloc[0] == 0  # None → ""


class TestDeduplicate:
    def test_removes_exact_duplicates(self):
        df = pd.DataFrame({
            "func_before": ["void foo() {}", "void foo() {}", "void bar() {}"],
            "vul": [1, 0, 0],
        })
        result = _deduplicate(df)
        assert len(result) == 2

    def test_no_duplicates_no_change(self):
        df = pd.DataFrame({
            "func_before": ["void foo() {}", "void bar() {}"],
            "vul": [1, 0],
        })
        result = _deduplicate(df)
        assert len(result) == 2


class TestExtractSubsetEndToEnd:
    def test_full_pipeline(self, bigvul_df, tmp_path):
        # Write synthetic BigVul CSV
        input_csv = tmp_path / "bigvul.csv"
        # Need to rename columns back to raw format for _read_bigvul_csv
        raw_df = bigvul_df.rename(columns={
            "cwe_id": "CWE ID",
            "cve_id": "CVE ID",
            "cve_desc": "Summary",
        })
        raw_df.to_csv(input_csv, index=False)

        output_csv = tmp_path / "subset.csv"
        manifest = tmp_path / "manifest.json"
        report = tmp_path / "report.md"

        result = extract_subset(
            input_csv=str(input_csv),
            output_csv=str(output_csv),
            output_manifest=str(manifest),
            output_report=str(report),
            deduplicate=True,
            seed=42,
        )

        assert len(result) > 0
        assert "sample_id" in result.columns
        assert "project" in result.columns
        assert "func_before" in result.columns
        assert "vul" in result.columns

        # Check files written
        assert output_csv.exists()
        assert manifest.exists()
        assert report.exists()

        # Check manifest
        with open(manifest) as f:
            m = json.load(f)
        assert "summary" in m
        assert m["summary"]["total_rows"] == len(result)

        # Check report
        report_text = report.read_text()
        assert "Subset Report" in report_text

    def test_keeps_all_cwe_rows(self, bigvul_df, tmp_path):
        """Should keep all CWE-119 rows after deduplication."""
        input_csv = tmp_path / "bigvul.csv"
        raw_df = bigvul_df.rename(columns={
            "cwe_id": "CWE ID",
            "cve_id": "CVE ID",
            "cve_desc": "Summary",
        })
        raw_df.to_csv(input_csv, index=False)

        result = extract_subset(
            input_csv=str(input_csv),
            output_csv=str(tmp_path / "subset.csv"),
            output_manifest=str(tmp_path / "manifest.json"),
            output_report=str(tmp_path / "report.md"),
            deduplicate=False,
            seed=42,
        )

        # Without dedup, should keep all CWE-119 rows
        cwe119_count = (bigvul_df["cwe_id"] == "CWE-119").sum()
        assert len(result) == cwe119_count
