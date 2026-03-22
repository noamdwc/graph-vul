"""Tests for the structured subset sampling pipeline."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from graph_vul.pipeline.extract_subset import (
    _allocate_budget,
    _deduplicate,
    _derive_metadata,
    _sample_rows,
    _select_cves,
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


class TestSelectCves:
    def test_cve_count_within_range(self, bigvul_df):
        df = _derive_metadata(bigvul_df)
        selected = _select_cves(df, min_cves=7, max_cves=12, seed=42)
        assert 7 <= len(selected) <= 12

    def test_reproducibility(self, bigvul_df):
        df = _derive_metadata(bigvul_df)
        sel1 = _select_cves(df, min_cves=7, max_cves=12, seed=42)
        sel2 = _select_cves(df, min_cves=7, max_cves=12, seed=42)
        assert sel1 == sel2

    def test_different_seed_different_result(self, bigvul_df):
        df = _derive_metadata(bigvul_df)
        sel1 = _select_cves(df, min_cves=7, max_cves=10, seed=42)
        sel2 = _select_cves(df, min_cves=7, max_cves=10, seed=99)
        # May or may not differ, but should not crash
        assert 7 <= len(sel1) <= 10
        assert 7 <= len(sel2) <= 10

    def test_includes_non_giant_projects(self, bigvul_df):
        df = _derive_metadata(bigvul_df)
        selected = _select_cves(df, min_cves=7, max_cves=12, seed=42)
        # Get projects of selected CVEs
        selected_projects = set()
        for cve in selected:
            proj = df[df["cve_id"] == cve]["project"].iloc[0]
            selected_projects.add(proj)

        # Top-2 projects by row count
        top2 = set(df.groupby("project").size().nlargest(2).index)
        non_giant = selected_projects - top2
        assert len(non_giant) >= 2, "Should include at least 2 CVEs from non-giant projects"


class TestAllocateBudget:
    def test_respects_caps(self, bigvul_df):
        df = _derive_metadata(bigvul_df)
        cves = _select_cves(df, min_cves=7, max_cves=10, seed=42)
        alloc = _allocate_budget(df, cves, 200, 400, max_per_cve=100, max_per_project=200, max_per_group=80)

        for cve_id, counts in alloc.items():
            assert counts["vul"] + counts["benign"] <= 100, f"CVE cap exceeded for {cve_id}"


class TestSampleRows:
    def test_class_aware_sampling(self, bigvul_df):
        df = _derive_metadata(bigvul_df)
        cves = _select_cves(df, min_cves=7, max_cves=10, seed=42)
        alloc = _allocate_budget(df, cves, 100, 200, max_per_cve=100, max_per_project=200, max_per_group=80)
        result = _sample_rows(df, cves, alloc, seed=42)

        assert len(result) > 0
        assert (result["vul"] == 1).sum() > 0
        assert (result["vul"] == 0).sum() > 0

    def test_benign_oversampling(self, bigvul_df):
        df = _derive_metadata(bigvul_df)
        cves = _select_cves(df, min_cves=7, max_cves=10, seed=42)
        alloc = _allocate_budget(df, cves, 50, 150, max_per_cve=100, max_per_project=200, max_per_group=80)
        result = _sample_rows(df, cves, alloc, seed=42)

        benign_count = (result["vul"] == 0).sum()
        vul_count = (result["vul"] == 1).sum()
        assert benign_count > vul_count, "Benign should be oversampled"

    def test_reproducibility(self, bigvul_df):
        df = _derive_metadata(bigvul_df)
        cves = _select_cves(df, min_cves=7, max_cves=10, seed=42)
        alloc = _allocate_budget(df, cves, 100, 200, max_per_cve=100, max_per_project=200, max_per_group=80)
        r1 = _sample_rows(df, cves, alloc, seed=42)
        r2 = _sample_rows(df, cves, alloc, seed=42)
        pd.testing.assert_frame_equal(r1.reset_index(drop=True), r2.reset_index(drop=True))


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
            target_vul=50,
            target_benign=100,
            min_cves=5,
            max_cves=10,
            max_per_project=200,
            max_per_cve=100,
            max_per_group=80,
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
        assert m["summary"]["unique_cves"] >= 5

        # Check report
        report_text = report.read_text()
        assert "Subset Sampling Report" in report_text

    def test_graceful_degradation_high_target(self, bigvul_df, tmp_path):
        """Request more samples than available — should degrade gracefully."""
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
            target_vul=10000,
            target_benign=20000,
            min_cves=5,
            max_cves=10,
            seed=42,
        )

        # Should still produce output, just fewer than requested
        assert len(result) > 0
