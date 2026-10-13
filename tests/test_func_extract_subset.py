"""Tests for the function-level subset extraction pipeline."""

import json

import pandas as pd

from graph_vul.pipeline import func_extract_subset as subset_module


def _write_raw_bigvul(df: pd.DataFrame, path) -> None:
    df.to_csv(path, index=False)


class TestExtractFuncSubset:
    def test_keeps_all_labeled_cwes_not_just_curated_lists(self, tmp_path):
        df = pd.DataFrame(
            [
                {
                    "cwe_id": "CWE-119",
                    "cve_id": "CVE-1",
                    "cve_desc": "buffer overflow",
                    "project": "ProjectA",
                    "func_before": "int a = 0;",
                    "vul": 1,
                },
                {
                    "cwe_id": "CWE-999",
                    "cve_id": "CVE-2",
                    "cve_desc": "rare cwe",
                    "project": "ProjectB",
                    "func_before": "int b = 0;",
                    "vul": 0,
                },
                {
                    "cwe_id": None,
                    "cve_id": "CVE-3",
                    "cve_desc": "missing label",
                    "project": "ProjectC",
                    "func_before": "int c = 0;",
                    "vul": 1,
                },
            ]
        )
        input_csv = tmp_path / "bigvul.csv"
        _write_raw_bigvul(df, input_csv)

        result = subset_module.extract_func_subset(
            input_csv=str(input_csv),
            output_csv=str(tmp_path / "func_subset.csv"),
            output_manifest=str(tmp_path / "manifest.json"),
            output_report=str(tmp_path / "report.md"),
            deduplicate=False,
            seed=42,
        )

        assert set(result["cwe_id"]) == {"CWE-119", "CWE-999"}
        assert len(result) == 2

    def test_preserves_first_duplicate_and_token_threshold(self, tmp_path):
        tokens_512 = " ".join(f"tok{i}" for i in range(512))
        tokens_513 = " ".join(f"tok{i}" for i in range(513))
        df = pd.DataFrame(
            [
                {
                    "cwe_id": "CWE-119",
                    "cve_id": "CVE-1",
                    "cve_desc": "first duplicate",
                    "project": "ProjectA",
                    "func_before": "duplicate();",
                    "vul": 1,
                },
                {
                    "cwe_id": "CWE-119",
                    "cve_id": "CVE-2",
                    "cve_desc": "second duplicate",
                    "project": "ProjectA",
                    "func_before": "duplicate();",
                    "vul": 1,
                },
                {
                    "cwe_id": "CWE-125",
                    "cve_id": "CVE-3",
                    "cve_desc": "kept at threshold",
                    "project": "ProjectB",
                    "func_before": tokens_512,
                    "vul": 1,
                },
                {
                    "cwe_id": "CWE-125",
                    "cve_id": "CVE-4",
                    "cve_desc": "too long",
                    "project": "ProjectB",
                    "func_before": tokens_513,
                    "vul": 0,
                },
            ]
        )
        input_csv = tmp_path / "bigvul.csv"
        _write_raw_bigvul(df, input_csv)

        result = subset_module.extract_func_subset(
            input_csv=str(input_csv),
            output_csv=str(tmp_path / "func_subset.csv"),
            output_manifest=str(tmp_path / "manifest.json"),
            output_report=str(tmp_path / "report.md"),
            deduplicate=True,
            seed=42,
        )

        assert len(result) == 2
        assert result.iloc[0]["cve_id"] == "CVE-1"
        assert "CVE-4" not in set(result["cve_id"])
        assert "CVE-3" in set(result["cve_id"])

    def test_removes_only_oversized_project_cwe_groups_and_reports_stages(
        self,
        tmp_path,
        monkeypatch,
    ):
        monkeypatch.setattr(subset_module, "FUNC_MAX_PROJECT_ROWS", 2)
        df = pd.DataFrame(
            [
                {
                    "cwe_id": "CWE-119",
                    "cve_id": "CVE-1",
                    "cve_desc": "small group row 1",
                    "project": "ProjectSmall",
                    "func_before": "a();",
                    "vul": 1,
                },
                {
                    "cwe_id": "CWE-119",
                    "cve_id": "CVE-2",
                    "cve_desc": "small group row 2",
                    "project": "ProjectSmall",
                    "func_before": "b();",
                    "vul": 0,
                },
                {
                    "cwe_id": "CWE-999",
                    "cve_id": "CVE-3",
                    "cve_desc": "large group row 1",
                    "project": "ProjectLarge",
                    "func_before": "c();",
                    "vul": 1,
                },
                {
                    "cwe_id": "CWE-999",
                    "cve_id": "CVE-4",
                    "cve_desc": "large group row 2",
                    "project": "ProjectLarge",
                    "func_before": "d();",
                    "vul": 0,
                },
                {
                    "cwe_id": "CWE-999",
                    "cve_id": "CVE-5",
                    "cve_desc": "large group row 3",
                    "project": "ProjectLarge",
                    "func_before": "e();",
                    "vul": 0,
                },
            ]
        )
        input_csv = tmp_path / "bigvul.csv"
        output_csv = tmp_path / "func_subset.csv"
        manifest_path = tmp_path / "manifest.json"
        report_path = tmp_path / "report.md"
        _write_raw_bigvul(df, input_csv)

        result = subset_module.extract_func_subset(
            input_csv=str(input_csv),
            output_csv=str(output_csv),
            output_manifest=str(manifest_path),
            output_report=str(report_path),
            deduplicate=False,
            seed=42,
        )

        assert set(result["project"]) == {"ProjectSmall"}
        assert len(result) == 2

        manifest = json.loads(manifest_path.read_text())
        assert manifest["config"]["subset_scope"] == "all_labeled_cwes"
        assert manifest["project_cwe_filter"]["removed_group_count"] == 1
        assert [stage["stage"] for stage in manifest["filter_stages"]] == [
            "raw_labeled_input",
            "deduplication",
            "token_length_filter",
            "project_cwe_max_filter",
            "drop_invalid_rows",
        ]
        assert manifest["filter_stages"][-2]["after_rows"] == 2
        assert manifest["top_cwes"][0]["cwe_id"] == "CWE-119"
        assert manifest["token_length_summary"]["count"] == 2

        report_text = report_path.read_text()
        assert "## Filter Stages" in report_text
        assert "## Top CWEs By Row Count" in report_text
        assert "## Token-Length Summary" in report_text
        assert "Removed oversized (project, cwe_id) groups" in report_text
