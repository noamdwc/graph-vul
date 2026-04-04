"""Tests for the function-length partition pipeline stage."""

import pandas as pd
import pytest

from graph_vul.pipeline.partition_by_length import compute_token_count, partition


def _make_df(token_counts: list[int]) -> pd.DataFrame:
    """Build a minimal func_subset DataFrame with controlled token counts."""
    rows = []
    for i, tc in enumerate(token_counts):
        # Create func_before with exactly `tc` whitespace tokens
        code = " ".join([f"tok{j}" for j in range(tc)])
        rows.append({
            "sample_id": f"s{i}",
            "func_before": code,
            "vul": i % 2,
            "cwe_id": f"CWE-{100 + i % 3}",
            "project": f"proj{i % 2}",
        })
    return pd.DataFrame(rows)


class TestComputeTokenCount:
    def test_adds_column(self):
        df = _make_df([10, 50, 100])
        result = compute_token_count(df)
        assert "code_token_count" in result.columns
        assert list(result["code_token_count"]) == [10, 50, 100]

    def test_preserves_existing_column(self):
        df = _make_df([10])
        df["code_token_count"] = [999]
        result = compute_token_count(df)
        assert result["code_token_count"].iloc[0] == 999


class TestPartition:
    def test_basic_split(self):
        df = _make_df([10, 30, 64, 65, 100, 200])
        short, other = partition(df, threshold=64)
        assert len(short) == 3  # 10, 30, 64
        assert len(other) == 3  # 65, 100, 200

    def test_boundary_inclusive(self):
        df = _make_df([64])
        short, other = partition(df, threshold=64)
        assert len(short) == 1
        assert len(other) == 0

    def test_boundary_exclusive(self):
        df = _make_df([65])
        short, other = partition(df, threshold=64)
        assert len(short) == 0
        assert len(other) == 1

    def test_row_count_preserved(self):
        df = _make_df([1, 20, 50, 64, 65, 128, 300])
        short, other = partition(df, threshold=64)
        assert len(short) + len(other) == len(df)

    def test_no_overlap(self):
        df = _make_df([10, 64, 65, 200])
        short, other = partition(df, threshold=64)
        short_ids = set(short["sample_id"])
        other_ids = set(other["sample_id"])
        assert short_ids.isdisjoint(other_ids)

    def test_all_short(self):
        df = _make_df([1, 10, 64])
        short, other = partition(df, threshold=64)
        assert len(short) == 3
        assert len(other) == 0

    def test_all_other(self):
        df = _make_df([65, 100, 500])
        short, other = partition(df, threshold=64)
        assert len(short) == 0
        assert len(other) == 3

    def test_columns_preserved(self):
        df = _make_df([10, 100])
        short, other = partition(df, threshold=64)
        expected_cols = set(df.columns) | {"code_token_count"}
        assert set(short.columns) == expected_cols
        assert set(other.columns) == expected_cols


class TestMainCLI:
    def test_end_to_end(self, tmp_path):
        df = _make_df([10, 30, 64, 65, 100])
        input_path = tmp_path / "func_subset.csv"
        df.to_csv(input_path, index=False)

        from graph_vul.pipeline.partition_by_length import main

        out_dir = tmp_path / "partitions"
        main(input_path, out_dir)

        short_df = pd.read_csv(out_dir / "short" / "func_subset.csv")
        other_df = pd.read_csv(out_dir / "other" / "func_subset.csv")

        assert len(short_df) == 3
        assert len(other_df) == 2
        assert len(short_df) + len(other_df) == len(df)
