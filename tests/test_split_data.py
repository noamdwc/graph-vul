"""Tests for the leakage-aware split pipeline."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from graph_vul.pipeline.split_data import (
    _assign_splits,
    _select_held_out_projects,
    _validate_splits,
    split_data,
)


class TestSelectHeldOutProjects:
    def test_correct_count(self, subset_df):
        held_out = _select_held_out_projects(subset_df, n_held_out=3, seed=42)
        assert len(held_out) == 3

    def test_held_out_have_both_classes(self, subset_df):
        held_out = _select_held_out_projects(subset_df, n_held_out=3, seed=42)
        for proj in held_out:
            proj_df = subset_df[subset_df["project"] == proj]
            assert proj_df["vul"].nunique() >= 2, f"Held-out project {proj} should have both classes"

    def test_reproducibility(self, subset_df):
        h1 = _select_held_out_projects(subset_df, n_held_out=3, seed=42)
        h2 = _select_held_out_projects(subset_df, n_held_out=3, seed=42)
        assert sorted(h1) == sorted(h2)

    def test_prefers_non_giant_projects(self, subset_df):
        top2 = set(subset_df.groupby("project").size().nlargest(2).index)
        held_out = _select_held_out_projects(subset_df, n_held_out=3, seed=42)
        # At least some held-out projects should be non-giant (may fall back if needed)
        non_giant = [p for p in held_out if p not in top2]
        # With enough candidate projects, should prefer non-giant
        n_candidates = subset_df.groupby("project").apply(
            lambda g: (g["vul"].sum() > 0) and ((g["vul"] == 0).sum() > 0)
        ).sum()
        if n_candidates > 5:
            assert len(non_giant) >= 1, "Should prefer at least 1 non-giant project"


class TestAssignSplits:
    def test_all_rows_assigned(self, subset_df):
        held_out = _select_held_out_projects(subset_df, n_held_out=2, seed=42)
        assignments = _assign_splits(subset_df, held_out, seed=42)
        assert len(assignments) == len(subset_df)
        assert set(assignments["split"].unique()) <= {"train", "val", "test"}

    def test_zero_project_overlap(self, subset_df):
        held_out = _select_held_out_projects(subset_df, n_held_out=2, seed=42)
        assignments = _assign_splits(subset_df, held_out, seed=42)
        merged = subset_df.merge(assignments, on="sample_id")

        train_projects = set(merged[merged["split"] == "train"]["project"])
        val_projects = set(merged[merged["split"] == "val"]["project"])
        test_projects = set(merged[merged["split"] == "test"]["project"])

        assert train_projects & val_projects == set(), "Train and val should not share projects"
        assert train_projects & test_projects == set(), "Train and test should not share projects"
        assert val_projects & test_projects == set(), "Val and test should not share projects"

    def test_held_out_not_in_train(self, subset_df):
        held_out = _select_held_out_projects(subset_df, n_held_out=2, seed=42)
        assignments = _assign_splits(subset_df, held_out, seed=42)
        merged = subset_df.merge(assignments, on="sample_id")

        train_projects = set(merged[merged["split"] == "train"]["project"])
        for proj in held_out:
            assert proj not in train_projects, f"Held-out project {proj} should not be in train"

    def test_held_out_in_val_or_test(self, subset_df):
        held_out = _select_held_out_projects(subset_df, n_held_out=2, seed=42)
        assignments = _assign_splits(subset_df, held_out, seed=42)
        merged = subset_df.merge(assignments, on="sample_id")

        eval_projects = set(merged[merged["split"].isin(["val", "test"])]["project"])
        for proj in held_out:
            assert proj in eval_projects, f"Held-out project {proj} should be in val or test"

    def test_deterministic(self, subset_df):
        held_out = _select_held_out_projects(subset_df, n_held_out=2, seed=42)
        a1 = _assign_splits(subset_df, held_out, seed=42)
        a2 = _assign_splits(subset_df, held_out, seed=42)
        pd.testing.assert_frame_equal(a1, a2)


class TestValidateSplits:
    def test_no_warnings_on_valid_split(self, subset_df):
        held_out = _select_held_out_projects(subset_df, n_held_out=2, seed=42)
        assignments = _assign_splits(subset_df, held_out, seed=42)
        warnings = _validate_splits(subset_df, assignments)
        # Should have no overlap warnings
        overlap_warnings = [w for w in warnings if "OVERLAP" in w]
        assert len(overlap_warnings) == 0

    def test_warns_on_empty_split(self):
        df = pd.DataFrame({
            "sample_id": range(10),
            "project": ["A"] * 10,
            "vul": [1] * 5 + [0] * 5,
        })
        assignments = pd.DataFrame({
            "sample_id": range(10),
            "split": ["train"] * 10,
        })
        warnings = _validate_splits(df, assignments)
        empty_warnings = [w for w in warnings if "empty" in w.lower()]
        assert len(empty_warnings) >= 1


class TestSplitDataEndToEnd:
    def test_full_pipeline(self, subset_df, tmp_path):
        input_csv = tmp_path / "subset.csv"
        subset_df.to_csv(input_csv, index=False)

        output_dir = tmp_path / "splits"
        split_data(
            input_csv=str(input_csv),
            output_dir=str(output_dir),
            n_held_out=2,
            seed=42,
        )

        # Check files
        assert (output_dir / "assignments.csv").exists()
        assert (output_dir / "split_report.md").exists()

        # Check assignments
        assignments = pd.read_csv(output_dir / "assignments.csv")
        assert set(assignments.columns) == {"sample_id", "split"}
        assert set(assignments["split"].unique()) <= {"train", "val", "test"}
        assert len(assignments) == len(subset_df)

        # Check report
        report = (output_dir / "split_report.md").read_text()
        assert "Split Diagnostics Report" in report
        assert "Overlap Verification" in report

    def test_project_disjoint_end_to_end(self, subset_df, tmp_path):
        input_csv = tmp_path / "subset.csv"
        subset_df.to_csv(input_csv, index=False)

        output_dir = tmp_path / "splits"
        split_data(input_csv=str(input_csv), output_dir=str(output_dir), seed=42)

        assignments = pd.read_csv(output_dir / "assignments.csv")
        merged = subset_df.merge(assignments, on="sample_id")

        # Verify project disjointness
        for s1, s2 in [("train", "val"), ("train", "test"), ("val", "test")]:
            p1 = set(merged[merged["split"] == s1]["project"])
            p2 = set(merged[merged["split"] == s2]["project"])
            assert p1 & p2 == set(), f"Projects overlap between {s1} and {s2}: {p1 & p2}"
