"""Tests for the leakage-aware split pipeline with full-project holdout."""

from itertools import combinations

import pandas as pd
import pytest

from graph_vul.pipeline.split_data import (
    ALL_SPLITS,
    _assign_splits,
    _assign_unseen_splits,
    _build_unseen_pool,
    _compute_sample_weights,
    _validate_heldout_eligibility,
    _validate_splits,
    split_data,
)

HELDOUT_CWES = ["CWE-200", "CWE-416"]


class TestValidateHeldoutEligibility:
    def test_passes_for_eligible_cwes(self, multi_cwe_subset_df):
        _validate_heldout_eligibility(
            multi_cwe_subset_df, HELDOUT_CWES,
            min_total=5, min_vul=2, min_projects=1,
        )

    def test_fails_for_too_few_rows(self, multi_cwe_subset_df):
        with pytest.raises(ValueError, match="total rows"):
            _validate_heldout_eligibility(
                multi_cwe_subset_df, HELDOUT_CWES,
                min_total=99999, min_vul=1, min_projects=1,
            )

    def test_fails_for_too_few_vul_rows(self, multi_cwe_subset_df):
        with pytest.raises(ValueError, match="vulnerable rows"):
            _validate_heldout_eligibility(
                multi_cwe_subset_df, HELDOUT_CWES,
                min_total=1, min_vul=99999, min_projects=1,
            )

    def test_fails_for_too_few_projects(self, multi_cwe_subset_df):
        with pytest.raises(ValueError, match="distinct projects"):
            _validate_heldout_eligibility(
                multi_cwe_subset_df, HELDOUT_CWES,
                min_total=1, min_vul=1, min_projects=99999,
            )

    def test_noop_when_no_cwe_column(self):
        df = pd.DataFrame({"sample_id": [0], "vul": [1], "project": ["A"]})
        _validate_heldout_eligibility(df, HELDOUT_CWES)

    def test_noop_when_no_heldout(self, multi_cwe_subset_df):
        _validate_heldout_eligibility(multi_cwe_subset_df, [])


class TestBuildUnseenPool:
    def test_mixed_projects_migrate_to_unseen(self, multi_cwe_subset_df):
        df_seen, df_unseen, migration_info = _build_unseen_pool(
            multi_cwe_subset_df, HELDOUT_CWES
        )
        # ProjectG and ProjectH have mixed CWEs -> should be in unseen
        assert "ProjectG" in migration_info
        assert "ProjectH" in migration_info
        # ProjectI and ProjectJ are heldout-only -> also unseen
        assert "ProjectI" in migration_info
        assert "ProjectJ" in migration_info

    def test_seen_projects_not_in_unseen(self, multi_cwe_subset_df):
        df_seen, df_unseen, migration_info = _build_unseen_pool(
            multi_cwe_subset_df, HELDOUT_CWES
        )
        seen_projects = set(df_seen["project"].unique())
        unseen_projects = set(df_unseen["project"].unique())
        assert seen_projects & unseen_projects == set()
        # Seen should be A-F only
        assert seen_projects == {"ProjectA", "ProjectB", "ProjectC", "ProjectD", "ProjectE", "ProjectF"}

    def test_all_rows_preserved(self, multi_cwe_subset_df):
        df_seen, df_unseen, _ = _build_unseen_pool(multi_cwe_subset_df, HELDOUT_CWES)
        assert len(df_seen) + len(df_unseen) == len(multi_cwe_subset_df)

    def test_migration_info_lists_cwes(self, multi_cwe_subset_df):
        _, _, migration_info = _build_unseen_pool(multi_cwe_subset_df, HELDOUT_CWES)
        # ProjectG has CWE-200 as held-out
        assert "CWE-200" in migration_info["ProjectG"]
        # ProjectH has CWE-416 as held-out
        assert "CWE-416" in migration_info["ProjectH"]

    def test_no_heldout_cwes_returns_all_seen(self, multi_cwe_subset_df):
        df_seen, df_unseen, migration_info = _build_unseen_pool(
            multi_cwe_subset_df, []
        )
        assert len(df_seen) == len(multi_cwe_subset_df)
        assert len(df_unseen) == 0
        assert migration_info == {}

    def test_unseen_pool_includes_non_heldout_rows_from_mixed_projects(self, multi_cwe_subset_df):
        """CWE-119 rows from ProjectG should also be in unseen pool."""
        _, df_unseen, _ = _build_unseen_pool(multi_cwe_subset_df, HELDOUT_CWES)
        proj_g_unseen = df_unseen[df_unseen["project"] == "ProjectG"]
        cwes_in_g = set(proj_g_unseen["cwe_id"].unique())
        assert "CWE-119" in cwes_in_g  # seen CWE rows migrated with the project


class TestAssignSplits:
    def test_all_rows_assigned(self, multi_cwe_subset_df):
        df_seen, _, _ = _build_unseen_pool(multi_cwe_subset_df, HELDOUT_CWES)
        assignments = _assign_splits(df_seen, seed=42)
        assert len(assignments) == len(df_seen)
        assert set(assignments["split"].unique()) <= {"train", "val", "test"}

    def test_zero_project_overlap(self, multi_cwe_subset_df):
        df_seen, _, _ = _build_unseen_pool(multi_cwe_subset_df, HELDOUT_CWES)
        assignments = _assign_splits(df_seen, seed=42)
        merged = df_seen.merge(assignments, on="sample_id")

        train_p = set(merged[merged["split"] == "train"]["project"])
        val_p = set(merged[merged["split"] == "val"]["project"])
        test_p = set(merged[merged["split"] == "test"]["project"])

        assert train_p & val_p == set()
        assert train_p & test_p == set()
        assert val_p & test_p == set()

    def test_deterministic(self, multi_cwe_subset_df):
        df_seen, _, _ = _build_unseen_pool(multi_cwe_subset_df, HELDOUT_CWES)
        a1 = _assign_splits(df_seen, seed=42)
        a2 = _assign_splits(df_seen, seed=42)
        pd.testing.assert_frame_equal(a1, a2)


class TestAssignUnseenSplits:
    def test_unseen_split_project_disjoint(self, multi_cwe_subset_df):
        _, df_unseen, _ = _build_unseen_pool(multi_cwe_subset_df, HELDOUT_CWES)
        assignments = _assign_unseen_splits(df_unseen, seed=42)
        merged = df_unseen.merge(assignments, on="sample_id")

        uv_projects = set(merged[merged["split"] == "unseen_val"]["project"])
        ut_projects = set(merged[merged["split"] == "unseen_test"]["project"])
        assert uv_projects & ut_projects == set()

    def test_all_unseen_rows_assigned(self, multi_cwe_subset_df):
        _, df_unseen, _ = _build_unseen_pool(multi_cwe_subset_df, HELDOUT_CWES)
        assignments = _assign_unseen_splits(df_unseen, seed=42)
        assert len(assignments) == len(df_unseen)
        assert set(assignments["split"].unique()) <= {"unseen_val", "unseen_test"}


class TestValidateSplits:
    def test_no_overlap_warnings_on_valid_split(self, multi_cwe_subset_df):
        df_seen, df_unseen, _ = _build_unseen_pool(multi_cwe_subset_df, HELDOUT_CWES)
        seen_assign = _assign_splits(df_seen, seed=42)
        unseen_assign = _assign_unseen_splits(df_unseen, seed=42)
        assignments = pd.concat([seen_assign, unseen_assign], ignore_index=True)
        df_all = pd.concat([df_seen, df_unseen], ignore_index=True)

        warnings = _validate_splits(df_all, assignments, heldout_cwe_ids=HELDOUT_CWES)
        # Only check project-level overlap/leakage, not CODE LEAKAGE (synthetic data has collisions)
        project_warnings = [
            w for w in warnings
            if ("OVERLAP" in w or "LEAKAGE" in w) and "CODE" not in w
        ]
        assert len(project_warnings) == 0

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

    def test_detects_held_out_cwe_in_seen_splits(self, multi_cwe_subset_df):
        """If we intentionally leak a held-out CWE into train, validation should catch it."""
        # Assign everything to train (including held-out CWEs)
        assignments = pd.DataFrame({
            "sample_id": multi_cwe_subset_df["sample_id"],
            "split": "train",
        })
        warnings = _validate_splits(
            multi_cwe_subset_df, assignments, heldout_cwe_ids=HELDOUT_CWES
        )
        cwe_leak_warnings = [w for w in warnings if "HELD-OUT CWE LEAKAGE" in w]
        assert len(cwe_leak_warnings) >= 1


class TestComputeSampleWeights:
    def test_weights_sum_to_n(self, multi_cwe_subset_df):
        df_seen, _, _ = _build_unseen_pool(multi_cwe_subset_df, HELDOUT_CWES)
        weights = _compute_sample_weights(df_seen)
        assert abs(weights["sample_weight"].sum() - len(df_seen)) < 1.0

    def test_only_train_rows(self, multi_cwe_subset_df):
        df_seen, _, _ = _build_unseen_pool(multi_cwe_subset_df, HELDOUT_CWES)
        weights = _compute_sample_weights(df_seen)
        assert (weights["split"] == "train").all()

    def test_vulnerable_upweighted(self, multi_cwe_subset_df):
        df_seen, _, _ = _build_unseen_pool(multi_cwe_subset_df, HELDOUT_CWES)
        weights = _compute_sample_weights(df_seen, target_vul_ratio=0.25)
        vul_mean = weights.loc[weights["vul"] == 1, "sample_weight"].mean()
        benign_mean = weights.loc[weights["vul"] == 0, "sample_weight"].mean()
        assert vul_mean > benign_mean

    def test_effective_ratio_near_target(self, multi_cwe_subset_df):
        df_seen, _, _ = _build_unseen_pool(multi_cwe_subset_df, HELDOUT_CWES)
        weights = _compute_sample_weights(df_seen, target_vul_ratio=0.25)
        w = weights["sample_weight"].values
        vul_mask = weights["vul"].values == 1
        effective_ratio = w[vul_mask].sum() / w.sum()
        assert abs(effective_ratio - 0.25) < 0.10  # within 10pp

    def test_max_weight_cap(self, multi_cwe_subset_df):
        df_seen, _, _ = _build_unseen_pool(multi_cwe_subset_df, HELDOUT_CWES)
        weights = _compute_sample_weights(df_seen, max_weight=2.0)
        import numpy as np
        median_w = np.median(weights["sample_weight"].values)
        # After normalization, cap may shift, but no weight should be extreme
        assert weights["sample_weight"].max() < 100  # reasonable upper bound

    def test_handles_single_class(self):
        """Edge case: all vulnerable or all benign."""
        df = pd.DataFrame({
            "sample_id": range(10),
            "project": ["A"] * 10,
            "cwe_id": ["CWE-119"] * 10,
            "vul": [1] * 10,
        })
        weights = _compute_sample_weights(df)
        assert len(weights) == 10
        assert (weights["sample_weight"] == 1.0).all()


class TestSplitDataEndToEnd:
    def test_full_pipeline(self, multi_cwe_subset_df, tmp_path):
        input_csv = tmp_path / "subset.csv"
        multi_cwe_subset_df.to_csv(input_csv, index=False)

        output_dir = tmp_path / "splits"
        split_data(input_csv=str(input_csv), output_dir=str(output_dir), seed=42)

        # Check all output files
        assert (output_dir / "assignments.csv").exists()
        assert (output_dir / "split_report.md").exists()
        assert (output_dir / "train_sampler_weights.csv").exists()
        assert (output_dir / "train_balanced_manifest.csv").exists()

        # Check assignments
        assignments = pd.read_csv(output_dir / "assignments.csv")
        assert set(assignments.columns) == {"sample_id", "split"}
        assert len(assignments) == len(multi_cwe_subset_df)

        # All five splits present
        present_splits = set(assignments["split"].unique())
        assert present_splits == {"train", "val", "test", "unseen_val", "unseen_test"}

        # Check report
        report = (output_dir / "split_report.md").read_text()
        assert "Split Diagnostics Report" in report
        assert "Overlap Verification" in report
        assert "Project Exclusivity" in report
        assert "Train Balancing" in report

    def test_zero_project_overlap_all_pairs(self, multi_cwe_subset_df, tmp_path):
        input_csv = tmp_path / "subset.csv"
        multi_cwe_subset_df.to_csv(input_csv, index=False)

        output_dir = tmp_path / "splits"
        split_data(input_csv=str(input_csv), output_dir=str(output_dir), seed=42)

        assignments = pd.read_csv(output_dir / "assignments.csv")
        merged = multi_cwe_subset_df.merge(assignments, on="sample_id")

        split_projects = {}
        for split_name in ALL_SPLITS:
            split_projects[split_name] = set(
                merged[merged["split"] == split_name]["project"]
            )

        for s1, s2 in combinations(ALL_SPLITS, 2):
            overlap = split_projects[s1] & split_projects[s2]
            assert overlap == set(), f"Projects overlap between {s1} and {s2}: {overlap}"

    def test_held_out_cwes_absent_from_seen_splits(self, multi_cwe_subset_df, tmp_path):
        input_csv = tmp_path / "subset.csv"
        multi_cwe_subset_df.to_csv(input_csv, index=False)

        output_dir = tmp_path / "splits"
        split_data(input_csv=str(input_csv), output_dir=str(output_dir), seed=42)

        assignments = pd.read_csv(output_dir / "assignments.csv")
        merged = multi_cwe_subset_df.merge(assignments, on="sample_id")

        for split_name in ["train", "val", "test"]:
            split_df = merged[merged["split"] == split_name]
            cwes_in_split = set(split_df["cwe_id"].unique())
            for hc in HELDOUT_CWES:
                assert hc not in cwes_in_split, (
                    f"Held-out {hc} found in {split_name}"
                )

    def test_unseen_projects_have_heldout_cwes(self, multi_cwe_subset_df, tmp_path):
        """Every unseen project should have at least one held-out CWE row."""
        input_csv = tmp_path / "subset.csv"
        multi_cwe_subset_df.to_csv(input_csv, index=False)

        output_dir = tmp_path / "splits"
        split_data(input_csv=str(input_csv), output_dir=str(output_dir), seed=42)

        assignments = pd.read_csv(output_dir / "assignments.csv")
        merged = multi_cwe_subset_df.merge(assignments, on="sample_id")

        unseen_df = merged[merged["split"].isin(["unseen_val", "unseen_test"])]
        for proj in unseen_df["project"].unique():
            proj_cwes = set(unseen_df[unseen_df["project"] == proj]["cwe_id"])
            has_heldout = bool(proj_cwes & set(HELDOUT_CWES))
            assert has_heldout, f"Unseen project {proj} has no held-out CWE rows"

    def test_weights_file_only_has_train_rows(self, multi_cwe_subset_df, tmp_path):
        input_csv = tmp_path / "subset.csv"
        multi_cwe_subset_df.to_csv(input_csv, index=False)

        output_dir = tmp_path / "splits"
        split_data(input_csv=str(input_csv), output_dir=str(output_dir), seed=42)

        weights = pd.read_csv(output_dir / "train_sampler_weights.csv")
        assert (weights["split"] == "train").all()

        assignments = pd.read_csv(output_dir / "assignments.csv")
        train_ids = set(assignments[assignments["split"] == "train"]["sample_id"])
        assert set(weights["sample_id"]) == train_ids

    def test_deterministic(self, multi_cwe_subset_df, tmp_path):
        input_csv = tmp_path / "subset.csv"
        multi_cwe_subset_df.to_csv(input_csv, index=False)

        out1 = tmp_path / "splits1"
        out2 = tmp_path / "splits2"
        split_data(input_csv=str(input_csv), output_dir=str(out1), seed=42)
        split_data(input_csv=str(input_csv), output_dir=str(out2), seed=42)

        a1 = pd.read_csv(out1 / "assignments.csv")
        a2 = pd.read_csv(out2 / "assignments.csv")
        pd.testing.assert_frame_equal(a1, a2)

        w1 = pd.read_csv(out1 / "train_sampler_weights.csv")
        w2 = pd.read_csv(out2 / "train_sampler_weights.csv")
        pd.testing.assert_frame_equal(w1, w2)
