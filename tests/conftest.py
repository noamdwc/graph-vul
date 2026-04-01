"""Shared fixtures for graph-vul tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def bigvul_df():
    """Synthetic Big-Vul-like DataFrame with multiple CVEs, projects, and classes."""
    rng = np.random.RandomState(42)
    rows = []

    # Define projects and their CVEs (mimicking CWE-119 structure)
    project_cves = {
        "ProjectA": ["CVE-2020-0001", "CVE-2020-0002", "CVE-2020-0003"],
        "ProjectB": ["CVE-2020-0004", "CVE-2020-0005"],
        "ProjectC": ["CVE-2020-0006"],
        "ProjectD": ["CVE-2020-0007", "CVE-2020-0008"],
        "ProjectE": ["CVE-2020-0009"],
        "ProjectF": ["CVE-2020-0010"],
        "ProjectG": ["CVE-2020-0011"],
        "ProjectH": ["CVE-2020-0012"],
        "ProjectI": ["CVE-2020-0013"],
        "ProjectJ": ["CVE-2020-0014"],
    }

    for project, cves in project_cves.items():
        for cve_id in cves:
            # Each CVE gets some vulnerable and benign rows
            n_vul = rng.randint(5, 30)
            n_benign = rng.randint(20, 80)

            for _ in range(n_vul):
                code_len = rng.randint(50, 5000)
                rows.append({
                    "CWE ID": "CWE-119",
                    "CVE ID": cve_id,
                    "Summary": f"Vulnerability in {project} related to {cve_id}",
                    "project": project,
                    "func_before": "x" * code_len,
                    "vul": 1,
                })

            for _ in range(n_benign):
                code_len = rng.randint(50, 5000)
                rows.append({
                    "CWE ID": "CWE-119",
                    "CVE ID": cve_id,
                    "Summary": f"Function in {project}",
                    "project": project,
                    "func_before": "y" * code_len,
                    "vul": 0,
                })

    df = pd.DataFrame(rows)
    # Rename to match _read_bigvul_csv output
    df = df.rename(columns={
        "CWE ID": "cwe_id",
        "CVE ID": "cve_id",
        "Summary": "cve_desc",
    })
    return df


@pytest.fixture
def subset_df(bigvul_df):
    """A small subset DataFrame with project column, as produced by extract_subset."""
    # Take a sample from each of a few CVEs
    rng = np.random.RandomState(42)
    cves = bigvul_df["cve_id"].unique()[:8]
    pool = bigvul_df[bigvul_df["cve_id"].isin(cves)]

    parts = []
    for cve_id in cves:
        group = pool[pool["cve_id"] == cve_id]
        n = min(30, len(group))
        parts.append(group.sample(n=n, random_state=rng.randint(2**31)))

    subset = pd.concat(parts, ignore_index=True)
    subset = subset.sample(frac=1, random_state=42).reset_index(drop=True)
    subset.insert(0, "sample_id", range(len(subset)))

    keep = ["sample_id", "project", "func_before", "cve_id", "cwe_id", "vul"]
    return subset[keep]


@pytest.fixture
def multi_cwe_subset_df():
    """Subset with multiple CWEs including held-out CWEs sharing projects with seen CWEs.

    Layout:
    - ProjectA-F: seen CWEs only (CWE-119, CWE-125)
    - ProjectG-H: mixed — have both seen CWEs and held-out CWEs (CWE-200, CWE-416)
      -> these should migrate entirely to unseen pool
    - ProjectI-J: held-out CWEs only (CWE-200, CWE-416)
      -> these should also be in unseen pool
    """
    rng = np.random.RandomState(42)
    rows = []

    # Seen-only projects (6 projects, enough for train/val/test splitting)
    seen_projects = {
        "ProjectA": [("CWE-119", "CVE-2020-0001"), ("CWE-125", "CVE-2020-0002")],
        "ProjectB": [("CWE-119", "CVE-2020-0003")],
        "ProjectC": [("CWE-125", "CVE-2020-0004"), ("CWE-119", "CVE-2020-0005")],
        "ProjectD": [("CWE-119", "CVE-2020-0006")],
        "ProjectE": [("CWE-125", "CVE-2020-0007")],
        "ProjectF": [("CWE-119", "CVE-2020-0008")],
    }

    # Mixed projects (have both seen and held-out CWEs)
    mixed_projects = {
        "ProjectG": [("CWE-119", "CVE-2020-0009"), ("CWE-200", "CVE-2020-0010")],
        "ProjectH": [("CWE-125", "CVE-2020-0011"), ("CWE-416", "CVE-2020-0012")],
    }

    # Held-out-only projects
    heldout_projects = {
        "ProjectI": [("CWE-200", "CVE-2020-0013"), ("CWE-416", "CVE-2020-0014")],
        "ProjectJ": [("CWE-200", "CVE-2020-0015")],
    }

    for project_group in [seen_projects, mixed_projects, heldout_projects]:
        for project, cwe_cves in project_group.items():
            for cwe_id, cve_id in cwe_cves:
                n_vul = rng.randint(8, 25)
                n_benign = rng.randint(20, 60)
                for _ in range(n_vul):
                    code_len = rng.randint(50, 5000)
                    rows.append({
                        "cwe_id": cwe_id,
                        "cve_id": cve_id,
                        "project": project,
                        "func_before": "x" * code_len,
                        "vul": 1,
                    })
                for _ in range(n_benign):
                    code_len = rng.randint(50, 5000)
                    rows.append({
                        "cwe_id": cwe_id,
                        "cve_id": cve_id,
                        "project": project,
                        "func_before": "y" * code_len,
                        "vul": 0,
                    })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.insert(0, "sample_id", range(len(df)))
    return df


@pytest.fixture
def official_split_df_with_overlap():
    """Dataset with source_split column where some projects span multiple splits.

    Layout:
    - ProjectA: train only (80 rows)
    - ProjectB: val only (40 rows)
    - ProjectC: test only (40 rows)
    - ProjectD: train + val (60 train, 20 val) — multi-split
    - ProjectE: train + test (50 train, 15 test) — multi-split
    - ProjectF: train + val + test (40 train, 15 val, 10 test) — multi-split
    - ProjectG: val + test (20 val, 20 test) — multi-split
    """
    rng = np.random.RandomState(99)
    rows = []

    project_layout = [
        ("ProjectA", [("train", 80)]),
        ("ProjectB", [("val", 40)]),
        ("ProjectC", [("test", 40)]),
        ("ProjectD", [("train", 60), ("val", 20)]),
        ("ProjectE", [("train", 50), ("test", 15)]),
        ("ProjectF", [("train", 40), ("val", 15), ("test", 10)]),
        ("ProjectG", [("val", 20), ("test", 20)]),
    ]

    sample_id = 0
    for project, split_counts in project_layout:
        for source_split, count in split_counts:
            for i in range(count):
                token_count = rng.randint(10, 400)
                rows.append({
                    "sample_id": sample_id,
                    "project": project,
                    "source_split": source_split,
                    "func_before": "t " * token_count,
                    "cwe_id": rng.choice(["CWE-119", "CWE-125", "CWE-476"]),
                    "vul": 1 if i < max(count // 5, 2) else 0,
                })
                sample_id += 1

    return pd.DataFrame(rows)
