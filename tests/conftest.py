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

    keep = ["sample_id", "project", "func_before", "cve_id", "cve_desc", "vul"]
    return subset[keep]
