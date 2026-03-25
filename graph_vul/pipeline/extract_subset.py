"""Extract a structured, leakage-aware CWE-119 subset from the Big-Vul dataset.

Uses CVE diversity scoring, project+CVE grouping, metadata-aware sampling,
and configurable benign oversampling to produce a training-oriented subset.
"""

import hashlib
import json
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from graph_vul.config import (
    BIGVUL_CSV,
    CWE_ID,
    DEDUPLICATE_CODE,
    MAX_CVES,
    MAX_ROWS_PER_CVE,
    MAX_ROWS_PER_GROUP,
    MAX_ROWS_PER_PROJECT,
    MIN_CVES,
    SEED,
    SUBSET_CSV,
    SUBSET_MANIFEST,
    SUBSET_REPORT,
    TARGET_BENIGN_SAMPLES,
    TARGET_VULNERABLE_SAMPLES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_bigvul_csv(path: str) -> pd.DataFrame:
    src = Path(path)

    if zipfile.is_zipfile(src):
        logger.info("Detected ZIP-compressed Big-Vul dataset at %s", src)
        df = pd.read_csv(src, compression="zip", low_memory=False)
    else:
        try:
            df = pd.read_csv(src, low_memory=False)
        except UnicodeDecodeError:
            logger.warning("UTF-8 decode failed for %s; retrying with latin-1", src)
            df = pd.read_csv(src, encoding="latin-1", low_memory=False)

    return df.rename(
        columns={
            "CWE ID": "cwe_id",
            "CVE ID": "cve_id",
            "Summary": "cve_desc",
        }
    )


# ---------------------------------------------------------------------------
# Metadata derivation
# ---------------------------------------------------------------------------

def _derive_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Compute code_char_len, code_line_count, code_token_count and buckets."""
    df = df.copy()
    code = df["func_before"].fillna("")
    df["code_char_len"] = code.str.len()
    df["code_line_count"] = code.str.count("\n") + 1
    df["code_token_count"] = code.str.split().str.len().fillna(0).astype(int)

    for col in ("code_char_len", "code_line_count", "code_token_count"):
        bucket_col = col + "_bucket"
        try:
            df[bucket_col] = pd.qcut(
                df[col], q=4, labels=["short", "medium", "long", "very_long"],
                duplicates="drop",
            )
        except ValueError:
            df[bucket_col] = "medium"
    return df


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate func_before texts, keeping one per unique code."""
    before = len(df)
    df = df.copy()
    df["_code_hash"] = df["func_before"].fillna("").apply(
        lambda x: hashlib.sha256(x.encode("utf-8", errors="replace")).hexdigest()
    )
    df = df.drop_duplicates(subset="_code_hash", keep="first")
    df = df.drop(columns=["_code_hash"])
    after = len(df)
    if before > after:
        logger.info("Deduplication removed %d exact duplicates (%d → %d)", before - after, before, after)
    return df


# ---------------------------------------------------------------------------
# CVE scoring and selection
# ---------------------------------------------------------------------------

def _score_cve(cve_stats: dict, selected_projects: set, top2_projects: set) -> float:
    """Score a candidate CVE for selection.

    Higher is better. Rewards:
    - Having both vul and benign rows
    - Moderate size (not too tiny, not absurdly large)
    - Project diversity (bonus if project not yet selected)
    - Not being from the top-2 largest projects
    """
    vul = cve_stats["vul_count"]
    benign = cve_stats["benign_count"]
    project = cve_stats["project"]

    if vul < 3:
        return -1  # too few vulnerable samples

    # Base score: geometric mean of vul and benign availability (rewards balance)
    base = np.sqrt(min(vul, 200) * min(benign, 200))

    # Diversity bonus: prefer projects not yet selected
    diversity = 2.0 if project not in selected_projects else 0.5

    # Moderate-size bonus: diminishing returns above 300
    size_factor = min(vul + benign, 300) / 300.0

    # Metadata spread (use coefficient of variation of code lengths if available)
    spread = cve_stats.get("length_cv", 0.5)
    spread_factor = 0.5 + min(spread, 1.0) * 0.5

    # Penalty for top-2 giant projects (still selectable, just penalized)
    giant_penalty = 0.6 if project in top2_projects else 1.0

    return base * diversity * size_factor * spread_factor * giant_penalty


def _select_cves(
    df: pd.DataFrame,
    min_cves: int,
    max_cves: int,
    seed: int,
) -> list[str]:
    """Greedily select CVEs for diversity."""
    rng = np.random.RandomState(seed)

    # Compute per-CVE stats
    cve_groups = df.groupby("cve_id")
    cve_stats = {}
    for cve_id, group in cve_groups:
        projects = group["project"].unique()
        cve_stats[cve_id] = {
            "project": projects[0],  # each CWE-119 CVE maps to one project
            "vul_count": int((group["vul"] == 1).sum()),
            "benign_count": int((group["vul"] == 0).sum()),
            "total": len(group),
            "length_cv": group["code_char_len"].std() / max(group["code_char_len"].mean(), 1),
        }

    # Identify top-2 projects by total row count
    project_totals = df.groupby("project").size()
    top2_projects = set(project_totals.nlargest(2).index)

    # Filter out CVEs with < 3 vulnerable samples
    eligible = {k: v for k, v in cve_stats.items() if v["vul_count"] >= 3}
    if len(eligible) < min_cves:
        logger.warning(
            "Only %d eligible CVEs (need %d); relaxing vul threshold to 1",
            len(eligible), min_cves,
        )
        eligible = {k: v for k, v in cve_stats.items() if v["vul_count"] >= 1}

    selected = []
    selected_projects = set()
    remaining = dict(eligible)

    # Ensure at least 2 CVEs from non-top2 projects (held-out candidates)
    non_giant_cves = {k: v for k, v in remaining.items() if v["project"] not in top2_projects}
    non_giant_scored = sorted(
        non_giant_cves.items(),
        key=lambda x: _score_cve(x[1], selected_projects, top2_projects),
        reverse=True,
    )

    for cve_id, stats in non_giant_scored[:2]:
        selected.append(cve_id)
        selected_projects.add(stats["project"])
        del remaining[cve_id]

    # Greedily add remaining CVEs
    while len(selected) < max_cves and remaining:
        scored = [
            (cve_id, _score_cve(stats, selected_projects, top2_projects))
            for cve_id, stats in remaining.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        if not scored or scored[0][1] < 0:
            break

        # Among top candidates with similar scores, add randomness
        top_score = scored[0][1]
        candidates = [s for s in scored if s[1] >= top_score * 0.7]
        pick = candidates[rng.randint(len(candidates))]

        cve_id = pick[0]
        selected.append(cve_id)
        selected_projects.add(eligible[cve_id]["project"])
        del remaining[cve_id]

        if len(selected) >= min_cves:
            # Check if we have enough diversity; stop if adding more doesn't help
            unique_projects = len({eligible[c]["project"] for c in selected})
            if unique_projects >= min(len(selected), 8):
                # Good diversity, can stop if we have enough CVEs
                pass

    logger.info(
        "Selected %d CVEs from %d projects",
        len(selected),
        len({eligible[c]["project"] for c in selected}),
    )
    return selected


# ---------------------------------------------------------------------------
# Budget allocation and row sampling
# ---------------------------------------------------------------------------

def _allocate_budget(
    df: pd.DataFrame,
    selected_cves: list[str],
    target_vul: int,
    target_benign: int,
    max_per_cve: int,
    max_per_project: int,
    max_per_group: int,
) -> dict[str, dict[str, int]]:
    """Allocate vul/benign budgets per CVE, respecting caps."""
    cve_data = {}
    for cve_id in selected_cves:
        group = df[df["cve_id"] == cve_id]
        cve_data[cve_id] = {
            "project": group["project"].iloc[0],
            "avail_vul": int((group["vul"] == 1).sum()),
            "avail_benign": int((group["vul"] == 0).sum()),
        }

    n_cves = len(selected_cves)

    # Start with proportional allocation of vulnerable budget
    total_avail_vul = sum(c["avail_vul"] for c in cve_data.values())
    total_avail_benign = sum(c["avail_benign"] for c in cve_data.values())

    allocation = {}
    for cve_id, stats in cve_data.items():
        # Proportional share based on availability
        if total_avail_vul > 0:
            vul_share = int(target_vul * stats["avail_vul"] / total_avail_vul)
        else:
            vul_share = 0
        if total_avail_benign > 0:
            benign_share = int(target_benign * stats["avail_benign"] / total_avail_benign)
        else:
            benign_share = 0

        # Apply caps
        total_cap = min(max_per_cve, max_per_group)  # group = project+CVE here (1:1)
        vul_alloc = min(vul_share, stats["avail_vul"], total_cap)
        benign_alloc = min(benign_share, stats["avail_benign"], total_cap - vul_alloc)

        allocation[cve_id] = {"vul": vul_alloc, "benign": benign_alloc}

    # Apply per-project caps
    project_totals: dict[str, int] = {}
    for cve_id, alloc in allocation.items():
        proj = cve_data[cve_id]["project"]
        project_totals[proj] = project_totals.get(proj, 0) + alloc["vul"] + alloc["benign"]

    for proj, total in project_totals.items():
        if total > max_per_project:
            scale = max_per_project / total
            for cve_id, alloc in allocation.items():
                if cve_data[cve_id]["project"] == proj:
                    alloc["vul"] = int(alloc["vul"] * scale)
                    alloc["benign"] = int(alloc["benign"] * scale)

    return allocation


def _sample_rows(
    df: pd.DataFrame,
    selected_cves: list[str],
    allocation: dict[str, dict[str, int]],
    seed: int,
) -> pd.DataFrame:
    """Sample rows from each CVE group with metadata-aware stratification."""
    rng = np.random.RandomState(seed)
    sampled_parts = []

    for cve_id in selected_cves:
        group = df[df["cve_id"] == cve_id]
        alloc = allocation[cve_id]

        for label, target_n in [(1, alloc["vul"]), (0, alloc["benign"])]:
            pool = group[group["vul"] == label]
            if len(pool) == 0 or target_n == 0:
                continue

            n = min(target_n, len(pool))

            # Metadata-aware sampling: stratify by length bucket if possible
            bucket_col = "code_char_len_bucket"
            if bucket_col in pool.columns and pool[bucket_col].nunique() > 1:
                # Sample proportionally from each bucket
                per_bucket = dict(list(pool.groupby(bucket_col, observed=True)))
                parts = []
                remaining = n
                bucket_names = list(per_bucket.keys())
                rng.shuffle(bucket_names)

                for i, bucket_name in enumerate(bucket_names):
                    bucket_df = per_bucket[bucket_name]
                    # Proportional allocation with remainder going to last buckets
                    if i < len(bucket_names) - 1:
                        bucket_n = max(1, int(n * len(bucket_df) / len(pool)))
                        bucket_n = min(bucket_n, remaining, len(bucket_df))
                    else:
                        bucket_n = min(remaining, len(bucket_df))

                    if bucket_n > 0:
                        parts.append(bucket_df.sample(n=bucket_n, random_state=rng.randint(2**31)))
                        remaining -= bucket_n

                    if remaining <= 0:
                        break

                if parts:
                    sampled_parts.append(pd.concat(parts))
            else:
                sampled_parts.append(pool.sample(n=n, random_state=rng.randint(2**31)))

    if not sampled_parts:
        return pd.DataFrame()

    result = pd.concat(sampled_parts, ignore_index=True)
    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Manifest and report generation
# ---------------------------------------------------------------------------

def _generate_manifest(
    subset: pd.DataFrame,
    selected_cves: list[str],
    config: dict,
) -> dict:
    """Generate a JSON-serializable manifest of the subset."""
    cve_details = []
    for cve_id in selected_cves:
        rows = subset[subset["cve_id"] == cve_id]
        cve_details.append({
            "cve_id": cve_id,
            "project": rows["project"].iloc[0] if len(rows) > 0 else "unknown",
            "total_rows": int(len(rows)),
            "vulnerable": int((rows["vul"] == 1).sum()),
            "benign": int((rows["vul"] == 0).sum()),
        })

    project_counts = subset.groupby("project").size().to_dict()
    project_counts = {k: int(v) for k, v in project_counts.items()}

    # Metadata distributions
    meta_dist = {}
    for col in ("code_char_len_bucket", "code_line_count_bucket", "code_token_count_bucket"):
        if col in subset.columns:
            meta_dist[col] = subset[col].value_counts().to_dict()
            meta_dist[col] = {str(k): int(v) for k, v in meta_dist[col].items()}

    return {
        "config": config,
        "summary": {
            "total_rows": int(len(subset)),
            "vulnerable": int((subset["vul"] == 1).sum()),
            "benign": int((subset["vul"] == 0).sum()),
            "unique_cves": int(subset["cve_id"].nunique()),
            "unique_projects": int(subset["project"].nunique()),
        },
        "selected_cves": cve_details,
        "project_counts": project_counts,
        "metadata_distributions": meta_dist,
    }


def _generate_report(manifest: dict) -> str:
    """Generate a human-readable Markdown report."""
    m = manifest
    s = m["summary"]
    cfg = m["config"]

    lines = [
        "# Subset Sampling Report",
        "",
        "## Configuration",
        "",
        f"- Target vulnerable: {cfg.get('target_vulnerable_samples', 'N/A')}",
        f"- Target benign: {cfg.get('target_benign_samples', 'N/A')}",
        f"- CVE range: {cfg.get('min_cves', 'N/A')}–{cfg.get('max_cves', 'N/A')}",
        f"- Max rows per project: {cfg.get('max_rows_per_project', 'N/A')}",
        f"- Max rows per CVE: {cfg.get('max_rows_per_cve', 'N/A')}",
        f"- Max rows per group: {cfg.get('max_rows_per_group', 'N/A')}",
        f"- Deduplicate code: {cfg.get('deduplicate_code', 'N/A')}",
        f"- Seed: {cfg.get('seed', 'N/A')}",
        "",
        "## Summary",
        "",
        f"- **Total rows:** {s['total_rows']}",
        f"- **Vulnerable:** {s['vulnerable']}",
        f"- **Benign:** {s['benign']}",
        f"- **Benign/Vulnerable ratio:** {s['benign'] / max(s['vulnerable'], 1):.2f}",
        f"- **Unique CVEs:** {s['unique_cves']}",
        f"- **Unique projects:** {s['unique_projects']}",
        "",
        "## Selected CVEs",
        "",
        "| CVE | Project | Total | Vulnerable | Benign |",
        "|-----|---------|-------|------------|--------|",
    ]

    for cve in m["selected_cves"]:
        lines.append(
            f"| {cve['cve_id']} | {cve['project']} | {cve['total_rows']} | "
            f"{cve['vulnerable']} | {cve['benign']} |"
        )

    lines.extend(["", "## Project Distribution", ""])
    lines.append("| Project | Rows |")
    lines.append("|---------|------|")
    for proj, count in sorted(m["project_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"| {proj} | {count} |")

    if m["metadata_distributions"]:
        lines.extend(["", "## Metadata Distributions", ""])
        for col, dist in m["metadata_distributions"].items():
            lines.append(f"### {col}")
            lines.append("")
            lines.append("| Bucket | Count |")
            lines.append("|--------|-------|")
            for bucket, count in sorted(dist.items()):
                lines.append(f"| {bucket} | {count} |")
            lines.append("")

    # Oversampling details
    lines.extend([
        "## Benign Oversampling",
        "",
        f"- Requested vulnerable: {cfg.get('target_vulnerable_samples', 'N/A')}",
        f"- Requested benign: {cfg.get('target_benign_samples', 'N/A')}",
        f"- Achieved vulnerable: {s['vulnerable']}",
        f"- Achieved benign: {s['benign']}",
    ])

    if s["vulnerable"] < cfg.get("target_vulnerable_samples", 0):
        lines.append(f"- **WARNING:** Achieved vulnerable count below target")
    if s["benign"] < cfg.get("target_benign_samples", 0):
        lines.append(f"- **WARNING:** Achieved benign count below target")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_subset(
    input_csv: str | None = None,
    output_csv: str | None = None,
    output_manifest: str | None = None,
    output_report: str | None = None,
    target_vul: int = TARGET_VULNERABLE_SAMPLES,
    target_benign: int = TARGET_BENIGN_SAMPLES,
    min_cves: int = MIN_CVES,
    max_cves: int = MAX_CVES,
    max_per_project: int = MAX_ROWS_PER_PROJECT,
    max_per_cve: int = MAX_ROWS_PER_CVE,
    max_per_group: int = MAX_ROWS_PER_GROUP,
    deduplicate: bool = DEDUPLICATE_CODE,
    seed: int = SEED,
) -> pd.DataFrame:
    src = input_csv or str(BIGVUL_CSV)
    dst = output_csv or str(SUBSET_CSV)
    manifest_path = output_manifest or str(SUBSET_MANIFEST)
    report_path = output_report or str(SUBSET_REPORT)

    logger.info("Reading Big-Vul CSV from %s", src)
    df = _read_bigvul_csv(src)

    # Filter to CWE-119 (vulnerable) + all benign from same projects later
    vul_df = df[(df["cwe_id"] == CWE_ID) & (df["vul"] == 1)]
    all_cwe = df[df["cwe_id"] == CWE_ID]

    logger.info(
        "CWE-119 pool: %d total rows (%d vul, %d benign)",
        len(all_cwe),
        len(vul_df),
        len(all_cwe) - len(vul_df),
    )

    # Derive metadata
    all_cwe = _derive_metadata(all_cwe)

    # Deduplicate if configured
    if deduplicate:
        all_cwe = _deduplicate(all_cwe)

    # Select CVEs
    selected_cves = _select_cves(all_cwe, min_cves, max_cves, seed)

    # Filter to selected CVEs
    pool = all_cwe[all_cwe["cve_id"].isin(selected_cves)]

    # Allocate budget
    allocation = _allocate_budget(
        pool, selected_cves, target_vul, target_benign,
        max_per_cve, max_per_project, max_per_group,
    )

    # Sample rows
    subset = _sample_rows(pool, selected_cves, allocation, seed)

    if len(subset) == 0:
        logger.error("No rows sampled — check configuration and input data")
        return pd.DataFrame()

    # Graceful degradation warnings
    achieved_vul = int((subset["vul"] == 1).sum())
    achieved_benign = int((subset["vul"] == 0).sum())
    if achieved_vul < target_vul:
        logger.warning(
            "Achieved %d vulnerable samples (target: %d) — insufficient data in selected CVEs",
            achieved_vul, target_vul,
        )
    if achieved_benign < target_benign:
        logger.warning(
            "Achieved %d benign samples (target: %d) — insufficient data in selected CVEs",
            achieved_benign, target_benign,
        )

    # Keep only needed columns + project
    keep_cols = ["project", "func_before", "cve_id", "cwe_id", "vul"]
    keep_cols = [c for c in keep_cols if c in subset.columns]
    subset = subset[keep_cols].copy()
    subset.insert(0, "sample_id", range(len(subset)))

    # Write outputs
    subset.to_csv(dst, index=False)
    logger.info(
        "Wrote %d samples (%d vul, %d benign) to %s",
        len(subset), achieved_vul, achieved_benign, dst,
    )

    # Generate manifest and report
    config_dict = {
        "target_vulnerable_samples": target_vul,
        "target_benign_samples": target_benign,
        "min_cves": min_cves,
        "max_cves": max_cves,
        "max_rows_per_project": max_per_project,
        "max_rows_per_cve": max_per_cve,
        "max_rows_per_group": max_per_group,
        "deduplicate_code": deduplicate,
        "seed": seed,
    }

    # Re-derive metadata on final subset for manifest
    subset_with_meta = _derive_metadata(subset)
    manifest = _generate_manifest(subset_with_meta, selected_cves, config_dict)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote manifest to %s", manifest_path)

    report = _generate_report(manifest)
    with open(report_path, "w") as f:
        f.write(report)
    logger.info("Wrote report to %s", report_path)

    return subset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract structured CWE-119 subset from Big-Vul")
    parser.add_argument("--input", type=str, default=None, help="Path to Big-Vul CSV")
    parser.add_argument("--output", type=str, default=None, help="Path to output subset CSV")
    parser.add_argument("--manifest", type=str, default=None, help="Path to output manifest JSON")
    parser.add_argument("--report", type=str, default=None, help="Path to output report MD")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    extract_subset(
        input_csv=args.input,
        output_csv=args.output,
        output_manifest=args.manifest,
        output_report=args.report,
    )
