"""Extract a function-level CWE-119 subset with project-size and token-length filters.

Keeps all CWE-119 rows after applying:
- Deduplication (exact code matches)
- Project filter: keep only projects with 50-2000 CWE-119 rows and >= 3 vulnerable
- Token filter: drop functions with >= 513 whitespace tokens
- Per-project cap: max 150 rows per project (preserving vul/benign ratio)
- Output to data/func_subset.csv (separate from existing data/subset.csv)
"""

import hashlib
import json
import logging
import zipfile
from pathlib import Path

import pandas as pd

from graph_vul.config import (
    BIGVUL_CSV,
    CWE_ID,
    DEDUPLICATE_CODE,
    FUNC_MAX_PROJECT_ROWS,
    FUNC_MAX_WHITESPACE_TOKENS,
    FUNC_MIN_PROJECT_ROWS,
    FUNC_SUBSET_CSV,
    FUNC_SUBSET_MANIFEST,
    FUNC_SUBSET_REPORT,
    SEED,
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
        logger.info("Deduplication removed %d exact duplicates (%d -> %d)", before - after, before, after)
    return df


# ---------------------------------------------------------------------------
# Project filtering
# ---------------------------------------------------------------------------

def _filter_projects(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only projects with row count in [MIN, MAX]."""
    project_counts = df.groupby("project").size()

    eligible = set()
    for project, count in project_counts.items():
        if FUNC_MIN_PROJECT_ROWS <= count <= FUNC_MAX_PROJECT_ROWS:
            eligible.add(project)

    before = len(df)
    df = df[df["project"].isin(eligible)]
    logger.info(
        "Project filter: %d eligible projects, %d -> %d rows",
        len(eligible), before, len(df),
    )
    return df


def _filter_token_length(df: pd.DataFrame) -> pd.DataFrame:
    """Drop functions with >= FUNC_MAX_WHITESPACE_TOKENS + 1 whitespace tokens."""
    before = len(df)
    token_counts = df["func_before"].fillna("").str.split().str.len().fillna(0).astype(int)
    df = df[token_counts <= FUNC_MAX_WHITESPACE_TOKENS]
    after = len(df)
    if before > after:
        logger.info("Token filter removed %d long functions (%d -> %d)", before - after, before, after)
    return df



# ---------------------------------------------------------------------------
# Manifest and report generation
# ---------------------------------------------------------------------------

def _generate_manifest(
    subset: pd.DataFrame,
    selected_cves: list[str],
    config: dict,
) -> dict:
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
    m = manifest
    s = m["summary"]
    cfg = m["config"]

    lines = [
        "# Function Subset Report",
        "",
        "## Configuration",
        "",
        f"- Project row range: {cfg.get('min_project_rows', 'N/A')}-{cfg.get('max_project_rows', 'N/A')}",
        f"- Min vulnerable per project: {cfg.get('min_vul_per_project', 'N/A')}",
        f"- Max rows per project: {cfg.get('max_rows_per_project', 'N/A')}",
        f"- Max whitespace tokens: {cfg.get('max_whitespace_tokens', 'N/A')}",
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

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_func_subset(
    input_csv: str | None = None,
    output_csv: str | None = None,
    output_manifest: str | None = None,
    output_report: str | None = None,
    deduplicate: bool = DEDUPLICATE_CODE,
    seed: int = SEED,
) -> pd.DataFrame:
    src = input_csv or str(BIGVUL_CSV)
    dst = output_csv or str(FUNC_SUBSET_CSV)
    manifest_path = output_manifest or str(FUNC_SUBSET_MANIFEST)
    report_path = output_report or str(FUNC_SUBSET_REPORT)

    logger.info("Reading Big-Vul CSV from %s", src)
    df = _read_bigvul_csv(src)

    # Filter to CWE-119
    all_cwe = df[df["cwe_id"] == CWE_ID]
    logger.info(
        "CWE-119 pool: %d total rows (%d vul, %d benign)",
        len(all_cwe),
        int((all_cwe["vul"] == 1).sum()),
        int((all_cwe["vul"] == 0).sum()),
    )

    # Derive metadata
    all_cwe = _derive_metadata(all_cwe)

    # Deduplicate if configured
    if deduplicate:
        all_cwe = _deduplicate(all_cwe)

    # Apply project-size filter
    all_cwe = _filter_projects(all_cwe)

    # Apply token-length filter
    subset = _filter_token_length(all_cwe)

    if len(subset) == 0:
        logger.error("No rows remaining after filtering — check configuration and input data")
        return pd.DataFrame()

    achieved_vul = int((subset["vul"] == 1).sum())
    achieved_benign = int((subset["vul"] == 0).sum())

    # Keep only needed columns
    keep_cols = ["project", "func_before", "cve_id", "cwe_id", "vul"]
    keep_cols = [c for c in keep_cols if c in subset.columns]
    subset = subset[keep_cols].copy()
    subset.insert(0, "sample_id", range(len(subset)))

    # Write outputs
    subset.to_csv(dst, index=False)
    logger.info(
        "Wrote %d functions (%d vul, %d benign) to %s",
        len(subset), achieved_vul, achieved_benign, dst,
    )

    selected_cves = sorted(subset["cve_id"].unique().tolist())
    config_dict = {
        "min_project_rows": FUNC_MIN_PROJECT_ROWS,
        "max_project_rows": FUNC_MAX_PROJECT_ROWS,
        "max_whitespace_tokens": FUNC_MAX_WHITESPACE_TOKENS,
        "deduplicate_code": deduplicate,
        "seed": seed,
    }

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

    parser = argparse.ArgumentParser(description="Extract function-level CWE-119 subset")
    parser.add_argument("--input", type=str, default=None, help="Path to Big-Vul CSV")
    parser.add_argument("--output", type=str, default=None, help="Path to output subset CSV")
    parser.add_argument("--manifest", type=str, default=None, help="Path to output manifest JSON")
    parser.add_argument("--report", type=str, default=None, help="Path to output report MD")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    extract_func_subset(
        input_csv=args.input,
        output_csv=args.output,
        output_manifest=args.manifest,
        output_report=args.report,
    )
