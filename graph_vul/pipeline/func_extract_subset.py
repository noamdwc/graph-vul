"""Extract a normalized PrimeVul-backed function subset for fine-tuning."""

import hashlib
import json
import logging

import pandas as pd

from graph_vul.config import (
    DEDUPLICATE_CODE,
    FUNC_MAX_PROJECT_ROWS,
    FUNC_MAX_WHITESPACE_TOKENS,
    FUNC_SUBSET_CSV,
    FUNC_SUBSET_MANIFEST,
    FUNC_SUBSET_REPORT,
    SEED,
    TRAINING_DATASET_NAME,
)
from graph_vul.dataset.primevul_adapter import load_primevul_training_data

logger = logging.getLogger(__name__)


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


def _count_unique_cwes(df: pd.DataFrame) -> int:
    if "cwe_id" not in df.columns:
        return 0
    return int(df["cwe_id"].dropna().nunique())


def _make_stage_summary(
    stage: str,
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    **extra: int,
) -> dict:
    summary = {
        "stage": stage,
        "before_rows": int(len(before_df)),
        "after_rows": int(len(after_df)),
        "before_unique_cwes": _count_unique_cwes(before_df),
        "after_unique_cwes": _count_unique_cwes(after_df),
        "removed_rows": int(len(before_df) - len(after_df)),
    }
    summary.update({k: int(v) for k, v in extra.items()})
    return summary


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

def _filter_projects_per_cwe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop only oversized (project, cwe_id) groups."""
    if "project" not in df.columns or "cwe_id" not in df.columns:
        return df.copy(), {
            "removed_group_count": 0,
            "eligible_group_count": 0,
            "total_group_count": 0,
            "applied": False,
        }

    pair_counts = df.groupby(["project", "cwe_id"]).size().reset_index(name="_count")
    eligible = pair_counts[pair_counts["_count"] <= FUNC_MAX_PROJECT_ROWS][["project", "cwe_id"]]
    removed_groups = pair_counts[pair_counts["_count"] > FUNC_MAX_PROJECT_ROWS]

    before = len(df)
    filtered = df.merge(eligible, on=["project", "cwe_id"], how="inner")
    logger.info(
        "Per-CWE project filter removed %d oversized (project, cwe) pairs, %d -> %d rows",
        len(removed_groups), before, len(filtered),
    )
    return filtered, {
        "removed_group_count": int(len(removed_groups)),
        "eligible_group_count": int(len(eligible)),
        "total_group_count": int(len(pair_counts)),
        "applied": True,
    }


def _filter_token_length(df: pd.DataFrame) -> pd.DataFrame:
    """Drop functions with more than FUNC_MAX_WHITESPACE_TOKENS whitespace tokens."""
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

def _select_labeled_rows(df: pd.DataFrame) -> pd.DataFrame:
    labeled_mask = df["cwe_id"].notna() & df["cwe_id"].astype(str).str.strip().ne("")
    return df[labeled_mask].copy()


def _summarize_token_lengths(subset: pd.DataFrame) -> dict:
    if "code_token_count" not in subset.columns or subset.empty:
        return {
            "count": int(len(subset)),
            "min": 0,
            "median": 0.0,
            "mean": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0,
        }

    tokens = subset["code_token_count"].astype(int)
    return {
        "count": int(tokens.shape[0]),
        "min": int(tokens.min()),
        "median": float(tokens.median()),
        "mean": float(tokens.mean()),
        "p90": float(tokens.quantile(0.90)),
        "p95": float(tokens.quantile(0.95)),
        "max": int(tokens.max()),
    }


def _generate_manifest(
    subset: pd.DataFrame,
    config: dict,
    filter_stages: list[dict],
    project_cwe_filter: dict,
) -> dict:
    # Per-CWE breakdown
    cwe_details = []
    for cwe_id in sorted(subset["cwe_id"].unique()):
        cwe_rows = subset[subset["cwe_id"] == cwe_id]
        cwe_details.append({
            "cwe_id": cwe_id,
            "total_rows": int(len(cwe_rows)),
            "vulnerable": int((cwe_rows["vul"] == 1).sum()),
            "benign": int((cwe_rows["vul"] == 0).sum()),
            "projects": int(cwe_rows["project"].nunique()),
            "cves": int(cwe_rows["cve_id"].nunique()),
        })

    project_counts = subset.groupby("project").size().to_dict()
    project_counts = {k: int(v) for k, v in project_counts.items()}

    meta_dist = {}
    for col in ("code_char_len_bucket", "code_line_count_bucket", "code_token_count_bucket"):
        if col in subset.columns:
            meta_dist[col] = subset[col].value_counts().to_dict()
            meta_dist[col] = {str(k): int(v) for k, v in meta_dist[col].items()}

    top_cwes = sorted(
        (
            {"cwe_id": cwe["cwe_id"], "total_rows": cwe["total_rows"]}
            for cwe in cwe_details
        ),
        key=lambda item: (-item["total_rows"], item["cwe_id"]),
    )

    return {
        "config": config,
        "filter_stages": filter_stages,
        "project_cwe_filter": project_cwe_filter,
        "summary": {
            "total_rows": int(len(subset)),
            "vulnerable": int((subset["vul"] == 1).sum()),
            "benign": int((subset["vul"] == 0).sum()),
            "unique_cves": int(subset["cve_id"].nunique()),
            "unique_projects": int(subset["project"].nunique()),
            "unique_cwes": int(subset["cwe_id"].nunique()),
        },
        "cwe_details": cwe_details,
        "top_cwes": top_cwes,
        "project_counts": project_counts,
        "metadata_distributions": meta_dist,
        "token_length_summary": _summarize_token_lengths(subset),
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
        f"- Dataset: {cfg.get('dataset_name', 'N/A')}",
        f"- Subset scope: {cfg.get('subset_scope', 'N/A')}",
        f"- Max (project, cwe_id) rows: {cfg.get('max_project_rows', 'N/A')}",
        f"- Max whitespace tokens: {cfg.get('max_whitespace_tokens', 'N/A')}",
        f"- Deduplicate code: {cfg.get('deduplicate_code', 'N/A')}",
        f"- Seed: {cfg.get('seed', 'N/A')}",
        "",
        "## Filter Stages",
        "",
        "| Stage | Before rows | After rows | Before CWEs | After CWEs | Removed rows |",
        "|-------|-------------|------------|-------------|------------|--------------|",
    ]

    for stage in m.get("filter_stages", []):
        lines.append(
            f"| {stage['stage']} | {stage['before_rows']} | {stage['after_rows']} | "
            f"{stage['before_unique_cwes']} | {stage['after_unique_cwes']} | {stage['removed_rows']} |"
        )

    project_filter = m.get("project_cwe_filter", {})
    lines.extend([
        "",
        "## Project-CWE Filter",
        "",
        f"- **Applied:** {project_filter.get('applied', False)}",
        f"- **Removed oversized (project, cwe_id) groups:** {project_filter.get('removed_group_count', 0)}",
        f"- **Eligible (project, cwe_id) groups:** {project_filter.get('eligible_group_count', 0)}",
        f"- **Total observed (project, cwe_id) groups:** {project_filter.get('total_group_count', 0)}",
        "",
        "## Summary",
        "",
        f"- **Total rows:** {s['total_rows']}",
        f"- **Vulnerable:** {s['vulnerable']}",
        f"- **Benign:** {s['benign']}",
        f"- **Benign/Vulnerable ratio:** {s['benign'] / max(s['vulnerable'], 1):.2f}",
        f"- **Unique CWEs:** {s['unique_cwes']}",
        f"- **Unique CVEs:** {s['unique_cves']}",
        f"- **Unique projects:** {s['unique_projects']}",
        "",
        "## Top CWEs By Row Count",
        "",
        "| CWE | Rows |",
        "|-----|------|",
    ])

    for cwe in m.get("top_cwes", [])[:10]:
        lines.append(f"| {cwe['cwe_id']} | {cwe['total_rows']} |")

    lines.extend([
        "",
        "## Per-CWE Breakdown",
        "",
        "| CWE | Total | Vulnerable | Benign | Vul % | Projects | CVEs |",
        "|-----|-------|------------|--------|-------|----------|------|",
    ])

    for cwe in m["cwe_details"]:
        vul_pct = cwe["vulnerable"] / max(cwe["total_rows"], 1) * 100
        lines.append(
            f"| {cwe['cwe_id']} | {cwe['total_rows']} | "
            f"{cwe['vulnerable']} | {cwe['benign']} | {vul_pct:.1f} | "
            f"{cwe['projects']} | {cwe['cves']} |"
        )

    token_summary = m.get("token_length_summary", {})
    lines.extend([
        "",
        "## Token-Length Summary",
        "",
        f"- **Count:** {token_summary.get('count', 0)}",
        f"- **Min:** {token_summary.get('min', 0)}",
        f"- **Median:** {token_summary.get('median', 0.0):.1f}",
        f"- **Mean:** {token_summary.get('mean', 0.0):.1f}",
        f"- **P90:** {token_summary.get('p90', 0.0):.1f}",
        f"- **P95:** {token_summary.get('p95', 0.0):.1f}",
        f"- **Max:** {token_summary.get('max', 0)}",
    ])

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
    dst = output_csv or str(FUNC_SUBSET_CSV)
    manifest_path = output_manifest or str(FUNC_SUBSET_MANIFEST)
    report_path = output_report or str(FUNC_SUBSET_REPORT)

    if input_csv:
        logger.info(
            "Ignoring --input=%s because the function baseline now loads %s directly from Hugging Face",
            input_csv,
            TRAINING_DATASET_NAME,
        )
    logger.info("Loading %s training source", TRAINING_DATASET_NAME)
    df = load_primevul_training_data()
    stage_summaries: list[dict] = []

    all_cwe = _select_labeled_rows(df)
    stage_summaries.append(_make_stage_summary("raw_labeled_input", df, all_cwe))
    logger.info(
        "Normalized %s pool: %d rows across %d CWEs (%d vul, %d benign)",
        TRAINING_DATASET_NAME,
        len(all_cwe),
        _count_unique_cwes(all_cwe),
        int((all_cwe["vul"] == 1).sum()),
        int((all_cwe["vul"] == 0).sum()),
    )

    # Derive metadata
    all_cwe = _derive_metadata(all_cwe)

    # Deduplicate if configured
    pre_dedup = all_cwe
    if deduplicate:
        all_cwe = _deduplicate(all_cwe)
    stage_summaries.append(_make_stage_summary("deduplication", pre_dedup, all_cwe))

    # Apply token-length filter (before project filter so counts are accurate)
    pre_token_filter = all_cwe
    all_cwe = _filter_token_length(all_cwe)
    stage_summaries.append(_make_stage_summary("token_length_filter", pre_token_filter, all_cwe))

    # Apply per-CWE max-size filter
    pre_project_filter = all_cwe
    subset, project_filter_stats = _filter_projects_per_cwe(all_cwe)
    stage_summaries.append(
        _make_stage_summary(
            "project_cwe_max_filter",
            pre_project_filter,
            subset,
            removed_group_count=project_filter_stats["removed_group_count"],
        )
    )

    if len(subset) == 0:
        logger.error("No rows remaining after filtering — check configuration and input data")
        return pd.DataFrame()

    achieved_vul = int((subset["vul"] == 1).sum())
    achieved_benign = int((subset["vul"] == 0).sum())
    subset = subset.reset_index(drop=True)
    if "sample_id" not in subset.columns:
        subset.insert(0, "sample_id", range(len(subset)))
    subset_with_meta = subset.copy()

    # Keep only needed columns
    keep_cols = [
        "sample_id",
        "project",
        "commit_id",
        "func_before",
        "cve_id",
        "cwe_id",
        "vul",
        "source_split",
    ]
    keep_cols = [c for c in keep_cols if c in subset.columns]
    subset = subset[keep_cols].copy()

    # Write outputs
    subset.to_csv(dst, index=False)
    logger.info(
        "Wrote %d functions (%d vul, %d benign) to %s",
        len(subset), achieved_vul, achieved_benign, dst,
    )

    config_dict = {
        "dataset_name": TRAINING_DATASET_NAME,
        "subset_scope": "all_labeled_cwes",
        "max_project_rows": FUNC_MAX_PROJECT_ROWS,
        "max_whitespace_tokens": FUNC_MAX_WHITESPACE_TOKENS,
        "deduplicate_code": deduplicate,
        "seed": seed,
    }

    manifest = _generate_manifest(
        subset_with_meta,
        config_dict,
        filter_stages=stage_summaries,
        project_cwe_filter=project_filter_stats,
    )

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

    parser = argparse.ArgumentParser(description="Extract function-level PrimeVul subset")
    parser.add_argument("--input", type=str, default=None, help="Unused legacy input argument")
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
