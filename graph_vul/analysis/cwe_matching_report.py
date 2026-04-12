from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from graph_vul.config import MATCHING_LENGTH_BUCKET_EDGES, MATCHING_LENGTH_BUCKET_LABELS


def binary_metrics_df(df: pd.DataFrame, y_true_col: str = "y_true") -> dict:
    if df.empty:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "num_samples": 0,
            "num_vulnerable": 0,
            "num_benign": 0,
        }
    y_true = df[y_true_col].astype(int).to_numpy()
    y_pred = df["y_pred"].astype(int).to_numpy()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "num_samples": int(len(df)),
        "num_vulnerable": int(y_true.sum()),
        "num_benign": int(len(df) - y_true.sum()),
    }


def per_cwe_metrics(df: pd.DataFrame, y_true_col: str = "y_true") -> pd.DataFrame:
    if "cwe_id" not in df.columns or df.empty:
        return pd.DataFrame()
    rows = []
    for cwe_id, group_df in df.groupby("cwe_id", dropna=False):
        row = binary_metrics_df(group_df, y_true_col=y_true_col)
        row["cwe_id"] = cwe_id
        valid_retrieval = group_df["retrieval_hit_top1"].dropna() if "retrieval_hit_top1" in group_df.columns else pd.Series(dtype=float)
        if not valid_retrieval.empty:
            row["retrieval_top1"] = float(valid_retrieval.mean())
            row["retrieval_top3"] = float(group_df["retrieval_hit_top3"].dropna().mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["f1", "num_samples", "cwe_id"], ascending=[True, False, True])


def add_length_slice(df: pd.DataFrame) -> pd.DataFrame:
    if "token_length" not in df.columns:
        return df.copy()
    bins = MATCHING_LENGTH_BUCKET_EDGES + [np.inf]
    result = df.copy()
    result["length_slice"] = pd.cut(
        result["token_length"],
        bins=bins,
        labels=MATCHING_LENGTH_BUCKET_LABELS,
        include_lowest=True,
        right=False,
    )
    return result


def length_slice_metrics(df: pd.DataFrame, y_true_col: str = "y_true") -> pd.DataFrame:
    bucketed = add_length_slice(df)
    if "length_slice" not in bucketed.columns:
        return pd.DataFrame()
    rows = []
    for length_slice, group_df in bucketed.groupby("length_slice", dropna=False, observed=False):
        if pd.isna(length_slice):
            continue
        row = binary_metrics_df(group_df, y_true_col=y_true_col)
        row["length_slice"] = str(length_slice)
        valid_retrieval = group_df["retrieval_hit_top1"].dropna() if "retrieval_hit_top1" in group_df.columns else pd.Series(dtype=float)
        if not valid_retrieval.empty:
            row["retrieval_top1"] = float(valid_retrieval.mean())
            row["retrieval_top3"] = float(group_df["retrieval_hit_top3"].dropna().mean())
        rows.append(row)
    return pd.DataFrame(rows)


def candidate_cwe_matrix(cwe_store, model, device: torch.device) -> tuple[list[str], torch.Tensor]:
    candidate_ids = cwe_store.available_cwe_ids()
    if not candidate_ids:
        return [], torch.zeros(0, model.cwe_projection.out_features, device=device)
    candidate_embeddings = cwe_store.stack(candidate_ids, device=device)
    with torch.inference_mode():
        candidate_match = model.project_cwe_embeddings(candidate_embeddings)
    return candidate_ids, candidate_match


def retrieval_columns(
    function_match: torch.Tensor,
    candidate_ids: list[str],
    candidate_match: torch.Tensor,
    true_cwe_ids: list[str],
) -> dict[str, list]:
    n = function_match.shape[0]
    if n == 0 or not candidate_ids or candidate_match.numel() == 0:
        return {
            "retrieved_cwe_top1": [None] * n,
            "retrieval_rank": [None] * n,
            "retrieval_hit_top1": [None] * n,
            "retrieval_hit_top3": [None] * n,
        }
    scores = torch.matmul(function_match, candidate_match.T)
    topk = torch.topk(scores, k=min(3, scores.shape[1]), dim=1)
    rank_indices = torch.argsort(scores, dim=1, descending=True)
    candidate_index = {cwe_id: idx for idx, cwe_id in enumerate(candidate_ids)}

    retrieved_top1: list[str | None] = []
    retrieval_rank: list[int | None] = []
    retrieval_hit_top1: list[int | None] = []
    retrieval_hit_top3: list[int | None] = []
    for row_idx, true_cwe_id in enumerate(true_cwe_ids):
        retrieved_top1.append(candidate_ids[int(topk.indices[row_idx, 0])])
        true_idx = candidate_index.get(str(true_cwe_id))
        if true_idx is None:
            retrieval_rank.append(None)
            retrieval_hit_top1.append(None)
            retrieval_hit_top3.append(None)
            continue
        ordered = rank_indices[row_idx].tolist()
        rank = ordered.index(true_idx) + 1
        retrieval_rank.append(rank)
        retrieval_hit_top1.append(int(rank == 1))
        retrieval_hit_top3.append(int(rank <= 3))
    return {
        "retrieved_cwe_top1": retrieved_top1,
        "retrieval_rank": retrieval_rank,
        "retrieval_hit_top1": retrieval_hit_top1,
        "retrieval_hit_top3": retrieval_hit_top3,
    }


def summarize_predictions(df: pd.DataFrame, y_true_col: str = "y_true") -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    metrics = binary_metrics_df(df, y_true_col=y_true_col)
    if "retrieval_hit_top1" in df.columns:
        top1 = df["retrieval_hit_top1"].dropna()
        top3 = df["retrieval_hit_top3"].dropna()
        metrics["retrieval_top1"] = float(top1.mean()) if not top1.empty else None
        metrics["retrieval_top3"] = float(top3.mean()) if not top3.empty else None
    return metrics, per_cwe_metrics(df, y_true_col=y_true_col), length_slice_metrics(df, y_true_col=y_true_col)


def write_summary_artifacts(
    df: pd.DataFrame,
    output_dir: str | Path,
    prefix: str,
    *,
    y_true_col: str = "y_true",
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics, per_cwe_df, length_df = summarize_predictions(df, y_true_col=y_true_col)

    metrics_path = output_path / f"{prefix}_metrics.json"
    with metrics_path.open("w") as handle:
        json.dump(metrics, handle, indent=2)

    if not per_cwe_df.empty:
        per_cwe_df.to_csv(output_path / f"per_cwe_metrics_{prefix}.csv", index=False)
    if not length_df.empty:
        length_df.to_csv(output_path / f"length_slice_metrics_{prefix}.csv", index=False)

    lines = [f"# {prefix} summary", "", "## Overall", ""]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")
    if not per_cwe_df.empty:
        lines.extend(["", "## Worst CWE F1", "", per_cwe_df.head(10).to_string(index=False)])
    if not length_df.empty:
        lines.extend(["", "## Length slices", "", length_df.to_string(index=False)])
    (output_path / f"{prefix}_summary.md").write_text("\n".join(lines))
    return metrics
