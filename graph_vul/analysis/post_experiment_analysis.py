"""Reusable helpers for post-experiment evaluation notebooks."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics

RANDOM_SEED = 42

LABEL_ALIASES = ("y_true", "label", "vul", "target")
PRED_ALIASES = ("y_pred", "pred", "prediction", "pred_label")
SCORE_ALIASES = ("score", "prob", "probability", "positive_prob", "pred_score", "logit", "logits")
SPLIT_ALIASES = ("split",)
CWE_ALIASES = ("cwe_id", "cwe", "cweid")
PROJECT_ALIASES = ("project", "project_name", "repo")
UNSEEN_ALIASES = ("is_unseen_cwe", "unseen_cwe", "heldout_cwe")
ID_ALIASES = ("row_id", "example_id", "sample_id", "id")
TOKEN_LEN_ALIASES = ("token_length", "func_token_length", "n_tokens", "token_count")
LINE_LEN_ALIASES = ("line_count", "n_lines", "func_line_count")
CHAR_LEN_ALIASES = ("char_length", "n_chars", "func_char_count", "code_length_chars")
CODE_PREVIEW_ALIASES = ("func_before", "func_after", "code", "source", "snippet")

DEFAULT_LENGTH_BUCKETS: dict[str, list[float]] = {
    "token_length": [0, 64, 128, 256, 512, math.inf],
    "line_count": [0, 32, 64, 128, 256, math.inf],
    "char_length": [0, 256, 512, 1024, 2048, 4096, math.inf],
}


def _first_present(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    lowered = {col.lower(): col for col in columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    return None


def _coerce_binary(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float).round().astype("Int64")
    mapping = {
        "1": 1,
        "0": 0,
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
        "vulnerable": 1,
        "benign": 0,
        "positive": 1,
        "negative": 0,
    }
    lowered = series.astype(str).str.strip().str.lower()
    converted = lowered.map(mapping)
    return converted.astype("Int64")


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffixes = [s.lower() for s in path.suffixes]
    if suffixes[-2:] == [".json", ".gz"] or suffixes[-1:] == [".json"]:
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            with path.open() as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                return pd.DataFrame(payload)
            if isinstance(payload, dict):
                if "predictions" in payload and isinstance(payload["predictions"], list):
                    return pd.DataFrame(payload["predictions"])
                return pd.DataFrame(payload)
            raise ValueError(f"Unsupported JSON payload in {path}")
    if suffixes[-2:] == [".jsonl", ".gz"] or suffixes[-1:] == [".jsonl"]:
        return pd.read_json(path, lines=True)
    if suffixes[-1:] == [".parquet"]:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def add_code_preview(df: pd.DataFrame, limit: int = 160) -> pd.DataFrame:
    df = df.copy()
    preview_col = _first_present(df.columns.tolist(), CODE_PREVIEW_ALIASES)
    if preview_col is None:
        return df
    preview = df[preview_col].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["code_preview"] = preview.str.slice(0, limit)
    return df


def canonicalize_prediction_frame(
    df: pd.DataFrame,
    split_name: str | None = None,
    threshold: float = 0.5,
    score_kind_override: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = df.copy()
    original_columns = df.columns.tolist()
    columns = original_columns

    label_col = _first_present(columns, LABEL_ALIASES)
    pred_col = _first_present(columns, PRED_ALIASES)
    score_col = _first_present(columns, SCORE_ALIASES)
    split_col = _first_present(columns, SPLIT_ALIASES)
    cwe_col = _first_present(columns, CWE_ALIASES)
    project_col = _first_present(columns, PROJECT_ALIASES)
    unseen_col = _first_present(columns, UNSEEN_ALIASES)
    id_col = _first_present(columns, ID_ALIASES)
    token_len_col = _first_present(columns, TOKEN_LEN_ALIASES)
    line_len_col = _first_present(columns, LINE_LEN_ALIASES)
    char_len_col = _first_present(columns, CHAR_LEN_ALIASES)

    if label_col is None:
        raise ValueError(f"Missing required label column. Expected one of {LABEL_ALIASES}, found {original_columns}")

    df["y_true"] = _coerce_binary(df[label_col])

    score_kind = score_kind_override or "auto"
    if score_col is not None:
        df["score_raw"] = pd.to_numeric(df[score_col], errors="coerce")
        inferred_kind = "prob"
        name = score_col.lower()
        if score_kind == "auto":
            if "logit" in name:
                inferred_kind = "logit"
            elif df["score_raw"].dropna().between(0, 1).all():
                inferred_kind = "prob"
            else:
                inferred_kind = "unknown"
        else:
            inferred_kind = score_kind

        if inferred_kind == "logit":
            df["score"] = 1.0 / (1.0 + np.exp(-df["score_raw"].clip(-50, 50)))
            decision_values = df["score_raw"]
        else:
            df["score"] = df["score_raw"]
            decision_values = df["score"]
        score_kind = inferred_kind
        if pred_col is None:
            df["y_pred"] = (decision_values >= threshold).astype("Int64")
    else:
        score_kind = None

    if pred_col is not None:
        df["y_pred"] = _coerce_binary(df[pred_col])

    if "y_pred" not in df.columns:
        raise ValueError("Missing predicted labels. Provide y_pred or a score/prob/logit column.")

    df["split"] = split_name or (df[split_col].iloc[0] if split_col is not None and not df.empty else "unknown")
    if cwe_col is not None:
        df["cwe_id"] = df[cwe_col].astype(str).replace({"nan": np.nan})
    if project_col is not None:
        df["project"] = df[project_col].astype(str).replace({"nan": np.nan})
    if unseen_col is not None:
        df["is_unseen_cwe"] = _coerce_binary(df[unseen_col]).astype("Int64")
    if id_col is not None:
        df["row_id"] = df[id_col]
    if token_len_col is not None:
        df["token_length"] = pd.to_numeric(df[token_len_col], errors="coerce")
    if line_len_col is not None:
        df["line_count"] = pd.to_numeric(df[line_len_col], errors="coerce")
    if char_len_col is not None:
        df["char_length"] = pd.to_numeric(df[char_len_col], errors="coerce")

    df = add_code_preview(df)
    metadata = {
        "label_col": label_col,
        "pred_col": pred_col,
        "score_col": score_col,
        "score_kind": score_kind,
        "split_name": df["split"].iloc[0] if not df.empty else split_name,
        "has_score": "score" in df.columns,
        "original_columns": original_columns,
    }
    return df, metadata


def load_prediction_splits(
    split_paths: dict[str, str | Path | None],
    threshold: float = 0.5,
    score_kind_override: str | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    frames: dict[str, pd.DataFrame] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for split_name, path in split_paths.items():
        if not path:
            continue
        path_obj = Path(path)
        if not path_obj.exists():
            print(f"[WARN] Missing file for split '{split_name}': {path_obj}")
            continue
        frame = read_table(path_obj)
        frame, meta = canonicalize_prediction_frame(
            frame,
            split_name=split_name,
            threshold=threshold,
            score_kind_override=score_kind_override,
        )
        frames[split_name] = frame
        metadata[split_name] = meta
    return frames, metadata


def derive_unseen_flags(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    updated = {name: frame.copy() for name, frame in frames.items()}
    if any("is_unseen_cwe" in frame.columns for frame in updated.values()):
        return updated
    seen_cwes: set[str] = set()
    for split_name in ("train", "val", "test"):
        frame = updated.get(split_name)
        if frame is not None and "cwe_id" in frame.columns:
            seen_cwes.update(frame["cwe_id"].dropna().astype(str).unique())
    if not seen_cwes:
        if "unseen_test" in updated and "cwe_id" in updated["unseen_test"].columns:
            unseen_cwes = set(updated["unseen_test"]["cwe_id"].dropna().astype(str).unique())
            for frame in updated.values():
                if "cwe_id" in frame.columns:
                    frame["is_unseen_cwe"] = frame["cwe_id"].astype(str).isin(unseen_cwes).astype(int)
        return updated
    for frame in updated.values():
        if "cwe_id" in frame.columns:
            frame["is_unseen_cwe"] = (~frame["cwe_id"].astype(str).isin(seen_cwes)).astype(int)
    return updated


def summarize_split_frame(df: pd.DataFrame) -> pd.Series:
    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "positive_rate": float(df["y_true"].mean()) if len(df) else np.nan,
        "unique_cwes": int(df["cwe_id"].nunique()) if "cwe_id" in df.columns else np.nan,
        "unique_projects": int(df["project"].nunique()) if "project" in df.columns else np.nan,
        "score_available": "score" in df.columns,
    }
    return pd.Series(summary)


def validation_checks(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append({"check": "missing_y_true", "value": int(df["y_true"].isna().sum())})
    rows.append({"check": "missing_y_pred", "value": int(df["y_pred"].isna().sum())})
    if "score" in df.columns:
        rows.append({"check": "missing_score", "value": int(df["score"].isna().sum())})
        malformed_prob = 0
        if df["score"].notna().any():
            malformed_prob = int((~df["score"].dropna().between(0, 1)).sum())
        rows.append({"check": "malformed_probability_values", "value": malformed_prob})
    if "row_id" in df.columns:
        rows.append({"check": "duplicate_row_id", "value": int(df["row_id"].duplicated().sum())})
    if "cwe_id" in df.columns:
        rows.append({"check": "missing_cwe_id", "value": int(df["cwe_id"].isna().sum())})
    if "project" in df.columns:
        rows.append({"check": "missing_project", "value": int(df["project"].isna().sum())})
    return pd.DataFrame(rows)


def confusion_counts(y_true: pd.Series, y_pred: pd.Series) -> dict[str, int]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)}


def _safe_metric(fn, *args, **kwargs) -> float:
    try:
        return float(fn(*args, **kwargs))
    except ValueError:
        return np.nan


def compute_binary_metrics(df: pd.DataFrame) -> dict[str, Any]:
    clean = df.dropna(subset=["y_true", "y_pred"]).copy()
    if clean.empty:
        return {
            "support": 0,
            "positive_rate": np.nan,
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "TP": 0,
            "FP": 0,
            "TN": 0,
            "FN": 0,
            "fp_rate": np.nan,
            "fn_rate": np.nan,
        }
    y_true = clean["y_true"].astype(int)
    y_pred = clean["y_pred"].astype(int)
    counts = confusion_counts(y_true, y_pred)
    metrics_dict = {
        "support": int(len(clean)),
        "positive_rate": float(y_true.mean()),
        "accuracy": _safe_metric(metrics.accuracy_score, y_true, y_pred),
        "precision": _safe_metric(metrics.precision_score, y_true, y_pred, zero_division=0),
        "recall": _safe_metric(metrics.recall_score, y_true, y_pred, zero_division=0),
        "f1": _safe_metric(metrics.f1_score, y_true, y_pred, zero_division=0),
        **counts,
    }
    negatives = counts["FP"] + counts["TN"]
    positives = counts["TP"] + counts["FN"]
    metrics_dict["fp_rate"] = counts["FP"] / negatives if negatives else np.nan
    metrics_dict["fn_rate"] = counts["FN"] / positives if positives else np.nan
    if "score" in clean.columns and clean["score"].notna().nunique() > 1 and y_true.nunique() > 1:
        metrics_dict["roc_auc"] = _safe_metric(metrics.roc_auc_score, y_true, clean["score"])
        metrics_dict["pr_auc"] = _safe_metric(metrics.average_precision_score, y_true, clean["score"])
        metrics_dict["brier_score"] = _safe_metric(metrics.brier_score_loss, y_true, clean["score"])
    else:
        metrics_dict["roc_auc"] = np.nan
        metrics_dict["pr_auc"] = np.nan
        metrics_dict["brier_score"] = np.nan
    return metrics_dict


def aggregate_metrics_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split_name, frame in frames.items():
        row = compute_binary_metrics(frame)
        row["split"] = split_name
        rows.append(row)
    return pd.DataFrame(rows).set_index("split").sort_index()


def interpretation_for_test_vs_unseen(aggregate_df: pd.DataFrame) -> str:
    if "test" not in aggregate_df.index or "unseen_test" not in aggregate_df.index:
        return "Test/unseen_test comparison unavailable."
    test_row = aggregate_df.loc["test"]
    unseen_row = aggregate_df.loc["unseen_test"]
    delta_f1 = unseen_row["f1"] - test_row["f1"]
    delta_precision = unseen_row["precision"] - test_row["precision"]
    delta_recall = unseen_row["recall"] - test_row["recall"]

    if pd.isna(delta_f1):
        return "Test/unseen_test comparison unavailable."

    if abs(delta_f1) <= 0.03:
        gap_label = "small"
    elif abs(delta_f1) <= 0.08:
        gap_label = "moderate"
    else:
        gap_label = "large"

    if delta_recall < -0.03 and delta_precision >= -0.01:
        behavior = "The unseen pattern looks conservative: recall drops more than precision."
    elif delta_precision < -0.03 and delta_recall >= -0.01:
        behavior = "The unseen pattern suggests more false positives rather than conservative abstention."
    elif delta_f1 >= -0.03:
        behavior = "Generalization to unseen CWE looks stable at the aggregate level."
    else:
        behavior = "Aggregate performance drops on unseen CWE enough to warrant deeper per-CWE inspection."

    return (
        f"Unseen F1 change vs test is {delta_f1:+.3f} ({gap_label} gap). "
        f"Unseen precision change is {delta_precision:+.3f}; unseen recall change is {delta_recall:+.3f}. "
        f"{behavior}"
    )


def grouped_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group_value, group_df in df.groupby(group_col, dropna=False):
        row = compute_binary_metrics(group_df)
        row[group_col] = group_value
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("support", ascending=False)


def per_cwe_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if "cwe_id" not in df.columns:
        return pd.DataFrame()
    rows = []
    for cwe_id, group_df in df.groupby("cwe_id", dropna=False):
        row = compute_binary_metrics(group_df)
        row["cwe_id"] = cwe_id
        row["vulnerable_examples"] = int(group_df["y_true"].astype(int).sum())
        row["benign_examples"] = int((1 - group_df["y_true"].astype(int)).sum())
        if "is_unseen_cwe" in group_df.columns:
            row["is_unseen_cwe"] = int(group_df["is_unseen_cwe"].mode().iloc[0])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["support", "cwe_id"], ascending=[False, True])


def per_project_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if "project" not in df.columns:
        return pd.DataFrame()
    rows = []
    for project, group_df in df.groupby("project", dropna=False):
        row = compute_binary_metrics(group_df)
        row["project"] = project
        row["vulnerable_examples"] = int(group_df["y_true"].astype(int).sum())
        row["benign_examples"] = int((1 - group_df["y_true"].astype(int)).sum())
        if "cwe_id" in group_df.columns:
            row["unique_cwes"] = int(group_df["cwe_id"].nunique())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["support", "project"], ascending=[False, True])


def add_length_buckets(df: pd.DataFrame, column: str, edges: list[float]) -> pd.DataFrame:
    if column not in df.columns:
        return df
    labels = []
    for left, right in zip(edges[:-1], edges[1:]):
        if math.isinf(right):
            labels.append(f">{int(left)}")
        else:
            labels.append(f"{int(left)}-{int(right)}")
    result = df.copy()
    result[f"{column}_bucket"] = pd.cut(
        result[column],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    return result


def stratified_metrics(df: pd.DataFrame, column: str, edges: list[float]) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame()
    bucketed = add_length_buckets(df, column, edges)
    return grouped_metrics(bucketed.dropna(subset=[f"{column}_bucket"]), f"{column}_bucket")


def rarity_flags(df: pd.DataFrame, column: str, min_support: int = 20) -> pd.Series:
    counts = df[column].value_counts()
    return df[column].map(counts).fillna(0).lt(min_support)


def threshold_sweep(
    df: pd.DataFrame,
    thresholds: np.ndarray,
    score_kind: str | None = "prob",
) -> pd.DataFrame:
    if "score" not in df.columns:
        return pd.DataFrame()
    clean = df.dropna(subset=["y_true", "score"]).copy()
    if clean.empty:
        return pd.DataFrame()
    rows = []
    decision_values = clean["score"]
    if score_kind == "logit" and "score_raw" in clean.columns:
        decision_values = clean["score_raw"]
    for threshold in thresholds:
        predicted = (decision_values >= threshold).astype(int)
        temp = clean.assign(y_pred=predicted)
        row = compute_binary_metrics(temp)
        row["threshold"] = float(threshold)
        rows.append(row)
    return pd.DataFrame(rows)


def best_threshold_from_validation(val_df: pd.DataFrame, thresholds: np.ndarray, score_kind: str | None = "prob") -> dict[str, Any] | None:
    sweep = threshold_sweep(val_df, thresholds=thresholds, score_kind=score_kind)
    if sweep.empty:
        return None
    best = sweep.sort_values(["f1", "recall", "precision", "threshold"], ascending=[False, False, False, True]).iloc[0]
    return best.to_dict()


def bootstrap_metric_ci(
    df: pd.DataFrame,
    metric_name: str,
    n_bootstrap: int = 1000,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    clean = df.dropna(subset=["y_true", "y_pred"]).copy()
    if clean.empty:
        return {"metric": metric_name, "mean": np.nan, "lower": np.nan, "upper": np.nan, "n_bootstrap": n_bootstrap}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, len(clean), size=len(clean))
        sample = clean.iloc[sample_idx]
        values.append(compute_binary_metrics(sample)[metric_name])
    values_arr = np.asarray(values, dtype=float)
    return {
        "metric": metric_name,
        "mean": float(np.nanmean(values_arr)),
        "lower": float(np.nanpercentile(values_arr, 2.5)),
        "upper": float(np.nanpercentile(values_arr, 97.5)),
        "n_bootstrap": n_bootstrap,
    }


def bootstrap_summary(frames: dict[str, pd.DataFrame], metric_names: tuple[str, ...] = ("f1", "precision", "recall"), n_bootstrap: int = 1000) -> pd.DataFrame:
    rows = []
    for split_name in ("test", "unseen_test"):
        frame = frames.get(split_name)
        if frame is None:
            continue
        for metric_name in metric_names:
            result = bootstrap_metric_ci(frame, metric_name, n_bootstrap=n_bootstrap)
            result["split"] = split_name
            rows.append(result)
    return pd.DataFrame(rows)


def error_frame(df: pd.DataFrame, error_type: str) -> pd.DataFrame:
    clean = df.dropna(subset=["y_true", "y_pred"]).copy()
    y_true = clean["y_true"].astype(int)
    y_pred = clean["y_pred"].astype(int)
    if error_type == "fn":
        return clean[(y_true == 1) & (y_pred == 0)].copy()
    if error_type == "fp":
        return clean[(y_true == 0) & (y_pred == 1)].copy()
    raise ValueError("error_type must be 'fn' or 'fp'")


def bucket_scores(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    if "score" not in df.columns:
        return df
    result = df.copy()
    edges = np.linspace(0.0, 1.0, bins + 1)
    labels = [f"{edges[i]:.1f}-{edges[i + 1]:.1f}" for i in range(len(edges) - 1)]
    result["score_bucket"] = pd.cut(result["score"], bins=edges, labels=labels, include_lowest=True)
    return result


def sample_error_rows(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    desired = [col for col in ("row_id", "project", "cwe_id", "y_true", "y_pred", "score", "token_length", "code_preview") if col in df.columns]
    if not desired:
        return df.head(n)
    return df[desired].head(n)


def make_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_figure(fig: plt.Figure, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")


def plot_metric_bars(df: pd.DataFrame, category_col: str, metrics_to_plot: list[str], title: str, figsize: tuple[int, int] = (8, 4)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(df))
    width = 0.25
    for idx, metric_name in enumerate(metrics_to_plot):
        ax.bar(x + (idx - (len(metrics_to_plot) - 1) / 2) * width, df[metric_name], width=width, label=metric_name)
    ax.set_xticks(x)
    ax.set_xticklabels(df[category_col].astype(str).tolist(), rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Metric")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_support_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str, hue_col: str | None = None, figsize: tuple[int, int] = (6, 4)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    if hue_col and hue_col in df.columns:
        for hue_value, group_df in df.groupby(hue_col):
            ax.scatter(group_df[x_col], group_df[y_col], alpha=0.75, label=str(hue_value))
        ax.legend(title=hue_col)
    else:
        ax.scatter(df[x_col], df[y_col], alpha=0.75)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def plot_threshold_curves(sweeps: dict[str, pd.DataFrame], metrics_to_plot: tuple[str, ...] = ("precision", "recall", "f1"), figsize: tuple[int, int] = (8, 5)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    for split_name, sweep_df in sweeps.items():
        if sweep_df.empty:
            continue
        for metric_name in metrics_to_plot:
            ax.plot(sweep_df["threshold"], sweep_df[metric_name], label=f"{split_name} {metric_name}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric")
    ax.set_ylim(0, 1)
    ax.set_title("Threshold sensitivity")
    ax.grid(True, alpha=0.2)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


def plot_score_histogram(series_map: dict[str, pd.Series], title: str, figsize: tuple[int, int] = (8, 4), bins: int = 20) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    for label, series in series_map.items():
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if clean.empty:
            continue
        ax.hist(clean, bins=bins, alpha=0.5, density=True, label=label)
    ax.set_title(title)
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    return fig


def reliability_table(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    if "score" not in df.columns:
        return pd.DataFrame()
    clean = df.dropna(subset=["score", "y_true"]).copy()
    if clean.empty:
        return pd.DataFrame()
    clean["prob_bin"] = pd.cut(clean["score"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    table = clean.groupby("prob_bin", observed=False).agg(
        support=("y_true", "size"),
        mean_pred=("score", "mean"),
        empirical_rate=("y_true", "mean"),
    ).reset_index()
    return table


def plot_reliability(table: pd.DataFrame, title: str = "Reliability plot", figsize: tuple[int, int] = (5, 5)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.plot(table["mean_pred"], table["empirical_rate"], marker="o")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def confirm_unseen_split_design(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    test_frame = frames.get("test")
    unseen_frame = frames.get("unseen_test")
    if unseen_frame is not None and "is_unseen_cwe" in unseen_frame.columns:
        rows.append({
            "check": "unseen_test_all_heldout",
            "value": bool(unseen_frame["is_unseen_cwe"].dropna().eq(1).all()),
        })
    if test_frame is not None and "is_unseen_cwe" in test_frame.columns:
        rows.append({
            "check": "test_contains_only_seen_cwe",
            "value": bool(test_frame["is_unseen_cwe"].dropna().eq(0).all()),
        })
        rows.append({
            "check": "test_unseen_share",
            "value": float(test_frame["is_unseen_cwe"].mean()),
        })
    return pd.DataFrame(rows)


def make_report_summary(
    aggregate_df: pd.DataFrame,
    per_cwe_df: pd.DataFrame | None = None,
    per_project_df: pd.DataFrame | None = None,
    threshold_note: str | None = None,
) -> str:
    lines = []
    if "test" in aggregate_df.index:
        row = aggregate_df.loc["test"]
        lines.append(
            "Test: "
            f"accuracy={row['accuracy']:.3f}, precision={row['precision']:.3f}, recall={row['recall']:.3f}, f1={row['f1']:.3f}."
        )
    if "unseen_test" in aggregate_df.index:
        row = aggregate_df.loc["unseen_test"]
        lines.append(
            "Unseen test: "
            f"accuracy={row['accuracy']:.3f}, precision={row['precision']:.3f}, recall={row['recall']:.3f}, f1={row['f1']:.3f}."
        )
    lines.append(interpretation_for_test_vs_unseen(aggregate_df))
    if per_cwe_df is not None and not per_cwe_df.empty:
        weakest = per_cwe_df.sort_values(["f1", "support"], ascending=[True, False]).head(3)["cwe_id"].astype(str).tolist()
        lines.append(f"Weak CWEs by F1: {', '.join(weakest)}.")
    if per_project_df is not None and not per_project_df.empty:
        weakest = per_project_df.sort_values(["f1", "support"], ascending=[True, False]).head(3)["project"].astype(str).tolist()
        lines.append(f"Weak projects by F1: {', '.join(weakest)}.")
    if threshold_note:
        lines.append(threshold_note)
    return "\n".join(f"- {line}" for line in lines)


def red_flag_summary(
    aggregate_df: pd.DataFrame,
    per_cwe_df: pd.DataFrame | None = None,
    per_project_df: pd.DataFrame | None = None,
    length_tables: dict[str, pd.DataFrame] | None = None,
    unseen_recall_drop_threshold: float = 0.10,
) -> list[str]:
    flags: list[str] = []
    if {"test", "unseen_test"} <= set(aggregate_df.index):
        test_recall = aggregate_df.loc["test", "recall"]
        unseen_recall = aggregate_df.loc["unseen_test", "recall"]
        if pd.notna(test_recall) and pd.notna(unseen_recall) and (test_recall - unseen_recall) > unseen_recall_drop_threshold:
            flags.append(f"Unseen recall drops by {(test_recall - unseen_recall):.3f}, above the red-flag threshold.")
    if per_cwe_df is not None and not per_cwe_df.empty and "is_unseen_cwe" in per_cwe_df.columns:
        weak_unseen = per_cwe_df[(per_cwe_df["is_unseen_cwe"] == 1) & (per_cwe_df["support"] >= 5) & (per_cwe_df["recall"] <= 0.05)]
        if not weak_unseen.empty:
            flags.append(f"{len(weak_unseen)} supported unseen CWE(s) have near-zero recall.")
    if per_project_df is not None and not per_project_df.empty:
        top_project_error_share = (
            per_project_df.assign(total_errors=per_project_df["FP"] + per_project_df["FN"])
            .sort_values("total_errors", ascending=False)
            .head(1)
        )
        if not top_project_error_share.empty and top_project_error_share["total_errors"].sum() > 0:
            total_errors = int((per_project_df["FP"] + per_project_df["FN"]).sum())
            share = float(top_project_error_share["total_errors"].iloc[0] / total_errors) if total_errors else 0.0
            if share >= 0.4:
                flags.append(f"One project contributes {share:.1%} of all evaluation errors.")
    if length_tables:
        for column, table in length_tables.items():
            if not table.empty and "fn_rate" in table.columns:
                worst = table.sort_values("fn_rate", ascending=False).head(1)
                if not worst.empty and worst["fn_rate"].iloc[0] >= 0.5 and worst["support"].iloc[0] >= 10:
                    flags.append(f"{column} bucket {worst.iloc[0][f'{column}_bucket']} has high FN rate ({worst['fn_rate'].iloc[0]:.2f}).")
    return flags
