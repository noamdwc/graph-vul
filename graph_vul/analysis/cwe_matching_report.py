"""Canonical report from hashed prediction exports; no notebook-derived metrics."""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from graph_vul.artifacts import file_sha256, write_json
from graph_vul.model.cwe_matching import parse_cwe_ids


def metrics(frame):
    labels, predictions = frame["y_true"], frame["y_pred"]
    benign = frame[labels == 0]
    vulnerable = frame[labels == 1]
    return {
        "rows": len(frame),
        "positive_support": int(labels.sum()),
        "pr_auc": float(average_precision_score(labels, frame["margin"]))
        if labels.sum()
        else None,
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "benign_support": len(benign),
        "false_positive_rate": float(benign["y_pred"].mean()) if len(benign) else None,
        "annotated_vulnerable_support": int(
            vulnerable["retrieval_hit_top1"].notna().sum()
        ),
        "retrieval_top1": float(vulnerable["retrieval_hit_top1"].mean())
        if vulnerable["retrieval_hit_top1"].notna().any()
        else None,
        "retrieval_top3": float(vulnerable["retrieval_hit_top3"].mean())
        if vulnerable["retrieval_hit_top3"].notna().any()
        else None,
    }


def generate_report(manifest_paths, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    summaries, per_cwe = [], []
    preparation_hashes = None
    candidates = None
    checkpoints = {}
    comparison_conditions = None
    pilot = None
    for manifest_path in map(Path, manifest_paths):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("contract") != 1:
            raise ValueError("Historical exports cannot enter the canonical report")
        preparation = manifest["provenance"]["preparation"]
        is_pilot = bool(preparation.get("pilot", False))
        if pilot is not None and is_pilot != pilot:
            raise ValueError(
                "Cannot mix pilot diagnostics with full experiment results"
            )
        pilot = is_pilot
        hashes = preparation["hashes"]
        if preparation_hashes is not None and preparation_hashes != hashes:
            raise ValueError("Cannot compare runs with different preparation artifacts")
        preparation_hashes = hashes
        if candidates is not None and manifest["candidate_ids"] != candidates:
            raise ValueError("Cannot compare different inference catalogs")
        candidates = manifest["candidate_ids"]
        config = manifest["resolved_config"]
        provenance = manifest["provenance"]
        conditions = {
            "code_source": provenance["code"]["source_sha256"],
            "cwe_embeddings": provenance["cwe_embeddings_sha256"],
            "budget": provenance["budget"],
            "training_controls": {
                name: config[name]
                for name in (
                    "encoder_lr",
                    "head_lr",
                    "weight_decay",
                    "head_dropout",
                    "freeze_encoder_epochs",
                    "freeze_encoder_fully",
                    "unfreeze_encoder_lr",
                    "matching_negatives_per_positive",
                    "matching_loss_lambda",
                    "matching_triplet_margin",
                )
            },
        }
        if comparison_conditions is not None and conditions != comparison_conditions:
            raise ValueError(
                "Cannot compare different code, embeddings or training budgets/settings"
            )
        comparison_conditions = conditions
        run_key = (config["ablation_mode"], manifest["provenance"]["seed"])
        checkpoint_identity = (
            manifest["checkpoint_sha256"],
            manifest["calibration_threshold"],
        )
        if run_key in checkpoints and checkpoints[run_key] != checkpoint_identity:
            raise ValueError(
                "Internal/external exports refer to different model or calibration"
            )
        checkpoints[run_key] = checkpoint_identity
        for split, export in manifest["splits"].items():
            path = manifest_path.parent / export["file"]
            if file_sha256(path) != export["sha256"]:
                raise ValueError(f"Prediction artifact changed: {path}")
            frame = pd.read_csv(path)
            if len(frame) != export["rows"] or not frame["sample_id"].is_unique:
                raise ValueError(f"Export count/identity mismatch: {path}")
            identity = {
                "experiment": config["ablation_mode"],
                "seed": manifest["provenance"]["seed"],
                "checkpoint": manifest["checkpoint_sha256"],
                "split": split,
            }
            summaries.append({**identity, "slice": "all", **metrics(frame)})
            short = frame[frame["token_length"] <= 64]
            if len(short):
                summaries.append({**identity, "slice": "0-64 tokens", **metrics(short)})
            annotations = frame["cwe_id"].apply(parse_cwe_ids)
            for cwe in sorted({cwe for values in annotations for cwe in values}):
                # Per-CWE binary metric compares its positive support with labeled benign rows.
                selected = frame[
                    (frame["y_true"] == 0)
                    | annotations.apply(lambda values: cwe in values)
                ]
                per_cwe.append({**identity, "cwe": cwe, **metrics(selected)})
    summary = pd.DataFrame(summaries)
    if summary.duplicated(["experiment", "seed", "split", "slice"]).any():
        raise ValueError("Duplicate run/split exports would double-count report rows")
    summary.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(per_cwe).to_csv(output / "per_cwe.csv", index=False)
    # Shared-seed comparison: show all runs and mean/std, never silently claim three completed seeds.
    aggregate = (
        summary.groupby(["experiment", "split", "slice"])
        .agg(
            seeds=("seed", "nunique"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
        )
        .reset_index()
    )
    aggregate.to_csv(output / "seed_summary.csv", index=False)
    columns = [
        "experiment",
        "seed",
        "split",
        "slice",
        "rows",
        "positive_support",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "retrieval_top1",
        "retrieval_top3",
    ]
    lines = [
        "# Graph-Vul integration pilot: execution diagnostics"
        if pilot
        else "# Graph-Vul matching results",
        "Pilot metrics verify a bounded workflow run; they do not estimate model performance."
        if pilot
        else "",
        "",
        "Generated only from versioned, hash-verified prediction artifacts. Unseen splits are overlapping diagnostic slices; row counts are not summed across them.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in summary[columns].itertuples(index=False, name=None):
        values = [
            f"{value:.4f}"
            if isinstance(value, float) and pd.notna(value)
            else str(value)
            for value in row
        ]
        lines.append("| " + " | ".join(values) + " |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    write_json(
        output / "sources.json",
        {str(path): file_sha256(path) for path in manifest_paths},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifests", dest="manifest_paths", nargs="+", type=Path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    generate_report(**vars(parser.parse_args()))
