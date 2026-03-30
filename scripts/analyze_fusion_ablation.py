"""Compute detailed metrics from fusion ablation prediction CSVs."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from graph_vul.analysis.post_experiment_analysis import (
    DEFAULT_LENGTH_BUCKETS,
    canonicalize_prediction_frame,
    compute_binary_metrics,
    stratified_metrics,
)

DATA_DIR = ROOT / "data" / "fusion_ablation"
OUTPUT_DIR = ROOT / "outputs"

EXPERIMENTS = {
    "E0_late_only": "E0: Late fusion only (baseline)",
    "E1_last2": "E1: Last 2 layers + late",
    "E2_last4": "E2: Last 4 layers + late",
    "E3_all_layers": "E3: All 12 layers + late",
    "E4_last4_no_late_graph": "E4: Last 4 layers, no late",
}


def load_metrics_json(exp_dir: Path) -> dict:
    path = exp_dir / "func_finetune_metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def analyze_predictions(exp_dir: Path, split: str) -> dict:
    pred_path = exp_dir / "predictions" / f"{split}_predictions.csv"
    if not pred_path.exists():
        return {}

    df = pd.read_csv(pred_path)
    df, _ = canonicalize_prediction_frame(df, split_name=split)

    overall = compute_binary_metrics(df)
    result = {
        f"{split}_f1": overall["f1"],
        f"{split}_precision": overall["precision"],
        f"{split}_recall": overall["recall"],
        f"{split}_accuracy": overall["accuracy"],
        f"{split}_support": overall["support"],
    }

    # Token-length bucket metrics
    token_buckets = DEFAULT_LENGTH_BUCKETS["token_length"]
    strat = stratified_metrics(df, "token_length", token_buckets)
    if not strat.empty:
        for _, brow in strat.iterrows():
            bucket_col = "token_length_bucket"
            bucket_name = brow.get(bucket_col, str(brow.name))
            # Convert bucket label to key: "0-64" -> "0_64"
            bucket_key = str(bucket_name).replace("-", "_").replace(">", "gt")
            result[f"{split}_{bucket_key}_f1"] = brow.get("f1")
            result[f"{split}_{bucket_key}_precision"] = brow.get("precision")
            result[f"{split}_{bucket_key}_recall"] = brow.get("recall")
            result[f"{split}_{bucket_key}_support"] = brow.get("support")

    return result


def main():
    rows = []
    for tag, label in EXPERIMENTS.items():
        exp_dir = DATA_DIR / tag
        row = {"experiment": tag, "label": label}

        # Load saved metrics JSON
        saved = load_metrics_json(exp_dir)
        for k, v in saved.items():
            row[f"saved_{k}"] = v

        # Compute from prediction CSVs
        for split in ("test", "unseen_test"):
            row.update(analyze_predictions(exp_dir, split))

        rows.append(row)

    df = pd.DataFrame(rows)
    output_path = OUTPUT_DIR / "fusion_ablation_results.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}\n")

    # Print summary tables
    print("=" * 80)
    print("UNSEEN TEST METRICS (from saved JSON)")
    print("=" * 80)
    json_cols = ["experiment", "label", "saved_unseen_test_f1", "saved_unseen_test_precision",
                 "saved_unseen_test_recall", "saved_unseen_test_accuracy"]
    available = [c for c in json_cols if c in df.columns]
    print(df[available].to_string(index=False))

    print()
    print("=" * 80)
    print("OVERALL TEST METRICS (computed from predictions)")
    print("=" * 80)
    test_cols = ["experiment", "test_f1", "test_precision", "test_recall", "test_accuracy"]
    available = [c for c in test_cols if c in df.columns]
    print(df[available].to_string(index=False))

    print()
    print("=" * 80)
    print("TOKEN-LENGTH BUCKET F1 (test split)")
    print("=" * 80)
    bucket_f1_cols = ["experiment"] + [c for c in df.columns if c.startswith("test_") and c.endswith("_f1") and c != "test_f1"]
    available = [c for c in bucket_f1_cols if c in df.columns]
    print(df[available].to_string(index=False))

    print()
    print("=" * 80)
    print("SHORT FUNCTION (0-64 tokens) DETAIL (test split)")
    print("=" * 80)
    short_cols = ["experiment"] + [c for c in df.columns if "0_64" in c and c.startswith("test_")]
    available = [c for c in short_cols if c in df.columns]
    print(df[available].to_string(index=False))

    print()
    print("=" * 80)
    print("UNSEEN TEST METRICS (computed from predictions)")
    print("=" * 80)
    unseen_cols = ["experiment", "unseen_test_f1", "unseen_test_precision", "unseen_test_recall"]
    available = [c for c in unseen_cols if c in df.columns]
    print(df[available].to_string(index=False))

    print()
    print("=" * 80)
    print("TOKEN-LENGTH BUCKET F1 (unseen_test split)")
    print("=" * 80)
    unseen_bucket_cols = ["experiment"] + [c for c in df.columns if c.startswith("unseen_test_") and c.endswith("_f1") and c != "unseen_test_f1"]
    available = [c for c in unseen_bucket_cols if c in df.columns]
    print(df[available].to_string(index=False))


if __name__ == "__main__":
    main()
