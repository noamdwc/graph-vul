"""Compare completed E2/E3 exports; run in a fresh Python process after Colab installs."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graph_vul.analysis.post_experiment_analysis import compute_binary_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2-run", type=Path, required=True)
    parser.add_argument("--e3-run", type=Path, required=True)
    args = parser.parse_args()
    e2_run, e3_run = args.e2_run.resolve(), args.e3_run.resolve()
    rows = []
    fields = ("support", "f1", "precision", "recall", "TP", "FP", "TN", "FN")
    for split in ("test", "unseen_test"):
        e2 = pd.read_csv(e2_run / "predictions" / f"{split}_predictions.csv").set_index("sample_id").sort_index()
        e3 = pd.read_csv(e3_run / "predictions" / f"{split}_predictions.csv").set_index("sample_id").sort_index()
        if not e2.index.is_unique or not e3.index.is_unique or not e2.index.equals(e3.index):
            raise ValueError(f"Different or duplicate sample IDs: {split}")
        pd.testing.assert_frame_equal(e2[["y_true", "token_length"]], e3[["y_true", "token_length"]])
        for experiment, frame in (("E2_last4", e2), ("E3_all_layers", e3)):
            for cohort, data in (("all", frame), ("short_0_64", frame[frame.token_length.between(0, 64)])):
                values = compute_binary_metrics(data)
                rows.append({"experiment": experiment, "split": split, "cohort": cohort,
                             "positive_support": values["TP"] + values["FN"],
                             **{field: values[field] for field in fields}})
    comparison = pd.DataFrame(rows)
    print(comparison.to_string(index=False))
    short_unseen = comparison[(comparison.split == "unseen_test") & (comparison.cohort == "short_0_64")].set_index("experiment")
    delta = float(short_unseen.loc["E3_all_layers", "f1"] - short_unseen.loc["E2_last4", "f1"])
    result = {"reference_E3_run": e3_run.name, "E2_run": e2_run.name,
              "short_definition": "0 <= token_length <= 64", "threshold": 0.5,
              "E3_minus_E2_short_unseen_f1": delta, "metrics": rows}
    (e2_run / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print("E3 minus E2 short unseen-test F1:", delta)
    print("Download the entire E2 run folder:", e2_run)


if __name__ == "__main__":
    main()
