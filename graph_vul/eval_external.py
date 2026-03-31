"""Tier-1 external evaluation: run trained checkpoint on PrimeVul & DiverseVul."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from graph_vul.config import (
    FINETUNE_BATCH_SIZE,
    FINETUNE_CHECKPOINT_PATH,
    PROCESSED_DIR,
    UNIXCODER_MODEL_NAME,
)
from graph_vul.dataset.external_datasets import EXTERNAL_DATASETS, ExternalEvalDataset
from graph_vul.model.func_finetune_lit_module import FuncFinetuneLitModule

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("data/analysis/external_eval_tier1")
BIGVUL_PREDICTIONS_DIR = Path("data/analysis/func_finetune_predictions")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> dict:
    """Compute binary classification metrics."""
    metrics: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "num_samples": int(len(y_true)),
        "num_vulnerable": int(y_true.sum()),
        "num_benign": int(len(y_true) - y_true.sum()),
    }
    # AUC metrics require both classes present
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
        metrics["pr_auc"] = float(average_precision_score(y_true, probs))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    return metrics


def _log_unseen_cwes(
    dataset_name: str,
    cwe_ids: list[str],
    cwe_embeddings_dir: Path,
) -> tuple[int, int]:
    """Log count/fraction of samples with no pre-computed CWE embedding."""
    total = len(cwe_ids)
    unseen = sum(
        1 for cid in cwe_ids
        if cid and not (cwe_embeddings_dir / f"{cid}.pt").exists()
    )
    frac = unseen / total if total > 0 else 0.0
    logger.info(
        "%s: %d / %d samples (%.1f%%) have unseen CWE IDs (zero-vector fallback)",
        dataset_name, unseen, total, frac * 100,
    )
    return unseen, total


def _load_bigvul_test_metrics() -> dict | None:
    """Try to load BigVul test predictions and compute comparable metrics."""
    test_path = BIGVUL_PREDICTIONS_DIR / "test_predictions.csv"
    if not test_path.exists():
        logger.info("BigVul test predictions not found at %s — skipping comparison", test_path)
        return None
    df = pd.read_csv(test_path)
    if "y_true" not in df.columns or "y_pred" not in df.columns or "prob" not in df.columns:
        logger.warning("BigVul test predictions missing required columns")
        return None
    return compute_metrics(
        df["y_true"].values,
        df["y_pred"].values,
        df["prob"].values,
    )


def _generate_summary(
    all_metrics: dict[str, dict],
    output_dir: Path,
) -> None:
    """Write summary.md comparing all datasets."""
    cols = ["accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"]
    header = "| Dataset | Samples | Vul | Benign | " + " | ".join(c.replace("_", " ").upper() for c in cols) + " |"
    sep = "|" + "|".join(["---"] * (len(cols) + 4)) + "|"

    rows = []
    for name, m in all_metrics.items():
        vals = []
        for c in cols:
            v = m.get(c)
            vals.append(f"{v:.4f}" if v is not None else "N/A")
        row = f"| {name} | {m.get('num_samples', 'N/A')} | {m.get('num_vulnerable', 'N/A')} | {m.get('num_benign', 'N/A')} | " + " | ".join(vals) + " |"
        rows.append(row)

    summary = f"# External Evaluation — Tier 1 Summary\n\n{header}\n{sep}\n" + "\n".join(rows) + "\n"
    summary_path = output_dir / "summary.md"
    summary_path.write_text(summary)
    logger.info("Wrote summary to %s", summary_path)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def evaluate_external(
    checkpoint_path: str | Path = FINETUNE_CHECKPOINT_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    batch_size: int = FINETUNE_BATCH_SIZE,
    threshold: float = 0.5,
    num_workers: int = 4,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cwe_embeddings_dir = PROCESSED_DIR / "cwe_embeddings"

    device = _device()
    logger.info("Loading checkpoint from %s on %s", checkpoint_path, device)
    tokenizer = AutoTokenizer.from_pretrained(UNIXCODER_MODEL_NAME)
    model = FuncFinetuneLitModule.load_from_checkpoint(str(checkpoint_path), map_location=device)
    model.to(device)
    model.eval()
    model.freeze()

    combined_metrics: dict[str, dict] = {}

    # BigVul test baseline (if available)
    bigvul_metrics = _load_bigvul_test_metrics()
    if bigvul_metrics is not None:
        combined_metrics["bigvul_test"] = bigvul_metrics

    for dataset_name, loader_fn in EXTERNAL_DATASETS.items():
        logger.info("=== Evaluating %s ===", dataset_name)
        df = loader_fn()

        # Log unseen CWEs
        _log_unseen_cwes(dataset_name, df["cwe_id"].tolist(), cwe_embeddings_dir)

        # Build dataset and loader
        dataset = ExternalEvalDataset(
            df=df,
            tokenizer=tokenizer,
            cwe_embeddings_dir=cwe_embeddings_dir,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

        # Inference
        logits_parts: list[torch.Tensor] = []
        probs_parts: list[torch.Tensor] = []
        preds_parts: list[torch.Tensor] = []
        token_lengths_parts: list[torch.Tensor] = []

        progress = tqdm(loader, desc=f"Evaluating {dataset_name}", unit="batch")
        with torch.inference_mode():
            for batch in progress:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                cwe_embedding = batch["cwe_embedding"].to(device)

                logits = model(input_ids, attention_mask, cwe_embedding, graph_batch=None).squeeze(-1)
                probs = torch.sigmoid(logits)
                preds = (probs >= threshold).long()

                logits_parts.append(logits.detach().cpu())
                probs_parts.append(probs.detach().cpu())
                preds_parts.append(preds.detach().cpu())
                token_lengths_parts.append(batch["attention_mask"].sum(dim=1).detach().cpu())

        # Build predictions DataFrame
        pred_df = df.copy()
        pred_df["y_pred"] = torch.cat(preds_parts).numpy()
        pred_df["prob"] = torch.cat(probs_parts).numpy()
        pred_df["logit"] = torch.cat(logits_parts).numpy()
        pred_df["token_length"] = torch.cat(token_lengths_parts).numpy()

        # Save predictions
        pred_path = output_dir / f"{dataset_name}_predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        logger.info("Wrote %s predictions: %d rows -> %s", dataset_name, len(pred_df), pred_path)

        # Compute and save per-dataset metrics
        y_true = pred_df["vul"].values
        y_pred = pred_df["y_pred"].values
        prob_vals = pred_df["prob"].values
        ds_metrics = compute_metrics(y_true, y_pred, prob_vals)

        metrics_path = output_dir / f"{dataset_name}_metrics.json"
        with metrics_path.open("w") as f:
            json.dump(ds_metrics, f, indent=2)
        logger.info("Wrote %s metrics -> %s", dataset_name, metrics_path)

        combined_metrics[dataset_name] = ds_metrics

    # Combined metrics file (DVC-tracked)
    combined_path = output_dir / "metrics.json"
    with combined_path.open("w") as f:
        json.dump(combined_metrics, f, indent=2)
    logger.info("Wrote combined metrics -> %s", combined_path)

    # Summary report
    _generate_summary(combined_metrics, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="External evaluation on Tier-1 datasets")
    parser.add_argument("--checkpoint", type=Path, default=FINETUNE_CHECKPOINT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=FINETUNE_BATCH_SIZE)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    evaluate_external(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        threshold=args.threshold,
        num_workers=args.num_workers,
    )
