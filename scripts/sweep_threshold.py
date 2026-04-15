"""Sweep classification threshold on a saved checkpoint without retraining."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import graph_vul.config as cfg
from graph_vul.dataset.func_finetune_dataset import FuncFinetuneDataset, available_splits, finetune_graph_collate
from graph_vul.model.func_finetune_lit_module import FuncFinetuneLitModule

logger = logging.getLogger(__name__)


def sweep_and_evaluate(
    checkpoint_path: Path,
    partition: str | None = "other",
    batch_size: int = 8,
    sweep_range: tuple[float, float] = (-4.0, 4.0),
    sweep_step: float = 0.05,
) -> dict:
    # Override config for partition
    if partition in ("short", "other"):
        cfg.FUNC_SUBSET_CSV = cfg.DATA_DIR / "partitions" / partition / "func_subset.csv"
        cfg.FUNC_SPLITS_DIR = cfg.DATA_DIR / "partitions" / partition / "splits"
        logger.info("Using partition '%s'", partition)

    tokenizer = AutoTokenizer.from_pretrained(cfg.CODE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    split_names = available_splits(cfg.FUNC_SPLITS_DIR)

    # Load model from checkpoint
    model = FuncFinetuneLitModule.load_from_checkpoint(str(checkpoint_path))
    model.eval()

    trainer = pl.Trainer(accelerator="auto", precision="16-mixed", logger=False)

    use_graph = model.hparams.get("graph_dim", 0) > 0
    collate_fn = finetune_graph_collate if use_graph else None

    # Collect logits for each split
    eval_splits = ["val", "test"]
    for s in ["unseen_val", "unseen_test"]:
        if s in split_names:
            eval_splits.append(s)

    split_data = {}
    for split_name in eval_splits:
        ds = FuncFinetuneDataset(split=split_name, tokenizer=tokenizer, use_graph=use_graph)
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
        predictions = trainer.predict(model, dataloaders=loader)
        logits = torch.cat([p["logits"] for p in predictions]).cpu().numpy()
        labels = torch.cat([p["labels"] for p in predictions]).cpu().numpy()
        split_data[split_name] = {"logits": logits, "labels": labels}
        logger.info("Collected %d predictions for %s", len(logits), split_name)

    # Sweep thresholds on val
    thresholds = np.arange(sweep_range[0], sweep_range[1] + sweep_step, sweep_step)
    val_logits = split_data["val"]["logits"]
    val_labels = split_data["val"]["labels"]

    best_f1 = -1.0
    best_thresh = 0.0
    sweep_results = []
    for thresh in thresholds:
        preds = (val_logits >= thresh).astype(int)
        f1 = f1_score(val_labels, preds, zero_division=0)
        sweep_results.append({"threshold": float(thresh), "f1": f1})
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)

    best_prob = 1 / (1 + np.exp(-best_thresh))
    logger.info(
        "Threshold sweep: best val F1=%.4f at threshold=%.2f (prob=%.4f)",
        best_f1, best_thresh, best_prob,
    )

    # Evaluate all splits at optimal threshold + default (0.0)
    results = {
        "optimal_threshold": best_thresh,
        "optimal_threshold_prob": best_prob,
        "optimal_val_f1": best_f1,
    }

    print(f"\n{'='*60}")
    print(f"Optimal threshold: {best_thresh:.2f} (prob={best_prob:.4f})")
    print(f"{'='*60}\n")

    for split_name, data in split_data.items():
        logits, labels = data["logits"], data["labels"]
        for thresh, label in [(0.0, "default_0.0"), (best_thresh, "optimal")]:
            preds = (logits >= thresh).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            prec = precision_score(labels, preds, zero_division=0)
            rec = recall_score(labels, preds, zero_division=0)
            acc = (preds == labels).mean()

            results[f"{split_name}_{label}_f1"] = f1
            results[f"{split_name}_{label}_precision"] = prec
            results[f"{split_name}_{label}_recall"] = rec
            results[f"{split_name}_{label}_accuracy"] = acc

        # Print comparison
        d = results
        print(f"  {split_name}:")
        print(f"    {'':20s} {'F1':>8s} {'Prec':>8s} {'Rec':>8s} {'Acc':>8s}")
        print(f"    {'threshold=0.0':20s} {d[f'{split_name}_default_0.0_f1']:8.4f} {d[f'{split_name}_default_0.0_precision']:8.4f} {d[f'{split_name}_default_0.0_recall']:8.4f} {d[f'{split_name}_default_0.0_accuracy']:8.4f}")
        print(f"    {'threshold=optimal':20s} {d[f'{split_name}_optimal_f1']:8.4f} {d[f'{split_name}_optimal_precision']:8.4f} {d[f'{split_name}_optimal_recall']:8.4f} {d[f'{split_name}_optimal_accuracy']:8.4f}")
        print()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep threshold on saved checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .ckpt file")
    parser.add_argument("--partition", type=str, default="other", help="Partition name (short/other/None)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None, help="Save results JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    results = sweep_and_evaluate(
        checkpoint_path=args.checkpoint,
        partition=args.partition if args.partition != "None" else None,
        batch_size=args.batch_size,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
