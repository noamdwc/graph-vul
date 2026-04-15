"""Training script for fine-tuning UniXcoder on function-level vulnerability detection."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer

from graph_vul.config import (
    DATA_DIR,
    EARLY_STOPPING_PATIENCE,
    FINETUNE_BATCH_SIZE,
    FINETUNE_ENCODER_LR,
    FINETUNE_GRAD_ACCUM_STEPS,
    FINETUNE_HEAD_LR,
    FINETUNE_NUM_EPOCHS,
    FUNC_SPLITS_DIR,
    FUNC_SUBSET_CSV,
    GRAPH_EDGE_VOCAB_PATH,
    GRAPH_HIDDEN_DIM,
    GRAPH_NODE_VOCAB_PATH,
    MATCHING_ABLATION_MODE,
    MATCHING_LOSS_LAMBDA,
    MATCHING_NEGATIVES_PER_POSITIVE,
    MATCHING_PROJECTION_DIM,
    MATCHING_TRIPLET_MARGIN,
    SEED,
    CODE_MODEL_NAME,
)
from graph_vul.dataset.func_finetune_dataset import FuncFinetuneDataset, available_splits, finetune_graph_collate
from graph_vul.model.func_finetune_lit_module import FuncFinetuneLitModule

logger = logging.getLogger(__name__)

CHECKPOINT_PATH = DATA_DIR / "func_finetune_model.ckpt"
METRICS_PATH = DATA_DIR / "func_finetune_metrics.json"


def _load_graph_vocab() -> tuple[int, int]:
    """Load graph vocabs and return (num_node_types, num_relations)."""
    with open(GRAPH_NODE_VOCAB_PATH) as f:
        node_vocab = json.load(f)
    with open(GRAPH_EDGE_VOCAB_PATH) as f:
        edge_vocab = json.load(f)
    return len(node_vocab), len(edge_vocab)


def _sweep_threshold(
    trainer: pl.Trainer,
    model: pl.LightningModule,
    val_loader: DataLoader,
    thresholds: np.ndarray | None = None,
) -> float:
    """Sweep classification thresholds on val set, return logit threshold maximizing F1."""
    if thresholds is None:
        thresholds = np.arange(-4.0, 4.05, 0.05)

    predictions = trainer.predict(model, dataloaders=val_loader, ckpt_path="best")
    all_logits = torch.cat([p["logits"] for p in predictions]).cpu().numpy()
    all_labels = torch.cat([p["labels"] for p in predictions]).cpu().numpy()

    best_f1 = -1.0
    best_thresh = 0.0
    sweep_results = []
    for thresh in thresholds:
        preds = (all_logits >= thresh).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        sweep_results.append({"threshold": float(thresh), "f1": f1})
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)

    best_prob = 1 / (1 + np.exp(-best_thresh))
    logger.info("Threshold sweep: best F1=%.4f at threshold=%.2f (prob=%.4f)", best_f1, best_thresh, best_prob)

    # Log sweep curve to W&B if available
    if trainer.logger and hasattr(trainer.logger, "experiment"):
        try:
            import wandb

            table = wandb.Table(columns=["threshold", "probability", "f1"])
            for r in sweep_results:
                prob = 1 / (1 + np.exp(-r["threshold"]))
                table.add_data(r["threshold"], prob, r["f1"])
            trainer.logger.experiment.log({"threshold_sweep": table})
            trainer.logger.experiment.log({
                "optimal_threshold": best_thresh,
                "optimal_threshold_prob": best_prob,
                "optimal_threshold_f1": best_f1,
            })
        except Exception:
            pass

    return best_thresh


def train(
    num_epochs: int = FINETUNE_NUM_EPOCHS,
    batch_size: int = FINETUNE_BATCH_SIZE,
    encoder_lr: float = FINETUNE_ENCODER_LR,
    head_lr: float = FINETUNE_HEAD_LR,
    grad_accum: int = FINETUNE_GRAD_ACCUM_STEPS,
    seed: int = SEED,
    use_graph: bool = False,
    fusion_layers: list[int] | None = list(range(12)),
    fusion_method: str = "scalar_gated_residual",
    final_graph_concat: bool = True,
    checkpoint_path: Path | None = None,
    metrics_path: Path | None = None,
    run_tag: str | None = None,
    freeze_encoder_epochs: int = 3,
    unfreeze_encoder_lr: float = 5e-6,
    focal_gamma: float = 0.0,
    focal_alpha: float = 0.25,
    head_dropout: float | None = None,
    freeze_encoder_fully: bool = False,
    ablation_mode: str = MATCHING_ABLATION_MODE,
    matching_projection_dim: int = MATCHING_PROJECTION_DIM,
    matching_negatives_per_positive: int = MATCHING_NEGATIVES_PER_POSITIVE,
    matching_loss_lambda: float = MATCHING_LOSS_LAMBDA,
    matching_triplet_margin: float = MATCHING_TRIPLET_MARGIN,
) -> None:
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_PATH
    if metrics_path is None:
        metrics_path = METRICS_PATH

    pl.seed_everything(seed)

    # Graph config
    graph_dim = 0
    num_node_types = 0
    num_relations = 0
    graph_stats: dict = {}

    if use_graph:
        num_node_types, num_relations = _load_graph_vocab()
        graph_dim = GRAPH_HIDDEN_DIM
        logger.info("Graph mode enabled: %d node types, %d edge types", num_node_types, num_relations)
        graph_stats = {
            "num_node_types": num_node_types,
            "num_relations": num_relations,
            "graph_hidden_dim": GRAPH_HIDDEN_DIM,
        }

    # Shared tokenizer — fine-tune backbone is Qwen3-0.6B; the frozen baseline
    # path still uses UNIXCODER_MODEL_NAME.
    tokenizer = AutoTokenizer.from_pretrained(CODE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Datasets
    split_names = available_splits(FUNC_SPLITS_DIR)
    train_ds = FuncFinetuneDataset(split="train", tokenizer=tokenizer, use_graph=use_graph)
    val_ds = FuncFinetuneDataset(split="val", tokenizer=tokenizer, use_graph=use_graph)
    test_ds = FuncFinetuneDataset(split="test", tokenizer=tokenizer, use_graph=use_graph)

    collate_fn = finetune_graph_collate if use_graph else None

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True,
        collate_fn=collate_fn,
    )
    val_dataloaders = [val_loader]
    if "unseen_val" in split_names:
        unseen_val_ds = FuncFinetuneDataset(split="unseen_val", tokenizer=tokenizer, use_graph=use_graph)
        unseen_val_loader = DataLoader(
            unseen_val_ds, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True,
            collate_fn=collate_fn,
        )
        val_dataloaders.append(unseen_val_loader)
    else:
        unseen_val_ds = None

    # Weighted resampling for class balance
    subset_meta = pd.read_csv(FUNC_SUBSET_CSV, usecols=["sample_id", "vul"])
    assignments = pd.read_csv(FUNC_SPLITS_DIR / "assignments.csv")
    train_assignments = assignments[assignments["split"] == "train"]
    train_meta = train_assignments.merge(subset_meta, on="sample_id")
    train_labels = torch.tensor(train_meta["vul"].values, dtype=torch.long)
    num_pos = train_labels.sum().item()
    num_neg = len(train_labels) - num_pos
    class_weights = {0: 1.0 / max(num_neg, 1), 1: 1.0 / max(num_pos, 1)}
    sample_weights = torch.tensor([class_weights[int(l.item())] for l in train_labels])
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, num_workers=4, persistent_workers=True,
        collate_fn=collate_fn,
    )

    logger.info(
        "Train: %d functions (%d pos, %d neg), Val: %d, Unseen Val: %d, Test: %d",
        len(train_ds), int(num_pos), int(num_neg), len(val_ds), len(unseen_val_ds) if unseen_val_ds is not None else 0, len(test_ds),
    )
    logger.info(
        "Effective batch size: %d (batch=%d x accum=%d)",
        batch_size * grad_accum, batch_size, grad_accum,
    )

    # Model
    model = FuncFinetuneLitModule(
        encoder_lr=encoder_lr,
        head_lr=head_lr,
        graph_dim=graph_dim,
        num_node_types=num_node_types,
        num_relations=num_relations,
        fusion_layers=fusion_layers,
        fusion_method=fusion_method,
        final_graph_concat=final_graph_concat,
        freeze_encoder_epochs=freeze_encoder_epochs,
        unfreeze_encoder_lr=unfreeze_encoder_lr,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        head_dropout=head_dropout,
        freeze_encoder_fully=freeze_encoder_fully,
        ablation_mode=ablation_mode,
        matching_projection_dim=matching_projection_dim,
        matching_negatives_per_positive=matching_negatives_per_positive,
        matching_loss_lambda=matching_loss_lambda,
        matching_triplet_margin=matching_triplet_margin,
    )

    # Logger
    default_run_name = "func-graph-finetune" if use_graph else "func-finetune"
    run_name = run_tag or default_run_name
    # Display label for W&B; reflects the active fine-tune backbone.
    model_name = "func-graph-finetune-qwen3-rgcn" if use_graph else "func-finetune-qwen3"
    wandb_logger = WandbLogger(
        project="graph-vul",
        name=run_name,
        config={
            "model": model_name,
            "encoder_lr": encoder_lr,
            "head_lr": head_lr,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "seed": seed,
            "use_graph": use_graph,
            "fusion_layers": fusion_layers,
            "fusion_method": fusion_method,
            "final_graph_concat": final_graph_concat,
            "run_tag": run_tag,
            "freeze_encoder_epochs": freeze_encoder_epochs,
            "unfreeze_encoder_lr": unfreeze_encoder_lr,
            "focal_gamma": focal_gamma,
            "focal_alpha": focal_alpha,
            "head_dropout": head_dropout,
            "freeze_encoder_fully": freeze_encoder_fully,
            "ablation_mode": ablation_mode,
            "matching_projection_dim": matching_projection_dim,
            "matching_negatives_per_positive": matching_negatives_per_positive,
            "matching_loss_lambda": matching_loss_lambda,
            "matching_triplet_margin": matching_triplet_margin,
            **graph_stats,
        },
    )

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(checkpoint_path.parent),
        filename=checkpoint_path.stem,
        monitor="val_f1",
        mode="max",
        save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_f1",
        mode="max",
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=0.005,
        verbose=True,
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        precision="16-mixed",
        accumulate_grad_batches=grad_accum,
        callbacks=[checkpoint_cb, early_stop_cb, LearningRateMonitor(logging_interval="epoch")],
        logger=wandb_logger,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_dataloaders)

    # Sweep threshold on val set to maximize F1
    best_threshold = _sweep_threshold(trainer, model, val_loader)
    model.threshold = best_threshold
    logger.info("Using optimized threshold=%.2f for test evaluation", best_threshold)

    # Test on seen CWEs
    model.set_test_prefix("test")
    trainer.test(model, dataloaders=test_loader, ckpt_path="best")

    # Test on unseen_test (unseen_val is already evaluated every epoch)
    unseen_splits = [split_name for split_name in ["unseen_test"] if split_name in split_names]
    for split_name in unseen_splits:
        ds = FuncFinetuneDataset(split=split_name, tokenizer=tokenizer, use_graph=use_graph)
        if len(ds) == 0:
            logger.warning("Skipping %s: no samples found", split_name)
            continue
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=4,
            collate_fn=collate_fn,
        )
        model.set_test_prefix(split_name)
        trainer.test(model, dataloaders=loader, ckpt_path="best")
        logger.info("Evaluated %s: %d samples", split_name, len(ds))

    # Write DVC metrics
    metrics = {
        "optimal_threshold": best_threshold,
        "ablation_mode": ablation_mode,
        "matching_projection_dim": matching_projection_dim,
        "matching_negatives_per_positive": matching_negatives_per_positive,
        "matching_loss_lambda": matching_loss_lambda,
        "matching_triplet_margin": matching_triplet_margin,
    }
    all_prefixes = ["val", "unseen_val", "test", "train"] + unseen_splits
    for prefix in all_prefixes:
        for metric in ("accuracy", "precision", "recall", "f1", "0_64_f1"):
            key = f"{prefix}_{metric}"
            val = trainer.callback_metrics.get(key, None)
            if val is not None:
                metrics[key] = float(val)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Wrote metrics to %s", metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune UniXcoder for vulnerability detection")
    parser.add_argument("--epochs", type=int, default=FINETUNE_NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=FINETUNE_BATCH_SIZE)
    parser.add_argument("--encoder-lr", type=float, default=FINETUNE_ENCODER_LR)
    parser.add_argument("--head-lr", type=float, default=FINETUNE_HEAD_LR)
    parser.add_argument("--grad-accum", type=int, default=FINETUNE_GRAD_ACCUM_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--use-graph", action="store_true", help="Enable graph (AST+CFG RGCN) features")
    parser.add_argument("--freeze-encoder-epochs", type=int, default=3, help="Epochs to freeze encoder (head-only warmup)")
    parser.add_argument("--unfreeze-encoder-lr", type=float, default=5e-6, help="Encoder LR after unfreezing")
    parser.add_argument("--focal-gamma", type=float, default=0.0, help="Focal loss gamma (0=standard BCE)")
    parser.add_argument("--focal-alpha", type=float, default=0.25, help="Focal loss alpha (positive class weight)")
    parser.add_argument("--head-dropout", type=float, default=None, help="Head dropout (default: use config FUNC_MLP_DROPOUT)")
    parser.add_argument("--freeze-encoder-fully", action="store_true", help="Keep encoder frozen for all epochs (head-only training)")
    parser.add_argument("--ablation-mode", type=str, default=MATCHING_ABLATION_MODE, choices=["E0", "E1", "E2", "E3", "E4"])
    parser.add_argument("--matching-projection-dim", type=int, default=MATCHING_PROJECTION_DIM)
    parser.add_argument("--matching-negatives-per-positive", type=int, default=MATCHING_NEGATIVES_PER_POSITIVE)
    parser.add_argument("--matching-loss-lambda", type=float, default=MATCHING_LOSS_LAMBDA)
    parser.add_argument("--matching-triplet-margin", type=float, default=MATCHING_TRIPLET_MARGIN)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        grad_accum=args.grad_accum,
        seed=args.seed,
        use_graph=args.use_graph,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        unfreeze_encoder_lr=args.unfreeze_encoder_lr,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        head_dropout=args.head_dropout,
        freeze_encoder_fully=args.freeze_encoder_fully,
        ablation_mode=args.ablation_mode,
        matching_projection_dim=args.matching_projection_dim,
        matching_negatives_per_positive=args.matching_negatives_per_positive,
        matching_loss_lambda=args.matching_loss_lambda,
        matching_triplet_margin=args.matching_triplet_margin,
    )
