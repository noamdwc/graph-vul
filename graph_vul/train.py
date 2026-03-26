"""Training script for GraphVul model with PyTorch Lightning."""

import argparse
import json
import logging

import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Subset as TorchSubset
from torch_geometric.loader import DataLoader

from graph_vul.config import (
    BATCH_SIZE,
    DATA_DIR,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    GAT_HIDDEN_DIM,
    LEARNING_RATE,
    NUM_EPOCHS,
    SEED,
    SPLITS_DIR,
    SUBSET_CSV,
)
from graph_vul.dataset.cpg_dataset import CPGDataset
from graph_vul.model.lit_module import GraphVulLitModule

logger = logging.getLogger(__name__)

METRICS_PATH = DATA_DIR / "metrics.json"
CHECKPOINT_PATH = DATA_DIR / "best_model.ckpt"


def train(
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    seed: int = SEED,
) -> None:
    pl.seed_everything(seed)

    # Data
    dataset = CPGDataset()

    assignments = pd.read_csv(SPLITS_DIR / "assignments.csv")
    train_idx = assignments[assignments["split"] == "train"]["sample_id"].tolist()
    val_idx = assignments[assignments["split"] == "val"]["sample_id"].tolist()

    train_subset = TorchSubset(dataset, train_idx)
    val_subset = TorchSubset(dataset, val_idx)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Weighted resampling to balance classes — read labels from CSV to avoid materializing all graphs
    subset_meta = pd.read_csv(SUBSET_CSV, usecols=["sample_id", "vul"])
    train_assignments = assignments[assignments["split"] == "train"]
    train_meta = train_assignments.merge(subset_meta, on="sample_id")
    train_labels = torch.tensor(train_meta["vul"].values, dtype=torch.long)
    num_pos = train_labels.sum().item()
    num_neg = len(train_labels) - num_pos
    class_weights = {0: 1.0 / num_neg, 1: 1.0 / num_pos}
    sample_weights = torch.tensor([class_weights[int(l.item())] for l in train_labels])
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_subset), replacement=True)
    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler, num_workers=2)
    logger.info("Train: %d samples (%d pos, %d neg), Val: %d samples", len(train_idx), int(num_pos), int(num_neg), len(val_idx))

    # Model
    model = GraphVulLitModule(lr=lr)

    # Logger
    wandb_logger = WandbLogger(
        project="graph-vul",
        config={
            "learning_rate": lr,
            "gat_hidden_channels": GAT_HIDDEN_DIM,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "seed": seed,
        },
    )

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(CHECKPOINT_PATH.parent),
        filename="best_model",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
    )

    early_stop_cb = EarlyStopping(
        monitor="val_f1",
        mode="max",
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA,
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        precision="16-mixed",
        accumulate_grad_batches=4,
        callbacks=[checkpoint_cb, early_stop_cb, LearningRateMonitor(logging_interval="epoch")],
        logger=wandb_logger,
        deterministic=True,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Write DVC metrics
    metrics = {
        "val_accuracy": trainer.callback_metrics.get("val_accuracy", 0.0),
        "val_precision": trainer.callback_metrics.get("val_precision", 0.0),
        "val_recall": trainer.callback_metrics.get("val_recall", 0.0),
        "val_f1": trainer.callback_metrics.get("val_f1", 0.0),
        "train_accuracy": trainer.callback_metrics.get("train_accuracy", 0.0),
        "train_precision": trainer.callback_metrics.get("train_precision", 0.0),
        "train_recall": trainer.callback_metrics.get("train_recall", 0.0),
        "train_f1": trainer.callback_metrics.get("train_f1", 0.0),
    }
    metrics = {k: float(v) for k, v in metrics.items()}

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Wrote metrics to %s", METRICS_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GraphVul model")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(num_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed)
