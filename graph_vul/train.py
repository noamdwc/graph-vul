"""Training script for GraphVul model with PyTorch Lightning."""

import argparse
import json
import logging

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.loader import DataLoader

from graph_vul.config import (
    BATCH_SIZE,
    DATA_DIR,
    GAT_HIDDEN_DIM,
    LEARNING_RATE,
    NUM_EPOCHS,
    SEED,
    SPLITS_DIR,
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

    with open(SPLITS_DIR / "train.json") as f:
        train_idx = json.load(f)
    with open(SPLITS_DIR / "val.json") as f:
        val_idx = json.load(f)

    train_loader = DataLoader(dataset[torch.tensor(train_idx)], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset[torch.tensor(val_idx)], batch_size=batch_size, shuffle=False)

    logger.info("Train: %d samples, Val: %d samples", len(train_idx), len(val_idx))

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

    # Trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        callbacks=[checkpoint_cb, LearningRateMonitor(logging_interval="epoch")],
        logger=wandb_logger,
        deterministic=True,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Write DVC metrics
    metrics = {
        "accuracy": trainer.callback_metrics.get("val_accuracy", 0.0),
        "precision": trainer.callback_metrics.get("val_precision", 0.0),
        "recall": trainer.callback_metrics.get("val_recall", 0.0),
        "f1": trainer.callback_metrics.get("val_f1", 0.0),
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
