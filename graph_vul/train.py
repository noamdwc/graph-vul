"""Training loop for GraphVul model with W&B experiment tracking."""

import argparse
import json
import logging
import random

import numpy as np
import torch
import wandb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from graph_vul.config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    DATA_DIR,
    GAT_HIDDEN_DIM,
    LEARNING_RATE,
    NUM_EPOCHS,
    SEED,
    VAL_SPLIT,
)
from graph_vul.dataset.cpg_dataset import CPGDataset
from graph_vul.model.classifier import GraphVulModel

logger = logging.getLogger(__name__)

METRICS_PATH = DATA_DIR / "metrics.json"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    seed: int = SEED,
) -> None:
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Initialize W&B
    wandb.init(
        project="graph-vul",
        config={
            "learning_rate": lr,
            "gat_hidden_channels": GAT_HIDDEN_DIM,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "seed": seed,
        },
    )

    try:
        _train_loop(num_epochs, batch_size, lr, seed, device)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        wandb.finish()


def _train_loop(
    num_epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> None:
    # Load dataset
    dataset = CPGDataset()
    labels = [int(dataset.get(i).y.item()) for i in range(len(dataset))]

    # Stratified train/val split
    train_idx, val_idx = train_test_split(
        list(range(len(dataset))),
        test_size=VAL_SPLIT,
        stratify=labels,
        random_state=seed,
    )

    train_dataset = dataset[torch.tensor(train_idx)]
    val_dataset = dataset[torch.tensor(val_idx)]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger.info("Train: %d samples, Val: %d samples", len(train_idx), len(val_idx))

    # Model, optimizer, loss
    model = GraphVulModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_f1 = 0.0
    best_metrics: dict[str, float] = {}

    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            logits = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_type=batch.edge_type,
                batch=batch.batch,
                text_vec=batch.text_vec,
            )

            loss = criterion(logits, batch.y.squeeze(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        # Validation
        metrics = evaluate(model, val_loader, device)

        logger.info(
            "Epoch %d/%d — Loss: %.4f | Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f",
            epoch,
            num_epochs,
            avg_loss,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
        )

        # Log to W&B
        wandb.log(
            {
                "train_loss": avg_loss,
                "val_loss": metrics.get("val_loss", avg_loss),
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            },
            step=epoch,
        )

        # Checkpoint on F1 improvement
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_metrics = {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
            CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            logger.info("Saved best model (F1=%.4f) to %s", best_f1, CHECKPOINT_PATH)

    logger.info("Training complete. Best validation F1: %.4f", best_f1)

    # Write metrics JSON for DVC
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(best_metrics, f, indent=2)
    logger.info("Wrote metrics to %s", METRICS_PATH)


@torch.no_grad()
def evaluate(
    model: GraphVulModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_type=batch.edge_type,
            batch=batch.batch,
            text_vec=batch.text_vec,
        )
        preds = (torch.sigmoid(logits) >= 0.5).long().cpu().tolist()
        labels = batch.y.squeeze(-1).long().cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels)

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GraphVul model")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(num_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed)
