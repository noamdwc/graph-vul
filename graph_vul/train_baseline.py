"""Training script for function-level baseline MLP with PyTorch Lightning."""

import argparse
import json
import logging

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, WeightedRandomSampler

from graph_vul.config import (
    DATA_DIR,
    FUNC_BATCH_SIZE,
    FUNC_CHECKPOINT_PATH,
    FUNC_LEARNING_RATE,
    FUNC_MLP_HIDDEN_DIM,
    FUNC_NUM_EPOCHS,
    FUNC_SPLITS_DIR,
    FUNC_SUBSET_CSV,
    GRAPH_EDGE_VOCAB_PATH,
    GRAPH_HIDDEN_DIM,
    GRAPH_NODE_VOCAB_PATH,
    SEED,
)
from graph_vul.dataset.func_dataset import FuncDataset, func_graph_collate
from graph_vul.model.func_lit_module import FuncLitModule

logger = logging.getLogger(__name__)

METRICS_PATH = DATA_DIR / "func_metrics.json"


def _load_graph_vocab() -> tuple[int, int]:
    """Load graph vocabs and return (num_node_types, num_relations)."""
    with open(GRAPH_NODE_VOCAB_PATH) as f:
        node_vocab = json.load(f)
    with open(GRAPH_EDGE_VOCAB_PATH) as f:
        edge_vocab = json.load(f)
    return len(node_vocab), len(edge_vocab)


def train(
    num_epochs: int = FUNC_NUM_EPOCHS,
    batch_size: int = FUNC_BATCH_SIZE,
    lr: float = FUNC_LEARNING_RATE,
    seed: int = SEED,
    use_graph: bool = False,
) -> None:
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

    # Datasets
    train_ds = FuncDataset(split="train", use_graph=use_graph)
    val_ds = FuncDataset(split="val", use_graph=use_graph)
    test_ds = FuncDataset(split="test", use_graph=use_graph)

    collate_fn = func_graph_collate if use_graph else None

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)

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
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2, collate_fn=collate_fn)

    logger.info(
        "Train: %d functions (%d pos, %d neg), Val: %d, Test: %d",
        len(train_ds), int(num_pos), int(num_neg), len(val_ds), len(test_ds),
    )

    # Model — sampler already balances classes, no pos_weight needed
    model = FuncLitModule(
        lr=lr,
        graph_dim=graph_dim,
        num_node_types=num_node_types,
        num_relations=num_relations,
    )

    # Logger
    run_name = "func-graph-baseline" if use_graph else "func-baseline"
    model_name = "func-graph-baseline-rgcn" if use_graph else "func-baseline-mlp"
    wandb_logger = WandbLogger(
        project="graph-vul",
        name=run_name,
        config={
            "model": model_name,
            "learning_rate": lr,
            "hidden_dim": FUNC_MLP_HIDDEN_DIM,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "seed": seed,
            "use_graph": use_graph,
            **graph_stats,
        },
    )

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(FUNC_CHECKPOINT_PATH.parent),
        filename="func_baseline_model",
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
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Test
    trainer.test(model, dataloaders=test_loader, ckpt_path="best")

    # Write DVC metrics
    metrics = {}
    for prefix in ("val", "test", "train"):
        for metric in ("accuracy", "precision", "recall", "f1"):
            key = f"{prefix}_{metric}"
            metrics[key] = float(trainer.callback_metrics.get(key, 0.0))

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Wrote metrics to %s", METRICS_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train function-level baseline MLP")
    parser.add_argument("--epochs", type=int, default=FUNC_NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=FUNC_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=FUNC_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--use-graph", action="store_true", help="Enable graph (AST+CFG RGCN) features")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(num_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed, use_graph=args.use_graph)
