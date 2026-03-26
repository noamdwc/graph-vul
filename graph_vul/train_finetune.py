"""Training script for fine-tuning UniXcoder on function-level vulnerability detection."""

import argparse
import json
import logging

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer

from graph_vul.config import (
    DATA_DIR,
    FINETUNE_BATCH_SIZE,
    FINETUNE_ENCODER_LR,
    FINETUNE_GRAD_ACCUM_STEPS,
    FINETUNE_HEAD_LR,
    FINETUNE_NUM_EPOCHS,
    FUNC_SPLITS_DIR,
    FUNC_SUBSET_CSV,
    SEED,
    UNIXCODER_MODEL_NAME,
)
from graph_vul.dataset.func_finetune_dataset import FuncFinetuneDataset
from graph_vul.model.func_finetune_lit_module import FuncFinetuneLitModule

logger = logging.getLogger(__name__)

CHECKPOINT_PATH = DATA_DIR / "func_finetune_model.ckpt"
METRICS_PATH = DATA_DIR / "func_finetune_metrics.json"


def train(
    num_epochs: int = FINETUNE_NUM_EPOCHS,
    batch_size: int = FINETUNE_BATCH_SIZE,
    encoder_lr: float = FINETUNE_ENCODER_LR,
    head_lr: float = FINETUNE_HEAD_LR,
    grad_accum: int = FINETUNE_GRAD_ACCUM_STEPS,
    seed: int = SEED,
) -> None:
    pl.seed_everything(seed)

    # Shared tokenizer
    tokenizer = AutoTokenizer.from_pretrained(UNIXCODER_MODEL_NAME)

    # Datasets
    train_ds = FuncFinetuneDataset(split="train", tokenizer=tokenizer)
    val_ds = FuncFinetuneDataset(split="val", tokenizer=tokenizer)
    test_ds = FuncFinetuneDataset(split="test", tokenizer=tokenizer)

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True,
    )

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
    )

    logger.info(
        "Train: %d functions (%d pos, %d neg), Val: %d, Test: %d",
        len(train_ds), int(num_pos), int(num_neg), len(val_ds), len(test_ds),
    )
    logger.info(
        "Effective batch size: %d (batch=%d x accum=%d)",
        batch_size * grad_accum, batch_size, grad_accum,
    )

    # Model
    model = FuncFinetuneLitModule(encoder_lr=encoder_lr, head_lr=head_lr)

    # Logger
    wandb_logger = WandbLogger(
        project="graph-vul",
        name="func-finetune",
        config={
            "model": "func-finetune-unixcoder",
            "encoder_lr": encoder_lr,
            "head_lr": head_lr,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "seed": seed,
        },
    )

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(CHECKPOINT_PATH.parent),
        filename="func_finetune_model",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        precision="16-mixed",
        accumulate_grad_batches=grad_accum,
        callbacks=[checkpoint_cb, LearningRateMonitor(logging_interval="epoch")],
        logger=wandb_logger,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Test on seen CWEs
    model.set_test_prefix("test")
    trainer.test(model, dataloaders=test_loader, ckpt_path="best")

    # Test on unseen CWE splits
    unseen_splits = ["unseen_val", "unseen_test"]
    for split_name in unseen_splits:
        ds = FuncFinetuneDataset(split=split_name, tokenizer=tokenizer)
        if len(ds) == 0:
            logger.warning("Skipping %s: no samples found", split_name)
            continue
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=4,
        )
        model.set_test_prefix(split_name)
        trainer.test(model, dataloaders=loader, ckpt_path="best")
        logger.info("Evaluated %s: %d samples", split_name, len(ds))

    # Write DVC metrics
    metrics = {}
    all_prefixes = ["val", "test", "train"] + unseen_splits
    for prefix in all_prefixes:
        for metric in ("accuracy", "precision", "recall", "f1"):
            key = f"{prefix}_{metric}"
            val = trainer.callback_metrics.get(key, None)
            if val is not None:
                metrics[key] = float(val)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Wrote metrics to %s", METRICS_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune UniXcoder for vulnerability detection")
    parser.add_argument("--epochs", type=int, default=FINETUNE_NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=FINETUNE_BATCH_SIZE)
    parser.add_argument("--encoder-lr", type=float, default=FINETUNE_ENCODER_LR)
    parser.add_argument("--head-lr", type=float, default=FINETUNE_HEAD_LR)
    parser.add_argument("--grad-accum", type=int, default=FINETUNE_GRAD_ACCUM_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        grad_accum=args.grad_accum,
        seed=args.seed,
    )
