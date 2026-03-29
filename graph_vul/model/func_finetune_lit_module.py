"""PyTorch Lightning module for UniXcoder fine-tuning."""

import pytorch_lightning as pl
import torch
import torchmetrics

from graph_vul.config import (
    FINETUNE_ENCODER_LR,
    FINETUNE_HEAD_LR,
    LR_SCHEDULER_FACTOR,
    MIN_LR,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
)
from graph_vul.model.func_finetune_model import FuncFinetuneModel


class FuncFinetuneLitModule(pl.LightningModule):
    def __init__(
        self,
        encoder_lr: float = FINETUNE_ENCODER_LR,
        head_lr: float = FINETUNE_HEAD_LR,
        weight_decay: float = WEIGHT_DECAY,
        lr_scheduler_factor: float = LR_SCHEDULER_FACTOR,
        warmup_epochs: int = WARMUP_EPOCHS,
        min_lr: float = MIN_LR,
        graph_dim: int = 0,
        num_node_types: int = 0,
        num_relations: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = FuncFinetuneModel(
            graph_dim=graph_dim,
            num_node_types=num_node_types,
            num_relations=num_relations,
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # Train metrics
        self.train_accuracy = torchmetrics.classification.BinaryAccuracy()
        self.train_precision = torchmetrics.classification.BinaryPrecision()
        self.train_recall = torchmetrics.classification.BinaryRecall()
        self.train_f1 = torchmetrics.classification.BinaryF1Score()

        # Val metrics
        self.val_accuracy = torchmetrics.classification.BinaryAccuracy()
        self.val_precision = torchmetrics.classification.BinaryPrecision()
        self.val_recall = torchmetrics.classification.BinaryRecall()
        self.val_f1 = torchmetrics.classification.BinaryF1Score()

        # Unseen val metrics (logged every epoch alongside val)
        self.unseen_val_accuracy = torchmetrics.classification.BinaryAccuracy()
        self.unseen_val_precision = torchmetrics.classification.BinaryPrecision()
        self.unseen_val_recall = torchmetrics.classification.BinaryRecall()
        self.unseen_val_f1 = torchmetrics.classification.BinaryF1Score()

        # Test metrics (one set per test dataloader)
        self._test_prefixes: list[str] = ["test"]
        self.test_accuracy = torchmetrics.classification.BinaryAccuracy()
        self.test_precision = torchmetrics.classification.BinaryPrecision()
        self.test_recall = torchmetrics.classification.BinaryRecall()
        self.test_f1 = torchmetrics.classification.BinaryF1Score()

    def forward(self, input_ids, attention_mask, cwe_embedding, graph_batch=None):
        return self.model(input_ids, attention_mask, cwe_embedding, graph_batch=graph_batch)

    def _shared_step(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        graph_batch = batch.get("graph_data")
        logits = self.model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["cwe_embedding"],
            graph_batch=graph_batch,
        ).squeeze(-1)
        loss = self.criterion(logits, batch["label"])
        preds = (logits >= 0).long()
        return loss, preds, batch["label"].long()

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, preds, labels = self._shared_step(batch)
        self.log("train_loss", loss.detach(), prog_bar=True, batch_size=len(labels))
        self.train_accuracy.update(preds, labels)
        self.train_precision.update(preds, labels)
        self.train_recall.update(preds, labels)
        self.train_f1.update(preds, labels)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train_accuracy", self.train_accuracy.compute())
        self.log("train_precision", self.train_precision.compute())
        self.log("train_recall", self.train_recall.compute())
        self.log("train_f1", self.train_f1.compute())
        self.train_accuracy.reset()
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_f1.reset()

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        _, preds, labels = self._shared_step(batch)
        if dataloader_idx == 0:
            self.val_accuracy.update(preds, labels)
            self.val_precision.update(preds, labels)
            self.val_recall.update(preds, labels)
            self.val_f1.update(preds, labels)
            self.log("val_accuracy", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
            self.log("val_precision", self.val_precision, on_step=False, on_epoch=True, add_dataloader_idx=False)
            self.log("val_recall", self.val_recall, on_step=False, on_epoch=True, add_dataloader_idx=False)
            self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        elif dataloader_idx == 1:
            self.unseen_val_accuracy.update(preds, labels)
            self.unseen_val_precision.update(preds, labels)
            self.unseen_val_recall.update(preds, labels)
            self.unseen_val_f1.update(preds, labels)
            self.log("unseen_val_accuracy", self.unseen_val_accuracy, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
            self.log("unseen_val_precision", self.unseen_val_precision, on_step=False, on_epoch=True, add_dataloader_idx=False)
            self.log("unseen_val_recall", self.unseen_val_recall, on_step=False, on_epoch=True, add_dataloader_idx=False)
            self.log("unseen_val_f1", self.unseen_val_f1, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)

    def set_test_prefix(self, prefix: str) -> None:
        """Set the metric prefix for the current test run (e.g. 'test', 'unseen_val')."""
        self._current_test_prefix = prefix

    def test_step(self, batch: dict, batch_idx: int) -> None:
        _, preds, labels = self._shared_step(batch)
        prefix = getattr(self, "_current_test_prefix", "test")
        self.test_accuracy.update(preds, labels)
        self.test_precision.update(preds, labels)
        self.test_recall.update(preds, labels)
        self.test_f1.update(preds, labels)
        self.log(f"{prefix}_accuracy", self.test_accuracy, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{prefix}_precision", self.test_precision, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{prefix}_recall", self.test_recall, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{prefix}_f1", self.test_f1, on_step=False, on_epoch=True, add_dataloader_idx=False)

    def configure_optimizers(self) -> dict:
        encoder_params = list(self.model.encoder.parameters())
        head_params = list(self.model.head.parameters())

        param_groups = [
            {"params": encoder_params, "lr": self.hparams.encoder_lr},
            {"params": head_params, "lr": self.hparams.head_lr},
        ]

        # Add graph encoder params to head LR group if present
        if self.model.graph_dim > 0:
            graph_params = list(self.model.graph_encoder.parameters())
            param_groups.append({"params": graph_params, "lr": self.hparams.head_lr})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.hparams.weight_decay)

        def get_lr_lambda(epoch: int) -> float:
            if epoch < self.hparams.warmup_epochs:
                return 1.0
            return self.hparams.lr_scheduler_factor ** (epoch - self.hparams.warmup_epochs)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda)
        return [optimizer], [scheduler]
