"""PyTorch Lightning module for the function-level baseline MLP."""

import pytorch_lightning as pl
import torch
import torchmetrics

from graph_vul.config import (
    FUNC_LEARNING_RATE,
    LR_SCHEDULER_FACTOR,
    MIN_LR,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
)
from graph_vul.model.func_classifier import FuncBaselineModel


class FuncLitModule(pl.LightningModule):
    def __init__(
        self,
        lr: float = FUNC_LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        lr_scheduler_factor: float = LR_SCHEDULER_FACTOR,
        warmup_epochs: int = WARMUP_EPOCHS,
        min_lr: float = MIN_LR,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = FuncBaselineModel()
        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

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

        # Test metrics
        self.test_accuracy = torchmetrics.classification.BinaryAccuracy()
        self.test_precision = torchmetrics.classification.BinaryPrecision()
        self.test_recall = torchmetrics.classification.BinaryRecall()
        self.test_f1 = torchmetrics.classification.BinaryF1Score()

    def forward(self, code_emb: torch.Tensor, cwe_emb: torch.Tensor) -> torch.Tensor:
        return self.model(code_emb, cwe_emb)

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        code_emb, cwe_emb, labels = batch
        logits = self.model(code_emb, cwe_emb).squeeze(-1)  # [N]
        loss = self.criterion(logits, labels)
        preds = (logits >= 0).long()
        return loss, preds, labels.long()

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
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

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        _, preds, labels = self._shared_step(batch)
        self.val_accuracy.update(preds, labels)
        self.val_precision.update(preds, labels)
        self.val_recall.update(preds, labels)
        self.val_f1.update(preds, labels)

    def on_validation_epoch_end(self) -> None:
        self.log("val_accuracy", self.val_accuracy.compute(), prog_bar=True)
        self.log("val_precision", self.val_precision.compute())
        self.log("val_recall", self.val_recall.compute())
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        _, preds, labels = self._shared_step(batch)
        self.test_accuracy.update(preds, labels)
        self.test_precision.update(preds, labels)
        self.test_recall.update(preds, labels)
        self.test_f1.update(preds, labels)

    def on_test_epoch_end(self) -> None:
        self.log("test_accuracy", self.test_accuracy.compute())
        self.log("test_precision", self.test_precision.compute())
        self.log("test_recall", self.test_recall.compute())
        self.log("test_f1", self.test_f1.compute())
        self.test_accuracy.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        def get_lr_lambda(epoch: int) -> float:
            if epoch < self.hparams.warmup_epochs:
                return 1.0
            if optimizer.param_groups[0]["lr"] < self.hparams.min_lr:
                return 1.0
            return self.hparams.lr_scheduler_factor ** (epoch - self.hparams.warmup_epochs)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda)
        return [optimizer], [scheduler]
