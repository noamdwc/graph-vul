"""PyTorch Lightning wrapper for GraphVulModel with graph-level prediction."""

import pytorch_lightning as pl
import torch
import torchmetrics

from graph_vul.config import LEARNING_RATE, WEIGHT_DECAY, LR_SCHEDULER_FACTOR, WARMUP_EPOCHS, MIN_LR
from graph_vul.model.classifier import GraphVulModel


class GraphVulLitModule(pl.LightningModule):
    def __init__(self,
                lr: float = LEARNING_RATE,
                weight_decay: float = WEIGHT_DECAY,
                lr_scheduler_factor: float = LR_SCHEDULER_FACTOR,
                warmup_epochs: int = WARMUP_EPOCHS,
                min_lr: float = MIN_LR):
        super().__init__()
        self.save_hyperparameters()
        self.model = GraphVulModel()
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


    def forward(self, **kwargs: torch.Tensor) -> torch.Tensor:
        return self.model(**kwargs)

    def training_step(self, batch: object, batch_idx: int) -> torch.Tensor:
        logits = self.model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_type=batch.edge_type,
            batch=batch.batch,
            text_vec=batch.text_vec,
        )
        loss = self.criterion(logits, batch.y)
        self.log("train_loss", loss.detach(), prog_bar=True, batch_size=batch.num_graphs)

        with torch.no_grad():
            preds = (logits >= 0).long()
            labels = batch.y.long()
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

    def validation_step(self, batch: object, batch_idx: int) -> None:
        logits = self.model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_type=batch.edge_type,
            batch=batch.batch,
            text_vec=batch.text_vec,
        )
        preds = (torch.sigmoid(logits) >= 0.5).long()
        labels = batch.y.long()

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


    def on_before_optimizer_step(self, optimizer):
        total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=float("inf"))
        self.log("grad_norm", total_norm.detach(), prog_bar=True)
  

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        def get_lr_lambda(epoch):
            if epoch < self.hparams.warmup_epochs:
                return 1.0
            if optimizer.param_groups[0]['lr'] < self.hparams.min_lr:
                return 1.0
            return self.hparams.lr_scheduler_factor ** (epoch - self.hparams.warmup_epochs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda)
        return [optimizer], [scheduler]
