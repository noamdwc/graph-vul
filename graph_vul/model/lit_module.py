"""PyTorch Lightning wrapper for GraphVulModel."""

import pytorch_lightning as pl
import torch
import torchmetrics

from graph_vul.config import LEARNING_RATE
from graph_vul.model.classifier import GraphVulModel


class GraphVulLitModule(pl.LightningModule):
    def __init__(self, lr: float = LEARNING_RATE):
        super().__init__()
        self.save_hyperparameters()
        self.model = GraphVulModel()
        self.criterion = torch.nn.BCEWithLogitsLoss()

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
        loss = self.criterion(logits, batch.y.squeeze(-1))
        self.log("train_loss", loss, prog_bar=True, batch_size=batch.y.size(0))
        return loss

    def validation_step(self, batch: object, batch_idx: int) -> None:
        logits = self.model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_type=batch.edge_type,
            batch=batch.batch,
            text_vec=batch.text_vec,
        )
        labels = batch.y.squeeze(-1).long()
        preds = (torch.sigmoid(logits) >= 0.5).long()

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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
