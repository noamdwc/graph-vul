"""PyTorch Lightning wrapper for GraphVulModel with node-level prediction."""

import pytorch_lightning as pl
import torch
import torchmetrics
from torch_geometric.utils import scatter

from graph_vul.config import LEARNING_RATE
from graph_vul.model.classifier import GraphVulModel


class GraphVulLitModule(pl.LightningModule):
    def __init__(self, lr: float = LEARNING_RATE):
        super().__init__()
        self.save_hyperparameters()
        self.model = GraphVulModel()
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # Graph-level metrics (aggregated from node predictions)
        self.val_accuracy = torchmetrics.classification.BinaryAccuracy()
        self.val_precision = torchmetrics.classification.BinaryPrecision()
        self.val_recall = torchmetrics.classification.BinaryRecall()
        self.val_f1 = torchmetrics.classification.BinaryF1Score()

    def forward(self, **kwargs: torch.Tensor) -> torch.Tensor:
        return self.model(**kwargs)

    def training_step(self, batch: object, batch_idx: int) -> torch.Tensor:
        # Per-node logits (N,)
        logits = self.model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_type=batch.edge_type,
            batch=batch.batch,
            text_vec=batch.text_vec,
        )
        # Per-node loss: batch.y is (N,) with per-node labels
        loss = self.criterion(logits, batch.y)
        self.log("train_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch: object, batch_idx: int) -> None:
        logits = self.model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_type=batch.edge_type,
            batch=batch.batch,
            text_vec=batch.text_vec,
        )

        # Aggregate to graph-level for metrics
        graph_logits = scatter(logits, batch.batch, dim=0, reduce="max")
        graph_preds = (torch.sigmoid(graph_logits) >= 0.5).long()
        # Graph labels: take first node's label per graph (all nodes share same label)
        graph_labels = scatter(batch.y, batch.batch, dim=0, reduce="max").long()

        self.val_accuracy.update(graph_preds, graph_labels)
        self.val_precision.update(graph_preds, graph_labels)
        self.val_recall.update(graph_preds, graph_labels)
        self.val_f1.update(graph_preds, graph_labels)

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
