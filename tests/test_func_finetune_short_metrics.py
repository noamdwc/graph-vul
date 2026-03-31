"""Tests for short-function metric logging in fine-tune Lightning module."""

import pytest
import pytorch_lightning as pl
import torch
import torchmetrics

from graph_vul.model.func_finetune_lit_module import FuncFinetuneLitModule


def _make_lit_for_metric_logging() -> tuple[FuncFinetuneLitModule, dict[str, float]]:
    lit = FuncFinetuneLitModule.__new__(FuncFinetuneLitModule)
    pl.LightningModule.__init__(lit)

    lit.val_accuracy = torchmetrics.classification.BinaryAccuracy()
    lit.val_precision = torchmetrics.classification.BinaryPrecision()
    lit.val_recall = torchmetrics.classification.BinaryRecall()
    lit.val_f1 = torchmetrics.classification.BinaryF1Score()
    lit.val_0_64_f1 = torchmetrics.classification.BinaryF1Score()

    lit.unseen_val_accuracy = torchmetrics.classification.BinaryAccuracy()
    lit.unseen_val_precision = torchmetrics.classification.BinaryPrecision()
    lit.unseen_val_recall = torchmetrics.classification.BinaryRecall()
    lit.unseen_val_f1 = torchmetrics.classification.BinaryF1Score()
    lit.unseen_val_0_64_f1 = torchmetrics.classification.BinaryF1Score()

    lit.test_accuracy = torchmetrics.classification.BinaryAccuracy()
    lit.test_precision = torchmetrics.classification.BinaryPrecision()
    lit.test_recall = torchmetrics.classification.BinaryRecall()
    lit.test_f1 = torchmetrics.classification.BinaryF1Score()
    lit.test_0_64_f1 = torchmetrics.classification.BinaryF1Score()

    logged: dict[str, float] = {}

    def capture_log(name, value, **kwargs):
        if hasattr(value, "compute"):
            logged[name] = float(value.compute())
        else:
            logged[name] = float(value)

    lit.log = capture_log
    return lit, logged


def test_validation_logs_short_function_f1_for_seen_and_unseen_val():
    lit, logged = _make_lit_for_metric_logging()

    preds = torch.tensor([1, 0, 1, 0], dtype=torch.long)
    labels = torch.tensor([1, 0, 0, 1], dtype=torch.long)
    token_counts = torch.tensor([64, 20, 65, 10], dtype=torch.long)

    lit._shared_step = lambda batch: (torch.tensor(0.0), preds, labels)
    batch = {"token_count": token_counts}

    lit.validation_step(batch, batch_idx=0, dataloader_idx=0)
    assert logged["val_0_64_f1"] == pytest.approx(2 / 3)

    logged.clear()
    lit.validation_step(batch, batch_idx=0, dataloader_idx=1)
    assert logged["unseen_val_0_64_f1"] == pytest.approx(2 / 3)


def test_test_step_logs_short_function_f1_with_prefixes():
    lit, logged = _make_lit_for_metric_logging()

    preds = torch.tensor([1, 0, 1, 1], dtype=torch.long)
    labels = torch.tensor([1, 0, 0, 1], dtype=torch.long)
    token_counts = torch.tensor([1, 64, 65, 200], dtype=torch.long)

    lit._shared_step = lambda batch: (torch.tensor(0.0), preds, labels)
    batch = {"token_count": token_counts}

    lit.set_test_prefix("test")
    lit.test_step(batch, batch_idx=0)
    assert logged["test_0_64_f1"] == 1.0

    logged.clear()
    lit.set_test_prefix("unseen_test")
    lit.test_step(batch, batch_idx=0)
    assert logged["unseen_test_0_64_f1"] == 1.0
