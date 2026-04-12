"""PyTorch Lightning module for UniXcoder fine-tuning."""

import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics

from graph_vul.config import (
    FINETUNE_ENCODER_LR,
    FINETUNE_HEAD_LR,
    LR_SCHEDULER_FACTOR,
    MATCHING_ABLATION_MODE,
    MATCHING_LOSS_LAMBDA,
    MATCHING_NEGATIVES_PER_POSITIVE,
    MATCHING_PROJECTION_DIM,
    MATCHING_TRIPLET_MARGIN,
    MIN_LR,
    PROCESSED_DIR,
    SEED,
    FUNC_SUBSET_CSV,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
    DEBERTA_EMBEDDING_DIM,
)
from graph_vul.model.cwe_matching import (
    CWEEmbeddingStore,
    CWENegativeSampler,
    build_matching_config,
    candidate_cwe_ids_from_subset,
    cosine_similarity_logits,
)
from graph_vul.model.func_finetune_model import FuncFinetuneModel

logger = logging.getLogger(__name__)

SHORT_FUNCTION_TOKEN_THRESHOLD = 64
FREEZE_ENCODER_EPOCHS = 3
UNFREEZE_ENCODER_LR = 5e-6


class BinaryFocalLoss(torch.nn.Module):
    """Focal loss for binary classification with logits.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    When gamma=0, this reduces to standard BCE.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)  # p_t = p if y=1, (1-p) if y=0
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting: alpha for positives, (1-alpha) for negatives
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * focal_weight * bce
        return loss.mean()


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
        fusion_layers: list[int] | None = None,
        fusion_method: str = "scalar_gated_residual",
        final_graph_concat: bool = True,
        freeze_encoder_epochs: int = FREEZE_ENCODER_EPOCHS,
        unfreeze_encoder_lr: float = UNFREEZE_ENCODER_LR,
        focal_gamma: float = 0.0,
        focal_alpha: float = 0.25,
        head_dropout: float | None = None,
        freeze_encoder_fully: bool = False,
        ablation_mode: str = MATCHING_ABLATION_MODE,
        matching_projection_dim: int = MATCHING_PROJECTION_DIM,
        matching_negatives_per_positive: int = MATCHING_NEGATIVES_PER_POSITIVE,
        matching_loss_lambda: float = MATCHING_LOSS_LAMBDA,
        matching_triplet_margin: float = MATCHING_TRIPLET_MARGIN,
        cwe_embeddings_dir: str | Path = PROCESSED_DIR / "cwe_embeddings",
        subset_csv: str | Path = FUNC_SUBSET_CSV,
        matching_seed: int = SEED,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.hparams["cwe_embeddings_dir"] = str(cwe_embeddings_dir)
        self.hparams["subset_csv"] = str(subset_csv)
        self.matching_config = build_matching_config(
            ablation_mode=ablation_mode,
            projection_dim=matching_projection_dim,
            negatives_per_positive=matching_negatives_per_positive,
            loss_lambda=matching_loss_lambda,
            triplet_margin=matching_triplet_margin,
        )
        self.model = FuncFinetuneModel(
            graph_dim=graph_dim,
            num_node_types=num_node_types,
            num_relations=num_relations,
            fusion_layers=fusion_layers,
            fusion_method=fusion_method,
            final_graph_concat=final_graph_concat,
            dropout=head_dropout if head_dropout is not None else None,
            matching_projection_dim=matching_projection_dim,
            matching_config=self.matching_config,
        )
        self.cwe_store = CWEEmbeddingStore(cwe_embeddings_dir=cwe_embeddings_dir, embedding_dim=DEBERTA_EMBEDDING_DIM)
        candidate_cwe_ids = candidate_cwe_ids_from_subset(subset_csv)
        if not candidate_cwe_ids:
            candidate_cwe_ids = self.cwe_store.available_cwe_ids()
        self.negative_sampler = CWENegativeSampler(candidate_cwe_ids=candidate_cwe_ids, seed=matching_seed)
        if focal_gamma > 0:
            self.criterion = BinaryFocalLoss(gamma=focal_gamma, alpha=focal_alpha)
            logger.info("Using focal loss: gamma=%.1f, alpha=%.2f", focal_gamma, focal_alpha)
        else:
            self.criterion = torch.nn.BCEWithLogitsLoss()
        self.threshold: float = 0.0  # logit-space threshold (0.0 = probability 0.5)

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
        self.val_0_64_f1 = torchmetrics.classification.BinaryF1Score()

        # Unseen val metrics (logged every epoch alongside val)
        self.unseen_val_accuracy = torchmetrics.classification.BinaryAccuracy()
        self.unseen_val_precision = torchmetrics.classification.BinaryPrecision()
        self.unseen_val_recall = torchmetrics.classification.BinaryRecall()
        self.unseen_val_f1 = torchmetrics.classification.BinaryF1Score()
        self.unseen_val_0_64_f1 = torchmetrics.classification.BinaryF1Score()

        # Test metrics (one set per test dataloader)
        self._test_prefixes: list[str] = ["test"]
        self.test_accuracy = torchmetrics.classification.BinaryAccuracy()
        self.test_precision = torchmetrics.classification.BinaryPrecision()
        self.test_recall = torchmetrics.classification.BinaryRecall()
        self.test_f1 = torchmetrics.classification.BinaryF1Score()
        self.test_0_64_f1 = torchmetrics.classification.BinaryF1Score()

        self._reset_low_token_tracking()

    def forward(self, input_ids, attention_mask, cwe_embedding, graph_batch=None):
        return self.model(input_ids, attention_mask, cwe_embedding, graph_batch=graph_batch)

    def _freeze_encoder(self) -> None:
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen")

    def _unfreeze_encoder(self) -> None:
        for param in self.model.encoder.parameters():
            param.requires_grad = True
        # Update the encoder LR in the optimizer to the lower unfreeze rate
        optimizer = self.optimizers()
        optimizer.param_groups[0]["lr"] = self.hparams.unfreeze_encoder_lr
        logger.info("Encoder unfrozen with lr=%.2e", self.hparams.unfreeze_encoder_lr)

    def on_train_start(self) -> None:
        if self.hparams.freeze_encoder_fully or self.hparams.freeze_encoder_epochs > 0:
            self._freeze_encoder()

    def on_train_epoch_start(self) -> None:
        if self.hparams.freeze_encoder_fully:
            return  # Never unfreeze
        if self.current_epoch == self.hparams.freeze_encoder_epochs:
            self._unfreeze_encoder()

    def _compute_matching_losses(
        self,
        batch: dict,
        outputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
        device = outputs["logits"].device
        labels = batch["label"]
        vul_mask = labels > 0.5
        zero = outputs["logits"].new_zeros(())
        if not bool(vul_mask.any()):
            return zero, zero, [], []

        positive_cwe_ids = [str(cwe_id) for cwe_id, is_vul in zip(batch["cwe_id"], vul_mask.tolist()) if is_vul]
        negative_cwe_ids: list[str] = []
        negative_sources: list[str] = []
        for positive_cwe_id in positive_cwe_ids:
            for _ in range(self.matching_config.negatives_per_positive):
                negative_cwe_id, source = self.negative_sampler.sample(
                    positive_cwe_id=positive_cwe_id,
                    use_hard_negatives=self.matching_config.use_hard_negatives,
                )
                negative_cwe_ids.append(negative_cwe_id)
                negative_sources.append(source)

        if not negative_cwe_ids:
            return zero, zero, [], []

        repeat_count = self.matching_config.negatives_per_positive
        function_match = outputs["function_match"][vul_mask].repeat_interleave(repeat_count, dim=0)
        positive_match = outputs["cwe_match"][vul_mask].repeat_interleave(repeat_count, dim=0)
        negative_embeddings = self.cwe_store.stack(negative_cwe_ids, device=device)
        negative_match = self.model.project_cwe_embeddings(negative_embeddings)

        negative_loss = zero
        if self.matching_config.use_negative_classification:
            pos_logits = cosine_similarity_logits(function_match, positive_match)
            neg_logits = cosine_similarity_logits(function_match, negative_match)
            pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
            neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
            negative_loss = 0.5 * (pos_loss + neg_loss)

        triplet_loss = zero
        if self.matching_config.use_triplet_loss:
            triplet_loss = F.triplet_margin_loss(
                function_match,
                positive_match,
                negative_match,
                margin=self.matching_config.triplet_margin,
                reduction="mean",
            )
        return negative_loss, triplet_loss, negative_cwe_ids, negative_sources

    def _shared_step(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        graph_batch = batch.get("graph_data")
        outputs = self.model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["cwe_embedding"],
            graph_batch=graph_batch,
            return_matching_outputs=True,
        )
        logits = outputs["logits"].squeeze(-1)
        bce_loss = self.criterion(logits, batch["label"])
        negative_loss, triplet_loss, negative_cwe_ids, negative_sources = self._compute_matching_losses(batch, outputs)
        loss = bce_loss + self.matching_config.loss_lambda * (negative_loss + triplet_loss)
        preds = (logits >= self.threshold).long()
        aux_metrics = {
            "loss_bce": float(bce_loss.detach()),
            "loss_wrong_cwe": float(negative_loss.detach()),
            "loss_triplet": float(triplet_loss.detach()),
            "negatives_sampled": float(len(negative_cwe_ids)),
            "hard_negative_rate": float(sum(source != "random" for source in negative_sources) / max(len(negative_sources), 1))
            if negative_sources else 0.0,
        }
        return loss, preds, batch["label"].long(), aux_metrics

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, preds, labels, aux_metrics = self._shared_step(batch)
        self.log("train_loss", loss.detach(), prog_bar=True, batch_size=len(labels))
        self.log("train_loss_bce", aux_metrics["loss_bce"], on_step=False, on_epoch=True, batch_size=len(labels))
        self.log("train_loss_wrong_cwe", aux_metrics["loss_wrong_cwe"], on_step=False, on_epoch=True, batch_size=len(labels))
        self.log("train_loss_triplet", aux_metrics["loss_triplet"], on_step=False, on_epoch=True, batch_size=len(labels))
        self.log("train_matching_negatives", aux_metrics["negatives_sampled"], on_step=False, on_epoch=True, batch_size=len(labels))
        self.log("train_hard_negative_rate", aux_metrics["hard_negative_rate"], on_step=False, on_epoch=True, batch_size=len(labels))
        self.train_accuracy.update(preds, labels)
        self.train_precision.update(preds, labels)
        self.train_recall.update(preds, labels)
        self.train_f1.update(preds, labels)

        token_counts = batch["token_count"]
        low_token_mask = token_counts <= SHORT_FUNCTION_TOKEN_THRESHOLD
        low_token_count = int(low_token_mask.sum().item())
        batch_size = len(labels)
        self._train_samples_seen += batch_size
        self._train_low_token_seen += low_token_count
        if low_token_count:
            low_token_ids = batch["sample_id"][low_token_mask].detach().cpu().tolist()
            self._train_low_token_sample_ids.update(int(sample_id) for sample_id in low_token_ids)
        self.log("train_low_token_count", float(low_token_count), on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(
            "train_low_token_frac",
            low_token_count / max(batch_size, 1),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train_accuracy", self.train_accuracy.compute())
        self.log("train_precision", self.train_precision.compute())
        self.log("train_recall", self.train_recall.compute())
        self.log("train_f1", self.train_f1.compute())
        self.log("train_low_token_seen", float(self._train_low_token_seen))
        self.log(
            "train_low_token_seen_frac",
            self._train_low_token_seen / max(self._train_samples_seen, 1),
        )
        if self._train_low_token_sample_ids:
            preview = ", ".join(str(sample_id) for sample_id in sorted(self._train_low_token_sample_ids)[:20])
            logger.info(
                "Epoch %d low-token train samples (<=%d tokens): %d/%d seen, %d unique sample_ids. First ids: %s",
                self.current_epoch,
                SHORT_FUNCTION_TOKEN_THRESHOLD,
                self._train_low_token_seen,
                self._train_samples_seen,
                len(self._train_low_token_sample_ids),
                preview,
            )
        else:
            logger.info(
                "Epoch %d low-token train samples (<=%d tokens): 0/%d seen",
                self.current_epoch,
                SHORT_FUNCTION_TOKEN_THRESHOLD,
                self._train_samples_seen,
            )
        self.train_accuracy.reset()
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_f1.reset()
        self._reset_low_token_tracking()

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        _, preds, labels, _ = self._shared_step(batch)
        short_f1_metric = None
        short_f1_name = None
        if dataloader_idx == 0:
            self.val_accuracy.update(preds, labels)
            self.val_precision.update(preds, labels)
            self.val_recall.update(preds, labels)
            self.val_f1.update(preds, labels)
            short_f1_metric = self.val_0_64_f1
            short_f1_name = "val_0_64_f1"
            self.log("val_accuracy", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
            self.log("val_precision", self.val_precision, on_step=False, on_epoch=True, add_dataloader_idx=False)
            self.log("val_recall", self.val_recall, on_step=False, on_epoch=True, add_dataloader_idx=False)
            self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        elif dataloader_idx == 1:
            self.unseen_val_accuracy.update(preds, labels)
            self.unseen_val_precision.update(preds, labels)
            self.unseen_val_recall.update(preds, labels)
            self.unseen_val_f1.update(preds, labels)
            short_f1_metric = self.unseen_val_0_64_f1
            short_f1_name = "unseen_val_0_64_f1"
            self.log("unseen_val_accuracy", self.unseen_val_accuracy, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
            self.log("unseen_val_precision", self.unseen_val_precision, on_step=False, on_epoch=True, add_dataloader_idx=False)
            self.log("unseen_val_recall", self.unseen_val_recall, on_step=False, on_epoch=True, add_dataloader_idx=False)
            self.log("unseen_val_f1", self.unseen_val_f1, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
        self._log_short_function_f1(batch["token_count"], preds, labels, short_f1_metric, short_f1_name)

    def set_test_prefix(self, prefix: str) -> None:
        """Set the metric prefix for the current test run (e.g. 'test', 'unseen_val')."""
        self._current_test_prefix = prefix

    def test_step(self, batch: dict, batch_idx: int) -> None:
        _, preds, labels, _ = self._shared_step(batch)
        prefix = getattr(self, "_current_test_prefix", "test")
        self.test_accuracy.update(preds, labels)
        self.test_precision.update(preds, labels)
        self.test_recall.update(preds, labels)
        self.test_f1.update(preds, labels)
        self._log_short_function_f1(batch["token_count"], preds, labels, self.test_0_64_f1, f"{prefix}_0_64_f1")
        self.log(f"{prefix}_accuracy", self.test_accuracy, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{prefix}_precision", self.test_precision, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{prefix}_recall", self.test_recall, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log(f"{prefix}_f1", self.test_f1, on_step=False, on_epoch=True, add_dataloader_idx=False)

    def predict_step(self, batch: dict, batch_idx: int) -> dict:
        graph_batch = batch.get("graph_data")
        outputs = self.model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["cwe_embedding"],
            graph_batch=graph_batch,
            return_matching_outputs=True,
        )
        logits = outputs["logits"].squeeze(-1)
        return {
            "logits": logits,
            "labels": batch["label"].long(),
            "function_match": outputs["function_match"],
        }

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

        # Add fusion module params if present
        if hasattr(self.model, "fusion_modules"):
            fusion_params = list(self.model.fusion_modules.parameters())
            param_groups.append({"params": fusion_params, "lr": self.hparams.head_lr})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.hparams.weight_decay)

        freeze_epochs = self.hparams.freeze_encoder_epochs

        def get_lr_lambda(epoch: int) -> float:
            # During frozen epochs, keep head LR constant (no decay)
            # After unfreeze, apply warmup then decay
            effective_epoch = epoch - freeze_epochs
            if effective_epoch < self.hparams.warmup_epochs:
                return 1.0
            return self.hparams.lr_scheduler_factor ** (effective_epoch - self.hparams.warmup_epochs)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda)
        return [optimizer], [scheduler]

    def _reset_low_token_tracking(self) -> None:
        self._train_samples_seen = 0
        self._train_low_token_seen = 0
        self._train_low_token_sample_ids: set[int] = set()

    def _log_short_function_f1(
        self,
        token_counts: torch.Tensor,
        preds: torch.Tensor,
        labels: torch.Tensor,
        metric: torchmetrics.classification.BinaryF1Score | None,
        metric_name: str | None,
    ) -> None:
        if metric is None or metric_name is None:
            return
        short_mask = token_counts <= SHORT_FUNCTION_TOKEN_THRESHOLD
        if not bool(short_mask.any()):
            return
        metric.update(preds[short_mask], labels[short_mask])
        self.log(metric_name, metric, on_step=False, on_epoch=True, add_dataloader_idx=False)
