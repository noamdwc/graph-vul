"""Joint pair-head and triplet supervision with all-candidate validation."""
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
import torchmetrics
from sklearn.metrics import average_precision_score

from graph_vul.config import (
    PROCESSED_DIR,
    FINETUNE_ENCODER_LR,
    FINETUNE_HEAD_LR,
    WEIGHT_DECAY,
)
from graph_vul.inference import predict_batch, score_batch
from graph_vul.model.cwe_matching import (
    CWEEmbeddingStore,
    CWENegativeSampler,
    build_matching_config,
    load_catalog,
    pair_targets,
    NONE,
)
from graph_vul.model.func_finetune_model import FuncFinetuneModel


class FuncFinetuneLitModule(pl.LightningModule):
    def __init__(
        self,
        encoder_lr=FINETUNE_ENCODER_LR,
        head_lr=FINETUNE_HEAD_LR,
        weight_decay=WEIGHT_DECAY,
        graph_dim=0,
        num_node_types=0,
        num_relations=0,
        freeze_encoder_epochs=3,
        unfreeze_encoder_lr=5e-6,
        head_dropout=0.3,
        freeze_encoder_fully=False,
        ablation_mode="pair",
        matching_projection_dim=128,
        matching_negatives_per_positive=1,
        matching_loss_lambda=0.1,
        matching_triplet_margin=0.2,
        matching_seed=42,
        cwe_embeddings_dir=PROCESSED_DIR / "cwe_embeddings",
        catalog=None,
        node_vocab=None,
        edge_vocab=None,
        provenance=None,
    ):
        super().__init__()
        catalog = catalog or load_catalog()
        self.save_hyperparameters()
        # Checkpoints carry the catalog and embeddings, so restoration needs no external files.
        self.hparams["catalog"] = catalog
        self.hparams["cwe_embeddings_dir"] = None
        self.candidate_ids = list(catalog["descriptions"]) + [NONE]
        self.matching_config = build_matching_config(
            ablation_mode,
            matching_projection_dim,
            matching_negatives_per_positive,
            matching_loss_lambda,
            matching_triplet_margin,
        )
        self.model = FuncFinetuneModel(
            graph_dim=graph_dim,
            num_node_types=num_node_types,
            num_relations=num_relations,
            dropout=head_dropout,
            matching_projection_dim=matching_projection_dim,
        )
        width = self.model.cwe_projection.in_features
        if cwe_embeddings_dir is None:
            # Shape allocation during checkpoint restoration; state_dict supplies the actual values.
            embeddings = torch.empty(len(self.candidate_ids) - 1, width)
        else:
            store = CWEEmbeddingStore(cwe_embeddings_dir, width)
            embeddings = store.stack(self.candidate_ids[:-1], torch.device("cpu"))
        self.register_buffer("real_cwe_embeddings", embeddings)
        self.register_buffer(
            "decision_threshold", torch.tensor(0.0, dtype=torch.float64)
        )
        self.register_buffer("calibrated", torch.tensor(False))
        self.negative_sampler = CWENegativeSampler(self.candidate_ids, matching_seed)
        self.val_f1 = torchmetrics.classification.BinaryF1Score()
        self.test_f1 = torchmetrics.classification.BinaryF1Score()

    def on_train_epoch_start(self):
        frozen = (
            self.hparams.freeze_encoder_fully
            or self.current_epoch < self.hparams.freeze_encoder_epochs
        )
        for param in self.model.encoder.parameters():
            param.requires_grad = not frozen

    def training_step(self, batch, batch_idx):
        features, candidates, scores = score_batch(self, batch)
        targets, supervised = pair_targets(
            batch["label"].tolist(), batch["cwe_ids"], self.candidate_ids, scores.device
        )
        pair_losses, triplet_losses = [], []
        function_match = F.normalize(self.model.function_projection(features), dim=-1)
        candidate_match = self.model.project_cwe_embeddings(candidates)
        for row in torch.where(supervised)[0].tolist():
            positives = torch.where(targets[row] == 1)[0].tolist()
            positive_ids = {self.candidate_ids[i] for i in positives}
            mining_scores = (
                scores[row].detach()
                if self.matching_config.use_hard_negatives
                else None
            )
            negatives = self.negative_sampler.select(
                positive_ids, self.matching_config.negatives_per_positive, mining_scores
            )
            # NONE is always a negative for annotated vulnerable functions, even under random sampling.
            if (
                NONE not in positive_ids
                and len(self.candidate_ids) - 1 not in negatives
            ):
                negatives.append(len(self.candidate_ids) - 1)
            pos_loss = F.binary_cross_entropy_with_logits(
                scores[row, positives], targets[row, positives]
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                scores[row, negatives], targets[row, negatives]
            )
            # Balance positive/negative groups per function to avoid catalog-size-dependent loss weights.
            pair_losses.append((pos_loss + neg_loss) / 2)
            if self.matching_config.use_triplet_loss:
                for positive in positives:
                    anchor = function_match[row].expand(len(negatives), -1)
                    positive_match = candidate_match[positive].expand(
                        len(negatives), -1
                    )
                    triplet_losses.append(
                        F.triplet_margin_loss(
                            anchor,
                            positive_match,
                            candidate_match[negatives],
                            margin=self.matching_config.triplet_margin,
                        )
                    )
        pair_loss = torch.stack(pair_losses).mean() if pair_losses else scores.sum() * 0
        triplet_loss = (
            torch.stack(triplet_losses).mean() if triplet_losses else scores.sum() * 0
        )
        loss = pair_loss + self.matching_config.loss_lambda * triplet_loss
        self.log("train_pair_loss", pair_loss, batch_size=len(features))
        self.log("train_triplet_loss", triplet_loss, batch_size=len(features))
        return loss

    def on_validation_epoch_start(self):
        self.validation_margins = []
        self.validation_labels = []

    def validation_step(self, batch, batch_idx):
        result = predict_batch(self, batch, require_calibration=False)
        labels = batch["label"].long()
        self.val_f1.update(result["y_pred"], labels)
        self.validation_margins.append(result["margin"].detach().cpu())
        self.validation_labels.append(labels.detach().cpu())
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        margins = torch.cat(self.validation_margins).numpy()
        labels = torch.cat(self.validation_labels).numpy()
        # Rank raw margins directly: sigmoid saturation must not change checkpoint selection.
        self.log(
            "val_pr_auc", float(average_precision_score(labels, margins)), prog_bar=True
        )

    def test_step(self, batch, batch_idx):
        result = predict_batch(self, batch)
        self.test_f1.update(result["y_pred"], batch["label"].long())
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        result = predict_batch(self, batch, require_calibration=False)
        return {"margin": result["margin"], "labels": batch["label"].long()}

    def on_save_checkpoint(self, checkpoint):
        checkpoint["matching_contract"] = 1
        checkpoint["resolved_matching_config"] = self.matching_config.to_dict()

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get("matching_contract") != 1:
            raise ValueError(
                "Historical checkpoints require retraining for the matching contract"
            )

    def configure_optimizers(self):
        encoder = list(self.model.encoder.parameters())
        encoder_ids = {id(p) for p in encoder}
        head = [p for p in self.parameters() if id(p) not in encoder_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder, "lr": self.hparams.encoder_lr},
                {"params": head, "lr": self.hparams.head_lr},
            ],
            weight_decay=self.hparams.weight_decay,
        )

        def encoder_schedule(epoch):
            if epoch < self.hparams.freeze_encoder_epochs:
                return 1.0
            return (
                self.hparams.unfreeze_encoder_lr / self.hparams.encoder_lr
            ) * 0.95 ** (epoch - self.hparams.freeze_encoder_epochs)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, [encoder_schedule, lambda epoch: 0.95**epoch]
        )
        return [optimizer], [scheduler]
