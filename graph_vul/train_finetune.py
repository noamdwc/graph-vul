"""Train one controlled experiment, calibrate its selected checkpoint, then export."""
import argparse
import hashlib
import json
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer

from graph_vul.artifacts import (
    code_identity,
    file_sha256,
    directory_sha256,
    preparation_identity,
    write_json,
)
from graph_vul.config import CODE_MODEL_NAME, CODE_MODEL_REVISION, GRAPH_HIDDEN_DIM
from graph_vul.dataset.func_finetune_dataset import (
    FuncFinetuneDataset,
    finetune_graph_collate,
)
from graph_vul.inference import calibrate_threshold
from graph_vul.model.func_finetune_lit_module import FuncFinetuneLitModule


def train(
    data_root,
    output_dir,
    ablation_mode="pair",
    seed=42,
    num_epochs=10,
    batch_size=8,
    grad_accum=8,
    encoder_lr=2e-5,
    head_lr=1e-3,
    freeze_encoder_epochs=3,
    unfreeze_encoder_lr=5e-6,
    freeze_encoder_fully=False,
    head_dropout=0.3,
    weight_decay=0.01,
    matching_negatives_per_positive=1,
    matching_loss_lambda=0.1,
    matching_triplet_margin=0.2,
    limit_train_batches=1.0,
):
    root, output = Path(data_root), Path(output_dir)
    prepared = preparation_identity(root)
    output.mkdir(parents=True, exist_ok=False)
    pl.seed_everything(seed, workers=True)
    catalog = json.loads((root / "cwe_embeddings/catalog.json").read_text())
    if "catalog_sha256" in prepared:
        catalog_sha = hashlib.sha256(
            json.dumps(catalog, sort_keys=True).encode()
        ).hexdigest()
        if catalog_sha != prepared["catalog_sha256"]:
            raise ValueError("Embedding catalog differs from preparation catalog")
    use_graph = ablation_mode != "pair"
    node_vocab = json.loads((root / "graphs/node_type_vocab.json").read_text())
    edge_vocab = json.loads((root / "graphs/edge_type_vocab.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        CODE_MODEL_NAME, revision=CODE_MODEL_REVISION
    )
    datasets = {
        split: FuncFinetuneDataset(
            split,
            tokenizer=tokenizer,
            subset_csv=root / "subset.csv",
            splits_dir=root / "splits",
            use_graph=use_graph,
            graph_artifacts_dir=root / "graphs",
        )
        for split in ("train", "val")
    }
    labels = torch.tensor(datasets["train"].labels, dtype=torch.long)
    # Dataset order, rather than assignment-file order, determines sampler weights.
    counts = torch.bincount(labels, minlength=2)
    if not bool((counts > 0).all()):
        raise ValueError("Training requires both classes")
    sampler = WeightedRandomSampler(1.0 / counts[labels], len(labels), replacement=True)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=finetune_graph_collate,
    )
    val_loader = DataLoader(
        datasets["val"], batch_size=batch_size, collate_fn=finetune_graph_collate
    )
    provenance = {
        "code": code_identity(),
        "preparation": prepared,
        "seed": seed,
        "cwe_embeddings_sha256": directory_sha256(root / "cwe_embeddings"),
        "budget": {
            "epochs": num_epochs,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "limit_train_batches": limit_train_batches,
        },
    }
    module = FuncFinetuneLitModule(
        encoder_lr=encoder_lr,
        head_lr=head_lr,
        weight_decay=weight_decay,
        graph_dim=GRAPH_HIDDEN_DIM if use_graph else 0,
        num_node_types=len(node_vocab),
        num_relations=len(edge_vocab),
        node_vocab=node_vocab,
        edge_vocab=edge_vocab,
        catalog=catalog,
        cwe_embeddings_dir=root / "cwe_embeddings",
        provenance=provenance,
        freeze_encoder_epochs=freeze_encoder_epochs,
        unfreeze_encoder_lr=unfreeze_encoder_lr,
        freeze_encoder_fully=freeze_encoder_fully,
        head_dropout=head_dropout,
        ablation_mode=ablation_mode,
        matching_seed=seed,
        matching_negatives_per_positive=matching_negatives_per_positive,
        matching_loss_lambda=matching_loss_lambda,
        matching_triplet_margin=matching_triplet_margin,
    )
    provenance["encoder_revision"] = CODE_MODEL_REVISION
    # All experiments use the same epoch budget. Select on annotation-independent validation PR-AUC.
    checkpoint = ModelCheckpoint(
        dirpath=output / "selection",
        filename="best",
        monitor="val_pr_auc",
        mode="max",
        save_top_k=1,
    )
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        accumulate_grad_batches=grad_accum,
        callbacks=[checkpoint],
        logger=False,
        gradient_clip_val=1.0,
        limit_train_batches=limit_train_batches,
    )
    trainer.fit(module, train_loader, val_loader)
    selected = FuncFinetuneLitModule.load_from_checkpoint(
        checkpoint.best_model_path, map_location="cpu"
    )
    prediction_trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="32-true",
        logger=False,
    )
    predictions = prediction_trainer.predict(selected, val_loader)
    margins = torch.cat([item["margin"].cpu() for item in predictions]).numpy()
    targets = torch.cat([item["labels"].cpu() for item in predictions]).numpy()
    threshold = calibrate_threshold(margins, targets)
    # Add calibration to the exact selected weights; never reload last-epoch weights for export/test.
    payload = torch.load(
        checkpoint.best_model_path, map_location="cpu", weights_only=True
    )
    payload["state_dict"]["decision_threshold"] = torch.tensor(
        threshold, dtype=torch.float64
    )
    payload["state_dict"]["calibrated"] = torch.tensor(True)
    payload["calibration"] = {
        "rule": "max_real_logit_minus_none_gt_threshold",
        "threshold": threshold,
        "split": "val",
        "rows": len(targets),
        "selected_checkpoint_sha256": file_sha256(checkpoint.best_model_path),
    }
    torch.save(payload, output / "model.ckpt")
    write_json(
        output / "run.json",
        {
            **provenance,
            "resolved_config": dict(module.hparams),
            "matching": module.matching_config.to_dict(),
            "calibration": payload["calibration"],
            "checkpoint_sha256": file_sha256(output / "model.ckpt"),
        },
    )
    from graph_vul.export_finetune_predictions import export_predictions

    export_predictions(output / "model.ckpt", output / "predictions", root, batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ablation-mode",
        choices=["pair", "graph_pair", "triplet_random", "triplet_hard"],
        default="pair",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", dest="num_epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=3)
    parser.add_argument("--unfreeze-encoder-lr", type=float, default=5e-6)
    parser.add_argument("--freeze-encoder-fully", action="store_true")
    parser.add_argument("--head-dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--matching-negatives-per-positive", type=int, default=1)
    parser.add_argument("--matching-loss-lambda", type=float, default=0.1)
    parser.add_argument("--matching-triplet-margin", type=float, default=0.2)
    parser.add_argument(
        "--limit-train-batches",
        type=float,
        default=1.0,
        help="Pilot only; 1.0 means all batches",
    )
    train(**vars(parser.parse_args()))
