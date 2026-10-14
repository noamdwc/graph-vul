"""Export the same complete-catalog decisions used by validation and the demo."""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from graph_vul.artifacts import (
    file_sha256,
    directory_sha256,
    preparation_identity,
    write_json,
)
from graph_vul.config import CODE_MODEL_NAME, CODE_MODEL_REVISION
from graph_vul.dataset.func_finetune_dataset import (
    FuncFinetuneDataset,
    available_splits,
    finetune_graph_collate,
)
from graph_vul.inference import predict_batch
from graph_vul.model.func_finetune_lit_module import FuncFinetuneLitModule


def load_model(checkpoint):
    module = FuncFinetuneLitModule.load_from_checkpoint(checkpoint, map_location="cpu")
    module.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return module.to(device)


def predict_dataset(module, dataset, batch_size=8):
    rows = []
    loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=finetune_graph_collate
    )
    device = next(module.parameters()).device
    with torch.inference_mode():
        for batch in loader:
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
            result = predict_batch(module, inputs)
            scores = result["scores"].cpu()
            ranking = torch.argsort(scores[:, :-1], dim=1, descending=True, stable=True)
            for i in range(len(scores)):
                sample_id = batch["sample_id"][i]
                if torch.is_tensor(sample_id):
                    sample_id = sample_id.item()
                top3 = [module.candidate_ids[j] for j in ranking[i, :3].tolist()]
                annotations = set(batch["cwe_ids"][i])
                vulnerable = int(batch["label"][i])
                retrieval_valid = vulnerable == 1 and bool(annotations)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "y_true": vulnerable,
                        "cwe_id": ";".join(sorted(annotations)),
                        "margin": float(result["margin"][i]),
                        "y_pred": int(result["y_pred"][i]),
                        "prediction": module.candidate_ids[int(result["chosen"][i])],
                        "top3": json.dumps(top3),
                        "scores": json.dumps(scores[i].tolist()),
                        "retrieval_hit_top1": int(top3[0] in annotations)
                        if retrieval_valid
                        else None,
                        "retrieval_hit_top3": int(bool(set(top3) & annotations))
                        if retrieval_valid
                        else None,
                        "token_length": int(batch["token_count"][i]),
                    }
                )
    return pd.DataFrame(rows)


def export_predictions(checkpoint_path, output_dir, data_root, batch_size=8):
    root, output = Path(data_root), Path(output_dir)
    prepared = preparation_identity(root)
    module = load_model(checkpoint_path)
    if (
        directory_sha256(root / "cwe_embeddings")
        != module.hparams.provenance["cwe_embeddings_sha256"]
    ):
        raise ValueError("CWE embedding artifact changed since training")
    if prepared != module.hparams.provenance["preparation"]:
        raise ValueError("Evaluation preparation differs from training")
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(
        CODE_MODEL_NAME, revision=CODE_MODEL_REVISION
    )
    subset = pd.read_csv(root / "subset.csv")
    exports = {}
    for split in available_splits(root / "splits"):
        if split == "train":
            continue
        dataset = FuncFinetuneDataset(
            split,
            tokenizer=tokenizer,
            subset_csv=root / "subset.csv",
            splits_dir=root / "splits",
            use_graph=module.model.graph_dim > 0,
            graph_artifacts_dir=root / "graphs",
        )
        if not len(dataset):
            continue
        predictions = predict_dataset(module, dataset, batch_size)
        predictions = predictions.merge(
            subset[["sample_id", "project"]], on="sample_id", validate="one_to_one"
        )
        path = output / f"{split}_predictions.csv"
        predictions.to_csv(path, index=False)
        exports[split] = {
            "file": path.name,
            "sha256": file_sha256(path),
            "rows": len(predictions),
        }
    write_json(
        output / "manifest.json",
        {
            "contract": 1,
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "candidate_ids": module.candidate_ids,
            "catalog_version": module.hparams.catalog["version"],
            "calibration_threshold": float(module.decision_threshold),
            "provenance": module.hparams.provenance,
            "resolved_config": dict(module.hparams),
            "splits": exports,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", dest="checkpoint_path", type=Path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    export_predictions(**vars(parser.parse_args()))
