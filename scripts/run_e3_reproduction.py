"""Run historical E3 or its last-four-layer E2 baseline on reconstructed inputs."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import snapshot_download

MODEL_REVISION = "5604afdc964f6c53782a6813140ade5216b99006"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def smoke_check(root, output, tokenizer, module_class, dataset_class, collate, fusion_layers):
    """Exercise real code, both labels, and present/missing graphs on CPU."""
    subset = pd.read_csv(root / "func_subset.csv")
    assignments = pd.read_csv(root / "func_splits/assignments.csv")
    train_ids = assignments.loc[assignments["split"] == "train", "sample_id"]
    train = subset[subset["sample_id"].isin(train_ids)].copy()
    graph_ids = {int(path.stem) for path in (root / "processed/graph_artifacts").glob("*.pt")}
    train["has_graph"] = train["sample_id"].isin(graph_ids)
    selected = []
    selected_ids = set()
    for has_graph in [True, False]:
        for label in [0, 1]:
            candidates = train[train["vul"] == label]
            if has_graph:
                candidates = candidates[candidates["has_graph"]]
            else:
                candidates = candidates[~candidates["sample_id"].isin(selected_ids)]
            row = candidates.loc[candidates["func_before"].str.len().idxmin()].copy()
            row["smoke_has_graph"] = has_graph
            selected.append(row)
            selected_ids.add(row["sample_id"])
    selected_frame = pd.DataFrame(selected)
    subset = selected_frame.drop(columns=["has_graph", "smoke_has_graph"])
    subset.to_csv(output / "smoke_subset.csv", index=False)
    splits = output / "smoke_splits"
    splits.mkdir()
    pd.DataFrame({"sample_id": subset["sample_id"], "split": "train"}).to_csv(splits / "assignments.csv", index=False)
    smoke_graphs = output / "smoke_graph_artifacts"
    smoke_graphs.mkdir()
    for vocab in ["node_type_vocab.json", "edge_type_vocab.json"]:
        (smoke_graphs / vocab).symlink_to(root / "processed/graph_artifacts" / vocab)
    for _, row in selected_frame.iterrows():
        if row["smoke_has_graph"]:
            sample_id = str(row["sample_id"])
            (smoke_graphs / f"{sample_id}.pt").symlink_to(root / "processed/graph_artifacts" / f"{sample_id}.pt")
    dataset = dataset_class("train", tokenizer=tokenizer, max_length=128,
                            subset_csv=output / "smoke_subset.csv", splits_dir=splits,
                            cwe_embeddings_dir=root / "processed/cwe_embeddings",
                            use_graph=True, graph_artifacts_dir=smoke_graphs)
    node_vocab = json.loads((smoke_graphs / "node_type_vocab.json").read_text())
    edge_vocab = json.loads((smoke_graphs / "edge_type_vocab.json").read_text())
    module = module_class(graph_dim=64, num_node_types=len(node_vocab), num_relations=len(edge_vocab),
                          fusion_layers=fusion_layers, final_graph_concat=True)
    module.train()
    batch = collate([dataset[index] for index in range(len(dataset))])
    optimizers, _ = module.configure_optimizers()
    optimizer = optimizers[0]
    optimizer.zero_grad()
    logits = module(batch["input_ids"], batch["attention_mask"], batch["cwe_embedding"], graph_batch=batch["graph_data"]).squeeze(-1)
    loss = module.criterion(logits, batch["label"])
    if not bool(torch.isfinite(loss)):
        raise ValueError("Nonfinite real-model smoke loss")
    loss.backward()
    graph_grad = module.model.graph_encoder.node_embedding.weight.grad
    if graph_grad is None or not bool(torch.isfinite(graph_grad).all()) or not bool(graph_grad.count_nonzero()):
        raise ValueError("Smoke batch did not produce finite, nonzero graph-encoder gradients")
    optimizer.step()
    module.eval()
    with torch.inference_mode():
        scores = module(batch["input_ids"], batch["attention_mask"], batch["cwe_embedding"], graph_batch=batch["graph_data"]).squeeze(-1)
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("Nonfinite smoke inference scores")
    return {"kind": "execution_smoke_not_performance", "rows": len(dataset), "max_length": 128,
            "sample_ids": subset["sample_id"].tolist(), "loss": loss.item(),
            "graph_gradient_nonzero": int(graph_grad.count_nonzero()), "logits": scores.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment", choices=("E2", "E3"), default="E3")
    parser.add_argument("--reference-run", type=Path, help="Completed E3 run.json; required for full E2 training")
    parser.add_argument("--smoke", action="store_true", help="Four real examples and one optimizer step on CPU; no performance estimate")
    args = parser.parse_args()
    if args.experiment == "E2" and not args.smoke and args.reference_run is None:
        parser.error("E2 requires --reference-run pointing to the completed E3 run.json")
    fusion_layers = list(range(8, 12)) if args.experiment == "E2" else list(range(12))
    run_tag = "E2_last4" if args.experiment == "E2" else "E3_all_layers"
    root, output = args.data_root.resolve(), args.output_dir.resolve()
    manifest = json.loads((root / "reconstruction.json").read_text())
    for name, digest in manifest["sha256"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Reconstructed input changed: {name}")
    if not args.smoke and not torch.cuda.is_available():
        raise RuntimeError("Full training requires a CUDA GPU; use the Colab notebook")
    os.environ.setdefault("WANDB_MODE", "disabled")
    torch.set_num_threads(4)
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    import graph_vul.config as config

    # Historical functions bind config paths at import time. Set the isolated
    # preparation before importing them; the repository's data directory is untouched.
    config.DATA_DIR = root
    config.PROCESSED_DIR = root / "processed"
    config.FUNC_SUBSET_CSV = root / "func_subset.csv"
    config.FUNC_SPLITS_DIR = root / "func_splits"
    config.GRAPH_ARTIFACTS_DIR = root / "processed/graph_artifacts"
    config.GRAPH_NODE_VOCAB_PATH = config.GRAPH_ARTIFACTS_DIR / "node_type_vocab.json"
    config.GRAPH_EDGE_VOCAB_PATH = config.GRAPH_ARTIFACTS_DIR / "edge_type_vocab.json"
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    config.UNIXCODER_MODEL_NAME = snapshot_download("microsoft/unixcoder-base", revision=MODEL_REVISION, local_files_only=offline)
    from transformers import AutoTokenizer
    import transformers
    import pytorch_lightning as pl
    from graph_vul.model.func_finetune_lit_module import FuncFinetuneLitModule
    from graph_vul.dataset.func_finetune_dataset import FuncFinetuneDataset, finetune_graph_collate

    pl.seed_everything(42, workers=True)
    provenance = {"kind": "reconstructed_E3_inputs", "experiment": run_tag, "historical_graph_identity_verified": False,
                  "preparation_manifest_sha256": sha256(root / "reconstruction.json"),
                  "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                  "encoder_revision": MODEL_REVISION,
                  "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "lightning": pl.__version__, "pandas": pd.__version__},
                  "configuration": {"seed": 42, "epochs": 10, "batch_size": 64, "grad_accum": 8,
                                    "encoder_lr": 2e-5, "head_lr": 1e-3, "use_graph": True,
                                    "fusion_layers": fusion_layers, "final_graph_concat": True,
                                    "selection": "val_f1", "threshold": 0.5}}
    if args.experiment == "E2" and not args.smoke:
        reference = json.loads(args.reference_run.read_text())
        for field in ("preparation_manifest_sha256", "encoder_revision", "versions"):
            if provenance[field] != reference[field]:
                raise ValueError(f"E2/E3 comparison mismatch: {field}")
        if reference["configuration"]["fusion_layers"] != list(range(12)):
            raise ValueError("Reference run must be all-layer E3")
        expected = reference["configuration"].copy()
        expected["fusion_layers"] = fusion_layers
        if provenance["configuration"] != expected:
            raise ValueError("E2/E3 training settings differ beyond fusion layers")
        # Runner/report changes are allowed; model, data, and training code must match.
        subprocess.run(["git", "diff", "--exit-code", reference["git_revision"], "HEAD", "--", "graph_vul"],
                       cwd=repo, check=True)
        provenance["reference_E3"] = {"run_json_sha256": sha256(args.reference_run),
                                      "git_revision": reference["git_revision"]}
    output.mkdir(parents=True, exist_ok=False)
    (output / "run.json").write_text(json.dumps(provenance, indent=2) + "\n")
    tokenizer = AutoTokenizer.from_pretrained(config.UNIXCODER_MODEL_NAME)
    if args.smoke:
        result = smoke_check(root, output, tokenizer, FuncFinetuneLitModule, FuncFinetuneDataset, finetune_graph_collate, fusion_layers)
        (output / "smoke.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return

    from graph_vul.train_finetune import train
    train(num_epochs=10, batch_size=64, encoder_lr=2e-5, head_lr=1e-3, grad_accum=8,
          seed=42, use_graph=True, fusion_layers=fusion_layers, final_graph_concat=True,
          checkpoint_path=output / "func_finetune_model.ckpt", metrics_path=output / "training_metrics.json",
          run_tag=run_tag + "_reconstructed")
    from graph_vul.export_finetune_predictions import export_predictions
    export_predictions(checkpoint_path=output / "func_finetune_model.ckpt", output_dir=output / "predictions",
                       splits=("test", "unseen_test"), batch_size=64, threshold=0.5,
                       subset_csv=root / "func_subset.csv", splits_dir=root / "func_splits")
    from graph_vul.analysis.post_experiment_analysis import compute_binary_metrics
    metrics = {split: compute_binary_metrics(pd.read_csv(output / f"predictions/{split}_predictions.csv"))
               for split in ("test", "unseen_test")}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
