"""External stress tests using checkpoint catalog, AST vocabulary and calibration."""
import argparse
import hashlib
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from graph_vul.artifacts import file_sha256, preparation_identity, write_json
from graph_vul.config import CODE_MODEL_NAME, CODE_MODEL_REVISION
from graph_vul.dataset.external_datasets import (
    EXTERNAL_DATASETS,
    EXTERNAL_DATASET_REVISIONS,
    ExternalEvalDataset,
)
from graph_vul.export_finetune_predictions import load_model, predict_dataset


def code_hash(code):
    return hashlib.sha256(" ".join(code.split()).encode()).hexdigest()


def evaluate(
    checkpoint_path,
    output_dir,
    data_root,
    datasets=("diversevul",),
    batch_size=8,
    max_samples=None,
):
    root, output = Path(data_root), Path(output_dir)
    prepared = preparation_identity(root)
    module = load_model(checkpoint_path)
    if prepared != module.hparams.provenance["preparation"]:
        raise ValueError("External evaluation preparation differs from training")
    output.mkdir(parents=True, exist_ok=False)
    # Exclude overlaps with the entire internal corpus, including internal evaluation rows.
    internal_hashes = set(
        pd.read_csv(root / "subset.csv")["func_before"].map(code_hash)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        CODE_MODEL_NAME, revision=CODE_MODEL_REVISION
    )
    exports = {}
    for name in datasets:
        frame = EXTERNAL_DATASETS[name]()
        hashes = frame["func_before"].map(code_hash)
        keep = ~hashes.isin(internal_hashes) & ~hashes.duplicated()
        excluded = int((~keep).sum())
        frame = frame[keep].reset_index(drop=True)
        source_rows = len(frame)
        if max_samples is not None:
            frame = frame.sample(
                n=min(max_samples, len(frame)), random_state=42
            ).reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"No independent external rows remain for {name}")
        use_graph = module.model.graph_dim > 0
        dataset = ExternalEvalDataset(
            frame,
            tokenizer=tokenizer,
            node_vocab=module.hparams.node_vocab if use_graph else None,
            edge_vocab=module.hparams.edge_vocab if use_graph else None,
        )
        result = predict_dataset(module, dataset, batch_size)
        path = output / f"{name}_predictions.csv"
        result.to_csv(path, index=False)
        exports[name] = {
            "file": path.name,
            "sha256": file_sha256(path),
            "rows": len(result),
            "normalized_input_sha256": hashlib.sha256(
                frame.to_csv(index=False).encode()
            ).hexdigest(),
            "excluded_overlap_or_duplicate_rows": excluded,
            "dataset_revision": EXTERNAL_DATASET_REVISIONS[name],
            "source_rows_after_overlap_exclusion": source_rows,
            "pilot_sample_limit": max_samples,
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
    parser.add_argument(
        "--datasets", nargs="+", choices=EXTERNAL_DATASETS, default=["diversevul"]
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-samples", type=int, help="Pilot only; sample a bounded external cohort"
    )
    evaluate(**vars(parser.parse_args()))
