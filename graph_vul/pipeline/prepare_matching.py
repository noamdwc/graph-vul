"""Fresh normalized preparation with dataset revision, leakage audit and hashes."""
import argparse
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path

import pandas as pd

from graph_vul.artifacts import code_identity, file_sha256, directory_sha256, write_json
from graph_vul.config import PRIMEVUL_DATASET_NAME, PRIMEVUL_DATASET_CONFIG
from graph_vul.dataset.primevul_adapter import load_primevul_training_data
from graph_vul.model.cwe_matching import load_catalog, parse_cwe_ids
from graph_vul.pipeline.func_extract_subset import extract_func_subset
from graph_vul.pipeline.split_data import split_data
from graph_vul.pipeline.treesitter_graphs import extract_treesitter_graphs


def prepare(output_dir, dataset_revision):
    if not re.fullmatch(r"[0-9a-f]{40}", dataset_revision):
        raise ValueError("Dataset revision must be an immutable commit SHA")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # DVC removes this stage's outputs but may retain the independent CWE embeddings.
    if any(path.name != "cwe_embeddings" for path in output.iterdir()):
        raise FileExistsError(f"Preparation outputs already exist: {output}")
    raw = load_primevul_training_data(revision=dataset_revision)
    raw.to_csv(output / "normalized.csv", index=False)
    unknown = raw[(raw["vul"] == 1) & raw["cwe_id"].eq("")]
    unknown.to_csv(output / "unknown_cwe_vulnerable.csv", index=False)
    subset = extract_func_subset(
        input_csv=str(output / "normalized.csv"),
        output_csv=str(output / "subset.csv"),
        output_manifest=str(output / "subset_manifest.json"),
        output_report=str(output / "subset_report.md"),
    )
    catalog = load_catalog()
    annotations = {cwe for value in subset["cwe_id"] for cwe in parse_cwe_ids(value)}
    missing = annotations - catalog["descriptions"].keys()
    if missing:
        raise ValueError(
            f"Catalog must be explicitly versioned to cover training annotations: {sorted(missing)}"
        )
    split_data(
        input_csv=str(output / "subset.csv"),
        output_dir=str(output / "splits"),
        seed=42,
        split_mode="project_disjoint",
    )
    assignments = pd.read_csv(output / "splits/assignments.csv")
    merged = subset.merge(assignments, on="sample_id", validate="one_to_one")
    if len(merged) != len(subset) or not assignments["sample_id"].is_unique:
        raise ValueError("Split coverage or unique assignment audit failed")
    if merged.groupby("project")["split"].nunique().max() != 1:
        raise ValueError("Project leakage detected")
    merged["code_hash"] = merged["func_before"].map(
        lambda code: hashlib.sha256(" ".join(code.split()).encode()).hexdigest()
    )
    if merged.groupby("code_hash")["split"].nunique().max() != 1:
        raise ValueError("Code leakage detected")
    extract_treesitter_graphs(
        str(output / "subset.csv"), str(output / "graphs"), str(output / "splits")
    )
    files = [
        "normalized.csv",
        "subset.csv",
        "splits/assignments.csv",
        "graphs/node_type_vocab.json",
        "graphs/edge_type_vocab.json",
    ]
    if (output / "splits/unseen_assignments.csv").exists():
        files.append("splits/unseen_assignments.csv")
    write_json(
        output / "preparation.json",
        {
            "contract": 1,
            "code": code_identity(),
            "dataset": {
                "name": PRIMEVUL_DATASET_NAME,
                "config": PRIMEVUL_DATASET_CONFIG,
                "revision": dataset_revision,
            },
            "catalog_version": catalog["version"],
            "catalog_sha256": hashlib.sha256(
                json.dumps(catalog, sort_keys=True).encode()
            ).hexdigest(),
            "unknown_vulnerable_excluded_from_training": len(unknown),
            "graph_parser": {
                "tree_sitter": importlib.metadata.version("tree-sitter"),
                "tree_sitter_c": importlib.metadata.version("tree-sitter-c"),
            },
            "leakage_audit": {
                "projects_overlap": 0,
                "code_overlap": 0,
                "rows": len(merged),
            },
            "directories": {"graphs": directory_sha256(output / "graphs")},
            "hashes": {file: file_sha256(output / file) for file in files},
        },
    )


def prepare_pilot(data_root, output_dir, samples_per_class=4):
    """Copy a bounded, labeled pilot from the shared preparation, retaining artifact identities."""
    import shutil
    from graph_vul.artifacts import preparation_identity

    root, output = Path(data_root), Path(output_dir)
    parent = preparation_identity(root)
    output.mkdir(parents=True, exist_ok=False)
    subset = pd.read_csv(root / "subset.csv")
    assignments = pd.read_csv(root / "splits/assignments.csv")
    assigned = subset.merge(assignments, on="sample_id", validate="one_to_one")
    parts = []
    for split in ("train", "val", "test"):
        for label in (0, 1):
            rows = assigned[
                (assigned["split"] == split) & (assigned["vul"] == label)
            ].head(samples_per_class)
            if rows.empty:
                raise ValueError(f"Pilot requires both classes in {split}")
            parts.append(rows)
    selected = pd.concat(parts, ignore_index=True)
    # Keep the full internal metadata for external overlap exclusion; assignments bound pilot work.
    shutil.copy2(root / "subset.csv", output / "subset.csv")
    (output / "splits").mkdir()
    selected[["sample_id", "split"]].to_csv(
        output / "splits/assignments.csv", index=False
    )
    (output / "graphs").mkdir()
    for name in ("node_type_vocab.json", "edge_type_vocab.json"):
        shutil.copy2(root / "graphs" / name, output / "graphs" / name)
    for sample_id in selected["sample_id"]:
        shutil.copy2(
            root / "graphs" / f"{sample_id}.pt", output / "graphs" / f"{sample_id}.pt"
        )
    shutil.copytree(root / "cwe_embeddings", output / "cwe_embeddings")
    hashes = {
        name: file_sha256(output / name)
        for name in (
            "subset.csv",
            "splits/assignments.csv",
            "graphs/node_type_vocab.json",
            "graphs/edge_type_vocab.json",
        )
    }
    write_json(
        output / "preparation.json",
        {
            "contract": 1,
            "pilot": True,
            "parent": parent,
            "catalog_sha256": parent["catalog_sha256"],
            "samples_per_class": samples_per_class,
            "hashes": hashes,
            "directories": {"graphs": directory_sha256(output / "graphs")},
        },
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-revision", required=True, help="Immutable Hugging Face commit SHA"
    )
    prepare(**vars(parser.parse_args()))
