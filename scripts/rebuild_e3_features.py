"""Recover cache-backed Joern graphs and missing CWE vectors without touching sources."""
import argparse
import csv
import hashlib
import json
import logging
import sys
import zipfile
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--joern-raw", type=Path,
                        help="Fresh per-sample DOT exports for functions absent from the exact-code cache")
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    csv.field_size_limit(10**8)
    prepared = args.output / "prepared"
    with (prepared / "func_subset.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (prepared / "func_splits/assignments.csv").open(newline="") as handle:
        assignments = {row["sample_id"]: row["split"] for row in csv.DictReader(handle)}
    links = args.output / "recovery/cpg_links"
    links.mkdir(parents=True, exist_ok=False)
    cache = args.source_repo / "data/cpg_cache"
    joern_raw = args.joern_raw.resolve() if args.joern_raw else None
    matched = []
    for row in rows:
        key = hashlib.sha256(row["func_before"].strip().encode()).hexdigest()
        cached = cache / key
        regenerated = joern_raw / row["sample_id"] if joern_raw else None
        if cached.is_dir() and any(cached.glob("*.dot")):
            source, origin = cached, "exact_source_hash_cache"
        elif regenerated and regenerated.is_dir() and any(regenerated.glob("*.dot")):
            source, origin = regenerated, "persistent_joern_export"
        else:
            continue
        (links / row["sample_id"]).symlink_to(source, target_is_directory=True)
        matched.append({"sample_id": row["sample_id"], "split": assignments[row["sample_id"]],
                        "cache_key": key, "source": origin})
    with (args.output / "recovery/cpg_mapping.json").open("x") as handle:
        json.dump(matched, handle, indent=2)
    print("Linked Joern graph exports:", len(matched), "of", len(rows), flush=True)
    if joern_raw and len(matched) != len(rows):
        raise ValueError(f"Full Joern rebuild is incomplete: linked {len(matched)} of {len(rows)} functions")

    graphs = prepared / "processed/graph_artifacts"
    graphs.mkdir(parents=True, exist_ok=False)
    from graph_vul.pipeline import extract_graphs
    # This historical producer keeps vocab paths as module globals even when an
    # explicit output directory is supplied. Redirect only those two output paths.
    extract_graphs.GRAPH_NODE_VOCAB_PATH = graphs / "node_type_vocab.json"
    extract_graphs.GRAPH_EDGE_VOCAB_PATH = graphs / "edge_type_vocab.json"
    extract_graphs.extract_graphs(input_dir=str(links), output_dir=str(graphs),
                                  subset_csv=str(prepared / "func_subset.csv"))

    embeddings = prepared / "processed/cwe_embeddings"
    embeddings.mkdir(parents=True, exist_ok=False)
    required = sorted({row["cwe_id"] for row in rows})
    archive = args.source_repo / "drive-download-20261014T074742Z-1-001.zip"
    copied, missing = [], []
    with zipfile.ZipFile(archive) as zipped:
        names = set(zipped.namelist())
        for cwe in required:
            name = f"processed/cwe_embeddings/{cwe}.pt"
            if name in names:
                vector = zipped.read(name)
                (embeddings / (cwe + ".pt")).write_bytes(vector)
                copied.append(cwe)
            else:
                missing.append(cwe)
    # The old REST producer embedded Description/Summary alone. The pinned local
    # catalog joins Name + '. ' + Description, so remove that name prefix here.
    catalog = json.loads((args.source_repo / "configs/cwe_catalog_v1.json").read_text())
    descriptions = {cwe: catalog["descriptions"][cwe].split(". ", 1)[1] for cwe in missing}
    (prepared / "regenerated_cwe_descriptions.json").write_text(json.dumps(descriptions, indent=2) + "\n")
    if missing:
        from graph_vul.pipeline.embed_utils import load_deberta, embed_texts
        tokenizer, encoder = load_deberta()
        for cwe in missing:
            vector = embed_texts([descriptions[cwe]], tokenizer, encoder)[0]
            torch.save(vector, embeddings / (cwe + ".pt"))
            print("Regenerated missing CWE vector", cwe, flush=True)

    graph_ids = {path.stem for path in graphs.glob("*.pt")}
    if joern_raw and len(graph_ids) != len(rows):
        raise ValueError(f"Full graph extraction is incomplete: wrote {len(graph_ids)} of {len(rows)} tensors")
    coverage = {}
    for split in sorted(set(assignments.values())):
        split_ids = {sid for sid, value in assignments.items() if value == split}
        coverage[split] = {"rows": len(split_ids), "graphs": len(split_ids & graph_ids),
                           "empty_graph_fallback": len(split_ids - graph_ids)}
    audit = {"graph_recovery": "Exact-source CPG cache plus fresh persistent Joern exports",
             "historical_graph_identity_verified": False,
             "historical_graph_identity_limitation": "Original vocabularies and graph-presence mask were not retained; rebuilding cannot prove their identity.",
             "cpg_matches": len(matched), "graph_coverage": coverage,
             "graph_sources": {origin: sum(item["source"] == origin for item in matched)
                               for origin in sorted({item["source"] for item in matched})},
             "node_types": len(json.loads((graphs / "node_type_vocab.json").read_text())),
             "relations": json.loads((graphs / "edge_type_vocab.json").read_text()),
             "copied_cwe_vectors": copied, "regenerated_cwe_vectors": missing,
             "regenerated_cwe_catalog": catalog["version"]}
    (args.output / "feature_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    manifest_path = prepared / "reconstruction.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({"graph_coverage": coverage, "node_types": audit["node_types"],
                     "relations": audit["relations"], "cpg_matches": len(matched),
                     "graph_sources": audit["graph_sources"]})
    manifest["sha256"] = {
        path.relative_to(prepared).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in prepared.rglob("*")
        if path.is_file() and path != manifest_path
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("FEATURE AUDIT", json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
