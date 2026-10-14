"""Rebuild the f9867245 BigVul inputs in a new, isolated artifact directory."""
import argparse
import hashlib
import json
import logging
import sys
import zipfile
from pathlib import Path

import pandas as pd


def file_hash(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    source = args.source_repo / "data/bigvul.csv"
    source_md5 = file_hash(source, "md5")
    if source_md5 != "95436ff6b6eb8b95c44f42dee5fe7541":
        raise ValueError("BigVul source differs from f9867245's recorded source")
    args.output.mkdir(parents=True, exist_ok=False)
    reference = args.output / "reference/E3_all_layers"
    reference.mkdir(parents=True)
    archive = args.source_repo / "E3_all_layers-20261014T074630Z-1-001.zip"
    names = ["func_finetune_model.ckpt", "func_finetune_metrics.json",
             "predictions/test_predictions.csv", "predictions/unseen_test_predictions.csv"]
    with zipfile.ZipFile(archive) as zipped:
        for name in names:
            target = reference / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open("E3_all_layers/" + name) as incoming, target.open("xb") as outgoing:
                for block in iter(lambda: incoming.read(1024 * 1024), b""):
                    outgoing.write(block)
            print("Recovered", target, flush=True)

    prepared = args.output / "prepared"
    prepared.mkdir()
    from graph_vul.pipeline import func_extract_subset
    from graph_vul.pipeline.split_data import split_data

    # The source expands to 10.8 GB. Read only columns consumed by the historical
    # preparation; this preserves their values/order without materializing patches.
    def read_bigvul_columns(path):
        frame = pd.read_csv(path, compression="zip", low_memory=False,
                            usecols=["project", "func_before", "CWE ID", "CVE ID", "vul"])
        return frame.rename(columns={"CWE ID": "cwe_id", "CVE ID": "cve_id"})

    func_extract_subset._read_bigvul_csv = read_bigvul_columns
    subset = func_extract_subset.extract_func_subset(
        input_csv=str(source), output_csv=str(prepared / "func_subset.csv"),
        output_manifest=str(prepared / "func_subset_manifest.json"),
        output_report=str(prepared / "func_subset_report.md"), seed=42)
    subset_md5 = file_hash(prepared / "func_subset.csv", "md5")
    print("Rebuilt subset", len(subset), "MD5", subset_md5,
          "matches historical", subset_md5 == "4f08bfe20f5c8e1ad79140990706f719", flush=True)
    split_data(input_csv=str(prepared / "func_subset.csv"),
               output_dir=str(prepared / "func_splits"), seed=42)
    assignments = pd.read_csv(prepared / "func_splits/assignments.csv")
    indexed = subset.set_index("sample_id")
    assigned = assignments.set_index("sample_id")["split"]
    compatibility = {}
    for split in ["test", "unseen_test"]:
        predictions = pd.read_csv(reference / f"predictions/{split}_predictions.csv")
        selected = indexed.reindex(predictions["sample_id"])
        discrepancies = {}
        for field in ["func_before", "project", "cwe_id", "vul"]:
            discrepancies[field] = int((selected[field].fillna("").astype(str).to_numpy()
                                        != predictions[field].fillna("").astype(str).to_numpy()).sum())
        discrepancies["split"] = int((assigned.reindex(predictions["sample_id"]).to_numpy() != split).sum())
        discrepancies["cohort_rows"] = abs(int((assignments["split"] == split).sum()) - len(predictions))
        compatibility[split] = {"rows": len(predictions), "discrepancies": discrepancies}
        print("Evaluation identity", split, compatibility[split], flush=True)

    report = {"source_md5": source_md5, "historical_subset_md5": "4f08bfe20f5c8e1ad79140990706f719",
              "rebuilt_subset_md5": subset_md5, "subset_rows": len(subset),
              "split_rows": assignments["split"].value_counts().to_dict(),
              "evaluation_identity": compatibility,
              "reference_sha256": {name: file_hash(reference / name) for name in names}}
    (args.output / "preparation_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    evaluation_identity_verified = not any(
        value for cohort in compatibility.values() for value in cohort["discrepancies"].values()
    )
    manifest = {"kind": "reconstructed_E3_inputs",
                "historical_commit": "f9867245e2cab78a79a7b617a346443e54d0348a",
                "source_csv_byte_identical": source_md5 == "95436ff6b6eb8b95c44f42dee5fe7541",
                "evaluation_identity_verified": evaluation_identity_verified,
                "historical_graph_identity_verified": False,
                "subset_rows": len(subset),
                "split_rows": assignments["split"].value_counts().to_dict(),
                "sha256": {path.relative_to(prepared).as_posix(): file_hash(path)
                           for path in prepared.rglob("*")
                           if path.is_file() and path.name != "reconstruction.json"}}
    (prepared / "reconstruction.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not evaluation_identity_verified:
        raise ValueError("Rebuilt preparation does not reproduce E3's evaluation identities; inspect preparation_audit.json")


if __name__ == "__main__":
    main()
