"""Behavioral contracts and a CPU training → calibration → export → report pilot."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import pytorch_lightning as pl
import torch
from torch_geometric.data import Batch

from graph_vul.analysis.cwe_matching_report import generate_report
from graph_vul.artifacts import file_sha256, write_json
from graph_vul.dataset.external_datasets import ExternalEvalDataset
from graph_vul.dataset.func_finetune_dataset import finetune_graph_collate
from graph_vul.export_finetune_predictions import load_model, predict_dataset
from graph_vul.inference import calibrate_threshold, decision, predict_batch
from graph_vul.model.cwe_matching import (
    CWEEmbeddingStore,
    CWENegativeSampler,
    NONE,
    pair_targets,
    parse_cwe_ids,
)
from graph_vul.model.func_finetune_lit_module import FuncFinetuneLitModule
from graph_vul.pipeline.treesitter_graphs import (
    extract_treesitter_graphs,
    function_graph,
)
from graph_vul.train_finetune import train


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = torch.nn.Embedding(32, 8)
        self.calls = 0

    def forward(self, input_ids, attention_mask):
        self.calls += 1
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class TinyTokenizer:
    def __call__(self, codes, truncation, max_length, padding, return_tensors):
        ids = [[min(len(word), 31) for word in code.split()[:8]] for code in codes]
        return {
            "input_ids": torch.tensor([row + [0] * (8 - len(row)) for row in ids]),
            "attention_mask": torch.tensor(
                [[1] * len(row) + [0] * (8 - len(row)) for row in ids]
            ),
        }


@pytest.fixture
def tiny_backbone(monkeypatch):
    monkeypatch.setattr(
        "graph_vul.model.func_finetune_model.AutoModel.from_pretrained",
        lambda *a, **kw: TinyEncoder(),
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *a, **kw: TinyTokenizer()
    )


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "prepared"
    root.mkdir()
    rows = [
        {
            "sample_id": i,
            "project": f"p{i//2}",
            "func_before": f"int f{i}() {{ return {i}; }}",
            "cwe_id": "CWE-119;CWE-125" if i % 2 else "",
            "vul": i % 2,
        }
        for i in range(6)
    ]
    pd.DataFrame(rows).to_csv(root / "subset.csv", index=False)
    (root / "splits").mkdir()
    pd.DataFrame(
        {
            "sample_id": range(6),
            "split": ["train", "train", "val", "val", "test", "test"],
        }
    ).to_csv(root / "splits/assignments.csv", index=False)
    extract_treesitter_graphs(
        str(root / "subset.csv"), str(root / "graphs"), str(root / "splits")
    )
    (root / "cwe_embeddings").mkdir()
    catalog = {
        "version": "test-v1",
        "descriptions": {"CWE-119": "buffer", "CWE-125": "read"},
    }
    write_json(root / "cwe_embeddings/catalog.json", catalog)
    for cwe in catalog["descriptions"]:
        torch.save(torch.randn(768), root / f"cwe_embeddings/{cwe}.pt")
    files = [
        "subset.csv",
        "splits/assignments.csv",
        "graphs/node_type_vocab.json",
        "graphs/edge_type_vocab.json",
    ]
    write_json(
        root / "preparation.json",
        {"hashes": {file: file_sha256(root / file) for file in files}},
    )
    return root


def test_pair_targets_preserve_secondary_labels_and_unknown_vulnerable():
    targets, supervised = pair_targets(
        [1, 1, 0],
        [["CWE-119", "CWE-125"], [], ["CWE-119"]],
        ["CWE-119", "CWE-125", "CWE-20", NONE],
    )
    assert targets.tolist() == [[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]
    assert supervised.tolist() == [True, False, True]
    assert parse_cwe_ids(np.array(["119", "CWE-125", "CWE-119"])) == [
        "CWE-119",
        "CWE-125",
    ]
    with pytest.raises(ValueError, match="outside"):
        pair_targets([1], [["CWE-999"]], ["CWE-119", NONE])


def test_negative_selection_excludes_every_positive_and_uses_model_scores():
    sampler = CWENegativeSampler(["CWE-119", "CWE-125", "CWE-20", NONE])
    scores = torch.tensor([10.0, 9.0, 0.1, 0.8])
    assert sampler.select({"CWE-119", "CWE-125"}, 1, scores) == [3]
    assert set(sampler.select({"CWE-119", "CWE-125"}, 20)) == {2, 3}
    assert sampler.select({NONE}, 1, scores) == [0]


def test_missing_embedding_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        CWEEmbeddingStore(tmp_path, 768).get("CWE-119")


def test_calibration_and_none_ties():
    scores = torch.tensor([[0.2, 0.1, 0.2], [2.0, 0.0, -1.0], [0.5, 0.1, 1.0]])
    result = decision(scores, 0.0)
    assert result["y_pred"].tolist() == [0, 1, 0]
    assert result["chosen"].tolist() == [2, 0, 2]
    threshold = calibrate_threshold(result["margin"].numpy(), [0, 1, 0])
    assert decision(scores, threshold)["y_pred"].tolist() == [0, 1, 0]


def test_external_graphs_and_inference_ignore_ground_truth(tiny_backbone, prepared):
    catalog = json.loads((prepared / "cwe_embeddings/catalog.json").read_text())
    nodes = json.loads((prepared / "graphs/node_type_vocab.json").read_text())
    edges = {"AST": 0}
    module = FuncFinetuneLitModule(
        graph_dim=8,
        num_node_types=len(nodes),
        num_relations=1,
        cwe_embeddings_dir=prepared / "cwe_embeddings",
        catalog=catalog,
        node_vocab=nodes,
        edge_vocab=edges,
    )
    module.eval()
    module.calibrated.fill_(True)
    frame = pd.DataFrame(
        [
            {
                "sample_id": "external",
                "func_before": "int x() { return 1; }",
                "vul": 1,
                "cwe_id": "CWE-119",
            }
        ]
    )
    dataset = ExternalEvalDataset(
        frame, tokenizer=TinyTokenizer(), node_vocab=nodes, edge_vocab=edges
    )
    batch = finetune_graph_collate([dataset[0]])
    with torch.no_grad():
        first = predict_batch(module, batch)
        batch["cwe_ids"] = [["CWE-999"]]
        batch["label"].zero_()
        second = predict_batch(module, batch)
    assert torch.equal(first["scores"], second["scores"])
    assert first["scores"].shape == (1, 3)
    assert module.model.encoder.calls == 2
    features = module.model.encode_functions(
        batch["input_ids"], batch["attention_mask"], batch["graph_data"]
    )
    candidates = torch.cat(
        [module.real_cwe_embeddings, module.model.none_embedding[None]]
    )
    assert torch.allclose(
        module.model.score_candidates(features, candidates, 1),
        first["scores"],
        atol=1e-6,
    )
    assert module.model.encoder.calls == 3
    with pytest.raises(ValueError, match="requires AST"):
        module.model.encode_functions(batch["input_ids"], batch["attention_mask"])


@pytest.mark.parametrize(
    "experiment", ["pair", "graph_pair", "triplet_random", "triplet_hard"]
)
def test_cpu_end_to_end_pilot(tiny_backbone, prepared, tmp_path, experiment):
    output = tmp_path / experiment
    train(
        prepared,
        output,
        ablation_mode=experiment,
        num_epochs=1,
        batch_size=2,
        grad_accum=1,
        freeze_encoder_epochs=0,
    )
    module = load_model(output / "model.ckpt")
    assert bool(module.calibrated)
    assert "model.none_embedding" in module.state_dict()
    before = float(module.decision_threshold)
    from graph_vul.eval_external import evaluate
    from unittest.mock import patch

    external = pd.DataFrame(
        [
            {
                "sample_id": f"ex{i}",
                "func_before": f"int ext{i}() {{ return {10+i}; }}",
                "vul": i % 2,
                "cwe_id": "CWE-119" if i % 2 else "",
            }
            for i in range(2)
        ]
    )
    with patch.dict(
        "graph_vul.eval_external.EXTERNAL_DATASETS", {"fixture": lambda: external}
    ), patch.dict(
        "graph_vul.eval_external.EXTERNAL_DATASET_REVISIONS",
        {"fixture": "synthetic-fixture"},
    ):
        evaluate(
            output / "model.ckpt",
            output / "external",
            prepared,
            datasets=["fixture"],
            batch_size=2,
        )
    manifests = [
        output / "predictions/manifest.json",
        output / "external/manifest.json",
    ]
    generate_report(manifests, output / "report")
    metrics = pd.read_csv(output / "report/metrics.csv")
    assert metrics.loc[metrics["slice"] == "all", "rows"].tolist() == [2, 2, 2]
    assert float(load_model(output / "model.ckpt").decision_threshold) == before
    payload = torch.load(output / "model.ckpt", weights_only=True)
    assert payload["calibration"]["rows"] == 2
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        manifest["provenance"]["preparation"]["pilot"] = True
        write_json(manifest_path, manifest)
    generate_report(manifests, output / "pilot-report")
    assert "do not estimate model performance" in (
        output / "pilot-report/report.md"
    ).read_text()
    manifest = json.loads(manifests[0].read_text())
    manifest["provenance"]["preparation"]["pilot"] = False
    write_json(manifests[0], manifest)
    with pytest.raises(ValueError, match="Cannot mix pilot"):
        generate_report(manifests, output / "mixed-report")
    manifest["provenance"]["preparation"]["pilot"] = True
    write_json(manifests[0], manifest)
    predictions = output / "predictions/test_predictions.csv"
    predictions.write_text(predictions.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        generate_report(manifests, output / "tampered-report")


def test_pair_head_and_none_receive_supervision_and_optimizer_owns_projections(
    tiny_backbone, prepared
):
    catalog = json.loads((prepared / "cwe_embeddings/catalog.json").read_text())
    module = FuncFinetuneLitModule(
        cwe_embeddings_dir=prepared / "cwe_embeddings",
        catalog=catalog,
        ablation_mode="triplet_hard",
    )
    module.log = lambda *args, **kwargs: None
    frame = pd.read_csv(prepared / "subset.csv").iloc[:2]
    dataset = ExternalEvalDataset(frame, tokenizer=TinyTokenizer())
    batch = finetune_graph_collate([dataset[0], dataset[1]])
    module.training_step(batch, 0).backward()
    assert module.model.head[0].weight.grad.abs().sum() > 0
    assert module.model.none_embedding.grad.abs().sum() > 0
    assert module.model.function_projection.weight.grad is not None
    optimizers, _ = module.configure_optimizers()
    owned = {
        id(param) for group in optimizers[0].param_groups for param in group["params"]
    }
    assert {id(param) for param in module.parameters()} == owned


def test_calibration_can_persist_all_positive_decision():
    scores = torch.tensor([[0.1, -0.5], [0.2, -0.5]])
    margins = decision(scores, 0)["margin"].numpy()
    # The lowest threshold lies immediately below min(margin); float64 preserves that strict boundary.
    threshold = np.nextafter(margins.min(), -np.inf)
    assert decision(scores, torch.tensor(threshold, dtype=torch.float64))[
        "y_pred"
    ].tolist() == [1, 1]


def test_deduplication_keeps_all_positive_annotations_and_rejects_conflicting_labels():
    from graph_vul.pipeline.func_extract_subset import _deduplicate

    frame = pd.DataFrame(
        {
            "func_before": ["int f() { return 0; }"] * 2,
            "cwe_id": ["CWE-119", "CWE-125"],
            "vul": [1, 1],
        }
    )
    merged = _deduplicate(frame)
    assert merged["cwe_id"].tolist() == ["CWE-119;CWE-125"]
    frame.loc[1, "vul"] = 0
    with pytest.raises(ValueError, match="contradictory"):
        _deduplicate(frame)


def test_calibration_matches_f1_contract_with_ties():
    from sklearn.metrics import f1_score

    margins = np.array([-2, -2, -1, 0, 0, 1, 2, 2], dtype=float)
    labels = np.array([0, 1, 0, 1, 0, 1, 1, 0])
    threshold = calibrate_threshold(margins, labels)
    alternatives = np.r_[np.nextafter(margins.min(), -np.inf), np.unique(margins)]
    expected = max(
        alternatives, key=lambda value: (f1_score(labels, margins > value), value)
    )
    assert threshold == expected


def test_pilot_preparation_bounds_work_but_keeps_full_overlap_metadata(
    prepared, tmp_path
):
    from graph_vul.pipeline.prepare_matching import prepare_pilot

    catalog = json.loads((prepared / "cwe_embeddings/catalog.json").read_text())
    import hashlib

    manifest_path = prepared / "preparation.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["catalog_sha256"] = hashlib.sha256(
        json.dumps(catalog, sort_keys=True).encode()
    ).hexdigest()
    write_json(manifest_path, manifest)
    pilot = prepare_pilot(prepared, tmp_path / "pilot", samples_per_class=1)
    assert file_sha256(pilot / "subset.csv") == file_sha256(prepared / "subset.csv")
    assert len(pd.read_csv(pilot / "splits/assignments.csv")) == 6
    assert json.loads((pilot / "preparation.json").read_text())["pilot"] is True


def test_ast_keeps_function_after_leading_comment():
    from graph_vul.pipeline.treesitter_graphs import parse_function_ast

    node_types, edges = parse_function_ast("/* context */ int f() { return 1; }")
    assert "function_definition" in node_types
    assert "return_statement" in node_types
    assert len(edges) == len(node_types) - 1


def test_project_cwe_cap_counts_secondary_annotations(monkeypatch):
    import graph_vul.pipeline.func_extract_subset as module

    monkeypatch.setattr(module, "FUNC_MAX_PROJECT_ROWS", 1)
    frame = pd.DataFrame(
        {
            "project": ["p", "p", "q"],
            "cwe_id": ["CWE-119;CWE-125", "CWE-125", "CWE-119"],
            "vul": [1, 1, 1],
        }
    )
    filtered, details = module._filter_projects_per_cwe(frame)
    assert filtered["project"].tolist() == ["q"]
    assert details["removed_group_count"] == 1
