## 1. Config & Vocab

- [x] 1.1 Add graph baseline constants to `config.py`: `GRAPH_ARTIFACTS_DIR`, `GRAPH_NODE_VOCAB_PATH`, `GRAPH_EDGE_VOCAB_PATH`, `GRAPH_NODE_EMBED_DIM` (64), `GRAPH_HIDDEN_DIM` (64), `GRAPH_NUM_RGCN_LAYERS` (2), `GRAPH_DROPOUT` (0.3)

## 2. Graph Preprocessing

- [x] 2.1 Create `pipeline/extract_graphs.py` — vocab builder: scan all CPG exports in `data/raw/`, collect unique node LABEL strings and edge type strings (AST+CFG only), write `node_type_vocab.json` and `edge_type_vocab.json` to `GRAPH_ARTIFACTS_DIR` (index 0 = `<UNK>`)
- [x] 2.2 In same script, add graph artifact extraction: parse each sample's CPG export (CSV/JSON/DOT reusing logic from `embed_code.py`), filter to AST+CFG edges, map to vocab IDs, save `{sample_id}.pt` with `node_type_ids`, `edge_index`, `edge_type_ids`
- [x] 2.3 Add skip-if-exists logic and summary statistics logging (num types, avg nodes/edges, missing count)

## 3. Graph Encoder Module

- [x] 3.1 Create `model/graph_encoder.py` with `FunctionGraphEncoder`: `nn.Embedding(num_node_types, node_embed_dim)` → 2× `RGCNConv` layers with ReLU + Dropout → `global_mean_pool` → `[B, graph_dim]`

## 4. Update Datasets

- [x] 4.1 Update `dataset/func_dataset.py`: add `use_graph: bool = False` and `graph_artifacts_dir` params to `FuncDataset.__init__`; when enabled, `__getitem__` loads graph `.pt` and returns `(code_emb, cwe_emb, graph_data, label)`
- [x] 4.2 Add fallback for missing graph artifacts (single UNK node, no edges)
- [x] 4.3 Update `dataset/func_finetune_dataset.py`: add `use_graph` param to `FuncFinetuneDataset`; when enabled, `__getitem__` includes `"graph_data"` in returned dict
- [x] 4.4 Add shared custom collate function (in `dataset/func_dataset.py` or a shared util) for batching graphs via `Batch.from_data_list` alongside stacked tensors; separate collate for tuple-based (baseline) and dict-based (finetune) batches

## 5. Update Models

- [x] 5.1 Update `model/func_classifier.py`: add `graph_dim: int = 0` and graph encoder params to `FuncBaselineModel.__init__`; when `graph_dim > 0`, instantiate `FunctionGraphEncoder`, adjust MLP input dim to `code_dim + graph_dim + cwe_dim`, update `forward()` to accept optional graph batch
- [x] 5.2 Update `model/func_finetune_model.py`: same pattern — add `graph_dim`, instantiate encoder when > 0, adjust head input dim, update `forward()` signature

## 6. Update Lightning Modules

- [x] 6.1 Update `model/func_lit_module.py`: update `_shared_step` to detect graph data in batch (4-tuple vs 3-tuple) and pass to model; pass `graph_dim` through to `FuncBaselineModel`
- [x] 6.2 Update `model/func_finetune_lit_module.py`: update `_shared_step` to pass `batch.get("graph_data")` to model when present; pass `graph_dim` through to `FuncFinetuneModel`

## 7. Update Training Scripts

- [x] 7.1 Update `train_baseline.py`: add `--use-graph` CLI flag; when set, load graph vocab, pass `use_graph=True` to dataset, use custom collate, instantiate model with `graph_dim=GRAPH_HIDDEN_DIM`, log graph stats to WandB config
- [x] 7.2 Update `train_finetune.py`: same `--use-graph` flag pattern; update WandB name/config, dataset, collate, and model instantiation

## 8. Update Colab Notebook

- [x] 8.1 Update `notebooks/colab_finetune.ipynb`: add `USE_GRAPH = False` toggle in config cell; when True, validate graph artifacts on Drive, install PyG deps, pass flag to `train()` call; update prerequisites docs

## 9. Verification

- [x] 9.1 Verify `python -m graph_vul.train_baseline` (without `--use-graph`) still runs unchanged
- [x] 9.2 Verify `python -m graph_vul.train_finetune` (without `--use-graph`) still runs unchanged
- [x] 9.3 Run `python -m graph_vul.pipeline.extract_graphs` to preprocess graph artifacts
- [x] 9.4 Run `python -m graph_vul.train_baseline --use-graph` end-to-end
- [x] 9.5 Run `python -m graph_vul.train_finetune --use-graph` end-to-end
