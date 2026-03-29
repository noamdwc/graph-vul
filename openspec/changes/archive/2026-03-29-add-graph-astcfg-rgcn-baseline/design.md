## Context

The project has three training paths: (1) a full graph model using GATv2 on DeBERTa-embedded CPG nodes, (2) a function baseline MLP on `[emb_code || emb_cwe]`, and (3) a UniXcoder fine-tuning model on `[encoder(code) || emb_cwe]`. Paths (2) and (3) ignore all structural information. The full graph model is expensive (DeBERTa per node, 4-layer GAT, mid-path text fusion). There is no middle ground.

Joern CPG exports already exist in `data/raw/{sample_id}/` with node types (LABEL column) and edge types (AST, CFG, PDG). The existing `embed_code.py` parses these but immediately embeds node text with DeBERTa — we need a parallel path that extracts only structural IDs.

Both the func baseline (`FuncBaselineModel` / `FuncDataset` / `train_baseline.py`) and the finetune path (`FuncFinetuneModel` / `FuncFinetuneDataset` / `train_finetune.py` / `colab_finetune.ipynb`) need updating. The change should be backwards-compatible: when graph features are disabled, behavior is identical to today.

## Goals / Non-Goals

**Goals:**
- Add a preprocessing stage that extracts per-function graph artifacts (node type IDs, edge index, edge type IDs) from existing Joern exports, filtering to AST+CFG edges only.
- Add a `FunctionGraphEncoder` (nn.Embedding → 2-layer RGCN → global_mean_pool) producing a fixed-size graph vector.
- Update the existing func baseline model/dataset/training to optionally fuse `[emb_code || emb_graph_ast_cfg || emb_cwe]`.
- Update the existing func finetune model/dataset/training/notebook to do the same.
- Keep all changes backwards-compatible — `--use-graph` flag or config toggle.

**Non-Goals:**
- PDG/dataflow/call graph edges (future work).
- Node text embeddings in the graph encoder (no DeBERTa, no UniXcoder for nodes).
- Target-node logic or node-level classification.
- Hyperparameter tuning — use reasonable defaults, ablate later.
- Creating separate model/dataset/training files — we modify the existing ones.

## Decisions

### D1: Reuse existing Joern CPG exports as input

Parse `data/raw/{sample_id}/` (CSV/JSON/DOT) to extract node types and edge types, reusing the same parsing logic from `embed_code.py`. Filter edges to AST (0) and CFG (1) only, drop PDG (2).

**Rationale:** Avoids a second Joern run. The exports already contain all needed structural information. Filtering to AST+CFG is a simple edge-type check.

**Alternative considered:** Re-run Joern with a custom script that only exports AST+CFG. Rejected — unnecessary overhead, same data is already available.

### D2: Stable vocab from a fixed JSON file

Ship two JSON files (`node_type_vocab.json`, `edge_type_vocab.json`) mapping Joern type strings to integer IDs. The preprocessing script builds these from the full corpus on first run, then they are frozen. Unknown types at inference time get an `<UNK>` ID (index 0).

**Rationale:** A fixed vocab ensures consistent IDs across preprocessing, training, and inference. JSON is human-readable and easy to version.

**Alternative considered:** Build vocab on-the-fly per split. Rejected — IDs would differ between train and test, breaking the embedding table.

### D3: Separate `.pt` graph artifact per sample

Store preprocessed graphs as `data/processed/graph_artifacts/{sample_id}.pt` containing `{"node_type_ids": LongTensor, "edge_index": LongTensor, "edge_type_ids": LongTensor}`. This parallels the existing `code_embeddings/{sample_id}.pt` pattern.

**Rationale:** Matches the project's existing per-sample `.pt` convention. Keeps graph data independent from code embeddings.

### D4: Update `FuncDataset` and `FuncFinetuneDataset` with optional graph loading

Add a `use_graph: bool = False` parameter. When True, `__getitem__` also loads the graph artifact and returns it alongside existing fields. A custom collate function handles batching variable-size graphs via PyG `Batch.from_data_list`.

**Rationale:** Keeps one dataset class per path (no proliferation). The flag makes it trivial to run with or without graph features. The collate function is needed because standard DataLoader can't batch PyG `Data` objects.

For `FuncDataset`, return changes from `(code_emb, cwe_emb, label)` to `(code_emb, cwe_emb, graph_data, label)` when enabled. For `FuncFinetuneDataset`, the dict gains a `"graph_data"` key.

### D5: FunctionGraphEncoder architecture

```
nn.Embedding(num_node_types, node_embed_dim)  # e.g., 64
→ RGCNConv(node_embed_dim, hidden_dim, num_relations)  # layer 1
→ ReLU → Dropout
→ RGCNConv(hidden_dim, hidden_dim, num_relations)  # layer 2
→ ReLU → Dropout
→ global_mean_pool → [batch_size, hidden_dim]  # e.g., 64
```

Use `torch_geometric.nn.RGCNConv` for relation-aware message passing. Two edge types (AST=0, CFG=1) so `num_relations=2`.

**Rationale:** RGCN is the standard choice for typed edges. 2 layers is sufficient for local structural patterns. `global_mean_pool` is simple and invariant to graph size.

**Alternative considered:** GCN ignoring edge types — rejected, edge type distinction is the point. GAT — rejected as overkill for type-only node features.

### D6: Update `FuncBaselineModel` and `FuncFinetuneModel` with optional graph input

Add `graph_dim: int = 0` to `__init__`. When `graph_dim > 0`, the MLP input size becomes `code_dim + graph_dim + cwe_dim` instead of `code_dim + cwe_dim`. The `forward()` method gains an optional `graph_emb` parameter. The `FunctionGraphEncoder` lives inside the model (instantiated when `graph_dim > 0`).

**Rationale:** Minimal change to existing models. When `graph_dim=0`, behavior is identical to current. The graph encoder is owned by the model so it trains end-to-end.

### D7: Update training scripts with `--use-graph` flag

Both `train_baseline.py` and `train_finetune.py` gain a `--use-graph` CLI flag. When set, they: (1) load graph vocab to determine num_node_types/num_edge_types, (2) pass `use_graph=True` to dataset, (3) use the custom collate, (4) instantiate model with `graph_dim > 0`, (5) log graph statistics to WandB.

**Rationale:** Single flag controls the entire graph feature path. Easy to compare runs with and without.

### D8: Update `colab_finetune.ipynb`

Add a `USE_GRAPH` toggle in the config cell. When True, the notebook validates that graph artifacts exist on Drive and passes `--use-graph` to the training call. Update the prerequisites section to document the additional data needed.

**Rationale:** The notebook is the production training path for finetune — it must support the new feature.

### D9: Handle missing graphs gracefully

If a sample's graph artifact `.pt` is missing, substitute a single-node graph with `node_type_id=0` (UNK) and no edges. Log the count of missing/empty graphs at startup.

**Rationale:** Not all samples may have successful Joern exports. A fallback graph ensures the batch doesn't crash. Logging lets us monitor coverage.

## Risks / Trade-offs

- **[Sparse node features]** Using only node type IDs (no text) means the graph encoder sees structure but not semantics. → Intentional for a minimal baseline. Text can be added in a follow-up.
- **[Vocab drift]** If Joern version changes, node type strings may change. → Mitigated by freezing vocab to JSON and using UNK for unknowns.
- **[Missing graph coverage]** Some samples in `func_subset.csv` may not have Joern exports (the graph model uses `subset.csv` which is CWE-119 only; func_subset covers all CWEs). → Preprocessing logs coverage. Samples without graphs get a fallback.
- **[Collate complexity]** Custom collate for PyG batching adds complexity to both dataset paths. → Isolated in one shared utility function. Standard collate used when `use_graph=False`.
- **[RGCN memory on large graphs]** Some functions may produce very large CPGs. → global_mean_pool handles variable sizes. Could add a max-nodes cap later if needed.
