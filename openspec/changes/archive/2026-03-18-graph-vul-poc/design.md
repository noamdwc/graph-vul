## Context

This is a greenfield POC for multi-modal vulnerability detection. The system ingests C/C++ source functions from the Big-Vul dataset, converts them to Code Property Graphs via Joern, and trains a neural classifier that fuses graph-structural features with natural-language CVE descriptions. The scope is deliberately narrow: CWE-119 (buffer overflow), ~1,000 samples, local training only.

No existing codebase or infrastructure exists. The project starts from an empty repository with only a POC plan document.

## Goals / Non-Goals

**Goals:**
- Validate that a GAT + DeBERTa fusion architecture can learn to distinguish vulnerable from benign C/C++ functions given their CPG structure and associated CVE text.
- Produce a reproducible, single-machine pipeline from raw Big-Vul CSV to trained model with logged metrics.
- Keep the codebase modular enough that individual components (graph extraction, dataset, model, training) can be iterated independently.

**Non-Goals:**
- Production-grade performance, scalability, or deployment.
- Support for vulnerability classes beyond CWE-119.
- Fine-tuning DeBERTa end-to-end (embeddings are frozen/pre-computed).
- Hyperparameter search or architecture optimization.
- CI/CD, containerization, or cloud training.

## Decisions

### 1. DeBERTa for both code node embeddings and CVE text embeddings

**Choice:** Use `microsoft/deberta-v3-base` to embed both CPG node content (code tokens, operator names) and CVE description text into the same latent space.

**Why over alternatives:**
- *vs. Word2Vec/GloVe for nodes*: DeBERTa captures sub-word and contextual semantics for code tokens (e.g., `memcpy`, `buf_len`) far better than static embeddings, at acceptable cost for a 1K-sample dataset.
- *vs. SentenceTransformers for text*: Using a single model family for both modalities keeps the embedding space consistent and reduces dependency surface. DeBERTa's disentangled attention is well-suited to both code and natural language.
- *vs. CodeBERT*: CodeBERT is more specialized for code but DeBERTa-v3-base outperforms it on general NLU benchmarks and the POC benefits from a model that handles both code tokens and English CVE descriptions in a unified way.

**Approach:** Pre-compute all embeddings at dataset build time. Extract the `[CLS]` token representation (or mean-pool over tokens) from DeBERTa. Freeze the model — no gradient flows through it during training.

### 2. Joern CLI for CPG extraction with JSON export

**Choice:** Use `joern-export --repr cpg14 --out <dir>` to produce per-function CPG files.

**Why:** Joern is the de facto standard for C/C++ CPG generation. JSON export is directly parseable in Python. The CLI approach avoids JVM API complexity for a POC.

**Alternative considered:** Tree-sitter (AST only, no data/control flow edges — insufficient for vulnerability detection).

### 3. PyG Data objects with heterogeneous edge types

**Choice:** Store AST, CFG, and PDG edges in a single PyG `Data` object using a unified `edge_index` with an `edge_type` attribute tensor.

**Why over separate graphs:** A single graph preserves cross-structure information flow through the GAT attention mechanism. Edge-type-aware attention (via edge attributes) lets the model learn which structural relationships matter for vulnerability detection.

### 4. GAT with 2 layers, 4 attention heads

**Choice:** 2-layer GATv2 with 4 heads, hidden dimension 128, followed by global mean pooling.

**Why:** 2 layers provide a 2-hop receptive field sufficient for local vulnerability patterns (e.g., unchecked buffer access). GATv2's dynamic attention is better than GAT for heterogeneous node types. 4 heads balance expressivity against overfitting on 1K samples. Global mean pool is simple and robust for variable-size graphs.

### 5. MLP fusion head

**Choice:** Concatenate the pooled code vector (128-d) and the CVE text vector (768-d from DeBERTa) into a 896-d vector, then pass through a 2-layer MLP (896 → 256 → 1) with dropout (0.3) and LeakyReLU.

**Why:** Simple concatenation + MLP is the standard baseline for multi-modal fusion. More complex fusion (cross-attention, gating) is premature for a POC.

### 6. Project structure

```
graph_vul/
├── data/
│   ├── raw/              # Big-Vul CSV, Joern CPG exports
│   └── processed/        # Pre-computed PyG Data objects + embeddings
├── pipeline/
│   ├── extract_subset.py  # Big-Vul subsetting
│   └── joern_export.py    # Joern CLI automation
├── dataset/
│   └── cpg_dataset.py     # PyG Dataset class
├── model/
│   ├── gat_encoder.py     # GAT branch
│   ├── text_encoder.py    # DeBERTa embedding wrapper
│   └── classifier.py      # Fusion MLP + full model
├── train.py               # Training loop
└── config.py              # Hyperparameters and paths
```

## Risks / Trade-offs

- **[Small dataset]** 1K samples is very small for deep learning. → Mitigation: Frozen DeBERTa embeddings (no fine-tuning), dropout, and low model capacity reduce overfitting risk. The POC is a feasibility check, not a benchmark.
- **[Joern failure on malformed code]** Some Big-Vul snippets may not parse. → Mitigation: Graceful skip with logging; aim for ≥800 usable samples from the 1K subset.
- **[DeBERTa memory for node embedding]** Large graphs (100+ nodes) generate many forward passes. → Mitigation: Batch node texts per graph; pre-compute once and cache to disk.
- **[Class imbalance in practice]** Even with balanced sampling, some CVE descriptions may be vague or duplicated. → Mitigation: Stratified train/val split; monitor per-class metrics.
