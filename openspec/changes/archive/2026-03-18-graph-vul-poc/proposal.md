## Why

Current vulnerability detection in C/C++ codebases relies on static analysis rules or manual code review, both of which miss subtle structural patterns that span data flow, control flow, and abstract syntax. By combining Code Property Graphs (CPGs) with natural-language CVE descriptions in a multi-modal neural architecture, we can learn to recognize vulnerability patterns that neither modality captures alone. This POC validates the approach on a tractable subset (CWE-119 buffer overflows from Big-Vul) before scaling to broader vulnerability classes.

## What Changes

- Introduce a data pipeline that extracts a ~1,000-sample balanced subset of Big-Vul (CWE-119) and uses Joern CLI to export Code Property Graphs as structured files.
- Create a custom PyTorch Geometric Dataset class that parses Joern CPG exports into graph tensors with node features initialized via DeBERTa embeddings of each node's code content.
- Build a dual-branch PyTorch model: a Graph Attention Network (GAT) branch for code structure and a DeBERTa-based branch for CVE text vectorization, fused through an MLP classification head.
- Scaffold a local training loop with Binary Cross-Entropy loss, train/validation split, and basic performance metrics (accuracy, precision, recall, F1).

## Capabilities

### New Capabilities
- `data-pipeline`: Automated extraction of Big-Vul CWE-119 subset and Joern-based CPG generation, producing structured graph exports for downstream consumption.
- `pyg-dataset`: Custom PyTorch Geometric Dataset that loads CPG exports, constructs edge indices and node feature tensors (using DeBERTa to embed each node's code tokens), and pairs each graph with its DeBERTa-encoded CVE text vector.
- `model-architecture`: Dual-branch neural network combining a GAT-based code graph encoder and a DeBERTa text vector ingestion path, fused via an MLP for binary vulnerability classification.
- `training-loop`: Local training scaffold with loss computation, optimizer step, validation evaluation, and metric reporting.

### Modified Capabilities
_(none -- greenfield project)_

## Impact

- **Dependencies**: Requires Joern CLI installed locally, plus Python packages: `torch`, `torch-geometric`, `transformers` (for DeBERTa), `pandas`, `numpy`.
- **Data**: Big-Vul dataset CSV must be available locally; Joern exports will be written to a project-local directory structure.
- **Compute**: Training is local CPU/GPU; DeBERTa inference adds a one-time embedding cost at dataset build time for both node features and CVE text vectors.
- **Code**: All new code -- no existing modules are modified.
