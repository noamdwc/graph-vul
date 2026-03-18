# Project Graph-Vul: Local Proof of Concept (POC) Plan

## Overview
This document outlines the first local Proof of Concept (POC) for Project Graph-Vul. The objective is to build a streamlined, end-to-end pipeline that generates latent representations of both source code and known weaknesses, and then classifies their intersection using a multi-modal neural architecture.

## 1. Data Scope & Subsetting
To ensure rapid iteration and feasible local training, the POC will operate on a heavily constrained subset of data.

* **Dataset Source:** Big-Vul. This provides real-world C/C++ functions mapped directly to CVEs.
* **Target Vulnerability:** CWE-119 (Buffer Overflow). This class is selected because its detection relies heavily on clear data and control flow paths, making it ideal for graph-based structural analysis.
* **Data Volume:** Extract a balanced subset of approximately 1,000 samples (500 vulnerable functions + 500 benign counterparts) along with their corresponding CVE description texts.

## 2. Environment & Tooling
The technology stack leverages industry-standard graph extraction and deep learning libraries.

* **Graph Extraction:** Joern (`joern-export` utility) to parse C/C++ into Code Property Graphs (CPGs).
* **Deep Learning Framework:** PyTorch and PyTorch Geometric (PyG) for graph processing and model training.
* **Semantic Embeddings:** Hugging Face `transformers` for generating initial node features and text-based CVE vectors.

## 3. Architecture Pipeline
The POC utilizes a dual-pipeline approach to independently process the code structure and the vulnerability semantics before fusing them for classification.

### Branch A: Code -> Graph -> Code Nodes -> Vector
1. **Code to Graph:** Process the raw C/C++ snippets through Joern to generate unified CPGs containing AST, CFG, and PDG structures. Export these as JSON or GraphML.
2. **Node Initialization:** Parse the graphs into PyTorch Geometric `Data` objects. Initialize each node's feature vector using a lightweight embedding model (e.g., Word2Vec or a small Hugging Face model) based on the node's semantic content (tokens, operations).
3. **Graph Neural Network:** Pass the graph through 2-3 layers of a Graph Attention Network (GAT) to aggregate structural features from neighboring nodes.
4. **Pooling (Code Vector):** Apply a global pooling operation (e.g., Global Mean Pool) to compress the processed graph into a single, fixed-size 1D tensor (`code_vector`).

### Branch B: CVE -> Vector
1. **Text Extraction:** Retrieve the natural language description of the CVE associated with the code snippet.
2. **Vectorization:** Use a pre-trained Sentence Transformer (e.g., `all-MiniLM-L6-v2` via Hugging Face) to embed the text into a dense semantic vector (`cve_vector`).

### Branch C: Multimodal Fusion & Prediction
1. **Fusion:** Concatenate the structural `code_vector` and the semantic `cve_vector` into a single representation.
2. **Classification Head:** Pass the fused vector through a Multi-Layer Perceptron (MLP) consisting of dense layers with Dropout and Leaky ReLU activations.
3. **Output:** A final Sigmoid activation yields a binary prediction (1 = Code contains the specific CVE pattern; 0 = Code is safe).

## 4. Execution Steps
1. **Data Engineering Script:** Write a script (Python/Bash) to automate the Joern CLI, ingesting the 1,000 Big-Vul samples and exporting the CPGs to a local directory.
2. **PyG Dataset Class:** Implement a custom PyTorch Geometric `Dataset` class to load the exported CPGs, tokenize node properties, construct the `edge_index`, and align them with the text-based CVE vectors.
3. **Model Definition:** Build the PyTorch `nn.Module` encapsulating the GAT layers, the text embedding ingestion, and the final MLP fusion head.
4. **Training Loop:** Train locally using Binary Cross-Entropy loss. Monitor validation metrics closely to mitigate overfitting on the small graph subset.