## 1. Setup

- [x] 1.1 Add `jupyter`, `matplotlib`, `seaborn`, `networkx` to dev/optional dependencies in `requirements.txt`
- [x] 1.2 Create `notebooks/` directory and empty `notebooks/eda.ipynb`

## 2. Raw Text EDA

- [x] 2.1 Add cells to load `data/subset_cwe119.csv` and display dataset overview (sample count, class distribution, DataFrame preview)
- [x] 2.2 Add cells for code text statistics: character length, line count, and token count distributions with histograms by label
- [x] 2.3 Add cells for CVE description statistics: length distribution, word count distribution, and top-N frequent terms bar chart

## 3. Processed Data EDA

- [x] 3.1 Add cells to load the PyG `CPGDataset` and display graph structure statistics (node/edge count distributions with summary stats)
- [x] 3.2 Add cells for edge type breakdown (AST/CFG/PDG proportions) with visualization
- [x] 3.3 Add cells for embedding analysis: dimensionality checks, value distribution stats, and 2D PCA/t-SNE scatter plot colored by label

## 4. Splits and Visualization

- [x] 4.1 Add cells to load train/val/test splits and display sample counts and class balance per split
- [x] 4.2 Add cells to render a sample CPG as a node-link diagram with edges colored by type
