## ADDED Requirements

### Requirement: Raw text dataset overview
The notebook SHALL load `data/subset_cwe119.csv` and display the total number of samples, class distribution (vulnerable vs benign), and a preview of the data columns.

#### Scenario: Dataset overview
- **WHEN** the raw text EDA section is executed
- **THEN** the notebook displays sample count, class counts with percentages, and a DataFrame preview

### Requirement: Code text statistics
The notebook SHALL compute and visualize statistics on the raw source code text, including character length distribution, line count distribution, and token count distribution.

#### Scenario: Code length distributions
- **WHEN** the code statistics cells are executed
- **THEN** histograms are displayed for code character length, line count, and token count, broken down by vulnerability label

### Requirement: CVE description statistics
The notebook SHALL compute and visualize statistics on CVE description text, including character length distribution, word count distribution, and most frequent terms.

#### Scenario: CVE text analysis
- **WHEN** the CVE statistics cells are executed
- **THEN** histograms are displayed for description length and word count, plus a bar chart of top-N most frequent terms

### Requirement: Graph structure statistics
The notebook SHALL load the processed PyG dataset and visualize node count and edge count distributions across all graphs.

#### Scenario: Node and edge distributions
- **WHEN** the graph structure cells are executed
- **THEN** histograms are displayed for nodes per graph and edges per graph, with summary statistics (mean, median, min, max)

### Requirement: Edge type breakdown
The notebook SHALL compute and visualize the distribution of edge types (AST, CFG, PDG) across the dataset.

#### Scenario: Edge type ratios
- **WHEN** the edge type analysis cell is executed
- **THEN** a bar chart or pie chart shows the proportion of each edge type across all graphs

### Requirement: Embedding analysis
The notebook SHALL analyze the DeBERTa node embeddings and CVE text embeddings, including dimensionality verification, value distribution statistics, and a 2D projection (e.g., PCA or t-SNE) colored by vulnerability label.

#### Scenario: Embedding distributions
- **WHEN** the embedding analysis cells are executed
- **THEN** the notebook displays embedding dimensionality, mean/std statistics, and a 2D scatter plot of embeddings colored by label

### Requirement: Data split statistics
The notebook SHALL load the train/val/test splits and display sample counts and class balance for each split.

#### Scenario: Split balance check
- **WHEN** the split statistics cell is executed
- **THEN** a table or grouped bar chart shows sample counts and class ratios for train, val, and test splits

### Requirement: Sample graph visualization
The notebook SHALL render at least one sample CPG as a node-link diagram with edge types color-coded.

#### Scenario: Graph rendering
- **WHEN** the graph visualization cell is executed
- **THEN** a networkx-rendered graph is displayed with nodes labeled and edges colored by type (AST, CFG, PDG)
