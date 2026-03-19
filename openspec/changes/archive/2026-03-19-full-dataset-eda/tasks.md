## 1. Notebook Setup

- [x] 1.1 Create `notebooks/full_dataset_eda.ipynb` with title, imports (pandas, matplotlib, seaborn, graph_vul.config), and a prerequisite note about running `dvc repro data_ingestion`
- [x] 1.2 Load `data/bigvul.csv` via `BIGVUL_CSV` config path, inspect columns, and fail early if expected columns are missing

## 2. Dataset Overview

- [x] 2.1 Display total row count, vulnerable vs benign counts with percentages, and DataFrame column list
- [x] 2.2 Show a preview of the first few rows

## 3. CVE Distribution Analysis

- [x] 3.1 Compute and display the number of unique CVE IDs
- [x] 3.2 Plot histogram of samples-per-CVE distribution
- [x] 3.3 Display top-30 CVEs table with sample counts and vulnerability ratios

## 4. CWE Distribution Analysis

- [x] 4.1 Compute and display the number of unique CWE IDs
- [x] 4.2 Plot bar chart of top-20 CWEs by sample count
- [x] 4.3 Display table with sample counts and class balance per CWE
- [x] 4.4 Plot grouped bar chart of vulnerable vs benign counts for top-20 CWEs

## 5. Code and Text Statistics

- [x] 5.1 Compute and plot character length, line count, and token count distributions for function source code, broken down by vulnerability label
- [x] 5.2 Compute and plot character length and word count distributions for CVE descriptions

## 6. Expansion Recommendations

- [x] 6.1 Identify candidate CWEs with >=200 vulnerable samples and >=30% benign representation
- [x] 6.2 Display ranked table of candidate CWEs sorted by total sample count with columns: CWE ID, total samples, vulnerable count, benign count, balance ratio
