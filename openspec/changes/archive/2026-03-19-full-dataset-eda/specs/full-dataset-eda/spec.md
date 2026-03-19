## ADDED Requirements

### Requirement: Full dataset overview
The notebook SHALL load `data/bigvul.csv` and display the total number of rows, the number of vulnerable vs benign samples, and a preview of available columns.

#### Scenario: Dataset overview
- **WHEN** the dataset overview cell is executed
- **THEN** the notebook displays total row count, vulnerable/benign counts with percentages, and a list of DataFrame columns

### Requirement: CVE distribution analysis
The notebook SHALL compute and display the number of distinct CVE IDs, the distribution of samples per CVE, and a table of the top-30 CVEs by sample count.

#### Scenario: CVE count and distribution
- **WHEN** the CVE analysis cells are executed
- **THEN** the notebook displays the total number of unique CVE IDs, a histogram of samples-per-CVE, and a table listing the top-30 CVEs with their sample counts and vulnerability ratios

### Requirement: CWE distribution analysis
The notebook SHALL compute and display the number of distinct CWE IDs, the distribution of samples per CWE, and a table of the top-20 CWEs by sample count.

#### Scenario: CWE count and distribution
- **WHEN** the CWE analysis cells are executed
- **THEN** the notebook displays the total number of unique CWE IDs, a bar chart of the top-20 CWEs by sample count, and a table with sample counts and class balance per CWE

### Requirement: Class balance per CWE
The notebook SHALL compute and visualize the vulnerable-to-benign ratio for each of the top-20 CWEs.

#### Scenario: Per-CWE class balance
- **WHEN** the class balance cells are executed
- **THEN** the notebook displays a grouped bar chart showing vulnerable and benign counts for each top-20 CWE

### Requirement: Code length statistics across full dataset
The notebook SHALL compute character length, line count, and token count distributions for the function source code across the full dataset.

#### Scenario: Full dataset code statistics
- **WHEN** the code statistics cells are executed
- **THEN** histograms are displayed for character length, line count, and token count, broken down by vulnerability label

### Requirement: CVE description statistics across full dataset
The notebook SHALL compute character length and word count distributions for CVE description text across the full dataset.

#### Scenario: Full dataset CVE text statistics
- **WHEN** the CVE text statistics cells are executed
- **THEN** histograms are displayed for description character length and word count

### Requirement: Expansion candidate recommendations
The notebook SHALL identify CWEs that are candidates for POC expansion based on having at least 200 vulnerable samples and a minimum 30% benign representation, and present them in a ranked table.

#### Scenario: Candidate CWE table
- **WHEN** the recommendations cell is executed
- **THEN** a table is displayed listing CWEs that meet the minimum data thresholds, sorted by total sample count descending, with columns for CWE ID, total samples, vulnerable count, benign count, and balance ratio
