import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CPG_CACHE_DIR = DATA_DIR / "cpg_cache"
BIGVUL_CSV = DATA_DIR / "bigvul.csv"
SUBSET_CSV = DATA_DIR / "subset.csv"
SUBSET_MANIFEST = DATA_DIR / "subset_manifest.json"
SUBSET_REPORT = DATA_DIR / "subset_report.md"
CHECKPOINT_PATH = DATA_DIR / "best_model.ckpt"

# Data pipeline
CWE_ID = "CWE-119"

# Subset sampling
TARGET_VULNERABLE_SAMPLES = 1000
TARGET_BENIGN_SAMPLES = 2000  # 2:1 benign oversampling for training
MIN_CVES = 7
MAX_CVES = 12
MAX_ROWS_PER_PROJECT = 500
MAX_ROWS_PER_CVE = 500
MAX_ROWS_PER_GROUP = 300  # per project+CVE combination
DEDUPLICATE_CODE = True

# DeBERTa
DEBERTA_MODEL_NAME = "microsoft/deberta-v3-base"
DEBERTA_EMBEDDING_DIM = 768

# GAT
GAT_HIDDEN_DIM = 128
GAT_HEADS = 4
GAT_LAYERS = 4
NUM_EDGE_TYPES = 3  # AST, CFG, PDG
FUSION_LAYER = 2

# Fusion MLP
MLP_HIDDEN_DIM = 256
MLP_DROPOUT = 0.3

# Node classifier
NODE_CLASSIFIER_HIDDEN_DIM = 128

# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SPLITS_DIR = DATA_DIR / "splits"
SPLIT_ASSIGNMENTS = SPLITS_DIR / "assignments.csv"
SPLIT_REPORT = SPLITS_DIR / "split_report.md"
HELD_OUT_PROJECT_COUNT = 3
# Joern export
JOERN_WORKERS = min(os.cpu_count() or 1, 4)

SEED = 42
WEIGHT_DECAY = 0.01
LR_SCHEDULER_FACTOR = 0.95
WARMUP_EPOCHS = 5
MIN_LR = 1e-7