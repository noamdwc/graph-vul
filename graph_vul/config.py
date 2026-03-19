import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CPG_CACHE_DIR = DATA_DIR / "cpg_cache"
BIGVUL_CSV = DATA_DIR / "bigvul.csv"
SUBSET_CSV = DATA_DIR / "subset_cwe119.csv"
CHECKPOINT_PATH = DATA_DIR / "best_model.ckpt"

# Data pipeline
CWE_ID = "CWE-119"
TARGET_SAMPLES = 1500  # per class (vulnerable / benign)

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
# Joern export
JOERN_WORKERS = min(os.cpu_count() or 1, 4)

SEED = 42
WEIGHT_DECAY = 0.01
LR_SCHEDULER_FACTOR = 0.95
WARMUP_EPOCHS = 5
MIN_LR = 1e-7