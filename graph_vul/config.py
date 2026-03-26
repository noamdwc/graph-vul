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

# Subset extraction
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
BATCH_SIZE = 8
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

EARLY_STOPPING_PATIENCE = 7
EARLY_STOPPING_MIN_DELTA = 0.005

SEED = 42
WEIGHT_DECAY = 0.01
LR_SCHEDULER_FACTOR = 0.95
WARMUP_EPOCHS = 5
MIN_LR = 1e-8

# UniXcoder
UNIXCODER_MODEL_NAME = "microsoft/unixcoder-base"
UNIXCODER_EMBEDDING_DIM = 768
FUNC_MAX_WHITESPACE_TOKENS = 512  # filter functions >= 513

# Function baseline paths
FUNC_SUBSET_CSV = DATA_DIR / "func_subset.csv"
FUNC_SUBSET_MANIFEST = DATA_DIR / "func_subset_manifest.json"
FUNC_SUBSET_REPORT = DATA_DIR / "func_subset_report.md"
FUNC_EMBEDDINGS_DIR = PROCESSED_DIR / "func_embeddings"
FUNC_SPLITS_DIR = DATA_DIR / "func_splits"
FUNC_CHECKPOINT_PATH = DATA_DIR / "func_baseline_model.ckpt"

# Function baseline sampling
FUNC_MIN_PROJECT_ROWS = 50
FUNC_MAX_PROJECT_ROWS = 2000

# Function baseline MLP
FUNC_MLP_HIDDEN_DIM = 128
FUNC_MLP_DROPOUT = 0.3

# Function baseline training
FUNC_BATCH_SIZE = 64
FUNC_LEARNING_RATE = 1e-3
FUNC_NUM_EPOCHS = 50

# UniXcoder fine-tuning
FINETUNE_ENCODER_LR = 2e-5
FINETUNE_HEAD_LR = 1e-3
FINETUNE_BATCH_SIZE = 8
FINETUNE_GRAD_ACCUM_STEPS = 8  # effective batch size = 64
FINETUNE_NUM_EPOCHS = 10
FINETUNE_CHECKPOINT_PATH = DATA_DIR / "func_finetune_model.ckpt"