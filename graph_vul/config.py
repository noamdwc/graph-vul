import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CPG_CACHE_DIR = DATA_DIR / "cpg_cache"
BIGVUL_CSV = DATA_DIR / "bigvul.csv"
TRAINING_DATASET_NAME = "primevul"
PRIMEVUL_DATASET_NAME = "colin/PrimeVul"
PRIMEVUL_DATASET_CONFIG = "default"
SUBSET_CSV = DATA_DIR / "subset.csv"
SUBSET_MANIFEST = DATA_DIR / "subset_manifest.json"
SUBSET_REPORT = DATA_DIR / "subset_report.md"
CHECKPOINT_PATH = DATA_DIR / "best_model.ckpt"

# Data pipeline
CWE_IDS = ["CWE-119", "CWE-125", "CWE-20", "CWE-476", "CWE-190", "CWE-787"]
HELDOUT_CWE_IDS = ["CWE-200", "CWE-416"]

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
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10
MIN_SPLIT_ROWS = 25
MIN_SPLIT_FRAC = 0.05
SPLIT_MAJOR_CWE_TOP_K = 5
SPLITS_DIR = DATA_DIR / "splits"
SPLIT_ASSIGNMENTS = SPLITS_DIR / "assignments.csv"
SPLIT_REPORT = SPLITS_DIR / "split_report.md"

# Held-out CWE eligibility thresholds
MIN_HELDOUT_TOTAL_ROWS = 30
MIN_HELDOUT_VUL_ROWS = 5
MIN_HELDOUT_PROJECTS = 2

# Unseen split ratio
UNSEEN_VAL_FRAC = 0.5

# Function length partition threshold
SHORT_FUNCTION_MAX_TOKENS = 64

# Fixed token-count bucket boundaries for evaluation
TOKEN_BUCKET_EDGES = [0, 64, 128, 256, 512]
TOKEN_BUCKET_LABELS = ["0-64", "64-128", "128-256", "256-512", "512+"]

# Minimum short-function (0-64 token) counts for evaluation splits
MIN_SHORT_VUL_PER_EVAL_SPLIT = 15
MIN_SHORT_BENIGN_PER_EVAL_SPLIT = 30

# Greedy allocator penalty weight for short-bucket deficit
SHORT_BUCKET_DEFICIT_WEIGHT = 15.0

# Train balancing
BALANCE_TARGET_VUL_RATIO = 0.25
BALANCE_CWE_SMOOTHING = 0.5
BALANCE_PROJECT_DOWNWEIGHT = 0.3
BALANCE_MAX_WEIGHT = 10.0
BALANCE_MAX_PROJECT_SHARE = 0.15
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

# Function baseline subset extraction
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

# Graph baseline (RGCN on AST+CFG)
GRAPH_ARTIFACTS_DIR = PROCESSED_DIR / "graph_artifacts"
GRAPH_NODE_VOCAB_PATH = GRAPH_ARTIFACTS_DIR / "node_type_vocab.json"
GRAPH_EDGE_VOCAB_PATH = GRAPH_ARTIFACTS_DIR / "edge_type_vocab.json"
GRAPH_NODE_EMBED_DIM = 64
GRAPH_HIDDEN_DIM = 64
GRAPH_NUM_RGCN_LAYERS = 2
GRAPH_DROPOUT = 0.3

# CWE matching supervision ablation
MATCHING_ABLATION_MODE = "E2"
MATCHING_PROJECTION_DIM = 128
MATCHING_NEGATIVES_PER_POSITIVE = 1
MATCHING_LOSS_LAMBDA = 0.1
MATCHING_TRIPLET_MARGIN = 0.2
MATCHING_LENGTH_BUCKET_EDGES = [0, 64, 128, 256]
MATCHING_LENGTH_BUCKET_LABELS = ["0-64", "64-128", "128-256", "256+"]
