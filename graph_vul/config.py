from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
BIGVUL_CSV = DATA_DIR / "bigvul.csv"
SUBSET_CSV = DATA_DIR / "subset_cwe119.csv"
CHECKPOINT_PATH = DATA_DIR / "best_model.pt"

# Data pipeline
CWE_ID = "CWE-119"
TARGET_SAMPLES = 500  # per class (vulnerable / benign)

# DeBERTa
DEBERTA_MODEL_NAME = "microsoft/deberta-v3-base"
DEBERTA_EMBEDDING_DIM = 768

# GAT
GAT_HIDDEN_DIM = 128
GAT_HEADS = 4
GAT_LAYERS = 2
NUM_EDGE_TYPES = 3  # AST, CFG, PDG

# Fusion MLP
MLP_HIDDEN_DIM = 256
MLP_DROPOUT = 0.3

# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
VAL_SPLIT = 0.2
SEED = 42
