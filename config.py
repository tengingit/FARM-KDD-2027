from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"
OUTPUT_DIR = ROOT / "results"
NUM_FOLDS = 10
DEVICE = "cuda"
SEED = 0

EPOCHS = 500
WARMUP_EPOCHS = 100
BATCH_SIZE = 512
HIDDEN_SIZE = 512
EMBEDDING_SIZE = 512
LEARNING_RATE = 0.1
SGD_MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
PROTOTYPE_MOMENTUM = 0.9
PROTOTYPE_WEIGHT = 0.1
LSE_TEMPERATURE = 5.0

# Zero-based dimension and class IDs. The exclusions apply only to worst-class metrics.
CLASS_EXCLUSIONS = {
    "Adult": {0: {6}, 1: {3, 5}, 2: {3, 4}},
    "BeLaE": {},
    "CoIL2000": {2: {5}, 3: {3}},
    "Edm": {1: {2}},
    "Enb": {},
    "Thyroid": {0: {2, 3}, 1: {3}, 2: {2}, 4: {1, 2}, 5: {1, 2, 3}},
}

# Thyroid uses the third feature matrix in its MATLAB data struct.
FEATURE_FIELD = {"Thyroid": 2}
