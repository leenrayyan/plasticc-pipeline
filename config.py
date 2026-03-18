"""
config.py — Central configuration: all paths and hyperparameters.

No other module should hard-code paths or magic numbers.
Import from here instead.
"""

import os

# Windows fix: bypass XetHub CDN (causes DNS failures on some networks)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# Suppress symlink warning on Windows (non-admin)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Paths — overridden dynamically in colab_setup.py for Colab environments
# ---------------------------------------------------------------------------
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "data")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")
FIG_DIR        = os.path.join(RESULTS_DIR, "figures")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

TRAINING_SET_PATH = os.path.join(DATA_DIR, "training_set.csv")
METADATA_PATH     = os.path.join(DATA_DIR, "training_set_metadata.csv")

# Ensure output directories exist at import time
for _d in [RESULTS_DIR, FIG_DIR, CHECKPOINT_DIR]:
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset parameters
# ---------------------------------------------------------------------------
TRUNCATION_FRACTIONS = [0.1, 0.3, 0.5, 1.0]   # early-observation fractions

# Pad / truncate all sequences to this fixed length for batching.
# 256 covers ~95 % of PLAsTiCC objects without excessive padding.
MAX_SEQ_LEN = 256

N_PASSBANDS = 6   # PLAsTiCC bands: u(0), g(1), r(2), i(3), z(4), Y(5)

# Feature vector per time-step fed to deep models:
#   [flux_norm, flux_err, time_delta_days, passband_norm]
# The spec specifies 3 features; we include normalised passband as the 4th
# because multi-band structure is critical for transient classification.
INPUT_DIM = 4

TEST_SIZE = 0.2   # fraction of objects held out for testing (stratified)

# PLAsTiCC integer target → human-readable class name
PLASTICC_CLASSES = {
     6: "µ-Lens-Single",
    15: "TDE",
    16: "EBE",
    42: "SNII",
    52: "SNIax",
    53: "Mira",
    62: "SNIbc",
    64: "KN",
    65: "M-dwarf",
    67: "SNIa-91bg",
    88: "AGN",
    90: "SNIa",
    92: "RRL",
    95: "SLSN-I",
}

# Rare classes driving the prioritization experiment (TDE=15, Kilonova=64)
RARE_CLASSES = [15, 64]

# ---------------------------------------------------------------------------
# Transformer hyperparameters (Model 2)
# ---------------------------------------------------------------------------
D_MODEL          = 64
N_HEADS          = 4
N_ENCODER_LAYERS = 2
DIM_FEEDFORWARD  = 256
DROPOUT          = 0.1
MAX_WAVELENGTH   = 10_000.0   # denominator in sinusoidal time encoding

# ---------------------------------------------------------------------------
# Training hyperparameters (Models 2, 3, 4)
# ---------------------------------------------------------------------------
BATCH_SIZE              = 64
LEARNING_RATE           = 1e-4
MAX_EPOCHS              = 50
EARLY_STOPPING_PATIENCE = 5
LR_WARMUP_EPOCHS        = 3    # linear warm-up epochs before cosine decay

# ---------------------------------------------------------------------------
# MC Dropout inference
# ---------------------------------------------------------------------------
MC_SAMPLES = 50   # stochastic forward passes per input

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
N_BINS = 10   # reliability-diagram confidence buckets

# ---------------------------------------------------------------------------
# Prioritization simulation
# ---------------------------------------------------------------------------
BUDGET_RANGE = list(range(10, 501, 10))   # K ∈ {10, 20, …, 500}

# ---------------------------------------------------------------------------
# XGBoost (Model 1)
# ---------------------------------------------------------------------------
XGB_PARAMS = {
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "eval_metric":      "mlogloss",
    "random_state":     SEED,
    "n_jobs":           -1,
    "tree_method":      "hist",   # GPU-friendly
}

# ---------------------------------------------------------------------------
# Pretrained model identifiers (Models 3 and 4)
# ---------------------------------------------------------------------------
ASTROMER_WEIGHTS = os.path.join(BASE_DIR, "astromer_weights")  # local cache

MOIRAI_MODEL_ID  = "Salesforce/moirai-1.0-R-small"
CHRONOS_MODEL_ID = "amazon/chronos-t5-small"
