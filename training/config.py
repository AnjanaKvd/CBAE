# training/config.py
"""
Training configuration constants for CBAE multi-stage training.
Follows CBAE_TRAINING_GUIDE.md Step 1.1.
"""

import torch

# ── Batch & Epoch Settings ──────────────────────────────────────────────────
BATCH_SIZE = 2  # DO NOT increase — checkpointed adjoint limits this
N_EPOCHS_CLEAN = 50
N_EPOCHS_ROBUSTNESS = 30
N_EPOCHS_BRIDGE = 20

# ── Optimizer ───────────────────────────────────────────────────────────────
LR_INIT = 1e-3
LR_MIN = 1e-5
GRAD_CLIP = 1.0
WEIGHT_DECAY = 1e-4

# ── Loss Weights ────────────────────────────────────────────────────────────
LOSS_WEIGHTS = {
    "render": 1.0,
    "smooth": 0.1,
    "alive": 0.5,
    "semantic": 0.1,
    "velocity": 1.0,  # set to 0.0 when ground truth velocity is not available
}

# ── Data Directories ───────────────────────────────────────────────────────
DATA_DIR_CLEAN = "data/synthetic/clean/"
DATA_DIR_ROBUST = "data/synthetic/robustness/"
DATA_DIR_BRIDGE = "data/synthetic/bridge/"
DATA_DIR_MIXED = "data/mixed/"
CHECKPOINT_DIR = "checkpoints/"
SAMPLE_OUTPUT_DIR = "training_samples/"
LOG_FILE = "training_log.csv"

# ── Canvas & Sequence ──────────────────────────────────────────────────────
CANVAS_SIZE = (512, 512)
SEQUENCE_LENGTH = 192  # frames per sequence
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
