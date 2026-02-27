"""
config.py — All hyperparameters and paths for Text-to-360° Video GAN.
USER MUST SET DATA_DIR and CSV_PATH before running.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR           = Path("/path/to/360_videos")       # ← USER MUST SET
CSV_PATH           = Path("/path/to/metadata.csv")     # ← USER MUST SET
BACKBONE_LOCAL_DIR = Path("./Wan2.1-T2V-1.3B")        # ← local HF download dir
OUTPUT_DIR         = Path("outputs")
CHECKPOINT_DIR     = Path("checkpoints")

# ── Video specs ────────────────────────────────────────────────────────────
T   = 100
FPS = 20
H   = 512
W   = 1024
C   = 3

# ── Backbone ───────────────────────────────────────────────────────────────
BACKBONE_STEPS    = 50
BACKBONE_GUIDANCE = 5.0

# ── Generator CNN ──────────────────────────────────────────────────────────
CNN_CHANNELS    = 64
DROPOUT_RATE    = 0.3
WRAP_PAD_H      = 30
WRAP_PAD_W      = 50
BOUNDARY_PIXELS = 50

# ── Training ───────────────────────────────────────────────────────────────
BATCH_SIZE        = 2
NUM_EPOCHS        = 100
LR                = 1e-4
N_CRITIC          = 5
LAMBDA_GP         = 10.0
LAMBDA_BOUNDARY   = 10.0
LOG_EVERY_N_STEPS = 50
SEED              = 42

# ── Hardware ───────────────────────────────────────────────────────────────
GPU_IDS = [0, 1, 2, 3]