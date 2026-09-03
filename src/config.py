"""Project-wide constants for stage 6.1 onwards.

Values come from Chapter 6 of the approved Phase-I report. Changing anything
here changes every notebook, so treat it as the single source of truth.
"""
from pathlib import Path

SEED = 42

# --- Stage 6.1 preprocessing -------------------------------------------------
SPACING_MM = (1.5, 1.5, 2.0)      # target voxel spacing, report Chapter 6.1
HU_MIN, HU_MAX = -100.0, 240.0    # pancreatic soft-tissue window -> [0, 1]
ORIENTATION = "RAS"

# --- Hardware envelope (RTX 4060, 8 GB VRAM) ---------------------------------
PATCH_SIZE = (96, 96, 96)
BATCH_SIZE = 2
CACHE_RATE = 0.3
NUM_WORKERS = 4

def parse_patch(text):
    """CLI helper: '96' -> (96, 96, 96); '64,64,32' -> (64, 64, 32).

    Both stage 6.2 entry points use this — a training run and its evaluation must
    agree on the patch size, so there is exactly one implementation.
    """
    parts = [int(x) for x in str(text).split(",")]
    return tuple(parts * 3) if len(parts) == 1 else tuple(parts)


# --- Labels in MSD Task07 Pancreas -------------------------------------------
LABEL_BACKGROUND = 0
LABEL_PANCREAS = 1
LABEL_TUMOR = 2

# --- Clinical threshold the project targets ----------------------------------
SMALL_TUMOR_CM = 2.0              # "sub-2 cm" bucket, equivalent spherical diameter

# --- Paths (relative to the repo root; data/ and checkpoints/ are gitignored) -
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
MSD_DIR = DATA_DIR / "Task07_Pancreas"
PANTS_DIR = DATA_DIR / "PanTS"
RESULTS_DIR = REPO_ROOT / "results"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
