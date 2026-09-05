"""
Central configuration for the holography -> segmentation pipeline.

Every script imports CONFIG from here. Override any value with an
environment variable of the same name, e.g.

    HOLO_WAVELENGTH=0.532 HOLO_DEPTH_MAX=3000 python Dataprep.py --raw

Paths are resolved relative to the repo root unless given as absolute.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("HOLO_DATA_ROOT", REPO_ROOT / "data"))


def _env(name, default, cast):
    """Read HOLO_<name> from the environment, falling back to default."""
    raw = os.environ.get(f"HOLO_{name}")
    return cast(raw) if raw is not None else default


CONFIG = {
    # ---- Optics ---------------------------------------------------------
    # Units are microns throughout. Defaults match a 650 nm laser diode on
    # a Raspberry Pi HQ sensor (IMX477, 1.55 um pitch) with the lens removed.
    "WAVELENGTH": _env("WAVELENGTH", 0.650, float),
    "PIXEL_SIZE": _env("PIXEL_SIZE", 1.55, float),

    # Depth scan for autofocus. Sign convention: positive depth propagates
    # the field AWAY from the sensor plane; for an in-line hologram the
    # object sits on the negative side, so scan negative depths to
    # back-propagate to it. Scan both signs if you are unsure of your setup.
    "DEPTH_MIN": _env("DEPTH_MIN", -3000.0, float),
    "DEPTH_MAX": _env("DEPTH_MAX", -200.0, float),
    "DEPTH_STEP": _env("DEPTH_STEP", 50.0, float),

    # Divide the hologram by its mean before propagating so the DC term
    # does not dominate the reconstruction. Set to False for raw intensity.
    "NORMALIZE_HOLOGRAM": _env("NORMALIZE_HOLOGRAM", True, lambda s: s.lower() in ("1", "true", "yes")),

    # ---- Image geometry ---------------------------------------------------
    "IMAGE_WIDTH": _env("IMAGE_WIDTH", 512, int),
    "IMAGE_HEIGHT": _env("IMAGE_HEIGHT", 512, int),
    "IMAGE_CHANNELS": 1,

    # ---- Directories -----------------------------------------------------
    "RAW_HOLOGRAM_DIR": DATA_ROOT / "raw_holograms",
    "RECONSTRUCTED_DIR": DATA_ROOT / "reconstructed",
    "TRAIN_DIR": DATA_ROOT / "train",          # contains images/ and masks/
    "VALIDATION_DIR": DATA_ROOT / "validation",  # contains images/ and masks/
    "MODEL_DIR": REPO_ROOT / "models",

    # ---- Training ----------------------------------------------------------
    "BATCH_SIZE": _env("BATCH_SIZE", 4, int),
    "EPOCHS": _env("EPOCHS", 100, int),
    "LEARNING_RATE": _env("LEARNING_RATE", 1e-4, float),
    "EARLY_STOPPING_PATIENCE": _env("EARLY_STOPPING_PATIENCE", 15, int),
    "REDUCE_LR_FACTOR": _env("REDUCE_LR_FACTOR", 0.5, float),
    "REDUCE_LR_PATIENCE": _env("REDUCE_LR_PATIENCE", 5, int),
}


def ensure_dirs():
    """Create every data/model directory the pipeline writes to."""
    for key in ("RAW_HOLOGRAM_DIR", "RECONSTRUCTED_DIR", "MODEL_DIR"):
        Path(CONFIG[key]).mkdir(parents=True, exist_ok=True)
    for split in ("TRAIN_DIR", "VALIDATION_DIR"):
        for sub in ("images", "masks"):
            (Path(CONFIG[split]) / sub).mkdir(parents=True, exist_ok=True)
