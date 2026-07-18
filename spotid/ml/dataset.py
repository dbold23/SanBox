"""On-the-fly synthetic training data for spot embeddings.

Every sample is rendered fresh from the generator — infinite data, zero
labeling. Identities used for training start at seed TRAIN_SEED_BASE so
the evaluation range (small seeds) is never seen during training.
"""

import cv2
import numpy as np

from ..render import ViewConfig, render_view
from ..shapes import generate_identity

PATCH = 96
TRAIN_SEED_BASE = 1_000_000

# Rendering config for training views: full tilt range, realistic noise.
TRAIN_VIEW = ViewConfig(tilt_max_deg=55)


def extract_patch(img: np.ndarray, center: np.ndarray, radius: float,
                  out: int = PATCH, margin: float = 1.5) -> np.ndarray:
    """Cut a scale-normalized square patch around a spot.

    ``radius`` is the spot's approximate RMS radius in pixels; the window
    spans ``margin`` times that on each side, clamped to the image with
    edge padding, and is resized to ``out`` x ``out`` float32 in [0, 1].
    """
    half = max(margin * radius, 8.0)
    x0, y0 = center[0] - half, center[1] - half
    x1, y1 = center[0] + half, center[1] + half
    h, w = img.shape
    ix0, iy0 = int(np.floor(x0)), int(np.floor(y0))
    ix1, iy1 = int(np.ceil(x1)), int(np.ceil(y1))
    pad_l, pad_t = max(0, -ix0), max(0, -iy0)
    pad_r, pad_b = max(0, ix1 - w), max(0, iy1 - h)
    crop = img[max(0, iy0):min(h, iy1), max(0, ix0):min(w, ix1)]
    if pad_l or pad_t or pad_r or pad_b:
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r,
                                  cv2.BORDER_REPLICATE)
    patch = cv2.resize(crop, (out, out), interpolation=cv2.INTER_AREA)
    return patch.astype(np.float32) / 255.0


def _augment(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Photometric augmentation beyond what the renderer already does:
    gamma, gain/bias, glare-like bright wash, extra noise."""
    p = patch
    p = np.clip(p, 0.0, 1.0) ** rng.uniform(0.7, 1.4)
    p = p * rng.uniform(0.85, 1.1) + rng.uniform(-0.08, 0.08)
    if rng.uniform() < 0.25:  # partial glare wash
        yy, xx = np.mgrid[0:p.shape[0], 0:p.shape[1]] / p.shape[0]
        c = rng.uniform(0.0, 1.0, 2)
        g = np.exp(-(((xx - c[0]) ** 2 + (yy - c[1]) ** 2)
                     / rng.uniform(0.05, 0.3)))
        s = rng.uniform(0.2, 0.55)
        p = p * (1 - s * g) + 0.95 * s * g
    p = p + rng.standard_normal(p.shape).astype(np.float32) * rng.uniform(0, 0.03)
    return np.clip(p, 0.0, 1.0).astype(np.float32)


def render_training_view(identity_seed: int, rng: np.random.Generator,
                         augment: bool = True) -> np.ndarray:
    """One PATCH x PATCH view of the given identity, cropped from a full
    rendered permutation using ground truth (with jitter, so the model
    tolerates imperfect segmentation-based crops at inference)."""
    contour = generate_identity(TRAIN_SEED_BASE + identity_seed
                                if identity_seed < TRAIN_SEED_BASE
                                else identity_seed)
    img, info = render_view(contour, rng, TRAIN_VIEW)
    poly = info["polygon_px"]
    center = poly.mean(axis=0)
    radius = float(np.sqrt(((poly - center) ** 2).sum(axis=1).mean()))
    if augment:
        center = center + rng.uniform(-0.12, 0.12, 2) * radius
        radius = radius * rng.uniform(0.85, 1.25)
    patch = extract_patch(img, center, radius)
    return _augment(patch, rng) if augment else patch


def make_batch(rng: np.random.Generator, n_ids: int, k_views: int,
               id_pool: int):
    """A (n_ids * k_views, 1, PATCH, PATCH) float32 batch plus labels."""
    ids = rng.choice(id_pool, size=n_ids, replace=False)
    xs, ys = [], []
    for label, ident in enumerate(ids):
        for _ in range(k_views):
            xs.append(render_training_view(int(ident), rng))
            ys.append(label)
    x = np.stack(xs)[:, None, :, :]
    return x, np.array(ys, np.int64)
