"""Segmentation and viewpoint-invariant spot descriptors.

The key idea: an out-of-plane view of a (locally) planar spot differs from
the canonical shape by approximately an affine transform. We therefore

  1. whiten the shape with the inverse square root of its second-moment
     matrix — this cancels the affine part exactly, leaving only an unknown
     rotation, then
  2. describe the whitened contour with rotation-invariant signatures
     (Fourier-descriptor magnitudes + radial histogram), and add Flusser
     affine moment invariants computed on the raw shape.

Two views of the same spot then map to (nearly) the same descriptor.
"""

import cv2
import numpy as np

FD_COEFFS = 24          # Fourier coefficients kept on each side (+/-k)
RESAMPLE_POINTS = 256   # contour resampling density
RADIAL_BINS = 24

# Weights of the descriptor blocks in the final concatenated vector.
BLOCK_WEIGHTS = {"fd": 1.0, "radial": 0.6, "flusser": 0.35}


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def _threshold_dark(img: np.ndarray, scale_frac: float = 1 / 24.0,
                    k_sigma: float = 1.4, floor_frac: float = 1.0,
                    abs_depth: float = 0.05) -> np.ndarray:
    """Binary mask of dark-on-light splotches, robust to smooth gradients,
    vignetting, gamma shifts and glossy sheen.

    Thresholds the high-pass residual with a *local* criterion: a pixel is
    spot if it sits clearly below its neighborhood, where "clearly" scales
    with the local residual spread. ``scale_frac`` sets the neighborhood
    size relative to the image (use larger for images of a single big
    spot). This keeps faded spots while ignoring smooth glare (whose
    residual is flat)."""
    f = img.astype(np.float32) / 255.0
    sigma_px = max(2.0, max(img.shape) * scale_frac)
    smooth = cv2.GaussianBlur(f, (0, 0), 1.2)
    lowpass = cv2.GaussianBlur(f, (0, 0), sigma_px)
    residual = smooth - lowpass
    # Local scale of the residual (robust-ish spread over a mid window).
    local_sq = cv2.GaussianBlur(residual * residual, (0, 0), sigma_px)
    sigma = np.sqrt(np.maximum(local_sq, 1e-8))
    # Global floor stops flat regions from amplifying sensor noise, and the
    # absolute depth floor keeps large empty backgrounds (far-away views)
    # from promoting texture ripples into spots.
    floor = floor_frac * float(np.median(sigma))
    gate = np.maximum(k_sigma * np.maximum(sigma, floor), abs_depth)
    binmask = (residual < -gate).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    binmask = cv2.morphologyEx(binmask, cv2.MORPH_OPEN, kernel)
    binmask = cv2.morphologyEx(binmask, cv2.MORPH_CLOSE, kernel)
    return binmask


def _is_crack(contour: np.ndarray) -> bool:
    """True for scratch/crack-like blobs: extremely thin and elongated.
    Legitimate elongated spots are streaks with real width; cracks are a
    few pixels wide and much longer."""
    (_, _), (w, h) = cv2.minAreaRect(contour.astype(np.float32))[:2]
    w, h = min(w, h), max(w, h)
    if h < 1e-6:
        return False
    return w < 4.0 and h / max(w, 1e-6) > 7.0


def segment_spot(img: np.ndarray) -> np.ndarray | None:
    """Extract the outer contour of the single largest splotch in ``img``.

    Returns an (N, 2) float contour in pixel coordinates, or None.
    """
    # A single spot fills much of the frame: use a wide neighborhood so the
    # spot interior is not absorbed into the local baseline.
    binmask = _threshold_dark(img, scale_frac=0.5)
    contours, _ = cv2.findContours(binmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < 40.0:
        return None
    return best.reshape(-1, 2).astype(np.float64)


def segment_all_spots(img: np.ndarray, min_area: float = 12.0,
                      max_area_fraction: float = 0.02) -> list[np.ndarray]:
    """Outer contours of every splotch in a multi-spot image.

    Filters out crack/scratch-like thin lines and anything larger than
    ``max_area_fraction`` of the image (shadows, image-scale artifacts)."""
    binmask = _threshold_dark(img)
    contours, _ = cv2.findContours(binmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_area = max_area_fraction * img.shape[0] * img.shape[1]
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area and not _is_crack(c):
            out.append(c.reshape(-1, 2).astype(np.float64))
    return out


# ---------------------------------------------------------------------------
# Affine normalization
# ---------------------------------------------------------------------------

def _polygon_moments(contour: np.ndarray) -> dict:
    m = cv2.moments(contour.astype(np.float32), binaryImage=False)
    if abs(m["m00"]) < 1e-9:
        return None
    return m


def _whiten_contour(contour: np.ndarray, m: dict) -> np.ndarray:
    """Map the contour so the filled shape has zero mean and identity
    covariance. The whitening matrix is symmetric positive definite, so no
    reflection is introduced; two affine-related shapes end up identical up
    to rotation."""
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    cov = np.array([[m["mu20"], m["mu11"]], [m["mu11"], m["mu02"]]]) / m["m00"]
    w, v = np.linalg.eigh(cov)
    w = np.clip(w, 1e-9, None)
    inv_sqrt = v @ np.diag(1.0 / np.sqrt(w)) @ v.T
    return (contour - np.array([cx, cy])) @ inv_sqrt.T


def _resample_closed(pts: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed polyline to n points uniformly by arc length."""
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total < 1e-9:
        return np.repeat(pts[:1], n, axis=0)
    t = np.linspace(0.0, total, n, endpoint=False)
    x = np.interp(t, s, closed[:, 0])
    y = np.interp(t, s, closed[:, 1])
    return np.stack([x, y], axis=1)


def _signed_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


# ---------------------------------------------------------------------------
# Descriptor blocks
# ---------------------------------------------------------------------------

def _fourier_block(whitened: np.ndarray) -> np.ndarray:
    pts = _resample_closed(whitened, RESAMPLE_POINTS)
    if _signed_area(pts) < 0:
        pts = pts[::-1]
    z = pts[:, 0] + 1j * pts[:, 1]
    z = z - z.mean()
    spec = np.fft.fft(z)
    # |Z_1| is the dominant, near-identical-for-all-blobs component; use it
    # only as the normalizer and keep the discriminative harmonics.
    ref = max(np.abs(spec[1]), 1e-9)
    ks = list(range(2, FD_COEFFS + 1)) + list(range(-FD_COEFFS, 0))
    mags = np.abs(spec[ks]) / ref
    # Log compression flattens the spectrum decay so higher harmonics count.
    return np.log1p(mags * 20.0)


def _radial_block(whitened: np.ndarray) -> np.ndarray:
    pts = _resample_closed(whitened, RESAMPLE_POINTS)
    r = np.linalg.norm(pts - pts.mean(axis=0), axis=1)
    mean_r = max(r.mean(), 1e-9)
    hist, _ = np.histogram(r / mean_r, bins=RADIAL_BINS, range=(0.0, 3.0))
    hist = hist.astype(np.float64)
    return hist / max(hist.sum(), 1e-9)


def _flusser_block(m: dict) -> np.ndarray:
    """Flusser-Suk affine moment invariants I1..I4 of the raw shape.

    Assumes the contour was oriented so that m00 > 0 (see describe_contour).
    """
    u20, u11, u02 = m["mu20"], m["mu11"], m["mu02"]
    u30, u21, u12, u03 = m["mu30"], m["mu21"], m["mu12"], m["mu03"]
    u00 = m["m00"]
    i1 = (u20 * u02 - u11 ** 2) / u00 ** 4
    i2 = (u30 ** 2 * u03 ** 2 - 6 * u30 * u21 * u12 * u03
          + 4 * u30 * u12 ** 3 + 4 * u21 ** 3 * u03
          - 3 * u21 ** 2 * u12 ** 2) / u00 ** 10
    i3 = (u20 * (u21 * u03 - u12 ** 2) - u11 * (u30 * u03 - u21 * u12)
          + u02 * (u30 * u12 - u21 ** 2)) / u00 ** 7
    i4 = (u20 ** 3 * u03 ** 2 - 6 * u20 ** 2 * u11 * u12 * u03
          - 6 * u20 ** 2 * u02 * u21 * u03 + 9 * u20 ** 2 * u02 * u12 ** 2
          + 12 * u20 * u11 ** 2 * u21 * u03
          + 6 * u20 * u11 * u02 * u30 * u03
          - 18 * u20 * u11 * u02 * u21 * u12
          - 8 * u11 ** 3 * u30 * u03 - 6 * u20 * u02 ** 2 * u30 * u12
          + 9 * u20 * u02 ** 2 * u21 ** 2
          + 12 * u11 ** 2 * u02 * u30 * u12
          - 6 * u11 * u02 ** 2 * u30 * u21 + u02 ** 3 * u30 ** 2) / u00 ** 11
    vals = np.array([i1, i2, i3, i4])
    # Signed log compression: invariants span many orders of magnitude.
    return np.sign(vals) * np.log1p(np.abs(vals) * np.array([1e2, 1e8, 1e5, 1e7]))


def describe_contour(contour: np.ndarray) -> np.ndarray | None:
    """Viewpoint-invariant descriptor of a spot's outer contour (pixel
    coords). Returns a 1-D float vector, or None for degenerate input."""
    if contour is None or len(contour) < 8:
        return None
    if _signed_area(contour) < 0:
        contour = contour[::-1]
    m = _polygon_moments(contour)
    if m is None or m["m00"] <= 0:
        return None
    whitened = _whiten_contour(contour, m)
    blocks = {
        "fd": _fourier_block(whitened),
        "radial": _radial_block(whitened),
        "flusser": _flusser_block(m),
    }
    parts = []
    for name, vec in blocks.items():
        norm = np.linalg.norm(vec)
        parts.append(BLOCK_WEIGHTS[name] * vec / max(norm, 1e-9))
    return np.concatenate(parts)


def describe_image(img: np.ndarray) -> np.ndarray | None:
    """Segment the largest spot in ``img`` and describe it."""
    contour = segment_spot(img)
    if contour is None:
        return None
    return describe_contour(contour)
