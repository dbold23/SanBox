"""Deterministic generation of organic splotch shapes.

An identity seed maps to one closed contour. The same seed always produces
the same shape, so a seed *is* the spot's identity.
"""

import numpy as np

# Number of points along the generated contour polyline.
CONTOUR_POINTS = 1024


def _harmonic_blob(rng: np.random.Generator, n_points: int) -> np.ndarray:
    """Star-convex base blob from a random radial harmonic spectrum."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    r = np.ones(n_points)
    for k in range(2, 14):
        amp = rng.uniform(0.02, 0.30) / (k ** 0.7)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        r += amp * np.cos(k * theta + phase)
    r = np.clip(r, 0.25, None)
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


def _smooth_warp(rng: np.random.Generator, pts: np.ndarray) -> np.ndarray:
    """Apply a random smooth displacement field so shapes stop being
    star-convex and gain organic, identity-rich structure.

    Amplitudes are kept small relative to the field wavelength so the warp
    stays a diffeomorphism and the contour cannot self-intersect.
    """
    out = pts.copy()
    for _ in range(3):
        fx = rng.uniform(0.6, 1.6, size=2)
        fy = rng.uniform(0.6, 1.6, size=2)
        px = rng.uniform(0.0, 2.0 * np.pi, size=2)
        py = rng.uniform(0.0, 2.0 * np.pi, size=2)
        ax, ay = rng.uniform(0.05, 0.16, size=2)
        x, y = out[:, 0], out[:, 1]
        dx = ax * np.sin(fx[0] * x + px[0]) * np.cos(fx[1] * y + px[1])
        dy = ay * np.sin(fy[0] * y + py[0]) * np.cos(fy[1] * x + py[1])
        out = out + np.stack([dx, dy], axis=1)
    return out


def generate_identity(seed: int, n_points: int = CONTOUR_POINTS) -> np.ndarray:
    """Return the canonical contour of spot identity ``seed``.

    The contour is an (n_points, 2) float array, centered on its centroid,
    with RMS radius scaled to 1.
    """
    rng = np.random.default_rng(seed)
    pts = _harmonic_blob(rng, n_points)
    pts = _smooth_warp(rng, pts)
    pts -= pts.mean(axis=0)
    rms = np.sqrt((pts ** 2).sum(axis=1).mean())
    return pts / rms
