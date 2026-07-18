"""Surfaces: many spots laid out on a plane, and angled views of them.

A *surface identity* is a seed that deterministically produces N spots
(each with its own shape identity, position, size and orientation) placed
without overlap on the unit plane. The surface's spot constellation is its
fingerprint, like the spot pattern on a whale shark's flank.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .render import ViewConfig, _rot2, _background, project_plane_points, \
    rasterize_polygons
from .shapes import generate_identity

# Spot contours on surfaces use fewer points: hundreds of polygons per view.
SURFACE_CONTOUR_POINTS = 256


@dataclass
class SurfaceSpot:
    """One spot as it lives on its surface (canonical layout)."""
    index: int              # index of the spot on its surface
    shape_seed: int         # identity of the splotch shape
    position: np.ndarray    # (2,) center in surface coords, roughly [-1, 1]
    radius: float           # RMS radius in surface coords
    angle: float            # orientation of the shape on the surface


class Surface:
    def __init__(self, surface_id: int, spots: list[SurfaceSpot]):
        self.surface_id = surface_id
        self.spots = spots
        self.positions = np.array([s.position for s in spots])

    def spot_contour(self, i: int) -> np.ndarray:
        """Contour of spot i in surface coordinates."""
        s = self.spots[i]
        base = generate_identity(s.shape_seed, SURFACE_CONTOUR_POINTS)
        return s.position + (base @ _rot2(s.angle).T) * s.radius


def generate_surface(
    surface_id: int,
    n_spots: int = 600,
    radius_range: tuple = (0.014, 0.034),
    min_gap: float = 1.35,
) -> Surface:
    """Deterministically generate a surface with ``n_spots`` spots.

    Spot shape seeds are derived from (surface_id, index) so every surface
    carries its own unique set of splotch shapes. Placement is rejection
    sampling with a spatial grid; ``min_gap`` scales the required
    center-to-center distance relative to the two spots' radii.
    """
    rng = np.random.default_rng(np.random.SeedSequence([77_000_017, surface_id]))
    spots: list[SurfaceSpot] = []
    placed = np.zeros((0, 3))  # x, y, radius
    grid: dict[tuple, list[int]] = {}
    cell = 2.0 * radius_range[1] * min_gap

    def cell_of(p):
        return (int(p[0] // cell), int(p[1] // cell))

    attempts = 0
    max_attempts = n_spots * 400
    while len(spots) < n_spots and attempts < max_attempts:
        attempts += 1
        pos = rng.uniform(-1.0, 1.0, size=2)
        radius = rng.uniform(*radius_range)
        cx, cy = cell_of(pos)
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((cx + dx, cy + dy), []):
                    other = placed[j]
                    limit = min_gap * (radius + other[2])
                    if np.hypot(pos[0] - other[0], pos[1] - other[1]) < limit:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            continue
        idx = len(spots)
        spots.append(SurfaceSpot(
            index=idx,
            shape_seed=int(rng.integers(0, 2**31 - 1)),
            position=pos.copy(),
            radius=radius,
            angle=rng.uniform(0.0, 2.0 * np.pi),
        ))
        placed = np.vstack([placed, [pos[0], pos[1], radius]])
        grid.setdefault((cx, cy), []).append(idx)
    if len(spots) < n_spots:
        raise RuntimeError(
            f"placed only {len(spots)}/{n_spots} spots; lower density")
    return Surface(surface_id, spots)


@dataclass
class SurfaceViewConfig:
    img_size: int = 1600
    tilt_max_deg: float = 50.0
    camera_distance: float = 8.0   # in units of surface half-extent
    contrast_range: tuple = (0.3, 0.55)
    noise_sigma_range: tuple = (0.0, 0.02)
    blur_sigma_range: tuple = (0.0, 0.8)
    # Fraction of the frame the projected surface spans.
    fill_range: tuple = (0.82, 0.95)


def render_surface_view(
    surface: Surface,
    rng: np.random.Generator,
    cfg: SurfaceViewConfig = SurfaceViewConfig(),
    tilt_deg: float | None = None,
):
    """Render the surface from a random viewpoint.

    Returns (image uint8, info). info["spot_centroids_px"] holds the
    ground-truth projected center of every spot (for evaluation);
    info["view"] the pose parameters.
    """
    size = cfg.img_size
    rotation = rng.uniform(0.0, 2.0 * np.pi)
    tilt = np.deg2rad(tilt_deg if tilt_deg is not None
                      else rng.uniform(0.0, cfg.tilt_max_deg))
    tilt_axis = rng.uniform(0.0, 2.0 * np.pi)
    roll = rng.uniform(0.0, 2.0 * np.pi)
    rot = _rot2(rotation)

    def proj(pts):
        return project_plane_points(pts @ rot.T, tilt, tilt_axis, roll,
                                    cfg.camera_distance)

    corners = proj(np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float))
    extent = np.abs(corners).max()
    frac = rng.uniform(*cfg.fill_range)
    scale = frac * size / 2.0 / extent
    center = size / 2.0 + rng.uniform(-0.02, 0.02, size=2) * size

    polys = []
    centroids = []
    for i in range(len(surface.spots)):
        c = surface.spot_contour(i)
        p = proj(c) * scale + center
        polys.append(p)
        centroids.append(p.mean(axis=0))

    bg = _background(size, rng)
    coverage = rasterize_polygons(polys, size)
    contrast = rng.uniform(*cfg.contrast_range)
    img = bg - contrast * coverage
    blur = rng.uniform(*cfg.blur_sigma_range)
    if blur > 0.05:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    noise = rng.uniform(*cfg.noise_sigma_range)
    if noise > 0:
        img = img + rng.standard_normal(img.shape) * noise
    img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)

    info = {
        "view": {
            "rotation": rotation,
            "tilt_deg": float(np.rad2deg(tilt)),
            "tilt_axis": tilt_axis,
            "roll": roll,
            "scale": scale,
        },
        "spot_centroids_px": np.array(centroids),
    }
    return img, info
