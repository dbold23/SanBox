"""Surfaces: many spots laid out on a plane, and angled views of them.

A *surface identity* is a seed that deterministically produces N spots
(each with its own shape identity, position, size, elongation, orientation
and darkness) placed without overlap on the unit plane. The surface's spot
constellation is its fingerprint, like the spot pattern on a whale shark's
flank.

The renderer aims at real-world imagery: uneven lighting, vignetting,
gamma shifts, glossy specular sheen, cracks and scratches, clutter blobs
that belong to no surface, missing spots, faded spots, and views from far
away or close up, all on top of perspective tilt.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .render import ViewConfig, _rot2, _background, project_plane_points
from .shapes import generate_identity, _harmonic_blob

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
    aspect: float = 1.0     # elongation (1 = round, 4 = long streak)
    darkness: float = 1.0   # 1 = full contrast, ~0.3 = faded


class Surface:
    def __init__(self, surface_id: int, spots: list["SurfaceSpot"]):
        self.surface_id = surface_id
        self.spots = spots
        self.positions = np.array([s.position for s in spots])

    def spot_contour(self, i: int) -> np.ndarray:
        """Contour of spot i in surface coordinates."""
        s = self.spots[i]
        base = generate_identity(s.shape_seed, SURFACE_CONTOUR_POINTS)
        # Area-preserving elongation, then in-plane orientation.
        stretch = np.diag([np.sqrt(s.aspect), 1.0 / np.sqrt(s.aspect)])
        return s.position + (base @ stretch.T @ _rot2(s.angle).T) * s.radius


def generate_surface(
    surface_id: int,
    n_spots: int = 600,
    radius_range: tuple = (0.010, 0.040),
    aspect_max: float = 4.0,
    faded_fraction: float = 0.15,
    min_gap: float = 1.35,
) -> Surface:
    """Deterministically generate a surface with ``n_spots`` spots.

    Spot sizes are log-uniform (many small specks, few large blobs), a
    random subset is elongated up to ``aspect_max``:1 streaks, and
    ``faded_fraction`` of spots carry low darkness (weak, washed-out
    marks). Placement is rejection sampling with a spatial grid;
    ``min_gap`` scales the required center distance relative to the two
    spots' effective radii.
    """
    rng = np.random.default_rng(np.random.SeedSequence([77_000_017, surface_id]))
    spots: list[SurfaceSpot] = []
    placed = np.zeros((0, 3))  # x, y, effective radius
    grid: dict[tuple, list[int]] = {}
    log_lo, log_hi = np.log(radius_range[0]), np.log(radius_range[1])
    max_eff = radius_range[1] * np.sqrt(aspect_max)
    cell = 2.0 * max_eff * min_gap

    def cell_of(p):
        return (int(p[0] // cell), int(p[1] // cell))

    attempts = 0
    max_attempts = n_spots * 400
    while len(spots) < n_spots and attempts < max_attempts:
        attempts += 1
        pos = rng.uniform(-1.0, 1.0, size=2)
        radius = float(np.exp(rng.uniform(log_lo, log_hi)))
        # Most spots roundish; a tail of long streaks.
        aspect = float(1.0 + (aspect_max - 1.0) * rng.beta(1.2, 4.0))
        eff = radius * np.sqrt(aspect)
        cx, cy = cell_of(pos)
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((cx + dx, cy + dy), []):
                    other = placed[j]
                    limit = min_gap * (eff + other[2])
                    if np.hypot(pos[0] - other[0], pos[1] - other[1]) < limit:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            continue
        faded = rng.uniform() < faded_fraction
        idx = len(spots)
        spots.append(SurfaceSpot(
            index=idx,
            shape_seed=int(rng.integers(0, 2**31 - 1)),
            position=pos.copy(),
            radius=radius,
            angle=rng.uniform(0.0, 2.0 * np.pi),
            aspect=aspect,
            darkness=float(rng.uniform(0.25, 0.5) if faded
                           else rng.uniform(0.7, 1.0)),
        ))
        placed = np.vstack([placed, [pos[0], pos[1], eff]])
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
    # Fraction of the frame the projected surface spans (low = far away).
    fill_range: tuple = (0.82, 0.95)
    # --- real-world degradations (ranges sampled per view) ---
    dropout_range: tuple = (0.0, 0.05)     # fraction of spots missing
    clutter_range: tuple = (0, 6)          # spurious blobs not in gallery
    gloss_range: tuple = (0, 2)            # glossy sheen patches
    gloss_strength: tuple = (0.35, 0.75)   # how much sheen washes out
    crack_range: tuple = (0, 2)            # dark scratch polylines
    gamma_range: tuple = (0.85, 1.2)
    gradient_strength: tuple = (0.0, 0.2)  # extra directional light
    vignette_strength: tuple = (0.0, 0.25)
    # Simulated sensor/detail loss: image downscaled by this factor and
    # upscaled back (0.5 = half resolution). 1.0 = full detail.
    resolution_range: tuple = (0.85, 1.0)


def harsh_view_config(**overrides) -> SurfaceViewConfig:
    """A deliberately punishing configuration: everything at once."""
    cfg = SurfaceViewConfig(
        contrast_range=(0.22, 0.5),
        noise_sigma_range=(0.01, 0.035),
        blur_sigma_range=(0.2, 1.4),
        fill_range=(0.45, 0.95),
        dropout_range=(0.05, 0.18),
        clutter_range=(5, 25),
        gloss_range=(2, 6),
        gloss_strength=(0.5, 0.95),
        crack_range=(1, 4),
        gamma_range=(0.65, 1.5),
        gradient_strength=(0.1, 0.4),
        vignette_strength=(0.1, 0.4),
        resolution_range=(0.5, 1.0),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _rasterize_weighted(polys, weights, size: int, supersample: int = 2):
    """Anti-aliased fill of polygons, each with its own weight in [0,1]."""
    s = supersample
    canvas = np.zeros((size * s, size * s), np.uint8)
    order = np.argsort(weights)  # darker spots drawn last where overlapping
    for i in order:
        cv2.fillPoly(canvas, [np.round(polys[i] * s).astype(np.int32)],
                     int(round(255 * float(weights[i]))))
    if s > 1:
        canvas = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    return canvas.astype(np.float32) / 255.0


def _random_crack(rng, size) -> np.ndarray:
    """A meandering polyline crossing part of the image."""
    n = rng.integers(8, 20)
    p = rng.uniform(0, size, 2)
    heading = rng.uniform(0, 2 * np.pi)
    pts = [p.copy()]
    step = size / n
    for _ in range(n):
        heading += rng.normal(0.0, 0.45)
        p = p + step * np.array([np.cos(heading), np.sin(heading)])
        pts.append(p.copy())
    return np.array(pts, np.float64)


def render_surface_view(
    surface: Surface,
    rng: np.random.Generator,
    cfg: SurfaceViewConfig = SurfaceViewConfig(),
    tilt_deg: float | None = None,
):
    """Render the surface from a random viewpoint with real-world defects.

    Returns (image uint8, info). info holds ground truth for evaluation:
      spot_centroids_px  projected center of every spot,
      drawn              False where the spot was dropped (missing/worn),
      obscured           True where glare covers the spot center,
      view               pose parameters.
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

    n = len(surface.spots)
    drawn = rng.uniform(size=n) >= rng.uniform(*cfg.dropout_range)
    polys, weights, centroids = [], [], []
    for i in range(n):
        p = proj(surface.spot_contour(i)) * scale + center
        centroids.append(p.mean(axis=0))
        if drawn[i]:
            polys.append(p)
            weights.append(surface.spots[i].darkness)

    # Clutter: blobs that belong to no enrolled surface (dirt, debris).
    n_clutter = int(rng.integers(cfg.clutter_range[0], cfg.clutter_range[1] + 1))
    for _ in range(n_clutter):
        blob = _harmonic_blob(np.random.default_rng(rng.integers(1 << 31)), 96)
        r = rng.uniform(0.006, 0.03) * size
        c = rng.uniform([0.05, 0.05], [0.95, 0.95]) * size
        polys.append(blob * r + c)
        weights.append(rng.uniform(0.4, 1.0))

    bg = _background(size, rng)
    coverage = _rasterize_weighted(polys, weights, size)
    contrast = rng.uniform(*cfg.contrast_range)
    img = bg - contrast * coverage

    # Cracks / scratches: thin dark meandering lines.
    n_cracks = int(rng.integers(cfg.crack_range[0], cfg.crack_range[1] + 1))
    if n_cracks:
        crack_layer = np.zeros((size, size), np.float32)
        for _ in range(n_cracks):
            pts = np.round(_random_crack(rng, size)).astype(np.int32)
            cv2.polylines(crack_layer, [pts], False, 1.0,
                          int(rng.integers(1, 3)), cv2.LINE_AA)
        crack_layer = cv2.GaussianBlur(crack_layer, (0, 0), 0.6)
        img = img - rng.uniform(0.3, 0.8) * contrast * crack_layer

    # Directional light gradient + vignette + gamma.
    yy, xx = np.mgrid[0:size, 0:size] / size
    gdir = rng.uniform(0.0, 2.0 * np.pi)
    img = img + rng.uniform(*cfg.gradient_strength) * (
        (xx - 0.5) * np.cos(gdir) + (yy - 0.5) * np.sin(gdir))
    cx, cy = rng.uniform(0.3, 0.7, 2)
    img = img * (1.0 - rng.uniform(*cfg.vignette_strength)
                 * 2.0 * ((xx - cx) ** 2 + (yy - cy) ** 2))

    # Glossy sheen: smooth bright patches that wash out what's beneath.
    n_gloss = int(rng.integers(cfg.gloss_range[0], cfg.gloss_range[1] + 1))
    gloss = np.zeros((size, size), np.float32)
    for _ in range(n_gloss):
        blob = _harmonic_blob(np.random.default_rng(rng.integers(1 << 31)), 96)
        r = rng.uniform(0.06, 0.22) * size
        c = rng.uniform([0.0, 0.0], [1.0, 1.0]) * size
        patch = np.zeros((size, size), np.float32)
        cv2.fillPoly(patch, [np.round(blob * r + c).astype(np.int32)], 1.0)
        gloss = np.maximum(gloss, patch)
    if n_gloss:
        gloss = cv2.GaussianBlur(gloss, (0, 0), rng.uniform(8.0, 25.0))
        strength = rng.uniform(*cfg.gloss_strength)
        sheen = rng.uniform(0.85, 0.98)
        img = img * (1.0 - strength * gloss) + sheen * strength * gloss

    img = np.clip(img, 0.0, 1.0) ** rng.uniform(*cfg.gamma_range)

    blur = rng.uniform(*cfg.blur_sigma_range)
    if blur > 0.05:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    res = rng.uniform(*cfg.resolution_range)
    if res < 0.999:
        small = cv2.resize(img, None, fx=res, fy=res,
                           interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (size, size), interpolation=cv2.INTER_LINEAR)
    noise = rng.uniform(*cfg.noise_sigma_range)
    if noise > 0:
        img = img + rng.standard_normal(img.shape) * noise
    img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)

    centroids = np.array(centroids)
    obscured = np.zeros(n, bool)
    if n_gloss:
        ic = np.round(centroids).astype(int)
        inside = ((ic[:, 0] >= 0) & (ic[:, 0] < size)
                  & (ic[:, 1] >= 0) & (ic[:, 1] < size))
        obscured[inside] = gloss[ic[inside, 1], ic[inside, 0]] > 0.45

    info = {
        "view": {
            "rotation": rotation,
            "tilt_deg": float(np.rad2deg(tilt)),
            "tilt_axis": tilt_axis,
            "roll": roll,
            "scale": scale,
        },
        "spot_centroids_px": centroids,
        "drawn": drawn,
        "obscured": obscured,
        "n_clutter": n_clutter,
    }
    return img, info
