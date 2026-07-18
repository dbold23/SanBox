"""Render permutations of spot identities.

Geometry is applied analytically to contour points (exact — no image-warp
interpolation), then the polygon is rasterized into a scene with lighting
gradients, texture, noise and blur. A permutation is:

  in-plane rotation x scale x translation x out-of-plane tilt (true
  perspective projection) x photometric variation.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ViewConfig:
    img_size: int = 256
    # Fraction of image the spot's nominal diameter spans.
    scale_range: tuple = (0.28, 0.48)
    # Max out-of-plane tilt in degrees. 0 = always fronto-parallel.
    tilt_max_deg: float = 55.0
    # Camera distance in units of spot size; smaller = stronger perspective.
    # 8 = weak perspective (near-affine), which the whitening-based
    # descriptor cancels well; below ~5 the projective distortion visibly
    # exceeds the affine model and accuracy at high tilt drops.
    camera_distance: float = 8.0
    # Contrast between spot and background, in [0, 1] intensity units.
    contrast_range: tuple = (0.25, 0.6)
    noise_sigma_range: tuple = (0.0, 0.035)
    blur_sigma_range: tuple = (0.0, 1.2)


def _rot2(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def project_plane_points(
    pts: np.ndarray,
    tilt_rad: float,
    tilt_axis_rad: float,
    roll_rad: float,
    camera_distance: float,
) -> np.ndarray:
    """Perspective-project points lying on the z=0 plane after tilting the
    plane out of fronto-parallel.

    ``tilt_axis_rad`` picks the in-plane axis the plane rotates about,
    ``tilt_rad`` is the out-of-plane angle, ``roll_rad`` an in-plane camera
    roll applied after projection. Focal length is normalized out; the
    projection is scaled so a unit-RMS shape keeps roughly unit RMS size.
    """
    p = pts @ _rot2(tilt_axis_rad).T
    x, y = p[:, 0], p[:, 1]
    # Rotate about the (new) x axis by tilt: y' = y cos t, z' = y sin t.
    ct, st = np.cos(tilt_rad), np.sin(tilt_rad)
    z = camera_distance + y * st
    u = camera_distance * x / z
    v = camera_distance * (y * ct) / z
    q = np.stack([u, v], axis=1) @ _rot2(roll_rad - tilt_axis_rad).T
    return q


def render_view(
    contour: np.ndarray,
    rng: np.random.Generator,
    cfg: ViewConfig = ViewConfig(),
    tilt_deg: float | None = None,
    rotation: float | None = None,
):
    """Render one random permutation of ``contour``.

    ``tilt_deg`` / ``rotation`` override the random pose (used for
    stratified enrollment and demos). Returns (image uint8 HxW, info dict);
    info contains the ground-truth view parameters and the projected
    polygon in pixel coordinates.
    """
    size = cfg.img_size
    if rotation is None:
        rotation = rng.uniform(0.0, 2.0 * np.pi)
    tilt = np.deg2rad(tilt_deg if tilt_deg is not None
                      else rng.uniform(0.0, cfg.tilt_max_deg))
    tilt_axis = rng.uniform(0.0, 2.0 * np.pi)
    roll = rng.uniform(0.0, 2.0 * np.pi)

    pts = contour @ _rot2(rotation).T
    pts = project_plane_points(pts, tilt, tilt_axis, roll, cfg.camera_distance)

    # Fit into the frame: scale relative to current extent, random placement.
    extent = np.abs(pts).max()
    frac = rng.uniform(*cfg.scale_range)
    pts = pts * (frac * size / 2.0 / extent)
    lim = size / 2.0 - np.abs(pts).max() - 4.0
    lim = max(lim, 0.0)
    center = size / 2.0 + rng.uniform(-lim, lim, size=2)
    pts_px = pts + center

    img = _compose_scene(pts_px, rng, cfg)
    info = {
        "rotation": rotation,
        "tilt_deg": float(np.rad2deg(tilt)),
        "tilt_axis": tilt_axis,
        "roll": roll,
        "polygon_px": pts_px,
    }
    return img, info


def _background(size: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth light background: gradient + low-frequency texture."""
    base = rng.uniform(0.65, 0.85)
    yy, xx = np.mgrid[0:size, 0:size] / size
    gdir = rng.uniform(0.0, 2.0 * np.pi)
    gmag = rng.uniform(0.0, 0.15)
    bg = base + gmag * ((xx - 0.5) * np.cos(gdir) + (yy - 0.5) * np.sin(gdir))
    tex = rng.standard_normal((size // 8, size // 8))
    tex = cv2.resize(tex, (size, size), interpolation=cv2.INTER_CUBIC)
    tex = cv2.GaussianBlur(tex, (0, 0), 3.0)
    bg += 0.02 * tex / max(tex.std(), 1e-6)
    return bg


def rasterize_polygons(polys, size: int, supersample: int = 2) -> np.ndarray:
    """Anti-aliased fill of polygons -> float coverage mask in [0, 1]."""
    s = supersample
    mask = np.zeros((size * s, size * s), np.uint8)
    for poly in polys:
        cv2.fillPoly(mask, [np.round(poly * s).astype(np.int32)], 255)
    if s > 1:
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA)
    return mask.astype(np.float32) / 255.0


def _compose_scene(pts_px, rng, cfg: ViewConfig) -> np.ndarray:
    size = cfg.img_size
    bg = _background(size, rng)
    coverage = rasterize_polygons([pts_px], size)
    contrast = rng.uniform(*cfg.contrast_range)
    img = bg - contrast * coverage
    blur = rng.uniform(*cfg.blur_sigma_range)
    if blur > 0.05:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    noise = rng.uniform(*cfg.noise_sigma_range)
    if noise > 0:
        img = img + rng.standard_normal(img.shape) * noise
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
