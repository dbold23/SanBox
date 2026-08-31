"""The (arc-length s x signed offset r) strip chart and its inverse.

Chart convention (fixed across this prototype):
- The centerline is resampled to ``n_s`` stations uniform in arc length;
  station i sits at arc length s_i = i * S / (n_s - 1), S = total length.
- Chart row i, column j samples the image at station_i + r_j * left_normal_i,
  with r_j = -half_width + j * 2*half_width / (n_r - 1).
- ``rectify`` output has shape (n_s, n_r); samples outside the image or the
  mask are ``fill`` (NaN by default).
- ``chart_to_image`` / ``image_to_chart`` are mutual inverses on the strip
  interior (up to interpolation error), so findings measured in chart space
  can be annotated back onto the frame.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from centerline import arc_length, resample_polyline
from frames import tangents_normals_2d

__all__ = ["chart_grid", "rectify", "chart_to_image", "image_to_chart"]


def _stations(centerline, n_s):
    cl = resample_polyline(centerline, n_s)
    _, normals = tangents_normals_2d(cl)
    return cl, normals


def _r_values(half_width, n_r):
    return np.linspace(-half_width, half_width, int(n_r))


def chart_grid(centerline, half_width, n_s, n_r):
    """Image-space sample coordinates of the chart, shape (n_s, n_r, 2) as (x, y)."""
    cl, normals = _stations(centerline, n_s)
    r = _r_values(half_width, n_r)
    return cl[:, None, :] + r[None, :, None] * normals[:, None, :]


def rectify(image, centerline, half_width, n_s, n_r, mask=None, fill=np.nan):
    """Sample ``image`` on the chart grid with bilinear interpolation.

    Returns an (n_s, n_r) float array. Samples falling outside the image, or
    (when ``mask`` is given) outside the mask, are set to ``fill``.
    """
    grid = chart_grid(centerline, half_width, n_s, n_r)
    coords = [grid[..., 1].ravel(), grid[..., 0].ravel()]  # (row=y, col=x)
    strip = ndimage.map_coordinates(
        np.asarray(image, dtype=float), coords, order=1, mode="constant", cval=np.nan
    ).reshape(n_s, n_r)

    invalid = np.isnan(strip)
    if mask is not None:
        inside = ndimage.map_coordinates(
            np.asarray(mask, dtype=float), coords, order=1, mode="constant", cval=0.0
        ).reshape(n_s, n_r)
        invalid |= inside < 0.5
    strip = np.where(invalid, fill, np.where(np.isnan(strip), fill, strip))
    return strip


def chart_to_image(centerline, half_width, n_s, n_r, s_idx, r_idx):
    """Map fractional chart indices (s_idx, r_idx) to image (x, y).

    Accepts scalars or arrays; returns an (..., 2) array of (x, y).
    """
    cl, normals = _stations(centerline, n_s)
    s_idx = np.atleast_1d(np.asarray(s_idx, dtype=float))
    r_idx = np.atleast_1d(np.asarray(r_idx, dtype=float))

    i0 = np.clip(np.floor(s_idx).astype(int), 0, n_s - 2)
    f = np.clip(s_idx - i0, 0.0, 1.0)[:, None]
    pos = cl[i0] * (1 - f) + cl[i0 + 1] * f
    nrm = normals[i0] * (1 - f) + normals[i0 + 1] * f
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)

    r = -half_width + r_idx * (2.0 * half_width / (n_r - 1))
    return pos + r[:, None] * nrm


def image_to_chart(centerline, half_width, n_s, n_r, points):
    """Project image points onto the chart: returns (k, 2) of (s_idx, r_idx).

    Each point is projected onto the nearest centerline segment; r is the
    signed offset along the left normal at the projection.
    """
    cl, _ = _stations(centerline, n_s)
    total = arc_length(cl)[-1]
    step = total / (n_s - 1)

    pts = np.atleast_2d(np.asarray(points, dtype=float))
    a = cl[:-1]                      # (m, 2) segment starts
    d = cl[1:] - cl[:-1]             # (m, 2)
    seg_len2 = np.maximum(np.sum(d * d, axis=1), 1e-18)

    rel = pts[:, None, :] - a[None, :, :]                       # (k, m, 2)
    t = np.clip(np.sum(rel * d[None], axis=2) / seg_len2, 0.0, 1.0)  # (k, m)
    proj = a[None] + t[..., None] * d[None]
    dist2 = np.sum((pts[:, None, :] - proj) ** 2, axis=2)
    best = np.argmin(dist2, axis=1)                             # (k,)

    k = len(pts)
    idx = np.arange(k)
    tb = t[idx, best]
    db = d[best]
    seg_norm = np.sqrt(seg_len2[best])
    # signed offset along the left normal n = (-dy, dx)/|d|
    rel_b = pts - a[best]
    r = (rel_b[:, 0] * (-db[:, 1]) + rel_b[:, 1] * db[:, 0]) / seg_norm

    s_arc = (best + tb) * step
    s_idx = s_arc / step
    r_idx = (r + half_width) / (2.0 * half_width / (n_r - 1))
    return np.column_stack([s_idx, r_idx])
