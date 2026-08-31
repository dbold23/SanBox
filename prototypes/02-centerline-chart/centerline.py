"""Centerline extraction from a binary body mask.

Contract:
- ``extract_centerline(mask, n_stations, seed)`` returns an ``(n_stations, 2)``
  float array of (x, y) pixel coordinates, uniformly spaced in arc length,
  ordered head -> tail under the widest-end-first rule (the end whose mean
  half-width, read from the distance transform, is larger comes first).
- Fully deterministic; ``seed`` is accepted for API stability but no step here
  is stochastic.
- Robust to ragged masks: only the largest connected component is used and the
  path is a medial-weighted shortest path, which hugs the distance-transform
  ridge rather than the boundary.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

__all__ = ["extract_centerline", "arc_length", "resample_polyline"]

# Half of the 8-neighbourhood; the graph is used undirected.
_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1))


def arc_length(points):
    """Cumulative arc length of a polyline, shape (n,). arc_length[0] == 0."""
    pts = np.asarray(points, dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def resample_polyline(points, n):
    """Resample a polyline to ``n`` points uniformly spaced in arc length."""
    pts = np.asarray(points, dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-12])
    pts = pts[keep]
    s = arc_length(pts)
    target = np.linspace(0.0, s[-1], int(n))
    x = np.interp(target, s, pts[:, 0])
    y = np.interp(target, s, pts[:, 1])
    return np.column_stack([x, y])


def _largest_component(mask):
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        raise ValueError("mask is empty")
    if n == 1:
        return labels == 1
    sizes = ndimage.sum_labels(np.ones_like(labels), labels, index=np.arange(1, n + 1))
    return labels == (1 + int(np.argmax(sizes)))


def _medial_graph(mask, edt):
    """Sparse undirected graph over mask pixels.

    Edge weight = euclidean step length scaled by the mean inverse distance
    transform of its endpoints, so shortest paths prefer the medial ridge.
    """
    idx = np.full(mask.shape, -1, dtype=np.int64)
    n = int(mask.sum())
    idx[mask] = np.arange(n)
    inv = np.zeros(mask.shape)
    inv[mask] = 1.0 / np.maximum(edt[mask], 0.5)

    rows, cols, data = [], [], []
    h, w = mask.shape
    for dy, dx in _OFFSETS:
        a = mask[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
        b = mask[max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)]
        both = a & b
        if not both.any():
            continue
        ia = idx[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)][both]
        ib = idx[max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)][both]
        step = float(np.hypot(dy, dx))
        wa = inv[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)][both]
        wb = inv[max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)][both]
        rows.append(ia)
        cols.append(ib)
        data.append(step * 0.5 * (wa + wb))
    if not rows:
        raise ValueError(
            "mask's largest component has no 8-connected pixel pairs "
            "(single pixel or degenerate region) - no centerline exists"
        )
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)
    return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def _longest_medial_path(mask, edt):
    """Pixel path (k, 2) as (y, x): medial-weighted graph diameter path."""
    graph = _medial_graph(mask, edt)
    yy, xx = np.nonzero(mask)
    flat_ids = np.arange(len(yy))

    start = int(np.argmax(edt[yy, xx]))
    d0 = dijkstra(graph, directed=False, indices=start)
    a = int(np.argmax(np.where(np.isfinite(d0), d0, -1.0)))
    d1, pred = dijkstra(graph, directed=False, indices=a, return_predecessors=True)
    b = int(np.argmax(np.where(np.isfinite(d1), d1, -1.0)))

    path = []
    node = b
    while node != -9999 and node != a:
        path.append(node)
        node = int(pred[node])
    path.append(a)
    path = flat_ids[np.array(path[::-1])]
    return np.column_stack([yy[path], xx[path]])


def _smooth(path_xy, window):
    if window < 3 or len(path_xy) <= window:
        return path_xy
    pad = window // 2
    padded = np.pad(path_xy, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    return np.column_stack(
        [np.convolve(padded[:, k], kernel, mode="valid") for k in range(2)]
    )


def extract_centerline(mask, n_stations=256, seed=None):
    """Extract the medial centerline of an elongate binary mask.

    Returns an (n_stations, 2) array of (x, y) coordinates, uniform in arc
    length, oriented widest-end-first (head first for a fish silhouette).
    ``seed`` is unused; extraction is deterministic.
    """
    mask = _largest_component(np.asarray(mask).astype(bool))
    mask = ndimage.binary_fill_holes(mask)  # pepper holes distort the ridge
    edt = ndimage.distance_transform_edt(mask)

    path_yx = _longest_medial_path(mask, edt)
    path_xy = path_yx[:, ::-1].astype(float)

    window = min(31, max(5, 2 * int(0.02 * len(path_xy)) + 1))
    path_xy = _smooth(path_xy, window)
    stations = resample_polyline(path_xy, n_stations)

    # Widest-end-first orientation from the distance transform half-width.
    k = max(1, n_stations // 10)
    widths = ndimage.map_coordinates(
        edt, [stations[:, 1], stations[:, 0]], order=1, mode="nearest"
    )
    if float(widths[-k:].mean()) > float(widths[:k].mean()):
        stations = stations[::-1].copy()

    # Sanity signal for a non-tubular mask (a blob or disc): for a genuine
    # tube, extracted length ~ area / (2 * mean half-width along the path)
    # (measured 0.96-1.01 on straight/arc/tapered test tubes; a disc scores
    # ~0.65). KNOWN LIMITATION: a fully self-touching bend whose flanks fuse
    # passes this check (~0.98) because the fused region widens exactly as the
    # path shortcuts - that failure is undetectable from the mask alone, so
    # keep bends resolvable in the segmentation.
    seg = np.diff(stations, axis=0)
    extracted_len = float(np.hypot(seg[:, 0], seg[:, 1]).sum())
    mean_half_width = float(widths.mean())
    if mean_half_width > 0:
        expected_len = float(mask.sum()) / (2.0 * mean_half_width)
        if extracted_len < 0.8 * expected_len:
            warnings.warn(
                "extracted centerline length %.1f is well below the area-derived "
                "expectation %.1f - the mask does not look tubular (blob/disc, "
                "or badly degraded); treat this centerline as unreliable"
                % (extracted_len, expected_len),
                RuntimeWarning,
                stacklevel=2,
            )
    return stations
