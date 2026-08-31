"""Moving frames along a resampled centerline.

2D: unit tangent and left normal per station (central differences).
3D: rotation-minimizing frames via the double-reflection method
(Wang, Juettler, Zheng & Liu, ACM TOG 27(1), 2008), for the future mesh chart.
All outputs are unit-norm; frames are right-handed.
"""

from __future__ import annotations

import numpy as np

__all__ = ["tangents_normals_2d", "rotation_minimizing_frames"]


def _unit(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, 1e-12)


def tangents_normals_2d(polyline):
    """Unit tangent and left normal at each station of a 2D polyline.

    Tangents use central differences (one-sided at the ends). The left normal
    is the tangent rotated +90 degrees in (x, y): n = (-ty, tx). In image
    coordinates (y down) this is the clockwise side when viewed on screen;
    what matters downstream is only that the convention is fixed.
    Returns (tangents, normals), each (n, 2).
    """
    pts = np.asarray(polyline, dtype=float)
    t = _unit(np.gradient(pts, axis=0))
    n = np.column_stack([-t[:, 1], t[:, 0]])
    return t, n


def rotation_minimizing_frames(points, initial_normal=None):
    """Rotation-minimizing frames along a 3D polyline (double reflection).

    Args:
        points: (n, 3) polyline.
        initial_normal: optional (3,) vector; its component orthogonal to the
            first tangent seeds the frame. Default: the coordinate axis least
            aligned with the first tangent (deterministic).

    Returns:
        (tangents, normals, binormals), each (n, 3), with
        binormal = tangent x normal (right-handed, orthonormal).
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    t = _unit(np.gradient(pts, axis=0))

    if initial_normal is None:
        axis = int(np.argmin(np.abs(t[0])))
        initial_normal = np.eye(3)[axis]
    r0 = np.asarray(initial_normal, dtype=float)
    r0 = r0 - np.dot(r0, t[0]) * t[0]
    norm = np.linalg.norm(r0)
    if norm < 1e-9:
        raise ValueError("initial_normal is parallel to the first tangent")
    r0 = r0 / norm

    normals = np.empty_like(t)
    normals[0] = r0
    for i in range(n - 1):
        v1 = pts[i + 1] - pts[i]
        c1 = float(np.dot(v1, v1))
        if c1 < 1e-18:
            normals[i + 1] = normals[i]
            continue
        r_l = normals[i] - (2.0 / c1) * np.dot(v1, normals[i]) * v1
        t_l = t[i] - (2.0 / c1) * np.dot(v1, t[i]) * v1
        v2 = t[i + 1] - t_l
        c2 = float(np.dot(v2, v2))
        if c2 < 1e-18:
            normals[i + 1] = r_l
        else:
            normals[i + 1] = r_l - (2.0 / c2) * np.dot(v2, r_l) * v2
    # Re-orthonormalize against accumulated float drift.
    normals = _unit(normals - np.sum(normals * t, axis=1, keepdims=True) * t)
    binormals = np.cross(t, normals)
    return t, normals, binormals
