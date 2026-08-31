from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def rasterize_tube(curve_xy, half_width, shape):
    """Binary mask of all pixels within half_width of a dense polyline.

    Ground-truth tube generator, independent of the pipeline under test.
    curve_xy: (k, 2) of (x, y), densely sampled (spacing <= ~0.5 px).
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    pts = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    d, _ = cKDTree(curve_xy).query(pts, k=1)
    return (d.reshape(shape) <= half_width)


def dist_to_curve(points_xy, curve_xy):
    """Distance from each point to a densely sampled ground-truth curve."""
    d, _ = cKDTree(curve_xy).query(np.asarray(points_xy, dtype=float), k=1)
    return d


@pytest.fixture(scope="session")
def demo_measurements():
    """Run the strain demo once per session for eps in {0, 0.05}."""
    import strain_demo

    out = {}
    for eps in (0.0, 0.05):
        out[eps] = strain_demo.measure_epsilon(eps, seed=0)
    return out
