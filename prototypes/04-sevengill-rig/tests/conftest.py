"""Shared fixtures.  Building a mesh and extracting a centerline costs ~0.7 s,
so every derived object is session-scoped and the suite stays under a few
seconds."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mesh3d  # noqa: E402
import synth  # noqa: E402

N_STATIONS = 64


def dist_to_curve(points, curve, n_dense=4000):
    """Distance from each point to a densely resampled ground-truth polyline."""
    dense = mesh3d.resample_polyline(np.asarray(curve, dtype=float), n_dense)
    d, _ = cKDTree(dense).query(np.asarray(points, dtype=float))
    return d


@pytest.fixture(scope="session")
def straight():
    return synth.make_sevengill()


@pytest.fixture(scope="session")
def bare():
    return synth.make_sevengill(with_fins=False)


@pytest.fixture(scope="session")
def bent(straight):
    mesh, info = synth.bend(straight)
    return mesh, info


@pytest.fixture(scope="session")
def bent_s(straight):
    total = mesh3d.arc_length(straight.metadata["centerline"])[-1]
    mesh, info = synth.bend(straight, synth.s_curve(total, 90.0, N_STATIONS))
    return mesh, info


@pytest.fixture(scope="session")
def extracted(bent):
    mesh, _ = bent
    return mesh3d.extract_centerline_3d(mesh, n_stations=N_STATIONS)


@pytest.fixture(scope="session")
def chart(bent, extracted):
    """(mesh, centerline, frames, coords, detection) for the bent C-pose."""
    mesh, _ = bent
    centerline, _ = extracted
    frames = mesh3d.tube_frames(centerline)
    coords = mesh3d.tube_coords(mesh, centerline, frames)
    return mesh, centerline, frames, coords, mesh3d.detect_fins(mesh, coords)
