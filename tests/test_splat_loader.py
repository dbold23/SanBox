"""Tests for splat_loader module."""
import os
import tempfile
import numpy as np
import pytest

from tests.generate_test_ply import generate_test_room_ply
from wifi_placer.splat_loader import load_splat, _sigmoid, _normalize_quaternions


@pytest.fixture
def test_ply_path():
    """Generate a temporary test .ply file."""
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name
    generate_test_room_ply(path, n_wall_splats_per_surface=500, n_noise_splats=100)
    yield path
    os.unlink(path)


def test_sigmoid():
    x = np.array([-100, -1, 0, 1, 100], dtype=np.float64)
    y = _sigmoid(x)
    assert np.all(y >= 0) and np.all(y <= 1)
    assert np.isclose(y[2], 0.5)
    assert y[0] < 0.01
    assert y[4] > 0.99


def test_normalize_quaternions():
    quats = np.array([[2, 0, 0, 0], [0, 3, 0, 0]], dtype=np.float64)
    normed = _normalize_quaternions(quats)
    norms = np.linalg.norm(normed, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


def test_load_splat_basic(test_ply_path):
    data = load_splat(test_ply_path)
    assert data.num_gaussians > 0
    assert data.centers.shape == (data.num_gaussians, 3)
    assert data.scales.shape == (data.num_gaussians, 3)
    assert data.rotations.shape == (data.num_gaussians, 4)
    assert data.opacities.shape == (data.num_gaussians,)
    assert data.colors.shape == (data.num_gaussians, 3)


def test_opacities_in_range(test_ply_path):
    data = load_splat(test_ply_path)
    assert np.all(data.opacities >= 0) and np.all(data.opacities <= 1)


def test_scales_positive(test_ply_path):
    data = load_splat(test_ply_path)
    assert np.all(data.scales > 0)


def test_quaternions_unit_length(test_ply_path):
    data = load_splat(test_ply_path)
    norms = np.linalg.norm(data.rotations, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_colors_in_range(test_ply_path):
    data = load_splat(test_ply_path)
    assert np.all(data.colors >= 0) and np.all(data.colors <= 1)


def test_opacity_filter(test_ply_path):
    data_low = load_splat(test_ply_path, opacity_threshold=0.01)
    data_high = load_splat(test_ply_path, opacity_threshold=0.5)
    assert data_low.num_gaussians >= data_high.num_gaussians


def test_noise_filtered(test_ply_path):
    """Noise splats (opacity=0.05) should be filtered at default threshold (0.1)."""
    data = load_splat(test_ply_path)
    # All remaining should have opacity >= 0.1
    assert np.all(data.opacities >= 0.1)
