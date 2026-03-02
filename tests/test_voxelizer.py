"""Tests for voxelizer module."""
import os
import tempfile
import numpy as np
import pytest

from tests.generate_test_ply import generate_test_room_ply
from wifi_placer.splat_loader import load_splat
from wifi_placer.voxelizer import voxelize_gaussians


@pytest.fixture
def test_splat_data():
    """Load splat data from a synthetic room."""
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name
    generate_test_room_ply(path, n_wall_splats_per_surface=500, n_noise_splats=50)
    data = load_splat(path)
    os.unlink(path)
    return data


def test_voxelize_produces_grid(test_splat_data):
    occ = voxelize_gaussians(test_splat_data, voxel_size=0.1)
    assert occ.grid.shape == occ.shape
    assert occ.binary_grid.shape == occ.shape
    assert occ.voxel_size == 0.1


def test_grid_has_occupied_voxels(test_splat_data):
    occ = voxelize_gaussians(test_splat_data, voxel_size=0.1)
    assert occ.binary_grid.sum() > 0


def test_grid_has_free_voxels(test_splat_data):
    occ = voxelize_gaussians(test_splat_data, voxel_size=0.1)
    assert (~occ.binary_grid).sum() > 0


def test_density_normalized(test_splat_data):
    occ = voxelize_gaussians(test_splat_data, voxel_size=0.1)
    assert occ.grid.max() <= 1.0
    assert occ.grid.min() >= 0.0


def test_world_voxel_roundtrip(test_splat_data):
    occ = voxelize_gaussians(test_splat_data, voxel_size=0.1)
    # Test a few random voxel coordinates
    test_voxels = np.array([[5, 5, 5], [10, 10, 10]])
    world = occ.voxel_to_world(test_voxels)
    back = occ.world_to_voxel(world)
    np.testing.assert_array_equal(test_voxels, back)


def test_is_inside(test_splat_data):
    occ = voxelize_gaussians(test_splat_data, voxel_size=0.1)
    inside = np.array([[0, 0, 0]])
    outside = np.array([[-1, -1, -1]])
    assert occ.is_inside(inside)[0] == True
    assert occ.is_inside(outside)[0] == False
