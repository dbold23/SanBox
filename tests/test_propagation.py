"""Tests for propagation module."""
import numpy as np
import pytest

from wifi_placer.propagation import (
    fspl_db, amanatides_woo_traversal, count_wall_crossings,
    compute_rssi,
)
from wifi_placer.voxelizer import OccupancyGrid


def _make_empty_grid(shape=(20, 20, 10), voxel_size=0.1):
    """Create an empty occupancy grid for testing."""
    grid = np.zeros(shape, dtype=np.float64)
    binary = np.zeros(shape, dtype=bool)
    origin = np.array([0.0, 0.0, 0.0])
    return OccupancyGrid(grid, binary, voxel_size, origin, shape)


def _make_grid_with_wall(shape=(20, 20, 10), wall_x=10, voxel_size=0.1):
    """Create a grid with a wall at column wall_x (all y, all z)."""
    grid = np.zeros(shape, dtype=np.float64)
    binary = np.zeros(shape, dtype=bool)
    grid[wall_x, :, :] = 1.0
    binary[wall_x, :, :] = True
    origin = np.array([0.0, 0.0, 0.0])
    return OccupancyGrid(grid, binary, voxel_size, origin, shape)


def test_fspl_known_value():
    """FSPL at 1m, 2.4GHz should be ~40dB."""
    loss = fspl_db(1.0, 2.4e9)
    assert 39 < loss < 41


def test_fspl_at_10m():
    """FSPL at 10m should be 20dB more than at 1m."""
    loss_1m = fspl_db(1.0, 2.4e9)
    loss_10m = fspl_db(10.0, 2.4e9)
    np.testing.assert_allclose(loss_10m - loss_1m, 20.0, atol=0.1)


def test_fspl_zero_distance():
    assert fspl_db(0.0, 2.4e9) == 0.0


def test_ray_traversal_straight_x():
    """Ray along x-axis should visit consecutive x voxels."""
    start = np.array([0.5, 5.5, 5.5])
    end = np.array([9.5, 5.5, 5.5])
    visited = amanatides_woo_traversal(start, end, (20, 20, 10))
    x_vals = [v[0] for v in visited]
    assert x_vals == list(range(0, 10))
    # All should have same y, z
    assert all(v[1] == 5 for v in visited)
    assert all(v[2] == 5 for v in visited)


def test_ray_traversal_diagonal():
    """Diagonal ray should visit more voxels than straight."""
    start = np.array([0.5, 0.5, 0.5])
    end = np.array([5.5, 5.5, 5.5])
    visited = amanatides_woo_traversal(start, end, (20, 20, 10))
    assert len(visited) > 5  # Diagonal crosses more voxels


def test_wall_crossings_no_wall():
    """Ray through empty grid should count zero walls."""
    occ = _make_empty_grid()
    wall_types = np.zeros(occ.shape, dtype=np.int32)
    counts = count_wall_crossings(
        np.array([0.05, 1.0, 0.5]),
        np.array([1.95, 1.0, 0.5]),
        wall_types, occ,
    )
    assert counts["light"] == 0 and counts["medium"] == 0 and counts["heavy"] == 0


def test_wall_crossings_with_wall():
    """Ray through a wall should count wall voxels."""
    occ = _make_grid_with_wall(wall_x=10)
    wall_types = np.zeros(occ.shape, dtype=np.int32)
    wall_types[10, :, :] = 2  # medium wall
    counts = count_wall_crossings(
        np.array([0.05, 1.0, 0.5]),
        np.array([1.95, 1.0, 0.5]),
        wall_types, occ,
    )
    assert counts["medium"] >= 1


def test_rssi_decreases_with_distance():
    """RSSI should be lower at greater distance."""
    occ = _make_empty_grid()
    wall_types = np.zeros(occ.shape, dtype=np.int32)
    tx = np.array([1.0, 1.0, 0.5])
    rx_near = np.array([1.5, 1.0, 0.5])
    rx_far = np.array([1.9, 1.0, 0.5])
    rssi_near = compute_rssi(tx, rx_near, wall_types, occ)
    rssi_far = compute_rssi(tx, rx_far, wall_types, occ)
    assert rssi_near > rssi_far
