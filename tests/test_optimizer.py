"""Tests for optimizer module."""
import os
import tempfile
import numpy as np
import pytest

from tests.generate_test_ply import generate_test_room_ply
from wifi_placer.splat_loader import load_splat
from wifi_placer.voxelizer import voxelize_gaussians
from wifi_placer.floor_extractor import (
    detect_floor_level, get_free_space_mask, classify_wall_density,
)
from wifi_placer.optimizer import optimize_placement, _get_candidate_positions


@pytest.fixture
def room_setup():
    """Set up a voxelized synthetic room."""
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name
    generate_test_room_ply(
        path,
        room_dims=(3.0, 3.0, 3.0),
        n_wall_splats_per_surface=500,
        n_noise_splats=50,
    )
    data = load_splat(path)
    os.unlink(path)

    occ = voxelize_gaussians(data, voxel_size=0.2)
    floor_z = detect_floor_level(occ)
    free_mask = get_free_space_mask(occ, floor_z)
    wall_types = classify_wall_density(occ)

    return occ, floor_z, free_mask, wall_types


def test_candidate_positions_nonempty(room_setup):
    occ, floor_z, free_mask, _ = room_setup
    candidates = _get_candidate_positions(occ, free_mask, floor_z + 2.0)
    assert len(candidates) > 0


def test_optimize_returns_result(room_setup):
    occ, floor_z, free_mask, wall_types = room_setup
    result = optimize_placement(
        occ, wall_types, free_mask, floor_z,
        max_routers=1,
        seed=42,
        t_max=10.0, t_min=1.0, cooling_rate=0.9, steps_per_temp=5,
    )
    assert result.num_routers >= 1
    assert 0.0 <= result.coverage_fraction <= 1.0
    assert len(result.positions) == result.num_routers


def test_router_in_free_space(room_setup):
    """Optimized router should be placed in free space."""
    occ, floor_z, free_mask, wall_types = room_setup
    result = optimize_placement(
        occ, wall_types, free_mask, floor_z,
        max_routers=1,
        seed=42,
        t_max=10.0, t_min=1.0, cooling_rate=0.9, steps_per_temp=5,
    )
    for pos in result.positions:
        voxel = occ.world_to_voxel(np.array([pos]))
        if occ.is_inside(voxel)[0]:
            i, j, k = voxel[0]
            assert not occ.binary_grid[i, j, k], "Router placed inside a wall!"
