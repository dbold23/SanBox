"""
WiFi signal propagation model using COST231 multi-wall path loss.

Implements:
- Free Space Path Loss (FSPL)
- Amanatides & Woo voxel traversal for ray casting through occupancy grids
- RSSI computation with per-wall-type attenuation
- Batch coverage map computation
"""
import numpy as np

from wifi_placer.config import (
    FREQUENCIES, DEFAULT_TX_POWER_DBM, SPEED_OF_LIGHT, MIN_RSSI_DBM,
    WALL_ATTENUATION_DB, REFERENCE_WALL_THICKNESS,
)
from wifi_placer.voxelizer import OccupancyGrid


def fspl_db(distance_m: float, freq_hz: float) -> float:
    """
    Free Space Path Loss in dB.

    FSPL = 20*log10(d) + 20*log10(f) - 147.55
    where d is in meters and f is in Hz.
    """
    if distance_m <= 0:
        return 0.0
    return 20.0 * np.log10(distance_m) + 20.0 * np.log10(freq_hz) - 147.55


def amanatides_woo_traversal(
    start: np.ndarray,
    end: np.ndarray,
    grid_shape: tuple,
) -> list:
    """
    Traverse all voxels along a ray from start to end (in voxel coordinates).

    Implements the Amanatides & Woo (1987) algorithm. Returns a list of
    (i, j, k) voxel index tuples for every voxel the ray passes through.
    """
    direction = end - start
    ray_len = np.linalg.norm(direction)
    if ray_len < 1e-10:
        si, sj, sk = int(np.floor(start[0])), int(np.floor(start[1])), int(np.floor(start[2]))
        if 0 <= si < grid_shape[0] and 0 <= sj < grid_shape[1] and 0 <= sk < grid_shape[2]:
            return [(si, sj, sk)]
        return []

    visited = []
    current = np.floor(start).astype(int)
    end_voxel = np.floor(end).astype(int)

    step = np.zeros(3, dtype=int)
    t_max = np.full(3, np.inf)
    t_delta = np.full(3, np.inf)

    for a in range(3):
        if direction[a] > 1e-10:
            step[a] = 1
            t_max[a] = (current[a] + 1.0 - start[a]) / direction[a]
            t_delta[a] = 1.0 / direction[a]
        elif direction[a] < -1e-10:
            step[a] = -1
            t_max[a] = (current[a] - start[a]) / direction[a]
            t_delta[a] = -1.0 / direction[a]
        else:
            step[a] = 0
            t_max[a] = np.inf
            t_delta[a] = np.inf

    # Maximum steps to prevent infinite loops
    max_steps = int(np.sum(np.abs(end_voxel - current))) + 10

    for _ in range(max_steps):
        # Check bounds
        if (0 <= current[0] < grid_shape[0] and
            0 <= current[1] < grid_shape[1] and
            0 <= current[2] < grid_shape[2]):
            visited.append((current[0], current[1], current[2]))

        # Check if we've passed the endpoint
        if np.all(t_max > 1.0):
            break

        # Step along axis with smallest t_max
        axis = np.argmin(t_max)
        current[axis] += step[axis]
        t_max[axis] += t_delta[axis]

    return visited


def count_wall_crossings(
    start_world: np.ndarray,
    end_world: np.ndarray,
    wall_type_grid: np.ndarray,
    occupancy: OccupancyGrid,
) -> dict:
    """
    Cast a ray from start to end through the wall type grid.
    Count voxels of each wall type crossed.

    Returns: {"light": int, "medium": int, "heavy": int}
    """
    start_voxel = (start_world - occupancy.origin) / occupancy.voxel_size
    end_voxel = (end_world - occupancy.origin) / occupancy.voxel_size

    visited = amanatides_woo_traversal(start_voxel, end_voxel, occupancy.shape)

    counts = {"light": 0, "medium": 0, "heavy": 0}
    type_map = {1: "light", 2: "medium", 3: "heavy"}

    for i, j, k in visited:
        wtype = wall_type_grid[i, j, k]
        if wtype in type_map:
            counts[type_map[wtype]] += 1

    return counts


def compute_rssi(
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    wall_type_grid: np.ndarray,
    occupancy: OccupancyGrid,
    tx_power_dbm: float = DEFAULT_TX_POWER_DBM,
    freq_band: str = "5.0",
) -> float:
    """
    Compute RSSI (dBm) at rx_pos from a transmitter at tx_pos.

    RSSI = Tx_power - FSPL(d, f) - wall_loss
    """
    distance = np.linalg.norm(tx_pos - rx_pos)
    if distance < 0.01:
        return tx_power_dbm

    freq_hz = FREQUENCIES[freq_band]
    path_loss = fspl_db(distance, freq_hz)

    wall_counts = count_wall_crossings(tx_pos, rx_pos, wall_type_grid, occupancy)

    attenuation_table = WALL_ATTENUATION_DB[freq_band]
    voxel_wall_scale = occupancy.voxel_size / REFERENCE_WALL_THICKNESS

    wall_loss = 0.0
    for wtype, count in wall_counts.items():
        wall_loss += count * attenuation_table[wtype] * voxel_wall_scale

    return tx_power_dbm - path_loss - wall_loss


def compute_coverage_map(
    tx_positions: list,
    occupancy: OccupancyGrid,
    wall_type_grid: np.ndarray,
    free_space_coords: np.ndarray,
    tx_power_dbm: float = DEFAULT_TX_POWER_DBM,
    freq_band: str = "5.0",
    subsample: float = 1.0,
    rng: np.random.Generator = None,
) -> tuple:
    """
    Compute signal strength at free-space points from multiple transmitters.

    For each point, RSSI = max over all transmitters (best signal wins).

    Args:
        tx_positions: List of (3,) arrays, transmitter world positions.
        free_space_coords: (M, 3) world coordinates of free-space points.
        subsample: Fraction of points to evaluate (0-1). Use < 1 for speed.
        rng: Random generator for subsampling.

    Returns:
        rssi_map: (M,) RSSI values (NaN for unevaluated points if subsampled).
        coverage_fraction: Fraction of evaluated points above MIN_RSSI_DBM.
    """
    n_points = len(free_space_coords)
    rssi_map = np.full(n_points, -np.inf)

    if subsample < 1.0:
        if rng is None:
            rng = np.random.default_rng()
        n_sample = max(1, int(n_points * subsample))
        sample_idx = rng.choice(n_points, n_sample, replace=False)
    else:
        sample_idx = np.arange(n_points)

    for tx_pos in tx_positions:
        tx_pos = np.asarray(tx_pos, dtype=np.float64)
        for idx in sample_idx:
            rssi = compute_rssi(
                tx_pos, free_space_coords[idx],
                wall_type_grid, occupancy,
                tx_power_dbm, freq_band,
            )
            if rssi > rssi_map[idx]:
                rssi_map[idx] = rssi

    # Coverage fraction over evaluated points only
    evaluated = rssi_map[sample_idx]
    coverage = np.mean(evaluated >= MIN_RSSI_DBM) if len(evaluated) > 0 else 0.0

    return rssi_map, coverage
