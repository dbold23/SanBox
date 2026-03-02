"""
Simulated annealing optimizer for WiFi router placement.

Two-level optimization:
1. Outer loop: iterate router count (1, 2, ..., MAX_ROUTERS)
2. Inner loop: SA to find best positions for that count
Stops adding routers when marginal coverage improvement < threshold.
"""
import dataclasses
import numpy as np
from tqdm import tqdm

from wifi_placer.config import (
    DEFAULT_TX_POWER_DBM, MIN_RSSI_DBM,
    SA_T_MAX, SA_T_MIN, SA_COOLING_RATE, SA_STEPS_PER_TEMP,
    SA_MAX_ROUTERS, SA_COVERAGE_IMPROVEMENT_THRESHOLD,
)
from wifi_placer.voxelizer import OccupancyGrid
from wifi_placer.propagation import compute_coverage_map


@dataclasses.dataclass
class PlacementResult:
    """Result of the optimization."""
    positions: list          # List of (3,) arrays, world coordinates
    num_routers: int
    coverage_fraction: float
    rssi_map: np.ndarray     # RSSI at each free-space point
    free_space_coords: np.ndarray


def _get_candidate_positions(
    occupancy: OccupancyGrid,
    free_space_mask: np.ndarray,
    placement_height_z: float,
    height_tolerance: float = 0.2,
) -> np.ndarray:
    """
    Get candidate router positions: free-space voxels near placement height.

    Args:
        placement_height_z: World z-coordinate for router placement.
        height_tolerance: Accept voxels within this range of placement_height_z.

    Returns:
        (K, 3) array of world coordinates.
    """
    indices = np.argwhere(free_space_mask)  # (M, 3) ijk
    if len(indices) == 0:
        return np.empty((0, 3))

    world_coords = occupancy.voxel_to_world(indices)

    # Filter to near placement height
    z_mask = np.abs(world_coords[:, 2] - placement_height_z) <= height_tolerance
    candidates = world_coords[z_mask]

    if len(candidates) == 0:
        # Fallback: use all free-space points
        candidates = world_coords

    return candidates


def _simulated_annealing(
    n_routers: int,
    candidate_positions: np.ndarray,
    occupancy: OccupancyGrid,
    wall_type_grid: np.ndarray,
    free_space_coords: np.ndarray,
    tx_power_dbm: float,
    freq_band: str,
    initial_positions: list = None,
    rng: np.random.Generator = None,
    t_max: float = SA_T_MAX,
    t_min: float = SA_T_MIN,
    cooling_rate: float = SA_COOLING_RATE,
    steps_per_temp: int = SA_STEPS_PER_TEMP,
) -> tuple:
    """
    SA to optimize n_routers positions.

    Returns: (best_positions, best_coverage)
    """
    if rng is None:
        rng = np.random.default_rng()

    n_candidates = len(candidate_positions)
    if n_candidates == 0:
        return [], 0.0

    # Initialize positions
    if initial_positions is not None and len(initial_positions) > 0:
        current_indices = []
        for pos in initial_positions:
            dists = np.linalg.norm(candidate_positions - pos, axis=1)
            current_indices.append(np.argmin(dists))
        # Add random positions for new routers
        while len(current_indices) < n_routers:
            current_indices.append(rng.integers(0, n_candidates))
    else:
        current_indices = [rng.integers(0, n_candidates) for _ in range(n_routers)]

    current_positions = [candidate_positions[i] for i in current_indices]

    # Evaluate initial state (subsampled for speed)
    _, current_coverage = compute_coverage_map(
        current_positions, occupancy, wall_type_grid, free_space_coords,
        tx_power_dbm, freq_band, subsample=0.25, rng=rng,
    )

    best_positions = list(current_positions)
    best_coverage = current_coverage

    T = t_max
    max_grid_dim = max(occupancy.shape)

    # Count total iterations for progress bar
    n_temps = int(np.log(t_min / t_max) / np.log(cooling_rate)) + 1
    total_steps = n_temps * steps_per_temp
    pbar = tqdm(total=total_steps, desc=f"SA ({n_routers} routers)", leave=False)

    while T > t_min:
        for _ in range(steps_per_temp):
            pbar.update(1)

            # Pick a random router to perturb
            router_idx = rng.integers(0, n_routers)

            # Perturbation radius shrinks with temperature
            radius = max(1, int(T / SA_T_MAX * max_grid_dim * 0.3))

            # Find nearby candidates
            current_pos = current_positions[router_idx]
            dists = np.linalg.norm(candidate_positions - current_pos, axis=1)
            nearby_mask = dists <= radius * occupancy.voxel_size
            nearby_indices = np.where(nearby_mask)[0]

            if len(nearby_indices) == 0:
                new_idx = rng.integers(0, n_candidates)
            else:
                new_idx = rng.choice(nearby_indices)

            # Create neighbor state
            new_positions = list(current_positions)
            new_positions[router_idx] = candidate_positions[new_idx]

            _, new_coverage = compute_coverage_map(
                new_positions, occupancy, wall_type_grid, free_space_coords,
                tx_power_dbm, freq_band, subsample=0.25, rng=rng,
            )

            # Acceptance criterion
            delta = new_coverage - current_coverage
            if delta > 0 or rng.random() < np.exp(delta / (T / SA_T_MAX)):
                current_positions = new_positions
                current_coverage = new_coverage
                if current_coverage > best_coverage:
                    best_positions = list(current_positions)
                    best_coverage = current_coverage

        T *= cooling_rate

    pbar.close()
    return best_positions, best_coverage


def optimize_placement(
    occupancy: OccupancyGrid,
    wall_type_grid: np.ndarray,
    free_space_mask: np.ndarray,
    floor_z: float,
    tx_power_dbm: float = DEFAULT_TX_POWER_DBM,
    freq_band: str = "5.0",
    max_routers: int = SA_MAX_ROUTERS,
    placement_height: float = 2.0,
    seed: int = 42,
    t_max: float = SA_T_MAX,
    t_min: float = SA_T_MIN,
    cooling_rate: float = SA_COOLING_RATE,
    steps_per_temp: int = SA_STEPS_PER_TEMP,
) -> PlacementResult:
    """
    Find optimal number and positions of WiFi routers.

    Uses greedy-additive approach: optimize 1 router, then 2, etc.
    Stops when marginal improvement < threshold.
    """
    rng = np.random.default_rng(seed)

    # Get free-space points for evaluation
    free_indices = np.argwhere(free_space_mask)
    if len(free_indices) == 0:
        raise ValueError("No free space found in occupancy grid")
    free_space_coords = occupancy.voxel_to_world(free_indices)

    # Get candidate router positions
    placement_z = floor_z + placement_height
    candidates = _get_candidate_positions(
        occupancy, free_space_mask, placement_z,
    )

    if len(candidates) == 0:
        raise ValueError("No candidate positions for router placement")

    print(f"Free space points: {len(free_space_coords)}")
    print(f"Candidate router positions: {len(candidates)}")

    prev_coverage = 0.0
    best_positions = []

    for n in range(1, max_routers + 1):
        print(f"\nOptimizing for {n} router(s)...")

        positions, coverage = _simulated_annealing(
            n_routers=n,
            candidate_positions=candidates,
            occupancy=occupancy,
            wall_type_grid=wall_type_grid,
            free_space_coords=free_space_coords,
            tx_power_dbm=tx_power_dbm,
            freq_band=freq_band,
            initial_positions=best_positions,
            rng=rng,
            t_max=t_max,
            t_min=t_min,
            cooling_rate=cooling_rate,
            steps_per_temp=steps_per_temp,
        )

        # Full evaluation (not subsampled)
        rssi_map, coverage_full = compute_coverage_map(
            positions, occupancy, wall_type_grid, free_space_coords,
            tx_power_dbm, freq_band, subsample=1.0,
        )

        improvement = coverage_full - prev_coverage
        print(f"  Coverage: {coverage_full:.1%} (improvement: {improvement:+.1%})")

        if n > 1 and improvement < SA_COVERAGE_IMPROVEMENT_THRESHOLD:
            print(f"  Marginal improvement below {SA_COVERAGE_IMPROVEMENT_THRESHOLD:.0%} threshold, stopping.")
            # Recompute final map with previous best
            rssi_map, coverage_full = compute_coverage_map(
                best_positions, occupancy, wall_type_grid, free_space_coords,
                tx_power_dbm, freq_band, subsample=1.0,
            )
            break

        best_positions = positions
        prev_coverage = coverage_full

    return PlacementResult(
        positions=best_positions,
        num_routers=len(best_positions),
        coverage_fraction=prev_coverage,
        rssi_map=rssi_map,
        free_space_coords=free_space_coords,
    )
