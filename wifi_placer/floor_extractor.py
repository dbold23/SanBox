"""
Floor detection, 2D floor plan extraction, and wall density classification.

Identifies the floor plane from the occupancy grid's z-histogram,
creates 2D slices for visualization, and classifies occupied voxels
into wall material categories based on density.
"""
import numpy as np
from scipy.signal import find_peaks

from wifi_placer.config import (
    FLOOR_SEARCH_BAND, SLICE_HEIGHT_ABOVE_FLOOR, SLICE_THICKNESS,
    DENSITY_THRESHOLD,
)
from wifi_placer.voxelizer import OccupancyGrid


def detect_floor_level(occupancy: OccupancyGrid) -> float:
    """
    Find the floor height (world z-coordinate).

    Computes occupied-voxel count per z-layer and finds the lowest
    significant peak — that's the floor.
    """
    nz = occupancy.shape[2]
    z_counts = np.zeros(nz)
    for k in range(nz):
        z_counts[k] = occupancy.binary_grid[:, :, k].sum()

    # Normalize for peak detection
    if z_counts.max() > 0:
        z_norm = z_counts / z_counts.max()
    else:
        z_norm = z_counts

    peaks, properties = find_peaks(z_norm, height=0.3, distance=3)

    if len(peaks) > 0:
        # Take the lowest peak (smallest z index) as the floor
        floor_k = peaks[0]
    else:
        # Fallback: highest count in bottom 30%
        bottom = max(1, int(nz * 0.3))
        floor_k = np.argmax(z_counts[:bottom])

    # Convert voxel z-index to world coordinate
    floor_z = floor_k * occupancy.voxel_size + occupancy.origin[2] + occupancy.voxel_size / 2
    return floor_z


def extract_floor_plan(
    occupancy: OccupancyGrid,
    floor_z: float,
    height_above_floor: float = SLICE_HEIGHT_ABOVE_FLOOR,
    thickness: float = SLICE_THICKNESS,
) -> np.ndarray:
    """
    Extract a 2D floor plan at a given height above the floor.

    Returns (Nx, Ny) array where True = wall/obstacle, False = free space.
    Uses max-projection through a thin horizontal band.
    """
    slice_z = floor_z + height_above_floor
    z_min = slice_z - thickness
    z_max = slice_z + thickness

    # Convert to voxel z-indices
    k_min = max(0, int((z_min - occupancy.origin[2]) / occupancy.voxel_size))
    k_max = min(occupancy.shape[2], int((z_max - occupancy.origin[2]) / occupancy.voxel_size) + 1)

    if k_min >= k_max:
        return np.zeros((occupancy.shape[0], occupancy.shape[1]), dtype=bool)

    return occupancy.binary_grid[:, :, k_min:k_max].any(axis=2)


def get_free_space_mask(occupancy: OccupancyGrid, floor_z: float) -> np.ndarray:
    """
    3D boolean mask of free-space voxels (not occupied, above floor, below ceiling).
    Router candidates are restricted to these voxels.
    """
    # Find ceiling: highest peak in z-histogram
    nz = occupancy.shape[2]
    z_counts = np.zeros(nz)
    for k in range(nz):
        z_counts[k] = occupancy.binary_grid[:, :, k].sum()

    z_norm = z_counts / max(z_counts.max(), 1)
    peaks, _ = find_peaks(z_norm, height=0.3, distance=3)

    if len(peaks) >= 2:
        ceiling_k = peaks[-1]
    else:
        ceiling_k = nz - 1

    floor_k = max(0, int((floor_z - occupancy.origin[2]) / occupancy.voxel_size))

    mask = ~occupancy.binary_grid.copy()
    # Zero out below floor and above ceiling
    if floor_k > 0:
        mask[:, :, :floor_k] = False
    if ceiling_k < nz - 1:
        mask[:, :, ceiling_k + 1:] = False

    return mask


def classify_wall_density(occupancy: OccupancyGrid) -> np.ndarray:
    """
    Classify each voxel into a wall type based on density.

    Returns (Nx, Ny, Nz) int array:
      0 = free space
      1 = light wall   (density in [threshold, 0.6))
      2 = medium wall   (density in [0.6, 0.85))
      3 = heavy wall    (density >= 0.85)
    """
    result = np.zeros(occupancy.shape, dtype=np.int32)
    d = occupancy.grid

    result[d >= DENSITY_THRESHOLD] = 1          # light
    result[d >= 0.6] = 2                         # medium
    result[d >= 0.85] = 3                        # heavy

    return result
