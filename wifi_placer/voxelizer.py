"""
Voxelizer: Convert Gaussian splat data into a 3D occupancy grid.

Uses a KD-tree for efficient spatial queries and accumulates
Gaussian density contributions per voxel (isotropic approximation).
"""
import dataclasses
import warnings
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

from wifi_placer.config import (
    VOXEL_SIZE, DENSITY_THRESHOLD, GAUSSIAN_CUTOFF_SIGMAS, MAX_VOXEL_GRID_DIM,
)
from wifi_placer.splat_loader import GaussianSplatData


@dataclasses.dataclass
class OccupancyGrid:
    """A 3D voxel grid representing occupied/free space."""
    grid: np.ndarray         # (Nx, Ny, Nz) float64, density [0, 1]
    binary_grid: np.ndarray  # (Nx, Ny, Nz) bool, True = occupied
    voxel_size: float
    origin: np.ndarray       # (3,) world coords of grid[0,0,0] corner
    shape: tuple             # (Nx, Ny, Nz)

    def world_to_voxel(self, world_xyz: np.ndarray) -> np.ndarray:
        """Convert world coordinates (N, 3) to voxel indices (N, 3) int."""
        return ((world_xyz - self.origin) / self.voxel_size).astype(int)

    def voxel_to_world(self, voxel_ijk: np.ndarray) -> np.ndarray:
        """Convert voxel indices (N, 3) to world coordinates at voxel centers."""
        return voxel_ijk.astype(float) * self.voxel_size + self.origin + self.voxel_size / 2.0

    def is_inside(self, voxel_ijk: np.ndarray) -> np.ndarray:
        """Check if voxel indices are within grid bounds."""
        if voxel_ijk.ndim == 1:
            voxel_ijk = voxel_ijk[np.newaxis, :]
        return np.all(
            (voxel_ijk >= 0) & (voxel_ijk < np.array(self.shape)), axis=1
        )


def voxelize_gaussians(
    splat_data: GaussianSplatData,
    voxel_size: float = VOXEL_SIZE,
    density_threshold: float = DENSITY_THRESHOLD,
) -> OccupancyGrid:
    """
    Convert Gaussian splat data into a 3D occupancy grid.

    Algorithm:
    1. Compute bounding box expanded by max Gaussian extent.
    2. Create grid, build KD-tree on Gaussian centers.
    3. For each voxel, query nearby Gaussians and accumulate density.
    4. Normalize and threshold to binary.
    """
    centers = splat_data.centers
    scales = splat_data.scales
    opacities = splat_data.opacities

    # Max scale per Gaussian (isotropic radius)
    max_scales = np.max(scales, axis=1)
    cutoff_radius = GAUSSIAN_CUTOFF_SIGMAS * np.median(max_scales)

    # Bounding box with padding
    bbox_min = centers.min(axis=0) - cutoff_radius
    bbox_max = centers.max(axis=0) + cutoff_radius

    # Grid dimensions
    dims = np.ceil((bbox_max - bbox_min) / voxel_size).astype(int)

    # Safety: cap grid size
    if np.any(dims > MAX_VOXEL_GRID_DIM):
        scale_factor = np.max(dims) / MAX_VOXEL_GRID_DIM
        new_voxel_size = voxel_size * scale_factor
        warnings.warn(
            f"Grid too large ({dims}), increasing voxel_size "
            f"from {voxel_size:.3f} to {new_voxel_size:.3f}m"
        )
        voxel_size = new_voxel_size
        dims = np.ceil((bbox_max - bbox_min) / voxel_size).astype(int)

    dims = np.minimum(dims, MAX_VOXEL_GRID_DIM)
    nx, ny, nz = dims

    # Build KD-tree
    tree = cKDTree(centers)

    # Generate voxel centers
    ix = np.arange(nx)
    iy = np.arange(ny)
    iz = np.arange(nz)
    grid_ix, grid_iy, grid_iz = np.meshgrid(ix, iy, iz, indexing="ij")
    voxel_indices = np.column_stack([
        grid_ix.ravel(), grid_iy.ravel(), grid_iz.ravel()
    ])
    voxel_centers = (
        voxel_indices.astype(float) * voxel_size
        + bbox_min
        + voxel_size / 2.0
    )

    # Density accumulation
    density_grid = np.zeros(nx * ny * nz, dtype=np.float64)

    # Process in batches for memory efficiency
    batch_size = 50000
    n_voxels = len(voxel_centers)

    for start in tqdm(range(0, n_voxels, batch_size), desc="Voxelizing"):
        end = min(start + batch_size, n_voxels)
        batch_centers = voxel_centers[start:end]

        # Query KD-tree for nearby Gaussians
        neighbors = tree.query_ball_point(batch_centers, r=cutoff_radius, workers=-1)

        for i, neighbor_ids in enumerate(neighbors):
            if len(neighbor_ids) == 0:
                continue
            nids = np.array(neighbor_ids)
            g_centers = centers[nids]     # (K, 3)
            g_scales = scales[nids]       # (K, 3)
            g_opacities = opacities[nids] # (K,)

            # Isotropic Gaussian evaluation (ignore rotation)
            diff = batch_centers[i] - g_centers  # (K, 3)
            sq_dist_normalized = np.sum(diff ** 2 / (g_scales ** 2), axis=1)  # (K,)
            contributions = g_opacities * np.exp(-0.5 * sq_dist_normalized)
            density_grid[start + i] = np.sum(contributions)

    # Reshape and normalize
    density_grid = density_grid.reshape((nx, ny, nz))
    max_density = density_grid.max()
    if max_density > 0:
        density_grid /= max_density

    binary_grid = density_grid >= density_threshold

    return OccupancyGrid(
        grid=density_grid,
        binary_grid=binary_grid,
        voxel_size=voxel_size,
        origin=bbox_min.copy(),
        shape=(nx, ny, nz),
    )
