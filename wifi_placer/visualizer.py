"""
Visualization: 3D interactive view (Open3D) and 2D heatmap (matplotlib).
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from wifi_placer.config import (
    HEATMAP_COLORMAP, MIN_RSSI_DBM, ROUTER_MARKER_RADIUS, ROUTER_MARKER_COLOR,
)
from wifi_placer.splat_loader import GaussianSplatData
from wifi_placer.voxelizer import OccupancyGrid
from wifi_placer.optimizer import PlacementResult


def _rssi_to_color(rssi_values: np.ndarray, vmin: float = -90.0, vmax: float = -30.0) -> np.ndarray:
    """Map RSSI values to RGB colors using the heatmap colormap."""
    cmap = plt.get_cmap(HEATMAP_COLORMAP)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    normalized = norm(rssi_values)
    colors = cmap(normalized)[:, :3]  # Drop alpha
    return colors


def visualize_3d(
    splat_data: GaussianSplatData,
    occupancy: OccupancyGrid,
    result: PlacementResult,
    show_occupancy: bool = False,
    show_heatmap: bool = True,
):
    """
    Launch interactive Open3D 3D visualization.

    Shows: point cloud, signal heatmap, router markers, coordinate axes.
    """
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D not installed. Skipping 3D visualization.")
        print("Install with: pip install open3d")
        return

    geometries = []

    # 1. Original point cloud (Gaussian centers)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(splat_data.centers)
    pcd.colors = o3d.utility.Vector3dVector(splat_data.colors)
    geometries.append(pcd)

    # 2. Occupancy voxel grid (optional)
    if show_occupancy:
        occupied_indices = np.argwhere(occupancy.binary_grid)
        if len(occupied_indices) > 0:
            occupied_world = occupancy.voxel_to_world(occupied_indices)
            occ_pcd = o3d.geometry.PointCloud()
            occ_pcd.points = o3d.utility.Vector3dVector(occupied_world)
            occ_pcd.colors = o3d.utility.Vector3dVector(
                np.full((len(occupied_world), 3), 0.3)  # dark gray
            )
            geometries.append(occ_pcd)

    # 3. Signal strength heatmap
    if show_heatmap and result.rssi_map is not None:
        valid = result.rssi_map > -np.inf
        if np.any(valid):
            heatmap_coords = result.free_space_coords[valid]
            heatmap_rssi = result.rssi_map[valid]
            heatmap_colors = _rssi_to_color(heatmap_rssi)

            heat_pcd = o3d.geometry.PointCloud()
            heat_pcd.points = o3d.utility.Vector3dVector(heatmap_coords)
            heat_pcd.colors = o3d.utility.Vector3dVector(heatmap_colors)
            geometries.append(heat_pcd)

    # 4. Router markers (red spheres)
    for pos in result.positions:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=ROUTER_MARKER_RADIUS)
        sphere.translate(pos)
        sphere.paint_uniform_color(ROUTER_MARKER_COLOR)
        sphere.compute_vertex_normals()
        geometries.append(sphere)

    # 5. Coordinate axes
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    geometries.append(axes)

    print("Launching 3D visualization (close window to continue)...")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="WiFi Router Placement Optimizer",
        width=1280,
        height=720,
    )


def plot_floor_plan_heatmap(
    floor_plan: np.ndarray,
    rssi_map: np.ndarray,
    free_space_coords: np.ndarray,
    occupancy: OccupancyGrid,
    router_positions: list,
    floor_z: float,
    coverage_fraction: float,
    output_path: str = "coverage_heatmap.png",
    slice_height: float = 1.0,
    slice_thickness: float = 0.3,
):
    """
    Generate a 2D matplotlib floor plan heatmap.

    Shows walls in black, RSSI as colored overlay, routers as red stars.
    """
    # Extract 2D RSSI slice at the floor plan height
    slice_z = floor_z + slice_height
    z_coords = free_space_coords[:, 2]
    z_mask = np.abs(z_coords - slice_z) <= slice_thickness
    valid = (rssi_map > -np.inf) & z_mask

    nx, ny = floor_plan.shape

    # Create 2D RSSI grid
    rssi_2d = np.full((nx, ny), np.nan)
    if np.any(valid):
        coords_2d = free_space_coords[valid]
        rssi_vals = rssi_map[valid]

        voxel_idx = occupancy.world_to_voxel(coords_2d)
        for idx, rssi in zip(voxel_idx, rssi_vals):
            i, j = idx[0], idx[1]
            if 0 <= i < nx and 0 <= j < ny:
                if np.isnan(rssi_2d[i, j]) or rssi > rssi_2d[i, j]:
                    rssi_2d[i, j] = rssi

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Floor plan extent in world coordinates
    x_min = occupancy.origin[0]
    x_max = occupancy.origin[0] + nx * occupancy.voxel_size
    y_min = occupancy.origin[1]
    y_max = occupancy.origin[1] + ny * occupancy.voxel_size
    extent = [y_min, y_max, x_min, x_max]

    # Heatmap (transposed to align x/y with plot axes)
    im = ax.imshow(
        rssi_2d, cmap=HEATMAP_COLORMAP, vmin=-90, vmax=-30,
        extent=extent, origin="lower", aspect="equal", alpha=0.7,
    )

    # Walls overlay
    wall_display = np.where(floor_plan, 0.0, np.nan)
    ax.imshow(
        wall_display, cmap="binary", vmin=0, vmax=1,
        extent=extent, origin="lower", aspect="equal", alpha=0.8,
    )

    # Router positions
    for i, pos in enumerate(router_positions):
        ax.plot(pos[1], pos[0], "r*", markersize=20, markeredgecolor="black",
                markeredgewidth=1.5, label=f"Router {i+1}" if i == 0 else "")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Signal Strength (dBm)")

    ax.set_xlabel("Y (meters)")
    ax.set_ylabel("X (meters)")
    ax.set_title(f"WiFi Coverage Heatmap — {coverage_fraction:.1%} coverage above {MIN_RSSI_DBM} dBm")
    if router_positions:
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved heatmap to {output_path}")
    plt.show()


def print_results_table(result: PlacementResult, freq_band: str):
    """Print formatted CLI results table."""
    print()
    print("=" * 50)
    print("  WiFi Router Placement Results")
    print("=" * 50)
    print(f"  Frequency band:   {freq_band} GHz")
    print(f"  Number of routers: {result.num_routers}")
    print(f"  Overall coverage:  {result.coverage_fraction:.1%}")
    print()
    print(f"  {'Router':<8} {'X (m)':>8} {'Y (m)':>8} {'Z (m)':>8}")
    print(f"  {'------':<8} {'-------':>8} {'-------':>8} {'-------':>8}")
    for i, pos in enumerate(result.positions):
        print(f"  {i+1:<8} {pos[0]:>8.2f} {pos[1]:>8.2f} {pos[2]:>8.2f}")
    print("=" * 50)
