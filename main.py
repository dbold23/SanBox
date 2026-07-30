"""
WiFi Router Placement Optimizer from Gaussian Splats
=====================================================
Takes a Gaussian splat (.ply) of an indoor space and finds
optimal WiFi router positions using signal propagation
simulation and simulated annealing.

Usage:
    python main.py --input room.ply [options]
"""
import argparse
import time

import numpy as np

from wifi_placer.config import (
    DEFAULT_TX_POWER_DBM, MIN_RSSI_DBM, VOXEL_SIZE, SA_MAX_ROUTERS,
)
from wifi_placer.splat_loader import load_splat
from wifi_placer.voxelizer import voxelize_gaussians
from wifi_placer.floor_extractor import (
    detect_floor_level, extract_floor_plan, get_free_space_mask,
    classify_wall_density,
)
from wifi_placer.optimizer import optimize_placement
from wifi_placer.visualizer import (
    visualize_3d, plot_floor_plan_heatmap, print_results_table,
)


def main():
    parser = argparse.ArgumentParser(
        description="WiFi Router Placement Optimizer from Gaussian Splats",
    )
    parser.add_argument("--input", required=True, help="Path to .ply Gaussian splat file")
    parser.add_argument("--frequency", default="5.0", choices=["2.4", "5.0"],
                        help="WiFi band (default: 5.0)")
    parser.add_argument("--tx-power", type=float, default=DEFAULT_TX_POWER_DBM,
                        help=f"Transmit power in dBm (default: {DEFAULT_TX_POWER_DBM})")
    parser.add_argument("--min-rssi", type=float, default=MIN_RSSI_DBM,
                        help=f"Min acceptable RSSI in dBm (default: {MIN_RSSI_DBM})")
    parser.add_argument("--voxel-size", type=float, default=VOXEL_SIZE,
                        help=f"Voxel size in meters (default: {VOXEL_SIZE})")
    parser.add_argument("--max-routers", type=int, default=SA_MAX_ROUTERS,
                        help=f"Max routers to try (default: {SA_MAX_ROUTERS})")
    parser.add_argument("--placement-height", type=float, default=2.0,
                        help="Router height above floor in meters (default: 2.0)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip 3D visualization")
    parser.add_argument("--save-heatmap", default="coverage_heatmap.png",
                        help="Path to save 2D heatmap (default: coverage_heatmap.png)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for optimizer (default: 42)")
    args = parser.parse_args()

    print("=" * 50)
    print("  WiFi Router Placement Optimizer")
    print("  from Gaussian Splats")
    print("=" * 50)
    print(f"  Input:       {args.input}")
    print(f"  Frequency:   {args.frequency} GHz")
    print(f"  TX Power:    {args.tx_power} dBm")
    print(f"  Min RSSI:    {args.min_rssi} dBm")
    print(f"  Voxel size:  {args.voxel_size} m")
    print(f"  Max routers: {args.max_routers}")
    print("=" * 50)

    t0 = time.time()

    # Step 1: Load Gaussian splat
    print("\n[1/5] Loading Gaussian splat...")
    splat_data = load_splat(args.input)
    print(f"  Loaded {splat_data.num_gaussians} Gaussians")

    # Step 2: Voxelize to occupancy grid
    print("\n[2/5] Building occupancy grid...")
    occupancy = voxelize_gaussians(splat_data, voxel_size=args.voxel_size)
    n_occupied = occupancy.binary_grid.sum()
    print(f"  Grid: {occupancy.shape[0]}x{occupancy.shape[1]}x{occupancy.shape[2]} voxels")
    print(f"  Occupied: {n_occupied} voxels ({n_occupied / np.prod(occupancy.shape) * 100:.1f}%)")

    # Step 3: Floor extraction and wall classification
    print("\n[3/5] Detecting floor and classifying walls...")
    floor_z = detect_floor_level(occupancy)
    floor_plan = extract_floor_plan(occupancy, floor_z)
    free_space_mask = get_free_space_mask(occupancy, floor_z)
    wall_type_grid = classify_wall_density(occupancy)
    print(f"  Floor detected at z = {floor_z:.2f} m")
    print(f"  Free space voxels: {free_space_mask.sum()}")

    # Step 4: Optimize router placement
    print("\n[4/5] Optimizing router placement...")
    result = optimize_placement(
        occupancy=occupancy,
        wall_type_grid=wall_type_grid,
        free_space_mask=free_space_mask,
        floor_z=floor_z,
        tx_power_dbm=args.tx_power,
        freq_band=args.frequency,
        max_routers=args.max_routers,
        placement_height=args.placement_height,
        seed=args.seed,
    )

    # Print results
    print_results_table(result, args.frequency)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    # Step 5: Visualize
    print("\n[5/5] Generating visualizations...")

    # 2D heatmap (always)
    plot_floor_plan_heatmap(
        floor_plan=floor_plan,
        rssi_map=result.rssi_map,
        free_space_coords=result.free_space_coords,
        occupancy=occupancy,
        router_positions=result.positions,
        floor_z=floor_z,
        coverage_fraction=result.coverage_fraction,
        output_path=args.save_heatmap,
    )

    # 3D visualization (optional)
    if not args.no_viz:
        visualize_3d(splat_data, occupancy, result)


if __name__ == "__main__":
    main()
