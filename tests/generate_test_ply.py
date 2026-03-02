"""
Generate a synthetic Gaussian splat .ply file of a rectangular room.

Creates 4 walls + floor + ceiling with high-opacity Gaussians,
plus low-opacity noise in the interior for testing the filter.
"""
import argparse
import numpy as np
from plyfile import PlyData, PlyElement

from wifi_placer.config import SH_C0


def _inverse_sigmoid(y: float) -> float:
    """Logit function: inverse of sigmoid."""
    y = np.clip(y, 1e-7, 1.0 - 1e-7)
    return np.log(y / (1.0 - y))


def _scatter_on_plane(normal_axis: int, offset: float,
                      bounds_min: np.ndarray, bounds_max: np.ndarray,
                      n_points: int, thickness: float,
                      rng: np.random.Generator) -> np.ndarray:
    """
    Scatter points uniformly on a plane defined by normal_axis=offset,
    within the bounding box, with random perturbation along the normal.
    """
    axes = [0, 1, 2]
    tangent_axes = [a for a in axes if a != normal_axis]

    points = np.zeros((n_points, 3))
    points[:, normal_axis] = offset + rng.uniform(
        -thickness / 2, thickness / 2, n_points
    )
    for a in tangent_axes:
        points[:, a] = rng.uniform(bounds_min[a], bounds_max[a], n_points)

    return points


def generate_test_room_ply(
    output_path: str,
    room_dims: tuple = (5.0, 4.0, 3.0),
    wall_thickness: float = 0.15,
    n_wall_splats_per_surface: int = 2000,
    n_noise_splats: int = 500,
    seed: int = 42,
):
    """
    Generate a synthetic room .ply file.

    Args:
        output_path: Where to write the .ply file.
        room_dims: (width_x, depth_y, height_z) in meters.
        wall_thickness: Thickness of wall splat scatter.
        n_wall_splats_per_surface: Gaussians per wall/floor/ceiling.
        n_noise_splats: Low-opacity Gaussians in interior.
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)
    w, d, h = room_dims
    bmin = np.array([0.0, 0.0, 0.0])
    bmax = np.array([w, d, h])

    all_centers = []

    # 6 surfaces: floor (z=0), ceiling (z=h), walls at x=0, x=w, y=0, y=d
    surfaces = [
        (2, 0.0),   # floor
        (2, h),      # ceiling
        (0, 0.0),   # wall x=0
        (0, w),      # wall x=w
        (1, 0.0),   # wall y=0
        (1, d),      # wall y=d
    ]

    for normal_axis, offset in surfaces:
        pts = _scatter_on_plane(
            normal_axis, offset, bmin, bmax,
            n_wall_splats_per_surface, wall_thickness, rng,
        )
        all_centers.append(pts)

    wall_centers = np.concatenate(all_centers, axis=0)
    n_walls = len(wall_centers)

    # Interior noise (low opacity, should be filtered out)
    margin = wall_thickness * 2
    noise_centers = np.column_stack([
        rng.uniform(margin, w - margin, n_noise_splats),
        rng.uniform(margin, d - margin, n_noise_splats),
        rng.uniform(margin, h - margin, n_noise_splats),
    ])

    centers = np.concatenate([wall_centers, noise_centers], axis=0)
    n_total = len(centers)

    # Scales (log space): small splats for walls, slightly larger for noise
    wall_log_scales = np.full((n_walls, 3), np.log(0.02))
    noise_log_scales = np.full((n_noise_splats, 3), np.log(0.05))
    log_scales = np.concatenate([wall_log_scales, noise_log_scales], axis=0)

    # Rotations: identity quaternion
    rotations = np.tile([1.0, 0.0, 0.0, 0.0], (n_total, 1))

    # Opacity (logit space): high for walls, low for noise
    wall_opacity_logit = np.full(n_walls, _inverse_sigmoid(0.9))
    noise_opacity_logit = np.full(n_noise_splats, _inverse_sigmoid(0.05))
    opacity_logit = np.concatenate([wall_opacity_logit, noise_opacity_logit])

    # Color SH DC coefficients: gray walls, random noise
    gray_val = (0.5 - 0.5) / SH_C0  # f_dc that maps to RGB=0.5
    wall_f_dc = np.full((n_walls, 3), gray_val)
    noise_f_dc = rng.uniform(-1, 1, (n_noise_splats, 3))
    f_dc = np.concatenate([wall_f_dc, noise_f_dc], axis=0)

    # Build PLY vertex array
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ("opacity", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ]

    vertex_data = np.empty(n_total, dtype=dtype)
    vertex_data["x"] = centers[:, 0].astype(np.float32)
    vertex_data["y"] = centers[:, 1].astype(np.float32)
    vertex_data["z"] = centers[:, 2].astype(np.float32)
    vertex_data["scale_0"] = log_scales[:, 0].astype(np.float32)
    vertex_data["scale_1"] = log_scales[:, 1].astype(np.float32)
    vertex_data["scale_2"] = log_scales[:, 2].astype(np.float32)
    vertex_data["rot_0"] = rotations[:, 0].astype(np.float32)
    vertex_data["rot_1"] = rotations[:, 1].astype(np.float32)
    vertex_data["rot_2"] = rotations[:, 2].astype(np.float32)
    vertex_data["rot_3"] = rotations[:, 3].astype(np.float32)
    vertex_data["opacity"] = opacity_logit.astype(np.float32)
    vertex_data["f_dc_0"] = f_dc[:, 0].astype(np.float32)
    vertex_data["f_dc_1"] = f_dc[:, 1].astype(np.float32)
    vertex_data["f_dc_2"] = f_dc[:, 2].astype(np.float32)

    el = PlyElement.describe(vertex_data, "vertex")
    PlyData([el], text=False).write(output_path)
    print(f"Wrote {n_total} Gaussians to {output_path}")
    print(f"  Wall splats: {n_walls}, Noise splats: {n_noise_splats}")
    print(f"  Room dimensions: {w}m x {d}m x {h}m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic room .ply")
    parser.add_argument("--output", default="test_room.ply", help="Output path")
    parser.add_argument("--width", type=float, default=5.0)
    parser.add_argument("--depth", type=float, default=4.0)
    parser.add_argument("--height", type=float, default=3.0)
    parser.add_argument("--splats-per-surface", type=int, default=2000)
    args = parser.parse_args()
    generate_test_room_ply(
        args.output,
        room_dims=(args.width, args.depth, args.height),
        n_wall_splats_per_surface=args.splats_per_surface,
    )
