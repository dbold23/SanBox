"""
Gaussian Splat PLY loader.

Parses standard 3DGS .ply files and decodes stored values:
- opacity: sigmoid(raw) → [0, 1]
- scale: exp(raw) → positive sigma per axis
- color: 0.5 + SH_C0 * f_dc → RGB [0, 1]
- rotation: normalize quaternion to unit length
"""
import dataclasses
import numpy as np
from plyfile import PlyData

from wifi_placer.config import OPACITY_THRESHOLD, SH_C0


@dataclasses.dataclass
class GaussianSplatData:
    """Decoded Gaussian splat parameters."""
    centers: np.ndarray       # (N, 3) xyz world positions
    scales: np.ndarray        # (N, 3) sigma per axis (after exp)
    rotations: np.ndarray     # (N, 4) unit quaternions (w, x, y, z)
    opacities: np.ndarray     # (N,)   [0, 1] (after sigmoid)
    colors: np.ndarray        # (N, 3) RGB [0, 1] (from SH DC)
    num_gaussians: int


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    pos = x >= 0
    result = np.empty_like(x, dtype=np.float64)
    result[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    result[~pos] = exp_x / (1.0 + exp_x)
    return result


def _normalize_quaternions(quats: np.ndarray) -> np.ndarray:
    """Normalize each row of (N, 4) quaternion array to unit length."""
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return quats / norms


def load_splat(ply_path: str,
               opacity_threshold: float = OPACITY_THRESHOLD) -> GaussianSplatData:
    """
    Load a Gaussian Splatting .ply file and return decoded parameters.

    Args:
        ply_path: Path to the .ply file.
        opacity_threshold: Discard Gaussians with decoded opacity below this.

    Returns:
        GaussianSplatData with filtered and decoded arrays.
    """
    plydata = PlyData.read(ply_path)
    vertices = plydata["vertex"]

    # Positions
    centers = np.column_stack([
        np.asarray(vertices["x"], dtype=np.float64),
        np.asarray(vertices["y"], dtype=np.float64),
        np.asarray(vertices["z"], dtype=np.float64),
    ])

    # Scales (stored in log space)
    prop_names = [p.name for p in vertices.properties]
    if "scale_0" in prop_names:
        raw_scales = np.column_stack([
            np.asarray(vertices["scale_0"], dtype=np.float64),
            np.asarray(vertices["scale_1"], dtype=np.float64),
            np.asarray(vertices["scale_2"], dtype=np.float64),
        ])
        scales = np.exp(raw_scales)
    else:
        scales = np.full((len(centers), 3), 0.01)

    # Rotations (quaternions w, x, y, z)
    if "rot_0" in prop_names:
        raw_rots = np.column_stack([
            np.asarray(vertices["rot_0"], dtype=np.float64),
            np.asarray(vertices["rot_1"], dtype=np.float64),
            np.asarray(vertices["rot_2"], dtype=np.float64),
            np.asarray(vertices["rot_3"], dtype=np.float64),
        ])
        rotations = _normalize_quaternions(raw_rots)
    else:
        rotations = np.tile([1.0, 0.0, 0.0, 0.0], (len(centers), 1))

    # Opacity (stored as logit)
    if "opacity" in prop_names:
        raw_opacity = np.asarray(vertices["opacity"], dtype=np.float64)
        opacities = _sigmoid(raw_opacity)
    else:
        opacities = np.ones(len(centers))

    # Color from 0th-order spherical harmonics
    if "f_dc_0" in prop_names:
        f_dc = np.column_stack([
            np.asarray(vertices["f_dc_0"], dtype=np.float64),
            np.asarray(vertices["f_dc_1"], dtype=np.float64),
            np.asarray(vertices["f_dc_2"], dtype=np.float64),
        ])
        colors = np.clip(0.5 + SH_C0 * f_dc, 0.0, 1.0)
    else:
        colors = np.full((len(centers), 3), 0.5)

    # Filter by opacity threshold
    mask = opacities >= opacity_threshold
    centers = centers[mask]
    scales = scales[mask]
    rotations = rotations[mask]
    opacities = opacities[mask]
    colors = colors[mask]

    return GaussianSplatData(
        centers=centers,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        colors=colors,
        num_gaussians=len(centers),
    )
