"""
Sevengill skeleton definition: a 35-joint kinematic tree built on a SERIAL SPINE.

Drop-in replacement candidate for the SEVENGILL case of
``shark_pose/core/skeleton.py``. The public API surface mirrors that module exactly
(``SHARK_KEYPOINT_SEQUENCE``, ``NUM_JOINTS``, ``JOINT_NAME_TO_IDX``, ``MORPHOMETRIC_PAIRS``,
``KINEMATIC_TREE``, ``ROOT_JOINT``, ``ROOT_IDX``, ``get_parent_indices``, ``PARENT_INDICES``,
``get_kinematic_chain``, ``get_children``, ``build_adjacency_matrix``, ``ADJACENCY_MATRIX``,
``get_bone_pairs``, ``BONE_PAIRS``, ``NUM_BONES``) so that a species switch can select between
the two without any caller change.

What is different, and why:

* The white-shark tree is a STAR — nearly every joint parents directly to
  ``body_midpoint_dorsal``, leaving ``gill_slit`` and ``caudal_notch`` as the only two
  lateral-bending degrees of freedom for the whole animal. That is adequate for a thunniform
  lamnid and it is the wrong model class for an elongate, laterally flexible hexanchiform.
  Here the trunk is a SERIAL CHAIN of 13 joints from cranium to caudal axis, giving 12 axial
  bending segments, compressible to 4-6 modes (``NUM_BENDING_MODES``).

* The root is not a constructed mid-body point. ``body_midpoint_dorsal`` is a Type III point
  defined against a body axis that, on a bending animal, is a curve that changes every frame;
  a noisy root propagates error into every downstream joint. The root here is
  ``spine_00_cranium``, in the rigid anterior region, anchored by Type I landmarks (eye, naris,
  rictus) and immediately adjacent to the chart's arc-length origin at the first gill slit.

* Fin TIPS are leaves, never chain joints. They are Type III extremal points that slide.

* The axial chain continues INTO THE UPPER CAUDAL LOBE rather than branching symmetrically to
  both tips: the tail is strongly heterocercal, so the vertebral axis turns up into the long
  upper lobe. The weak lower lobe hangs off the chain as a leaf.

Keypoint schema: ``keypoints_sevengill_v1.yaml`` (Schema S1, 30 points). The 30 annotation
keypoints map onto 30 of the 35 joints; the five unmapped joints are unlabelled interior spine
stations (two branchial, two caudal-axis, one cranial) that exist to carry bending DOF.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
SPECIES: str = "Notorynchus cepedianus"
SCHEMA_NAME: str = "keypoints_sevengill_v1"
SCHEMA_VERSION: str = "S1"

# The identity signal is not bilaterally symmetric and cross-flank matching is near-chance;
# a mirrored left flank is a fabricated right flank. See the schema yaml.
ALLOW_HORIZONTAL_FLIP: bool = False

# ---------------------------------------------------------------------------
# Serial spine. Root-to-tail, one parent each, no branching within the chain.
# ---------------------------------------------------------------------------
SPINE_JOINTS: List[str] = [
    "spine_00_cranium",          # root; neurocranium, between the eyes
    "spine_01_branchial_1",      # station of gill slit 1  (chart arc-length origin, s = 0)
    "spine_02_branchial_7",      # station of gill slit 7  (posterior bound of the rigid head)
    "spine_03_trunk_01",         # midline_01, axis fraction 0.125
    "spine_04_trunk_02",         # midline_02, axis fraction 0.250
    "spine_05_trunk_03",         # midline_03, axis fraction 0.375
    "spine_06_trunk_04",         # midline_04, axis fraction 0.500
    "spine_07_trunk_05",         # midline_05, axis fraction 0.625
    "spine_08_trunk_06",         # midline_06, axis fraction 0.750
    "spine_09_trunk_07",         # midline_07, axis fraction 0.875
    "spine_10_precaudal",        # precaudal pit; the tube chart terminates here
    "spine_11_caudal_axis_1",    # axis continues INTO the upper lobe (heterocercal)
    "spine_12_caudal_axis_2",
]

NUM_SPINE_JOINTS: int = len(SPINE_JOINTS)          # 13
NUM_SPINE_SEGMENTS: int = NUM_SPINE_JOINTS - 1     # 12 axial bending segments

# Axis fractions of the snout-to-precaudal centerline for the seven midline semilandmarks,
# duplicated from the schema yaml so this module stands alone; the test suite asserts they agree.
MIDLINE_AXIS_FRACTIONS: Tuple[float, ...] = (0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875)

# ---------------------------------------------------------------------------
# Full joint sequence. ORDERED SO THAT EVERY PARENT PRECEDES ITS CHILDREN, which
# the white-shark module does not do (its root sits at index 14).
# Spine first, then appendage leaves.
# ---------------------------------------------------------------------------
_APPENDAGE_JOINTS: List[str] = [
    "snout_tip",
    "naris_anterior_margin",
    "eye_center",
    "mouth_rictus",
    "spiracle",
    "gill_slit_1_dorsal_origin",
    "gill_slit_1_ventral_terminus",
    "gill_slit_7_dorsal_origin",
    "gill_slit_7_ventral_terminus",
    "pectoral_origin",
    "pectoral_insertion",
    "pectoral_fin_tip",
    "pelvic_origin",
    "cloaca",
    "dorsal_fin_origin",
    "dorsal_fin_insertion",
    "dorsal_fin_apex",
    "anal_fin_origin",
    "anal_fin_insertion",
    "caudal_subterminal_notch",
    "caudal_upper_lobe_tip",
    "caudal_lower_lobe_tip",
]

JOINT_SEQUENCE: List[str] = SPINE_JOINTS + _APPENDAGE_JOINTS

# Legacy alias: in the white-shark module this name indexes the joint tensor. Keeping it means
# callers that do `from ...skeleton import SHARK_KEYPOINT_SEQUENCE` keep working. For sevengills
# the ANNOTATION keypoints are a different, smaller list — SEVENGILL_KEYPOINT_SEQUENCE below.
SHARK_KEYPOINT_SEQUENCE: List[str] = JOINT_SEQUENCE

NUM_JOINTS: int = len(JOINT_SEQUENCE)  # 35

JOINT_NAME_TO_IDX: Dict[str, int] = {
    name: idx for idx, name in enumerate(JOINT_SEQUENCE)
}

# ---------------------------------------------------------------------------
# Annotation keypoints (Schema S1 order, ids 0-29) and their joint mapping.
# ---------------------------------------------------------------------------
SEVENGILL_KEYPOINT_SEQUENCE: List[str] = [
    "snout_tip",                    # 0
    "naris_anterior_margin",        # 1
    "eye_center",                   # 2
    "mouth_rictus",                 # 3
    "spiracle",                     # 4
    "gill_slit_1_dorsal_origin",    # 5
    "gill_slit_1_ventral_terminus", # 6
    "gill_slit_7_dorsal_origin",    # 7
    "gill_slit_7_ventral_terminus", # 8
    "pectoral_origin",              # 9
    "pectoral_insertion",           # 10
    "pectoral_fin_tip",             # 11
    "pelvic_origin",                # 12
    "cloaca",                       # 13
    "dorsal_fin_origin",            # 14
    "dorsal_fin_insertion",         # 15
    "dorsal_fin_apex",              # 16
    "anal_fin_origin",              # 17
    "anal_fin_insertion",           # 18
    "precaudal_pit",                # 19
    "caudal_subterminal_notch",     # 20
    "caudal_upper_lobe_tip",        # 21
    "caudal_lower_lobe_tip",        # 22
    "midline_01",                   # 23
    "midline_02",                   # 24
    "midline_03",                   # 25
    "midline_04",                   # 26
    "midline_05",                   # 27
    "midline_06",                   # 28
    "midline_07",                   # 29
]

NUM_KEYPOINTS: int = len(SEVENGILL_KEYPOINT_SEQUENCE)  # 30

# Total map keypoint -> joint. Every keypoint has exactly one joint. Not surjective: the five
# unlabelled interior spine stations carry bending DOF and are never annotated.
KEYPOINT_TO_JOINT: Dict[str, str] = {
    "precaudal_pit": "spine_10_precaudal",
    "midline_01": "spine_03_trunk_01",
    "midline_02": "spine_04_trunk_02",
    "midline_03": "spine_05_trunk_03",
    "midline_04": "spine_06_trunk_04",
    "midline_05": "spine_07_trunk_05",
    "midline_06": "spine_08_trunk_06",
    "midline_07": "spine_09_trunk_07",
}
for _kp in SEVENGILL_KEYPOINT_SEQUENCE:
    KEYPOINT_TO_JOINT.setdefault(_kp, _kp)
del _kp

KEYPOINT_TO_JOINT_IDX: Dict[str, int] = {
    kp: JOINT_NAME_TO_IDX[joint] for kp, joint in KEYPOINT_TO_JOINT.items()
}

UNLABELLED_JOINTS: List[str] = [
    name for name in JOINT_SEQUENCE if name not in set(KEYPOINT_TO_JOINT.values())
]

# ---------------------------------------------------------------------------
# Morphometric measurement pairs, expressed in JOINT names.
# Compagno nomenclature; point-to-point on a possibly-bent animal, not caliper distances.
# `interdorsal_space` and `second_dorsal_height` are absent: no second dorsal fin.
# ---------------------------------------------------------------------------
MORPHOMETRIC_PAIRS: Dict[str, Tuple[str, str]] = {
    "total_length_proxy": ("snout_tip", "caudal_upper_lobe_tip"),
    "precaudal_length": ("snout_tip", "spine_10_precaudal"),
    "head_length": ("snout_tip", "gill_slit_7_dorsal_origin"),
    "prebranchial_length": ("snout_tip", "gill_slit_1_dorsal_origin"),
    "branchial_span": ("gill_slit_1_dorsal_origin", "gill_slit_7_dorsal_origin"),
    "gill_slit_1_height": ("gill_slit_1_dorsal_origin", "gill_slit_1_ventral_terminus"),
    "gill_slit_7_height": ("gill_slit_7_dorsal_origin", "gill_slit_7_ventral_terminus"),
    "prepectoral_length": ("snout_tip", "pectoral_origin"),
    "predorsal_length": ("snout_tip", "dorsal_fin_origin"),
    "prepelvic_length": ("snout_tip", "pelvic_origin"),
    "preanal_length": ("snout_tip", "anal_fin_origin"),
    "pectoral_anterior_margin": ("pectoral_origin", "pectoral_fin_tip"),
    "pectoral_base_chord": ("pectoral_origin", "pectoral_insertion"),
    "dorsal_base_chord": ("dorsal_fin_origin", "dorsal_fin_insertion"),
    "dorsal_height": ("dorsal_fin_origin", "dorsal_fin_apex"),
    "anal_base_chord": ("anal_fin_origin", "anal_fin_insertion"),
    "caudal_upper_lobe_length": ("spine_10_precaudal", "caudal_upper_lobe_tip"),
    "caudal_span": ("caudal_upper_lobe_tip", "caudal_lower_lobe_tip"),
}

# ---------------------------------------------------------------------------
# Kinematic tree: child -> parent.
# Root = spine_00_cranium (index 0), in the rigid anterior region.
# The 13 spine joints form a strict serial chain; everything else is a leaf or a
# two-deep appendage hanging off its nearest spine station.
#
# ⚠ The station assignments for the median and pelvic fins (which spine joint each parents to)
# are PROVISIONAL. No published Notorynchus cepedianus fin-station proportions were retrieved
# [UNVERIFIED]; they are placed from the qualitative anatomy — single dorsal far posterior, over
# or behind the pelvics, anal beneath and slightly behind it — and must be re-derived from the
# first annotated frames. Changing a parent here is a one-line edit and no other code depends on
# the specific station.
# ---------------------------------------------------------------------------
KINEMATIC_TREE: Dict[str, Optional[str]] = {
    # --- serial spine -------------------------------------------------------
    "spine_00_cranium": None,  # ROOT
    "spine_01_branchial_1": "spine_00_cranium",
    "spine_02_branchial_7": "spine_01_branchial_1",
    "spine_03_trunk_01": "spine_02_branchial_7",
    "spine_04_trunk_02": "spine_03_trunk_01",
    "spine_05_trunk_03": "spine_04_trunk_02",
    "spine_06_trunk_04": "spine_05_trunk_03",
    "spine_07_trunk_05": "spine_06_trunk_04",
    "spine_08_trunk_06": "spine_07_trunk_05",
    "spine_09_trunk_07": "spine_08_trunk_06",
    "spine_10_precaudal": "spine_09_trunk_07",
    "spine_11_caudal_axis_1": "spine_10_precaudal",
    "spine_12_caudal_axis_2": "spine_11_caudal_axis_1",
    # --- cranial leaves -----------------------------------------------------
    "snout_tip": "spine_00_cranium",
    "naris_anterior_margin": "spine_00_cranium",
    "eye_center": "spine_00_cranium",
    "mouth_rictus": "spine_00_cranium",
    "spiracle": "spine_00_cranium",
    # --- branchial leaves ---------------------------------------------------
    "gill_slit_1_dorsal_origin": "spine_01_branchial_1",
    "gill_slit_1_ventral_terminus": "spine_01_branchial_1",
    "gill_slit_7_dorsal_origin": "spine_02_branchial_7",
    "gill_slit_7_ventral_terminus": "spine_02_branchial_7",
    # --- pectoral -----------------------------------------------------------
    "pectoral_origin": "spine_02_branchial_7",
    "pectoral_insertion": "pectoral_origin",
    "pectoral_fin_tip": "pectoral_origin",
    # --- pelvic / median fins, all posterior (single dorsal over the pelvics) -
    "pelvic_origin": "spine_07_trunk_05",
    "cloaca": "spine_07_trunk_05",
    "dorsal_fin_origin": "spine_07_trunk_05",
    "dorsal_fin_insertion": "dorsal_fin_origin",
    "dorsal_fin_apex": "dorsal_fin_origin",
    "anal_fin_origin": "spine_08_trunk_06",
    "anal_fin_insertion": "anal_fin_origin",
    # --- caudal -------------------------------------------------------------
    "caudal_subterminal_notch": "spine_12_caudal_axis_2",
    "caudal_upper_lobe_tip": "spine_12_caudal_axis_2",
    "caudal_lower_lobe_tip": "spine_11_caudal_axis_1",
}

ROOT_JOINT: str = "spine_00_cranium"
ROOT_IDX: int = JOINT_NAME_TO_IDX[ROOT_JOINT]  # 0


def get_parent_indices() -> List[int]:
    """Return parent index for each joint (-1 for root).

    Ordered by JOINT_SEQUENCE.
    """
    parents = []
    for name in JOINT_SEQUENCE:
        parent_name = KINEMATIC_TREE[name]
        if parent_name is None:
            parents.append(-1)
        else:
            parents.append(JOINT_NAME_TO_IDX[parent_name])
    return parents


# Pre-computed parent index array
PARENT_INDICES: List[int] = get_parent_indices()


def get_kinematic_chain(joint_name: str) -> List[str]:
    """Return chain from root to the given joint (inclusive)."""
    chain = []
    current = joint_name  # type: Optional[str]
    while current is not None:
        chain.append(current)
        current = KINEMATIC_TREE[current]
    return list(reversed(chain))


def get_children(joint_name: str) -> List[str]:
    """Return direct children of a joint in the kinematic tree."""
    return [
        child
        for child, parent in KINEMATIC_TREE.items()
        if parent == joint_name
    ]


def build_adjacency_matrix() -> torch.Tensor:
    """Build symmetric adjacency matrix (35x35) for structure-aware loss.

    A[i,j] = 1 if joints i and j are connected by a bone in the kinematic tree.
    """
    adj = torch.zeros(NUM_JOINTS, NUM_JOINTS)
    for child, parent in KINEMATIC_TREE.items():
        if parent is not None:
            ci = JOINT_NAME_TO_IDX[child]
            pi = JOINT_NAME_TO_IDX[parent]
            adj[ci, pi] = 1.0
            adj[pi, ci] = 1.0
    return adj


# Pre-computed adjacency
ADJACENCY_MATRIX: torch.Tensor = build_adjacency_matrix()


def get_bone_pairs() -> List[Tuple[int, int]]:
    """Return list of (parent_idx, child_idx) bone pairs."""
    pairs = []
    for child, parent in KINEMATIC_TREE.items():
        if parent is not None:
            pairs.append((JOINT_NAME_TO_IDX[parent], JOINT_NAME_TO_IDX[child]))
    return pairs


BONE_PAIRS: List[Tuple[int, int]] = get_bone_pairs()
NUM_BONES: int = len(BONE_PAIRS)  # 34


# ---------------------------------------------------------------------------
# Axial bending. The point of the serial spine.
#
# Posture along an elongate swimmer is a 1-D field of tangent angle against arc length, and a
# small number of modes explains nearly all of it — four eigenworm modes cover >95% of C. elegans
# posture variance, and three PCs cover >90% of teleost larval tail shape [SEARCH-grade, from the
# programme scan; do not promote]. NUM_BENDING_MODES = 6 is the top of the 4-6 bracket the
# programme design specifies.
#
# ⚠ The basis below is an ANALYTIC placeholder (DCT-II over the tangent-angle profile), not a
# learned one. There is no annotated sevengill midline corpus yet, so a data-driven eigen-basis
# cannot be fitted. Replace `build_bending_basis` with PCA over real tangent-angle profiles as
# soon as the first annotation batch lands; the projection/reconstruction API does not change.
# ---------------------------------------------------------------------------
NUM_BENDING_MODES: int = 6


def get_spine_chain() -> List[str]:
    """Return the serial spine, root-to-tail.

    Derived from the tree rather than returned from the literal list, so that this function and
    ``KINEMATIC_TREE`` cannot silently disagree.
    """
    chain = [ROOT_JOINT]
    current = ROOT_JOINT
    while True:
        spine_children = [c for c in get_children(current) if c in set(SPINE_JOINTS)]
        if not spine_children:
            break
        if len(spine_children) != 1:
            raise ValueError(
                "spine joint {0} has {1} spine children; the spine must be serial".format(
                    current, len(spine_children)
                )
            )
        current = spine_children[0]
        chain.append(current)
    return chain


def build_bending_basis(
    n_modes: int = NUM_BENDING_MODES,
    n_segments: int = NUM_SPINE_SEGMENTS,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return an orthonormal ``(n_modes, n_segments)`` bending basis.

    Rows are DCT-II cosine modes over arc length, excluding the constant mode (a constant
    tangent-angle offset is a global heading rotation, not a bend).

    Only ``n_segments - 1`` usable modes exist, not ``n_segments``: mode ``k = n_segments``
    evaluates to ``cos(pi * (2i + 1) / 2)``, which is identically zero at every one of these
    sample points. Normalising that row divides by a norm that is pure floating-point noise and
    yields a garbage direction (measured ``max|BB^T - I| = 0.607`` at ``n_modes = 12``), so the
    top of the valid range is ``n_segments - 1``.
    """
    if not 1 <= n_modes <= n_segments - 1:
        raise ValueError(
            "n_modes must be in [1, n_segments - 1] = [1, {0}]; got {1}. DCT-II mode "
            "k = n_segments is identically zero at these sample points (the cosine is "
            "evaluated at odd multiples of pi/2), so it carries no bending direction and "
            "cannot be normalised.".format(n_segments - 1, n_modes)
        )
    rows = []
    for k in range(1, n_modes + 1):
        row = torch.tensor(
            [math.cos(math.pi * k * (2 * i + 1) / (2 * n_segments)) for i in range(n_segments)],
            dtype=dtype,
        )
        rows.append(row / torch.linalg.norm(row))
    return torch.stack(rows, dim=0)


BENDING_BASIS: torch.Tensor = build_bending_basis()


def spine_tangent_angles(positions: torch.Tensor) -> torch.Tensor:
    """Tangent-angle profile of a posed spine.

    ``positions`` is ``(..., NUM_SPINE_JOINTS, 2)`` in the image/body plane, ordered root-to-tail.
    Returns ``(..., NUM_SPINE_SEGMENTS)`` angles in radians, each the direction of one segment.
    """
    if positions.shape[-2] != NUM_SPINE_JOINTS or positions.shape[-1] != 2:
        raise ValueError(
            "expected (..., {0}, 2) spine positions".format(NUM_SPINE_JOINTS)
        )
    deltas = positions[..., 1:, :] - positions[..., :-1, :]
    return torch.atan2(deltas[..., 1], deltas[..., 0])


def project_to_bending_modes(
    angles: torch.Tensor, basis: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Project a ``(..., NUM_SPINE_SEGMENTS)`` tangent-angle profile onto the bending basis.

    The profile mean (global heading) is removed first, so the coefficients describe SHAPE.
    Returns ``(..., n_modes)``.
    """
    if basis is None:
        basis = BENDING_BASIS
    centred = angles - angles.mean(dim=-1, keepdim=True)
    return centred.to(basis.dtype) @ basis.transpose(0, 1)


def reconstruct_from_bending_modes(
    coeffs: torch.Tensor, basis: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Inverse of :func:`project_to_bending_modes`, up to the discarded global heading.

    LOSSY TWICE OVER. The global heading (the profile mean) is gone by construction, and so is
    every mode above ``n_modes``: the basis spans 6 of the 11 non-constant DCT modes available on
    a 12-segment spine, so 5 are discarded. Measured relative L2 residual of the round trip
    (``||centred - reconstructed|| / ||centred||``) with the shipped 6-mode basis: **1.8%** on a
    constant-curvature arc, **5.7%** on a one-wavelength undulation, **15.4%** on a
    two-wavelength one, and **66%** on white noise (the sqrt(5/11) floor). Truncation is cheap on
    smooth swimming postures and expensive on high-frequency profiles; a reconstructed profile is
    a smoothed profile, not the original.
    """
    if basis is None:
        basis = BENDING_BASIS
    return coeffs.to(basis.dtype) @ basis


def keypoints_to_joint_indices(
    keypoint_names: Optional[Sequence[str]] = None,
) -> List[int]:
    """Return the joint index each annotation keypoint maps to.

    Defaults to the full Schema S1 keypoint order, so the result indexes a ``(30,)`` keypoint
    tensor into the ``(35,)`` joint tensor.
    """
    if keypoint_names is None:
        keypoint_names = SEVENGILL_KEYPOINT_SEQUENCE
    return [KEYPOINT_TO_JOINT_IDX[name] for name in keypoint_names]
