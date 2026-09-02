"""Rig for a straight (de-bent) sevengill mesh: skeleton placement, LBS weights, FK.

Mesh-agnostic by construction. Nothing here touches trimesh, GLB files or the
de-bending code; the whole module speaks plain numpy arrays, so it runs equally on
the procedural test capsule and on the real Meshy-AI scan once module A has
straightened it.

INPUT CONTRACT (module A -> module B)
-------------------------------------
``centerline_straight``: (M, 3) float array, the straight rest-pose centerline
    polyline ordered SNOUT FIRST, TAIL LAST. Body axis convention is inherited from
    ``blender/operators/create_shark_armature.py``: snout at +X, tail at -X,
    dorsal +Z, lateral +/-Y. Only the ordering matters to this module; arc length is
    measured along the polyline as given.
``fin_info``: mapping ``fin_name -> {"insertion": (3,), "tip": (3,)}``, optionally
    with ``"parent"``: an explicit spine joint name overriding nearest-spine-joint
    attachment. ``insertion`` is the centroid of the fin's insertion (where the fin
    leaves the body wall); ``tip`` is its distal extremity. The strongly heterocercal
    caudal is TWO entries -- ``caudal_upper`` (the long lobe the vertebral axis turns
    into) and ``caudal_lower`` -- so the upper lobe gets its own root+tip pair.
    ``"insertion_centroid"`` is accepted as an alias for ``"insertion"``, and
    ``fin_info_from_detection`` turns module A's ``detect_fins(...).fins`` dict --
    which carries ``insertion_centroid`` and ``vertex_indices`` but no tip -- into
    this shape. Fin NAMES are not fixed here: whatever keys ``fin_info`` uses are the
    labels ``compute_weights`` expects and the ``<name>_fin_root/_fin_tip`` joints it
    creates.
``vertices``: (N, 3) rest-pose vertex positions of the straight mesh.
``labels``: (N,) per-vertex fin labels: the string ``"body"`` for trunk/head
    vertices, otherwise a key of ``fin_info``. Integer codes are accepted together
    with ``label_names``.

WHAT THE SKELETON IS
--------------------
The spine is NOT invented here: it is the 13-joint serial chain of
``phase1b/p0-sevengill-schema/skeleton_sevengill.py`` (its joint names, order,
parents), placed on the straight centerline at the arc-length fractions that module
declares. On top of that, each fin gets a two-joint appendage (root at the insertion
centroid, tip parented to the root) so fins can flap, pitch and flex -- 'plasticity'
the schema's leaf-only fin keypoints do not provide.

A NOTE ON ``shark_pose/model_3d/skinning.py``
---------------------------------------------
That module is the reference for semantics (rotation about the joint, parent-chain
composition, parents processed before children via topological sort) and this module
mirrors it -- with one deliberate divergence. There, the per-joint world transform is
already built as ``T(p) R T(-p)`` (a rotation ABOUT the rest joint) and is then
multiplied by ``rest_inv = T(-p)`` a second time, which subtracts the joint position
twice: at identity rotation that maps ``x -> x - p`` instead of ``x -> x``. Here the
world transform doubles as the skinning matrix, so identity rotations reproduce the
rest mesh exactly (``test_rig.py::test_identity_pose_is_rest_mesh``) and the result
agrees with glTF's ``worldTransform @ inverseBindMatrix`` term for term.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "BODY_LABEL",
    "DEFAULT_PRECAUDAL_FRACTION",
    "Skeleton",
    "spine_arclength_fractions",
    "build_skeleton",
    "fin_info_from_detection",
    "compute_weights",
    "DEFAULT_FIN_BLEND_RINGS",
    "vertex_adjacency",
    "fin_seam_rings",
    "prune_weights",
    "weights_to_indexed",
    "forward_kinematics",
    "posed_joints",
    "lbs",
    "topological_order",
    "axis_angle_rotmat",
    "rotmat_to_quat",
    "quat_to_rotmat",
]


# ---------------------------------------------------------------------------
# The sevengill schema skeleton. Imported, never vendored.
# ---------------------------------------------------------------------------
def _import_schema_skeleton():
    """Import ``skeleton_sevengill`` from the phase1b schema package.

    Path resolution: ``$SEVENGILL_SCHEMA_DIR`` if set, else
    ``<repo>/phase1b/p0-sevengill-schema`` relative to this file.
    """
    override = os.environ.get("SEVENGILL_SCHEMA_DIR")
    if override:
        candidate = override
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.abspath(
            os.path.join(here, os.pardir, os.pardir, "phase1b", "p0-sevengill-schema")
        )
    if not os.path.isfile(os.path.join(candidate, "skeleton_sevengill.py")):
        raise ImportError(
            "skeleton_sevengill.py not found in %r; set SEVENGILL_SCHEMA_DIR" % candidate
        )
    if candidate not in sys.path:
        sys.path.insert(0, candidate)
    import skeleton_sevengill  # noqa: E402  (path is only valid after the insert)

    return skeleton_sevengill


SCHEMA = _import_schema_skeleton()

SPINE_JOINTS: List[str] = list(SCHEMA.SPINE_JOINTS)
NUM_SPINE_JOINTS: int = len(SPINE_JOINTS)

BODY_LABEL: str = "body"

# ---------------------------------------------------------------------------
# Arc-length fractions of the spine stations.
#
# The schema pins seven of the thirteen: MIDLINE_AXIS_FRACTIONS = 0.125 .. 0.875 of
# the SNOUT-TO-PRECAUDAL axis, for spine_03_trunk_01 .. spine_09_trunk_07. The other
# six (three cranial/branchial, precaudal, two caudal-axis) are not given there and
# are declared below.
#
# The three head stations sit at 0.030 / 0.065 / 0.100 of precaudal length. That is
# FORWARD of published hexanchiform head proportions (snout to 7th gill slit is
# roughly a quarter of precaudal length) [UNVERIFIED - no measured Notorynchus
# cepedianus proportions were retrieved]. They are placed forward on purpose: the
# schema fixes trunk_01 at 0.125, and a serial chain whose stations are not monotone
# in arc length is not a spine. Monotonicity wins; the head segment is short and
# rigid anyway, so the cost is a slightly long branchial_7 -> trunk_01 bone. Override
# via ``spine_fractions=`` when real proportions land.
#
# The two caudal-axis stations are fractions of the POST-precaudal remainder, since
# the vertebral axis turns up into the long upper lobe past the precaudal pit.
# ---------------------------------------------------------------------------
HEAD_PRECAUDAL_FRACTIONS: Dict[str, float] = {
    "spine_00_cranium": 0.030,
    "spine_01_branchial_1": 0.065,
    "spine_02_branchial_7": 0.100,
}
CAUDAL_REMAINDER_FRACTIONS: Dict[str, float] = {
    "spine_11_caudal_axis_1": 0.40,
    "spine_12_caudal_axis_2": 0.80,
}
# Precaudal length as a fraction of the full centerline (snout tip -> caudal upper
# lobe tip). ~0.78 is the usual shark bracket [UNVERIFIED for this species].
DEFAULT_PRECAUDAL_FRACTION: float = 0.78


def spine_arclength_fractions(precaudal_fraction: float = DEFAULT_PRECAUDAL_FRACTION):
    """Arc-length fraction of the FULL centerline for each of the 13 spine joints.

    Returns a (13,) float array aligned with ``SPINE_JOINTS``, strictly increasing,
    first value > 0 and last value < 1 (the chain does not reach either mesh tip:
    snout_tip and caudal_upper_lobe_tip are schema leaves, not spine stations).
    """
    pf = float(precaudal_fraction)
    if not 0.2 < pf < 1.0:
        raise ValueError("precaudal_fraction must be in (0.2, 1.0); got %r" % precaudal_fraction)
    midline = tuple(SCHEMA.MIDLINE_AXIS_FRACTIONS)
    out = []
    for name in SPINE_JOINTS:
        if name in HEAD_PRECAUDAL_FRACTIONS:
            out.append(HEAD_PRECAUDAL_FRACTIONS[name] * pf)
        elif name in CAUDAL_REMAINDER_FRACTIONS:
            out.append(pf + CAUDAL_REMAINDER_FRACTIONS[name] * (1.0 - pf))
        elif name == "spine_10_precaudal":
            out.append(pf)
        else:
            idx = SPINE_JOINTS.index(name) - 3  # trunk_01 is the 4th spine joint
            out.append(midline[idx] * pf)
    frac = np.asarray(out, dtype=float)
    if not np.all(np.diff(frac) > 0):
        raise ValueError("spine arc-length fractions are not strictly increasing: %r" % frac)
    return frac


# ---------------------------------------------------------------------------
# Skeleton
# ---------------------------------------------------------------------------
class Skeleton(object):
    """A placed kinematic tree: names, parents, rest joint positions.

    Attributes:
        names: list of J joint names.
        parents: (J,) int array, -1 for the root. GUARANTEED ``parents[j] < j``
            for every joint, i.e. parents-before-children, so forward kinematics
            can iterate ``range(J)`` directly (the topological sort of
            ``shark_pose/model_3d/skinning.py`` is a no-op on this ordering, and
            ``topological_order`` asserts it).
        joints: (J, 3) rest-pose world positions.
        kinds: list of J strings, one of ``"spine"``, ``"fin_root"``, ``"fin_tip"``.
        fractions: (J,) arc-length fraction along the centerline for spine joints,
            NaN for fin joints.
        fins: mapping ``fin_name -> (root_index, tip_index)``.
    """

    def __init__(self, names, parents, joints, kinds, fractions, fins):
        self.names = list(names)
        self.parents = np.asarray(parents, dtype=int)
        self.joints = np.asarray(joints, dtype=float)
        self.kinds = list(kinds)
        self.fractions = np.asarray(fractions, dtype=float)
        self.fins = dict(fins)
        self._index = {name: i for i, name in enumerate(self.names)}
        n = len(self.names)
        if self.parents.shape != (n,) or self.joints.shape != (n, 3):
            raise ValueError("Skeleton array shapes disagree with the name list")
        for j, p in enumerate(self.parents):
            if p >= j and p != -1:
                raise ValueError(
                    "joint %r (index %d) has parent index %d: parents must precede children"
                    % (self.names[j], j, p)
                )
        topological_order(self.parents)  # raises on a cycle / bad parent

    # -- basics ------------------------------------------------------------
    def __len__(self):
        return len(self.names)

    @property
    def num_joints(self):
        return len(self.names)

    def index(self, name):
        """Index of a joint by name."""
        return self._index[name]

    def children(self, j):
        """Indices of the direct children of joint index ``j``."""
        return [i for i in range(len(self)) if self.parents[i] == j]

    @property
    def spine_indices(self):
        """(13,) int array: the serial spine, root-to-tail, in schema order."""
        return np.asarray([self._index[n] for n in SPINE_JOINTS], dtype=int)

    def descendants(self, j):
        """Indices of joint ``j`` and everything below it."""
        out = [int(j)]
        for i in range(int(j) + 1, len(self)):
            if self.parents[i] in out:
                out.append(i)
        return out

    def __repr__(self):
        return "Skeleton(%d joints, %d fins)" % (len(self), len(self.fins))


def topological_order(parents):
    """Joint indices with every parent before its children.

    Mirrors ``_topological_sort`` in ``shark_pose/model_3d/skinning.py``; raises
    ``ValueError`` on a cycle instead of looping forever.
    """
    parents = np.asarray(parents, dtype=int)
    n = len(parents)
    order = []
    placed = set()
    remaining = set(range(n))
    while remaining:
        progress = False
        for j in sorted(remaining):
            if parents[j] == -1 or int(parents[j]) in placed:
                order.append(j)
                placed.add(j)
                remaining.discard(j)
                progress = True
        if not progress:
            raise ValueError("kinematic tree has a cycle or a dangling parent: %r" % parents)
    return order


def _arc_length(polyline):
    pts = np.asarray(polyline, dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _interp_at_fractions(polyline, fractions):
    """Points on ``polyline`` at the given fractions of its total arc length."""
    pts = np.asarray(polyline, dtype=float)
    s = _arc_length(pts)
    total = s[-1]
    if total <= 0:
        raise ValueError("centerline has zero length")
    targets = np.asarray(fractions, dtype=float) * total
    return np.column_stack([np.interp(targets, s, pts[:, k]) for k in range(3)])


def build_skeleton(
    centerline_straight,
    fin_info,
    precaudal_fraction=DEFAULT_PRECAUDAL_FRACTION,
    spine_fractions=None,
):
    """Place the sevengill spine on a straight centerline and hang fin joints off it.

    Args:
        centerline_straight: (M, 3) straight rest-pose centerline, snout first.
        fin_info: ``{fin_name: {"insertion": (3,), "tip": (3,), "parent": name?}}``.
        precaudal_fraction: precaudal length / total centerline length.
        spine_fractions: optional (13,) override of the arc-length fractions.

    Returns:
        Skeleton with 13 spine joints (schema names/order/parents) followed by two
        joints per fin, named ``"<fin>_fin_root"`` and ``"<fin>_fin_tip"``. Fin
        ordering follows the iteration order of ``fin_info``.
    """
    cl = np.asarray(centerline_straight, dtype=float)
    if cl.ndim != 2 or cl.shape[1] != 3 or len(cl) < 2:
        raise ValueError("centerline_straight must be (M >= 2, 3); got %r" % (cl.shape,))

    if spine_fractions is None:
        frac = spine_arclength_fractions(precaudal_fraction)
    else:
        frac = np.asarray(spine_fractions, dtype=float)
        if frac.shape != (NUM_SPINE_JOINTS,):
            raise ValueError("spine_fractions must be (%d,)" % NUM_SPINE_JOINTS)
        if not np.all(np.diff(frac) > 0):
            raise ValueError("spine_fractions must be strictly increasing")

    names = list(SPINE_JOINTS)
    joints = list(_interp_at_fractions(cl, frac))
    parents = [-1] + list(range(NUM_SPINE_JOINTS - 1))
    kinds = ["spine"] * NUM_SPINE_JOINTS
    fractions = list(frac)

    spine_pos = np.asarray(joints)
    fins = {}
    for fin_name in fin_info:
        entry = fin_info[fin_name]
        if "insertion" in entry:
            insertion = np.asarray(entry["insertion"], dtype=float).reshape(3)
        else:
            insertion = np.asarray(entry["insertion_centroid"], dtype=float).reshape(3)
        tip = np.asarray(entry["tip"], dtype=float).reshape(3)
        explicit = entry.get("parent") if hasattr(entry, "get") else None
        if explicit is not None:
            if explicit not in SPINE_JOINTS:
                raise ValueError(
                    "fin %r parent %r is not a spine joint" % (fin_name, explicit)
                )
            parent_idx = SPINE_JOINTS.index(explicit)
        else:
            parent_idx = int(np.argmin(np.linalg.norm(spine_pos - insertion[None, :], axis=1)))
        root_idx = len(names)
        names.append("%s_fin_root" % fin_name)
        joints.append(insertion)
        parents.append(parent_idx)
        kinds.append("fin_root")
        fractions.append(np.nan)

        names.append("%s_fin_tip" % fin_name)
        joints.append(tip)
        parents.append(root_idx)
        kinds.append("fin_tip")
        fractions.append(np.nan)
        fins[fin_name] = (root_idx, root_idx + 1)

    return Skeleton(names, parents, np.asarray(joints, dtype=float), kinds, fractions, fins)


def _axis_frame_at(point, centerline):
    """Closest point on ``centerline`` (a straight or gently curved polyline) to
    ``point`` and the unit tangent there, pointing snout -> tail (``T = -X`` in
    the canonical straight pose).  With ``centerline=None`` the body axis is
    the X line through the origin, which is where ``mesh3d.debend`` puts the
    straight rest pose."""
    if centerline is None:
        return np.array([float(point[0]), 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])   # T = -X
    cl = np.asarray(centerline, dtype=float)
    seg, t = _project_on_polyline(np.asarray(point, dtype=float)[None, :], cl)
    a, b = cl[seg[0]], cl[seg[0] + 1]
    foot = a + t[0] * (b - a)
    tang = b - a
    return foot, tang / max(np.linalg.norm(tang), 1e-12)


def fin_info_from_detection(fins, vertices, centerline=None):
    """Adapt module A's ``detect_fins(...).fins`` dict to the ``fin_info`` contract.

    Module A reports, per fin, an ``insertion_centroid`` and the ``vertex_indices`` of
    the island, but no tip -- the tip is not a landmark it needs. Here the tip is the
    island's APEX: the labelled vertex that protrudes farthest from the body axis in
    the direction the insertion sits (dorsal for a dorsal fin, outboard for a
    pectoral).  The axis root->tip is what ``compute_weights`` grades the fin along,
    so it has to leave the body: the earlier rule, "farthest from the insertion
    centroid", picks a base corner on any island that is longer along the body than
    it is tall (a sevengill's low dorsal, anal and pelvic fins), which turns the fin
    drive into a fore-aft hinge and folds the blade over.

    Args:
        fins: ``{name: {"insertion_centroid": (3,), "vertex_indices": (k,) int}}``.
        vertices: (N, 3) rest-pose vertices of the straight mesh.
        centerline: optional (M, 3) straight centerline the rest pose was built on;
            the body axis near each fin is read from it.  Default: the +X axis
            through the origin (``mesh3d.straight_centerline``'s convention).

    Returns:
        ``{name: {"insertion": (3,), "tip": (3,)}}`` in the same order.
    """
    verts = np.asarray(vertices, dtype=float)
    out = {}
    for name in fins:
        entry = fins[name]
        insertion = np.asarray(
            entry["insertion"] if "insertion" in entry else entry["insertion_centroid"],
            dtype=float,
        ).reshape(3)
        if "tip" in entry:
            tip = np.asarray(entry["tip"], dtype=float).reshape(3)
        else:
            members = verts[np.asarray(entry["vertex_indices"], dtype=int)]
            if len(members) == 0:
                raise ValueError("fin %r has no vertices" % name)
            foot, tang = _axis_frame_at(insertion, centerline)
            radial = insertion - foot
            radial = radial - (radial @ tang) * tang
            norm = np.linalg.norm(radial)
            if name.startswith("caudal"):
                # A caudal lobe's long axis is AXIAL: its tip is the most
                # posterior vertex along the tail-ward tangent.  (Not "farthest
                # from the insertion": on a tapering lobe the innermost-radius
                # quartile that defines the insertion can sit mid-lobe, and the
                # farthest vertex from there is the base, pointing the bone
                # forward.)
                axial = (members - foot) @ tang
                tip = members[int(np.argmax(axial))]
                # ...and its ROOT must be where the lobe leaves the peduncle.
                # A mid-lobe root turns the fin drive into a lever: the part of
                # the lobe ahead of the root swings against the body and tears
                # away as thin slivers.  When the island is a true lobe (axial
                # extent >= 2x its radial extent) and the detected insertion is
                # not within the anterior quarter of it, the root is moved to the
                # centroid of the anterior 15% of the lobe.
                a_lo, a_hi = float(axial.min()), float(axial.max())
                rel = members - foot
                radial_ext = float(np.ptp(np.linalg.norm(rel - np.outer(axial, tang), axis=1)))
                a_ins = float((insertion - foot) @ tang)
                if (a_hi - a_lo) >= 2.0 * max(radial_ext, 1e-9) and a_ins > a_lo + 0.25 * (a_hi - a_lo):
                    slab = members[axial <= a_lo + 0.15 * (a_hi - a_lo)]
                    insertion = slab.mean(axis=0)
            elif norm > 1e-9:
                u = radial / norm
                rel = members - foot
                rel = rel - (rel @ tang)[:, None] * tang
                tip = members[int(np.argmax(rel @ u))]
            else:  # insertion on the axis: no side to prefer, fall back to reach
                tip = members[int(np.argmax(np.linalg.norm(members - insertion, axis=1)))]
        out[name] = {"insertion": insertion, "tip": tip}
        if "parent" in entry:
            out[name]["parent"] = entry["parent"]
    return out


# ---------------------------------------------------------------------------
# Skinning weights
# ---------------------------------------------------------------------------
def _normalise_labels(labels, n, label_names=None):
    arr = np.asarray(labels)
    if arr.shape != (n,):
        raise ValueError("labels must be (%d,); got %r" % (n, (arr.shape,)))
    if label_names is not None:
        names = list(label_names)
        return np.asarray([str(names[int(i)]) for i in arr])
    if arr.dtype.kind in "iub":
        raise ValueError("integer labels need label_names=")
    return np.asarray([str(x) for x in arr])  # unicode array: comparisons vectorise


def _project_on_polyline(points, polyline):
    """Closest-point projection onto a polyline.

    Returns ``(segment_index, t)`` with ``t`` in [0, 1] along that segment.
    """
    p = np.asarray(points, dtype=float)
    a = polyline[:-1]
    b = polyline[1:]
    ab = b - a                                            # (K, 3)
    denom = np.maximum((ab * ab).sum(axis=1), 1e-24)       # (K,)
    ap = p[:, None, :] - a[None, :, :]                     # (N, K, 3)
    t = np.clip((ap * ab[None, :, :]).sum(axis=2) / denom[None, :], 0.0, 1.0)  # (N, K)
    closest = a[None, :, :] + t[..., None] * ab[None, :, :]
    d = np.linalg.norm(p[:, None, :] - closest, axis=2)     # (N, K)
    k = np.argmin(d, axis=1)
    rows = np.arange(len(p))
    return k, t[rows, k]


#: Default width, in mesh edge rings, of the fin-base weight blend.  See
#: :func:`compute_weights`.
DEFAULT_FIN_BLEND_RINGS: int = 3


def vertex_adjacency(faces, n_vertices):
    """Undirected vertex adjacency of a triangle mesh, in CSR-ish form.

    Returns ``(neighbours, starts)``: vertex ``v``'s neighbours are
    ``neighbours[starts[v]:starts[v + 1]]``.  Isolated vertices get an empty slice.
    """
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    n = int(n_vertices)
    if len(f) and (f.min() < 0 or f.max() >= n):
        raise ValueError("faces index outside [0, %d)" % n)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    e = np.concatenate([e, e[:, ::-1]], axis=0)
    if len(e) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(n + 1, dtype=np.int64)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e = e[order]
    keep = np.ones(len(e), dtype=bool)
    keep[1:] = np.any(e[1:] != e[:-1], axis=1)
    e = e[keep]
    counts = np.bincount(e[:, 0], minlength=n)
    starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return np.ascontiguousarray(e[:, 1]), starts


def _expand(neighbours, starts, frontier):
    """All neighbours of every vertex in ``frontier``, concatenated."""
    lens = starts[frontier + 1] - starts[frontier]
    total = int(lens.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    base = np.repeat(starts[frontier], lens)
    within = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(lens) - lens, lens)
    return neighbours[base + within]


def fin_seam_rings(faces, labels, n_vertices, max_rings, body_label=BODY_LABEL):
    """Graph distance, in mesh edge rings, from each BODY vertex to a fin island.

    Multi-source BFS over the mesh edge graph, one source set per fin island, walking
    only through body vertices.  Returns ``(ring, owner)``: ``ring[v]`` is the number
    of edges from ``v`` to the nearest fin island (1 = directly adjacent to it), or
    ``-1`` when ``v`` is a fin vertex or farther than ``max_rings``; ``owner[v]`` is
    the label of that nearest island, ``""`` where ``ring[v] < 0``.  Ties go to
    whichever island BFS reached the vertex first, which is deterministic given the
    label order passed in.
    """
    lab = np.asarray(labels)
    n = int(n_vertices)
    neighbours, starts = vertex_adjacency(faces, n)
    is_body = lab == body_label
    ring = np.full(n, -1, dtype=np.int64)
    owner = np.zeros(n, dtype=object)
    owner[:] = ""
    for name in sorted(set(np.unique(lab).tolist()) - {body_label}):
        frontier = np.nonzero(lab == name)[0].astype(np.int64)
        for d in range(1, int(max_rings) + 1):
            if len(frontier) == 0:
                break
            nb = _expand(neighbours, starts, frontier)
            if len(nb) == 0:
                break
            nb = np.unique(nb)
            nb = nb[is_body[nb] & (ring[nb] < 0)]
            if len(nb) == 0:
                frontier = nb
                break
            ring[nb] = d
            owner[nb] = name
            frontier = nb
    return ring, owner


def compute_weights(
    vertices,
    labels,
    skeleton,
    sigma=None,
    max_influences=4,
    label_names=None,
    faces=None,
    fin_blend_rings=DEFAULT_FIN_BLEND_RINGS,
):
    """Linear blend skinning weights, (N, J), rows summing to 1.

    Body vertices (label ``"body"``) are bound to the spine by arc length: each
    vertex projects onto the spine polyline and is shared between its two bracketing
    spine joints in proportion to where it falls between them. With ``sigma`` given
    (in world units) the binding is instead a Gaussian in arc length over all spine
    joints, which widens the falloff and makes bends visibly smoother; it is then
    pruned back to ``max_influences``.

    Fin vertices are bound to that fin's root and tip by normalised distance along
    the fin axis (root -> tip), so the tip joint flexes the fin's distal half.

    THE FIN-BASE SEAM.  Those two rules meet at the edge of a fin island with nothing
    in between: the island is 100% fin root, the body vertices one ring outside it are
    100% spine, and the fin root carries the fin's own drive rotation on top of the
    spine's.  Under a clip that is a step discontinuity straight across a mesh edge --
    measured at up to 3.5% BL of edge-length change on the seam edges of the demo
    mesh, which is a visible tear at the fin base.  So when ``faces`` is given the
    fin-root weight is RAMPED across the base instead: a body vertex ``d`` edge rings
    outside an island (``1 <= d <= fin_blend_rings``) gets fin-root weight
    ``(R - d) / R``, which is 1 at the island boundary and 0 at ring ``R``, and its
    spine weights are scaled by the complement so the row still sums to 1.  Passing
    ``faces=None`` restores the old hard seam (and is what the mesh-free rig tests
    use).

    Args:
        vertices: (N, 3) rest-pose positions.
        labels: (N,) per-vertex ``"body"`` / fin name (or integer codes with
            ``label_names``).
        skeleton: :class:`Skeleton`.
        sigma: Gaussian arc-length falloff width in world units, or None.
        max_influences: glTF's JOINTS_0/WEIGHTS_0 limit, 4.
        label_names: names for integer ``labels``.
        faces: (F, 3) triangle indices of the SAME vertex array, used only to find
            the seam rings.  None disables the fin-base blend.
        fin_blend_rings: ``R`` above; must be >= 1.  1 is a hard seam again.

    At most ``max_influences`` (4, the glTF JOINTS_0/WEIGHTS_0 limit) nonzero
    entries per row; pruned weights are renormalised so rows still sum to 1.
    """
    verts = np.asarray(vertices, dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("vertices must be (N, 3)")
    n = len(verts)
    lab = _normalise_labels(labels, n, label_names)
    j = skeleton.num_joints
    weights = np.zeros((n, j), dtype=float)

    spine_idx = skeleton.spine_indices
    spine_pos = skeleton.joints[spine_idx]
    s_joint = _arc_length(spine_pos)

    is_body = lab == BODY_LABEL
    if np.any(is_body):
        body = verts[is_body]
        k, t = _project_on_polyline(body, spine_pos)
        if sigma is None:
            rows = np.nonzero(is_body)[0]
            weights[rows, spine_idx[k]] += 1.0 - t
            weights[rows, spine_idx[k + 1]] += t
        else:
            s_v = s_joint[k] + t * (s_joint[k + 1] - s_joint[k])
            g = np.exp(-0.5 * ((s_v[:, None] - s_joint[None, :]) / float(sigma)) ** 2)
            g = g / g.sum(axis=1, keepdims=True)
            weights[np.ix_(np.nonzero(is_body)[0], spine_idx)] = g

    for fin_name, (root_idx, tip_idx) in skeleton.fins.items():
        mask = lab == fin_name
        if not np.any(mask):
            continue
        root = skeleton.joints[root_idx]
        tip = skeleton.joints[tip_idx]
        axis = tip - root
        length = np.linalg.norm(axis)
        if length <= 0:
            raise ValueError("fin %r has a zero-length root->tip axis" % fin_name)
        u = axis / length
        t = np.clip(((verts[mask] - root[None, :]) @ u) / length, 0.0, 1.0)
        rows = np.nonzero(mask)[0]
        weights[rows, root_idx] = 1.0 - t
        weights[rows, tip_idx] = t

    unknown = set(np.unique(lab).tolist()) - set([BODY_LABEL]) - set(skeleton.fins)
    if unknown:
        raise ValueError("labels contain names with no fin in the skeleton: %r" % sorted(unknown))

    if faces is not None:
        rings = int(fin_blend_rings)
        if rings < 1:
            raise ValueError("fin_blend_rings must be >= 1; got %r" % (fin_blend_rings,))
        ring, owner = fin_seam_rings(faces, lab, n, rings)
        seam = np.nonzero((ring > 0) & (ring < rings))[0]
        for v in seam:
            root_idx = int(skeleton.fins[str(owner[v])][0])
            ramp = float(rings - int(ring[v])) / float(rings)
            weights[v, :] *= 1.0 - ramp
            weights[v, root_idx] += ramp

    row_sums = weights.sum(axis=1)
    if np.any(row_sums <= 0):
        bad = int(np.argmin(row_sums))
        raise ValueError("vertex %d received no weight (label %r)" % (bad, lab[bad]))
    weights /= row_sums[:, None]
    return prune_weights(weights, max_influences)


def prune_weights(weights, max_influences=4):
    """Keep the ``max_influences`` largest entries per row and renormalise."""
    w = np.array(weights, dtype=float, copy=True)
    k = int(max_influences)
    if w.shape[1] > k:
        cut = np.partition(w, -k, axis=1)[:, -k]
        w[w < cut[:, None]] = 0.0
        # ties at the cut can leave more than k survivors; drop the extras
        over = np.nonzero((w > 0).sum(axis=1) > k)[0]
        for r in over:
            keep = np.argsort(w[r])[::-1][:k]
            row = np.zeros_like(w[r])
            row[keep] = w[r][keep]
            w[r] = row
    w /= w.sum(axis=1, keepdims=True)
    return w


def weights_to_indexed(weights, max_influences=4):
    """Dense (N, J) weights -> glTF-shaped ``(joint_indices, joint_weights)``.

    Returns ``(N, 4) uint16`` indices and ``(N, 4) float32`` weights. Unused slots
    carry joint index 0 with weight 0 (what the glTF validator expects). The float32
    weights are re-balanced after the cast so each row sums to exactly 1.0 in float32
    -- the validator's ACCESSOR_WEIGHTS_NON_NORMALIZED check is unforgiving.
    """
    w = np.asarray(weights, dtype=float)
    k = int(max_influences)
    order = np.argsort(w, axis=1)[:, ::-1][:, :k]
    vals = np.take_along_axis(w, order, axis=1)
    order = np.where(vals > 0, order, 0)
    total = vals.sum(axis=1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError("some rows have no positive weight")
    vals = vals / total
    vals32 = vals.astype(np.float32)
    # push the float32 residual into the dominant influence
    resid = (np.float32(1.0) - vals32.sum(axis=1)).astype(np.float32)
    lead = np.argmax(vals32, axis=1)
    vals32[np.arange(len(vals32)), lead] = (
        vals32[np.arange(len(vals32)), lead] + resid
    ).astype(np.float32)
    return order.astype(np.uint16), vals32


# ---------------------------------------------------------------------------
# Forward kinematics and linear blend skinning
# ---------------------------------------------------------------------------
def forward_kinematics(skeleton, local_rotmats):
    """World transforms per joint, (J, 4, 4).

    ``local_rotmats`` is (J, 3, 3): joint ``j``'s rotation, expressed about its own
    rest position, in its parent's frame. The world transform is

        W_j = W_parent @ T(p_j) @ R_j @ T(-p_j)

    i.e. rotation ABOUT the rest joint, composed along the parent chain (the
    semantics of ``shark_pose/model_3d/skinning.py``). W_j doubles as the skinning
    matrix, so identity rotations give identity transforms. It equals glTF's
    ``globalJointTransform @ inverseBindMatrix`` for a rig whose bind pose has
    identity node rotations, which is exactly what ``gltf_export`` writes.
    """
    r = np.asarray(local_rotmats, dtype=float)
    j = skeleton.num_joints
    if r.shape != (j, 3, 3):
        raise ValueError("local_rotmats must be (%d, 3, 3); got %r" % (j, (r.shape,)))
    world = np.zeros((j, 4, 4), dtype=float)
    for i in topological_order(skeleton.parents):
        p = skeleton.joints[i]
        local = np.eye(4)
        local[:3, :3] = r[i]
        local[:3, 3] = p - r[i] @ p
        parent = int(skeleton.parents[i])
        world[i] = local if parent == -1 else world[parent] @ local
    return world


def posed_joints(skeleton, local_rotmats):
    """(J, 3) joint positions after applying ``local_rotmats``."""
    world = forward_kinematics(skeleton, local_rotmats)
    homo = np.concatenate([skeleton.joints, np.ones((skeleton.num_joints, 1))], axis=1)
    return np.einsum("jab,jb->ja", world, homo)[:, :3]


def lbs(vertices, weights, skeleton, local_rotmats):
    """Linear blend skinning: (N, 3) rest vertices -> (N, 3) posed vertices.

    ``weights`` is (N, J) with rows summing to 1. Identity rotations reproduce
    ``vertices`` exactly (bit-for-bit up to the blend's floating-point sum).
    """
    verts = np.asarray(vertices, dtype=float)
    w = np.asarray(weights, dtype=float)
    if w.shape != (len(verts), skeleton.num_joints):
        raise ValueError(
            "weights must be (%d, %d); got %r" % (len(verts), skeleton.num_joints, (w.shape,))
        )
    world = forward_kinematics(skeleton, local_rotmats)          # (J, 4, 4)
    blended = np.tensordot(w, world.reshape(-1, 16), axes=(1, 0)).reshape(-1, 4, 4)
    homo = np.concatenate([verts, np.ones((len(verts), 1))], axis=1)
    return np.einsum("nab,nb->na", blended, homo)[:, :3]


# ---------------------------------------------------------------------------
# Small rotation helpers (module C needs these to build animation curves)
# ---------------------------------------------------------------------------
def axis_angle_rotmat(axis, angle):
    """Rodrigues rotation matrix, (3, 3)."""
    a = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(a)
    if norm <= 0:
        raise ValueError("axis must be nonzero")
    a = a / norm
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def rotmat_to_quat(rotmats):
    """(..., 3, 3) rotation matrices -> (..., 4) quaternions in glTF (x, y, z, w) order."""
    r = np.asarray(rotmats, dtype=float)
    flat = r.reshape(-1, 3, 3)
    out = np.zeros((len(flat), 4), dtype=float)
    for i, m in enumerate(flat):
        trace = m[0, 0] + m[1, 1] + m[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            out[i] = [(m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s,
                      (m[1, 0] - m[0, 1]) * s, 0.25 / s]
        else:
            d = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
            if d == 0:
                s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
                out[i] = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s,
                          (m[2, 1] - m[1, 2]) / s]
            elif d == 1:
                s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
                out[i] = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s,
                          (m[0, 2] - m[2, 0]) / s]
            else:
                s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
                out[i] = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s,
                          (m[1, 0] - m[0, 1]) / s]
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out.reshape(r.shape[:-2] + (4,))


def quat_to_rotmat(quats):
    """(..., 4) quaternions in (x, y, z, w) order -> (..., 3, 3) rotation matrices."""
    q = np.asarray(quats, dtype=float)
    flat = q.reshape(-1, 4)
    flat = flat / np.linalg.norm(flat, axis=1, keepdims=True)
    x, y, z, w = flat[:, 0], flat[:, 1], flat[:, 2], flat[:, 3]
    m = np.empty((len(flat), 3, 3), dtype=float)
    m[:, 0, 0] = 1 - 2 * (y * y + z * z)
    m[:, 0, 1] = 2 * (x * y - z * w)
    m[:, 0, 2] = 2 * (x * z + y * w)
    m[:, 1, 0] = 2 * (x * y + z * w)
    m[:, 1, 1] = 1 - 2 * (x * x + z * z)
    m[:, 1, 2] = 2 * (y * z - x * w)
    m[:, 2, 0] = 2 * (x * z - y * w)
    m[:, 2, 1] = 2 * (y * z + x * w)
    m[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return m.reshape(q.shape[:-1] + (3, 3))
