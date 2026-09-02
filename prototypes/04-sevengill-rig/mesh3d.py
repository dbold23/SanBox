"""Mesh-agnostic tube chart for an elongate body: centerline, (s, r, phi), de-bend.

This is prototype 02's canonical chart lifted from a 2D silhouette to a 3D mesh.
The 2D chart rectified an image onto (arc length s) x (signed offset r); the 3D
chart maps every vertex of a tube-with-fins onto (s, r, phi) against a
rotation-minimising frame field, which is what makes *de-bending* possible: a
mesh whose rest pose is a lateral C-curve is re-embedded on a straight axis by
keeping (r, phi) and replacing the centerline.

Pipeline
--------
1. ``load_mesh``               GLB/OBJ in, UVs and materials intact.
2. ``extract_centerline_3d``   voxelise -> 3D EDT -> medial-weighted Dijkstra.
3. ``tube_frames``             rotation-minimising frames (prototype 02's
                               ``frames.rotation_minimizing_frames``, imported).
4. ``tube_coords``             per-vertex (s, r, phi) + station index.
5. ``debend`` / ``rebend``     re-embed on a straight / arbitrary centerline.
6. ``detect_fins``             per-vertex anatomical label + insertions.
7. ``estimate_roll`` /         audit the chart's orientation: body torsion and
   ``check_anatomy``           a dorsal/ventral (``up``) sign check.

Canonical straight pose (shared with ``create_shark_armature.py``): snout **+X**,
tail **-X**, dorsal **+Z**, animal's left **+Y**.  Arc length runs head -> tail,
so the body tangent is ``-X``; phi is measured from +Z toward +Y, i.e. phi = 0 is
dorsal, phi = +90 deg is the animal's left flank, phi = 180 deg is ventral.

Invertibility contract
----------------------
``tube_coords`` returns the *station index* alongside (s, r, phi) precisely so
that ``tube_to_points`` is an exact inverse.  The foot point is an unclamped
orthogonal projection onto the chosen segment, which lets a point sit beyond
either end of the chart (a snout tip ahead of s = 0, a caudal lobe behind
s = S); those overhangs are transported rigidly by the terminal frame, which is
what preserves a heterocercal tail through a de-bend.  Recovering the segment
from ``s`` alone would be ambiguous at corners, hence the explicit index.

Body roll is measured, not corrected
------------------------------------
``phi`` is seeded once, by ``up``, and the rotation-minimising frames carry that
seed down the body without twist.  So if the *mesh* is rolled about its own axis
-- a scan whose dorsal ridge spirals, or a body whose true up varies per station
-- the chart inherits the roll, and because ``debend`` preserves ``(r, phi)`` by
contract the de-bent rest pose is straight but still rolled.  Both the fin
priors above and the rig's joint schema assume an unrolled body, so a large roll
drifts fins across phi sectors and mis-names them.  ``estimate_roll`` measures
the drift and ``debend`` warns once it exceeds ``_ROLL_WARN_RAD`` over the body;
the fixes are to pre-unroll the mesh before charting, or to supply a per-station
up (a twisting frame field), which this module does not implement.

All steps are deterministic; ``seed`` arguments exist for API stability only.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import NamedTuple

import numpy as np
import trimesh
from scipy import ndimage
from scipy.signal import savgol_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

# Prototype 02 owns the rotation-minimising frames; import rather than vendor.
_P02 = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "02-centerline-chart"
)
if _P02 not in sys.path:
    sys.path.insert(0, _P02)
from frames import rotation_minimizing_frames  # noqa: E402  (path shim above)

__all__ = [
    "arc_length",
    "resample_polyline",
    "load_mesh",
    "extract_centerline_3d",
    "tube_frames",
    "canonical_frames",
    "straight_centerline",
    "TubeCoords",
    "tube_coords",
    "tube_to_points",
    "map_points",
    "debend",
    "rebend",
    "detect_fins",
    "FinDetection",
    "estimate_roll",
    "check_anatomy",
    "rotation_minimizing_frames",
    "FIN_LABELS",
]

FIN_LABELS = (
    "body",
    "pectoral_L",
    "pectoral_R",
    "dorsal",
    "pelvic_L",
    "pelvic_R",
    "anal",
    "caudal_upper",
    "caudal_lower",
)

# --- anatomical priors used to name fin islands (degrees / arc fractions) ----
# phi sectors: |phi| <= _PHI_DORSAL is the dorsal midline, |phi| >= _PHI_VENTRAL
# the ventral midline, and everything between is a paired lateral fin.  The
# ventral bound is 160 deg, not 135 deg, because sevengill pelvics sit
# ventrolaterally around |phi| ~ 140 deg and must not be mistaken for the anal.
_PHI_DORSAL = 45.0
_PHI_VENTRAL = 160.0
# An island is caudal if it is the posterior-most median island in its sector
# AND its centroid lies past this fraction of the chart.  The sevengill dorsal
# sits at s ~ 0.75 and the caudal root at s ~ 0.95, so 0.85 separates them.
_CAUDAL_S_MIN = 0.85
# Pectoral/pelvic split for lateral islands.
_PECTORAL_S_MAX = 0.50

# Per-name prior (mean s as a fraction of chart length, phi in degrees), used
# only to arbitrate a *collision*: when two disjoint islands both classify as
# one name, the island whose (s, phi) centroid sits closest to its prior keeps
# the name and the others are demoted (see ``detect_fins``).  The numbers are
# the island centroids measured on the procedural sevengill; caudal values
# exceed 1.0 because the lobes overhang the end of the chart by design.
_FIN_PRIORS = {
    "pectoral_L": (0.35, 115.0),
    "pectoral_R": (0.35, -115.0),
    "pelvic_L": (0.69, 145.0),
    "pelvic_R": (0.69, -145.0),
    "dorsal": (0.77, 0.0),
    "anal": (0.84, 180.0),
    "caudal_upper": (1.10, 0.0),
    "caudal_lower": (1.08, 180.0),
}
# Scales that make the two mismatch terms commensurate: being a quarter of the
# chart away in s is as wrong as being 45 deg away in phi.
_PRIOR_S_SCALE = 0.25
_PRIOR_PHI_SCALE = 45.0
# Two islands claiming one name are merged when their inclusive station ranges
# overlap or are separated by at most this many stations -- "touching".  That
# is the legitimate case (one caudal fin arriving as two islands); anything
# further apart is a collision and is arbitrated by the priors above.
_MERGE_STATION_GAP = 1

# Body roll (torsion about the body axis).  ``estimate_roll`` fits the drift of
# the dorsal ridge in phi against arc length; the warning fires when the fitted
# drift over the *whole* body exceeds this many radians (0.35 rad = 20 deg end
# to end), which is where fin-naming priors start to move across sectors.
_ROLL_WARN_RAD = 0.35
# Fit inputs: stations needed for a fit at all, and vertices needed per station
# for that station's maximum-radius direction to mean anything.
_ROLL_MIN_STATIONS = 8
_ROLL_MIN_PER_STATION = 6
# A station contributes only if its fattest candidate reaches this fraction of
# the station's robust radius (skips near-axis caps, where phi is noise).
_ROLL_MIN_RADIUS_FRAC = 0.5
# ...and only vertices within this fraction above the envelope are candidates.
# Fin *roots* keep the ``body`` label by design (they sit under the detection
# margin of 0.30), so without this ceiling the fattest "body" vertex of a finned
# station is a fin root and the fit tracks the fins, not the ridge.
_ROLL_BAND_FRAC = 0.20
# ...and stations within this many stations of a *detected* fin are dropped
# whole.  ``station_range`` reports the detected blade; the fin's root rows sit
# under the detection margin, keep the ``body`` label by design and run a
# station or two further at each end, and there they are the fattest
# body-labelled vertex of their station.  Without the pad those end stations
# leak into the fit, one per fin, each free to pick the left or the right blade
# of a bilaterally symmetric animal -- a coin flip that ``np.unwrap`` then
# turns into a pi offset over the whole tail of the fit.
_ROLL_FIN_STATION_PAD = 2


# ---------------------------------------------------------------------------
# Polyline helpers (prototype 02's ``arc_length``/``resample_polyline``,
# generalised from 2D to N-D; the 2D versions index columns 0 and 1 explicitly).
# ---------------------------------------------------------------------------

def arc_length(points):
    """Cumulative arc length of an (n, d) polyline; ``arc_length[0] == 0``."""
    pts = np.asarray(points, dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def resample_polyline(points, n):
    """Resample an (m, d) polyline to ``n`` points uniform in arc length."""
    pts = np.asarray(points, dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    pts = pts[np.concatenate([[True], seg > 1e-12])]
    s = arc_length(pts)
    target = np.linspace(0.0, s[-1], int(n))
    return np.column_stack(
        [np.interp(target, s, pts[:, k]) for k in range(pts.shape[1])]
    )


def _unit(v, axis=-1):
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), 1e-12)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_mesh(path, report=True):
    """Load a GLB/OBJ/PLY mesh, keeping UVs, textures and materials.

    A ``Scene`` with one geometry is unwrapped (node transform baked in); a
    multi-geometry scene is concatenated, which is lossless for geometry and
    UVs but keeps only the first material -- reported, not silently dropped.

    ``mesh.metadata`` gains ``source_path``, ``extents``, ``scale``,
    ``units`` (as declared by the file, else None) and ``n_geometries``.
    """
    obj = trimesh.load(str(path), process=False, force=None)
    n_geom = 1
    if isinstance(obj, trimesh.Scene):
        geoms = [g for g in obj.dump(concatenate=False)
                 if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError("no triangle geometry in %s" % path)
        n_geom = len(geoms)
        if n_geom == 1:
            mesh = geoms[0]
        else:
            warnings.warn(
                "scene %s has %d geometries; concatenating (geometry and UVs "
                "are preserved, per-geometry materials are not)" % (path, n_geom),
                RuntimeWarning,
                stacklevel=2,
            )
            mesh = trimesh.util.concatenate(geoms)
    elif isinstance(obj, trimesh.Trimesh):
        mesh = obj
    else:
        raise TypeError("unsupported geometry type %r from %s" % (type(obj), path))

    mesh.metadata.update(
        {
            "source_path": str(path),
            "extents": np.asarray(mesh.extents, dtype=float),
            "scale": float(np.max(mesh.extents)),
            "units": mesh.units,
            "n_geometries": n_geom,
        }
    )
    if report:
        e = mesh.extents
        print(
            "load_mesh: %s  %d verts / %d faces  extents=(%.4f, %.4f, %.4f) %s  uv=%s"
            % (
                os.path.basename(str(path)), len(mesh.vertices), len(mesh.faces),
                e[0], e[1], e[2], mesh.units or "(units undeclared)",
                getattr(mesh.visual, "uv", None) is not None,
            )
        )
    return mesh


# ---------------------------------------------------------------------------
# 3D centerline: voxelise -> EDT -> medial-weighted Dijkstra
# ---------------------------------------------------------------------------

def _neighbour_offsets():
    """Half of the 26-neighbourhood (the graph is used undirected)."""
    offs = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dz, dy, dx) < (0, 0, 0):
                    offs.append((dz, dy, dx))
    return tuple(offs)


_OFFSETS_3D = _neighbour_offsets()


def _voxelize(mesh, pitch):
    """Solid occupancy grid + its grid->world transform."""
    vg = mesh.voxelized(pitch=float(pitch))
    occ = np.asarray(vg.matrix, dtype=bool)
    occ = np.pad(occ, 1, mode="constant", constant_values=False)
    filled = ndimage.binary_fill_holes(occ)
    transform = np.asarray(vg.transform, dtype=float).copy()
    transform[:3, 3] -= transform[:3, :3] @ np.ones(3)  # undo the pad
    return filled, transform


def _largest_component(mask, structure=None):
    if structure is None:
        structure = np.ones((3, 3, 3), dtype=int)
    labels, n = ndimage.label(mask, structure=structure)
    if n == 0:
        raise ValueError("occupancy grid is empty; pitch is probably too coarse")
    if n == 1:
        return labels == 1
    sizes = ndimage.sum_labels(np.ones_like(labels), labels, np.arange(1, n + 1))
    return labels == (1 + int(np.argmax(sizes)))


def _voxel_graph(mask, weight=None):
    """Sparse graph over ``mask`` voxels.

    Edge weight is the euclidean step length, optionally scaled by the mean of
    ``weight`` at its endpoints.  With ``weight = 1/EDT`` shortest paths hug the
    medial ridge, exactly as in prototype 02's 2D ``_medial_graph``.
    """
    idx = np.full(mask.shape, -1, dtype=np.int64)
    n = int(mask.sum())
    idx[mask] = np.arange(n)
    dims = mask.shape

    rows, cols, data = [], [], []
    for off in _OFFSETS_3D:
        sa = tuple(slice(max(0, -d), dim - max(0, d)) for d, dim in zip(off, dims))
        sb = tuple(slice(max(0, d), dim - max(0, -d)) for d, dim in zip(off, dims))
        both = mask[sa] & mask[sb]
        if not both.any():
            continue
        step = float(np.linalg.norm(off))
        w = np.full(int(both.sum()), step)
        if weight is not None:
            w *= 0.5 * (weight[sa][both] + weight[sb][both])
        rows.append(idx[sa][both])
        cols.append(idx[sb][both])
        data.append(w)
    if not rows:
        raise ValueError(
            "no 26-connected voxel pairs in the largest component; the mesh is "
            "not resolvable at this pitch"
        )
    return coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsr()


def _farthest(graph, source):
    d = dijkstra(graph, directed=False, indices=int(source))
    return int(np.argmax(np.where(np.isfinite(d), d, -1.0)))


def _smooth(path, window):
    if window < 3 or len(path) <= window:
        return path
    pad = window // 2
    padded = np.pad(path, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    return np.column_stack(
        [np.convolve(padded[:, k], kernel, mode="valid") for k in range(path.shape[1])]
    )


def extract_centerline_3d(
    mesh,
    voxel_pitch=None,
    n_stations=64,
    core_radius_frac=0.17,
    core_pitch_mult=1.5,
    end_trim_frac=0.03,
    max_cells=6_000_000,
    seed=None,
):
    """Arc-length-uniform 3D centerline of a bent tube-with-fins, head first.

    Method (prototype 02's algorithm, lifted to voxels):
      1. voxelise + flood-fill to a solid occupancy grid;
      2. 3D euclidean distance transform (EDT) in world units;
      3. keep only the **thick core** ``EDT >= tau`` -- this is what stops fins
         from diverting the path.  A fin is a plate whose EDT is its
         half-thickness; a body tube's EDT is its radius.  ``tau =
         max(core_pitch_mult * pitch, core_radius_frac * max EDT)`` sits between
         the two: with the default 0.17, a sevengill peduncle (radius ~0.02 BL)
         survives while fin plates (half-thickness <~0.01 BL) do not.  THIS IS
         THE TUNING KNOB; raise it for fleshier fins, lower it for a very
         slender peduncle.  The core also excludes the caudal fin, so the chart
         ends at the peduncle and the heterocercal tail is carried through a
         de-bend as a rigid overhang instead of being straightened out.
      4. endpoints = the two ends of the **unweighted** geodesic diameter of the
         core (double sweep).  Unweighted, not medial-weighted: a weighted
         "farthest" node is the most *expensive* one, which is a thin extremity,
         not the most distant one.
      5. path = medial-weighted Dijkstra between them (weight = step / EDT), so
         it rides the ridge rather than the boundary;
      6. box smooth; **trim ``end_trim_frac`` of the path length off each end**
         -- a medial path terminates on the *boundary* of the core, not on the
         medial axis, so its last few percent lean off-axis and, being the
         terminal frame, they steer every overhanging vertex (a caudal lobe sits
         ~0.25 BL past the chart end, so a 20 deg terminal tangent error becomes
         a 9%-of-BL displacement).  Measured on the bent synthetic: trimming 3%
         cut the peak centerline deviation from 0.0084 to 0.0013 BL and the
         terminal tangent error from 24 deg to ~6 deg;
      7. resample to ``n_stations`` uniform in arc length and Savitzky-Golay
         smooth at that resolution (``mode='interp'`` fits a polynomial at the
         ends rather than padding, which a box filter cannot do).

    Head-first rule: the end whose mean EDT over the first/last 10% of stations
    is larger comes first.  FAILURE MODE: this assumes the head is girthier than
    the peduncle, which holds for sharks but not for a body that thickens
    posteriorly (a tadpole, a gravid female with a slender head, or a mesh whose
    caudal fin survived the core threshold and inflated the tail EDT).  Check the
    reported ``head_width``/``tail_width`` in the returned info and flip if
    needed.

    Args:
        mesh: ``trimesh.Trimesh`` (any pose).
        voxel_pitch: world-unit voxel size; default ``max(extents) / 128``,
            coarsened if the grid would exceed ``max_cells``.
        n_stations: output stations.
        core_radius_frac, core_pitch_mult: the ``tau`` knobs above.
        end_trim_frac: fraction of the raw medial path discarded at each end.
        seed: unused; extraction is deterministic.

    Returns:
        ``(centerline, info)``: an ``(n_stations, 3)`` array uniform in arc
        length and oriented head-first, plus a dict with ``pitch``, ``tau``,
        ``radius`` (EDT sampled at each station), ``head_width``,
        ``tail_width``, ``length`` and ``n_core_voxels``.
    """
    extents = np.asarray(mesh.extents, dtype=float)
    pitch = float(voxel_pitch) if voxel_pitch else float(np.max(extents)) / 128.0
    cells = np.prod(np.maximum(extents / pitch + 3.0, 1.0))
    if cells > max_cells:
        pitch *= float((cells / max_cells) ** (1.0 / 3.0))

    mask, transform = _voxelize(mesh, pitch)
    mask = _largest_component(mask)
    edt = ndimage.distance_transform_edt(mask, sampling=pitch)

    tau = max(core_pitch_mult * pitch, core_radius_frac * float(edt.max()))
    core = mask & (edt >= tau)
    if core.sum() < 8:
        warnings.warn(
            "the thick-core threshold tau=%.5g emptied the grid; falling back to "
            "the full mask, so fins may divert the centerline" % tau,
            RuntimeWarning,
            stacklevel=2,
        )
        core = mask
        tau = 0.0
    core = _largest_component(core)

    plain = _voxel_graph(core)
    inv = np.zeros(core.shape)
    inv[core] = 1.0 / np.maximum(edt[core], 0.5 * pitch)
    medial = _voxel_graph(core, weight=inv)

    # trimesh voxel matrices are indexed (x, y, z), and ``transform`` maps that
    # index triple straight to world coordinates -- not the (z, y, x) order a
    # numpy image would use.
    ix, iy, iz = np.nonzero(core)
    a = _farthest(plain, int(np.argmax(edt[ix, iy, iz])))
    b = _farthest(plain, a)

    _, pred = dijkstra(medial, directed=False, indices=a, return_predecessors=True)
    order, node = [], b
    while node != -9999 and node != a:
        order.append(node)
        node = int(pred[node])
    order.append(a)
    order = np.asarray(order[::-1], dtype=np.int64)

    ijk = np.column_stack([ix[order], iy[order], iz[order]]).astype(float)
    pts = trimesh.transform_points(ijk, transform)

    window = min(31, max(5, 2 * int(0.03 * len(pts)) + 1))
    dense = resample_polyline(_smooth(pts, window), max(8 * int(n_stations), 256))
    k = int(round(float(end_trim_frac) * len(dense)))
    if 2 * k < len(dense) - 4:
        dense = dense[k:len(dense) - k]
    stations = resample_polyline(dense, n_stations)
    w = min(len(stations) - (1 - len(stations) % 2), 5)
    if w >= 5:
        stations = resample_polyline(
            savgol_filter(stations, w, 3, axis=0, mode="interp"), n_stations
        )

    grid = trimesh.transform_points(stations, np.linalg.inv(transform))
    radius = ndimage.map_coordinates(
        edt, [grid[:, 0], grid[:, 1], grid[:, 2]], order=1, mode="nearest"
    )
    k = max(1, int(n_stations) // 10)
    head_w, tail_w = float(radius[:k].mean()), float(radius[-k:].mean())
    if tail_w > head_w:
        stations = stations[::-1].copy()
        radius = radius[::-1].copy()
        head_w, tail_w = tail_w, head_w

    length = float(arc_length(stations)[-1])
    volume = float(mask.sum()) * pitch ** 3
    expected = volume / max(np.pi * float(np.mean(radius) ** 2), 1e-12)
    if length < 0.6 * expected:
        warnings.warn(
            "extracted centerline length %.4g is far below the volume-derived "
            "expectation %.4g; the mesh does not look tubular (blob, or a bend "
            "so tight its flanks fuse) -- treat this centerline as unreliable"
            % (length, expected),
            RuntimeWarning,
            stacklevel=2,
        )

    return stations, {
        "pitch": pitch,
        "tau": tau,
        "radius": radius,
        "head_width": head_w,
        "tail_width": tail_w,
        "length": length,
        "n_core_voxels": int(core.sum()),
    }


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

def tube_frames(centerline, up=(0.0, 0.0, 1.0)):
    """Rotation-minimising frames along a centerline.

    Thin wrapper over prototype 02's ``rotation_minimizing_frames`` that seeds
    the normal with the dorsal direction, so phi = 0 means "up" on the animal.
    A planar centerline (a lateral C-curve) with an out-of-plane seed keeps a
    twist-free, out-of-plane normal all the way down -- that property is what
    makes de-bending a purely lateral straightening.

    Returns ``(tangents, normals, binormals)``, each ``(n, 3)``, right-handed
    with ``binormal = tangent x normal``.
    """
    return rotation_minimizing_frames(np.asarray(centerline, dtype=float), up)


def straight_centerline(centerline, n=None):
    """The canonical straight axis matching ``centerline``'s arc length.

    Snout at ``+S/2``, tail at ``-S/2`` along X, centred on the origin, with the
    same station spacing (so station indices and arc length transfer unchanged).
    """
    s = arc_length(np.asarray(centerline, dtype=float))
    if n is not None:
        s = np.linspace(0.0, s[-1], int(n))
    return np.column_stack([0.5 * s[-1] - s, np.zeros_like(s), np.zeros_like(s)])


def canonical_frames(n):
    """Constant frames of the straight pose: T = -X, N = +Z (dorsal), B = +Y."""
    n = int(n)
    return (
        np.tile([-1.0, 0.0, 0.0], (n, 1)),
        np.tile([0.0, 0.0, 1.0], (n, 1)),
        np.tile([0.0, 1.0, 0.0], (n, 1)),
    )


# ---------------------------------------------------------------------------
# Tube coordinates
# ---------------------------------------------------------------------------

class TubeCoords(NamedTuple):
    """Per-point tube chart coordinates.

    s: arc length of the foot point along the centerline (may fall outside
       [0, total_length] for material overhanging either end).
    r: perpendicular distance from the axis.
    phi: angle in the cross-section, from the normal (+Z when straight, i.e.
       dorsal) toward the binormal (+Y, the animal's left), in (-pi, pi].
    station: index of the centerline *segment* the foot point lies on.  Carried
       explicitly because ``s`` alone is ambiguous at corners; see the module
       docstring's invertibility contract.
    """

    s: np.ndarray
    r: np.ndarray
    phi: np.ndarray
    station: np.ndarray
    total_length: float
    n_stations: int


def _segments(centerline):
    cl = np.asarray(centerline, dtype=float)
    if cl.ndim != 2 or cl.shape[1] != 3 or len(cl) < 2:
        raise ValueError("centerline must be (n >= 2, 3)")
    d = cl[1:] - cl[:-1]
    seg = np.linalg.norm(d, axis=1)
    if np.any(seg < 1e-12):
        raise ValueError("centerline has a zero-length segment")
    return cl[:-1], d, seg, np.concatenate([[0.0], np.cumsum(seg)])


def _frame_at(d, seg, frames, station, t):
    """Frame at (segment ``station``, parameter ``t``), orthonormal.

    The tangent is the *segment* direction, so the perpendicular projection is
    exact; the normal is the linearly interpolated RMF normal re-orthogonalised
    against it.  ``t`` is clamped for the frame only -- the position is free to
    extrapolate past either end, where the terminal frame is held constant.
    """
    _, normals, _ = frames
    tc = np.clip(t, 0.0, 1.0)[:, None]
    tang = d[station] / seg[station][:, None]
    nrm = normals[station] * (1.0 - tc) + normals[station + 1] * tc
    nrm = _unit(nrm - np.sum(nrm * tang, axis=1, keepdims=True) * tang)
    return tang, nrm, np.cross(tang, nrm)


def tube_coords(mesh_or_points, centerline, frames=None):
    """Project points onto the tube chart.

    Args:
        mesh_or_points: a ``trimesh.Trimesh`` or an (k, 3) array.
        centerline: (n, 3) polyline, head first.
        frames: output of ``tube_frames``; computed if omitted.

    Returns:
        ``TubeCoords``.  Deterministic: the foot segment is the argmin of the
        clamped point-segment distance (first index wins on ties), after which
        the projection parameter is *un*clamped so that ``r`` is exactly
        perpendicular and no material is lost off the ends.
    """
    pts = (
        np.asarray(mesh_or_points.vertices, dtype=float)
        if isinstance(mesh_or_points, trimesh.Trimesh)
        else np.asarray(mesh_or_points, dtype=float)
    )
    if frames is None:
        frames = tube_frames(centerline)
    a, d, seg, cum = _segments(centerline)

    rel = pts[:, None, :] - a[None, :, :]
    t_all = np.einsum("kmi,mi->km", rel, d) / (seg ** 2)[None, :]
    proj = a[None] + np.clip(t_all, 0.0, 1.0)[..., None] * d[None]
    station = np.argmin(np.sum((pts[:, None, :] - proj) ** 2, axis=2), axis=1)

    t = t_all[np.arange(len(pts)), station]
    foot = a[station] + t[:, None] * d[station]
    _, nrm, bnm = _frame_at(d, seg, frames, station, t)

    v = pts - foot
    rn = np.sum(v * nrm, axis=1)
    rb = np.sum(v * bnm, axis=1)
    return TubeCoords(
        s=cum[station] + t * seg[station],
        r=np.hypot(rn, rb),
        phi=np.arctan2(rb, rn),
        station=station.astype(np.int64),
        total_length=float(cum[-1]),
        n_stations=len(np.asarray(centerline)),
    )


def tube_to_points(coords, centerline, frames=None):
    """Inverse of ``tube_coords`` on the same (centerline, frames)."""
    if frames is None:
        frames = tube_frames(centerline)
    a, d, seg, cum = _segments(centerline)
    st = np.asarray(coords.station, dtype=np.int64)
    if st.max(initial=0) >= len(seg):
        raise ValueError("station index exceeds the centerline's segment count")
    t = (np.asarray(coords.s, dtype=float) - cum[st]) / seg[st]
    foot = a[st] + t[:, None] * d[st]
    _, nrm, bnm = _frame_at(d, seg, frames, st, t)
    r = np.asarray(coords.r, dtype=float)[:, None]
    phi = np.asarray(coords.phi, dtype=float)[:, None]
    return foot + r * (np.cos(phi) * nrm + np.sin(phi) * bnm)


def map_points(points, src_centerline, src_frames, dst_centerline, dst_frames):
    """Transport points from one centerline to another through the tube chart.

    Both centerlines must have the same station count (station indices are
    carried across).  Arc length is rescaled by the length ratio and ``r``/``phi``
    are preserved, so the map is an isometry in ``s`` when the lengths match and
    a uniform axial stretch otherwise.
    """
    src_centerline = np.asarray(src_centerline, dtype=float)
    dst_centerline = np.asarray(dst_centerline, dtype=float)
    if len(src_centerline) != len(dst_centerline):
        raise ValueError(
            "source and target centerlines must have the same station count "
            "(%d vs %d); resample one first"
            % (len(src_centerline), len(dst_centerline))
        )
    c = tube_coords(points, src_centerline, src_frames)
    ratio = float(arc_length(dst_centerline)[-1]) / max(c.total_length, 1e-12)
    return tube_to_points(c._replace(s=c.s * ratio), dst_centerline, dst_frames)


def _remap(mesh, src_cl, src_fr, dst_cl, dst_fr):
    """Copy of ``mesh`` with vertices transported; faces/UVs/visual untouched."""
    out = mesh.copy()
    out.vertices = map_points(
        np.asarray(mesh.vertices, dtype=float), src_cl, src_fr, dst_cl, dst_fr
    )
    return out


def debend(mesh, centerline, frames=None, up=(0.0, 0.0, 1.0), check_roll=True):
    """Straighten a bent mesh onto the canonical +X axis.

    Vertex positions change; ``faces``, ``visual`` (UVs, texture, material) and
    ``metadata`` are carried over untouched, which is what lets a textured
    Meshy GLB survive the operation.

    De-bending keeps ``(r, phi)`` and replaces the centerline, so it removes the
    *bend* and nothing else: a body rolled about its own axis comes out straight
    and still rolled.  With ``check_roll`` (default) ``estimate_roll`` measures
    that torsion first and warns past ``_ROLL_WARN_RAD``; pass ``False`` to skip
    the measurement (it costs one extra chart and fin detection).

    Returns ``(straight_mesh, straight_centerline)``.
    """
    if frames is None:
        frames = tube_frames(centerline, up=up)
    if check_roll:
        _warn_if_rolled(mesh, centerline, frames)
    target = straight_centerline(centerline)
    out = _remap(mesh, centerline, frames, target, canonical_frames(len(target)))
    out.metadata["centerline"] = target
    out.metadata["bent_centerline"] = np.asarray(centerline, dtype=float)
    return out, target


def rebend(straight_mesh, target_centerline, source_centerline=None,
           target_frames=None, up=(0.0, 0.0, 1.0)):
    """Inverse of ``debend``: re-embed a straight mesh on ``target_centerline``."""
    target_centerline = np.asarray(target_centerline, dtype=float)
    src = (
        straight_centerline(target_centerline)
        if source_centerline is None
        else np.asarray(source_centerline, dtype=float)
    )
    if target_frames is None:
        target_frames = tube_frames(target_centerline, up=up)
    out = _remap(
        straight_mesh, src, canonical_frames(len(src)),
        target_centerline, target_frames,
    )
    out.metadata["centerline"] = target_centerline
    return out


# ---------------------------------------------------------------------------
# Fin detection
# ---------------------------------------------------------------------------

class FinDetection(NamedTuple):
    """Per-vertex fin labels plus per-fin insertion geometry.

    labels: (V,) array of strings drawn from ``FIN_LABELS``.
    fins: name -> dict with ``vertex_indices``, ``n_vertices``,
        ``station_range`` (inclusive min/max centerline station touched),
        ``s_range`` (arc length), ``insertion_centroid`` (3D centroid of the
        island's innermost quartile, i.e. where the fin meets the body) and
        ``phi_centroid`` (radians).  Keys named ``unassigned_island_<k>`` are
        islands that lost a name collision: same fields plus ``unassigned``
        (True) and ``collided_with`` (the name they lost).  Their vertices keep
        the ``body`` label, so a consumer that binds fins by name must skip that
        prefix.
    envelope: (n_stations,) robust body radius per station.
    """

    labels: np.ndarray
    fins: dict
    envelope: np.ndarray


def _radius_envelope(coords, mask, n_stations, percentile, smooth):
    env = np.full(n_stations, np.nan)
    st = coords.station[mask]
    r = coords.r[mask]
    if len(st):
        order = np.argsort(st, kind="stable")
        st, r = st[order], r[order]
        bounds = np.searchsorted(st, np.arange(n_stations + 1))
        for i in range(n_stations):
            lo, hi = bounds[i], bounds[i + 1]
            if hi > lo:
                env[i] = np.percentile(r[lo:hi], percentile)
    good = np.isfinite(env)
    if not good.any():
        raise ValueError("no vertices project onto the centerline")
    env = np.interp(np.arange(n_stations), np.flatnonzero(good), env[good])
    return ndimage.median_filter(env, size=int(smooth), mode="nearest")


def _classify(islands):
    """Name islands from their (s, phi) centroids against anatomical priors.

    Every median island past ``_CAUDAL_S_MIN`` is caudal, not just the
    posterior-most one: a caudal fin routinely arrives as more than one island
    (the two lobes, or a lobe split by a hole in the scan), and calling only the
    last one caudal leaves its siblings claiming ``dorsal``/``anal`` from the far
    end of the body.  ``detect_fins`` then merges the ones that touch.
    """
    names = {}
    dorsal_sector, ventral_sector = [], []
    for key, isl in islands.items():
        phi_deg = abs(np.rad2deg(isl["phi_centroid"]))
        if phi_deg <= _PHI_DORSAL:
            dorsal_sector.append(key)
        elif phi_deg >= _PHI_VENTRAL:
            ventral_sector.append(key)
        else:
            lateral = "pectoral" if isl["s_frac"] < _PECTORAL_S_MAX else "pelvic"
            names[key] = "%s_%s" % (lateral, "L" if isl["phi_centroid"] > 0 else "R")

    for sector, median_name in ((dorsal_sector, "dorsal"), (ventral_sector, "anal")):
        if not sector:
            continue
        sector = sorted(sector, key=lambda k: islands[k]["s_frac"])
        while sector and islands[sector[-1]]["s_frac"] >= _CAUDAL_S_MIN:
            names[sector.pop()] = "caudal"  # split by phi, per vertex, below
        for key in sector:
            names[key] = median_name
    return names


def _island_entry(members, coords, verts):
    """Geometry record for one island; shared by named fins and demotions."""
    inner = members[coords.r[members] <= np.percentile(coords.r[members], 25.0)]
    return {
        "vertex_indices": members,
        "n_vertices": int(len(members)),
        "station_range": (int(coords.station[members].min()),
                          int(coords.station[members].max())),
        "s_range": (float(coords.s[members].min()),
                    float(coords.s[members].max())),
        "insertion_centroid": verts[inner].mean(axis=0),
        "phi_centroid": float(np.arctan2(np.mean(np.sin(coords.phi[members])),
                                         np.mean(np.cos(coords.phi[members])))),
    }


def _merge_touching(groups, coords):
    """Cluster islands whose inclusive station ranges overlap or touch.

    One anatomical fin can legitimately arrive as several islands -- a caudal
    whose two lobes are separate components is the standard case -- but only
    when those islands occupy the *same* stretch of the body.  Islands more than
    ``_MERGE_STATION_GAP`` stations apart are kept separate and arbitrated by
    ``detect_fins``.  Returns merged member arrays ordered by first station.
    """
    spans = [(int(coords.station[g].min()), int(coords.station[g].max()), g)
             for g in groups]
    spans.sort(key=lambda t: (t[0], t[1]))
    clusters = []
    for lo, hi, g in spans:
        if clusters and lo <= clusters[-1][1] + _MERGE_STATION_GAP:
            c_lo, c_hi, c_g = clusters[-1]
            clusters[-1] = (c_lo, max(c_hi, hi), np.union1d(c_g, g))
        else:
            clusters.append((lo, hi, g))
    return [c[2] for c in clusters]


def _prior_mismatch(name, members, coords):
    """How far an island's (s, phi) centroid sits from ``name``'s prior.

    Unitless: ``_PRIOR_S_SCALE`` of chart length in s counts the same as
    ``_PRIOR_PHI_SCALE`` degrees in phi.  Names with no prior score 0, which
    hands the tie-break to island size.
    """
    if name not in _FIN_PRIORS:
        return 0.0
    s0, phi0 = _FIN_PRIORS[name]
    s_frac = float(np.mean(coords.s[members])) / max(coords.total_length, 1e-12)
    phi = np.arctan2(np.mean(np.sin(coords.phi[members])),
                     np.mean(np.cos(coords.phi[members])))
    d = phi - np.deg2rad(phi0)
    dphi = abs(np.arctan2(np.sin(d), np.cos(d)))
    return float(np.hypot((s_frac - s0) / _PRIOR_S_SCALE,
                          np.rad2deg(dphi) / _PRIOR_PHI_SCALE))


def _island_desc(members, coords):
    total = max(coords.total_length, 1e-12)
    return "%d verts, stations %d-%d, s %.3f-%.3f" % (
        len(members), int(coords.station[members].min()),
        int(coords.station[members].max()),
        float(coords.s[members].min()) / total,
        float(coords.s[members].max()) / total,
    )


def _warn_missing_fins(fins, margin, floor_frac, min_island):
    """G4: a fin too slender to clear the envelope must not vanish quietly."""
    missing = [n for n in FIN_LABELS if n != "body" and n not in fins]
    if missing:
        warnings.warn(
            "detect_fins: %d of the %d expected fins were not found (%s) at "
            "margin=%.3g, floor_frac=%.3g, min_island=%d -- a blade that never "
            "reaches (1 + margin) * envelope + floor_frac * S is invisible to "
            "the detector.  Lower the margin, or check that the mesh really has "
            "them (and that ``up`` is right, which decides their names)."
            % (len(missing), len(FIN_LABELS) - 1, ", ".join(missing),
               margin, floor_frac, int(min_island)),
            RuntimeWarning,
            stacklevel=3,
        )


def detect_fins(mesh, coords, percentile=60.0, margin=0.30, floor_frac=0.002,
                envelope_smooth=5, min_island=8, passes=2, check=True):
    """Label every vertex ``body`` or with a fin name, from the tube chart.

    A vertex protrudes when ``r > (1 + margin) * envelope[station] +
    floor_frac * S``, where ``envelope`` is the ``percentile``-th percentile of
    ``r`` per station -- a robust stand-in for the local body radius, recomputed
    over the non-protruding vertices for ``passes`` iterations so that large
    fins cannot inflate their own envelope.  Protruding vertices are grouped
    into islands by connected components of the mesh edge graph restricted to
    them, and each island is named from its (s, phi) centroid (see the
    ``_PHI_*``/``_CAUDAL_S_MIN`` priors at the top of this module).  An island
    named caudal is split per-vertex into ``caudal_upper``/``caudal_lower`` at
    ``|phi| = 90 deg``, so a single wrap-around caudal fin and two separate lobes
    both come out right.

    Two islands may only share a name when their station ranges overlap or touch
    (``_MERGE_STATION_GAP``) -- one fin arriving as several components.  Islands
    that classify alike but sit at different places along the body are *not* one
    fin: the one whose (s, phi) centroid best matches the anatomical prior keeps
    the name, the others are demoted to ``unassigned_island_<k>`` entries in the
    returned ``fins`` dict, and a ``RuntimeWarning`` names the collision.
    Demoted vertices keep the ``body`` label, so a downstream rig that binds
    ``fins`` by name must skip the ``unassigned_island_`` prefix (there are no
    vertices carrying that label to weight).

    With ``check`` (default), the detection is audited: any of the eight
    expected fins that were not found is reported with the envelope margin in
    use, and ``check_anatomy`` flags a probably-flipped ``up`` vector.

    Note: a fin's *root* vertices sit at body radius and are therefore labelled
    ``body``.  That is deliberate -- the label marks the protruding blade -- and
    the returned ``station_range`` / ``insertion_centroid`` are what a rig should
    use to bind the base.

    Returns ``FinDetection``.
    """
    n_stations = int(coords.n_stations) - 1
    keep = np.ones(len(coords.r), dtype=bool)
    env = None
    for _ in range(max(1, int(passes))):
        env = _radius_envelope(coords, keep, n_stations, percentile, envelope_smooth)
        thresh = (1.0 + margin) * env[coords.station] + floor_frac * coords.total_length
        keep = coords.r <= thresh
    protruding = ~keep

    labels = np.full(len(coords.r), "body", dtype=object)
    fins = {}
    if protruding.any():
        idx = np.flatnonzero(protruding)
        remap = np.full(len(protruding), -1, dtype=np.int64)
        remap[idx] = np.arange(len(idx))
        e = np.asarray(mesh.edges_unique, dtype=np.int64)
        e = e[protruding[e[:, 0]] & protruding[e[:, 1]]]
        graph = coo_matrix(
            (np.ones(len(e)), (remap[e[:, 0]], remap[e[:, 1]])),
            shape=(len(idx), len(idx)),
        ).tocsr()
        n_comp, comp = connected_components(graph, directed=False)

        islands = {}
        for c in range(n_comp):
            members = idx[comp == c]
            if len(members) < int(min_island):
                continue
            phis = coords.phi[members]
            islands[c] = {
                "members": members,
                "s_frac": float(np.mean(coords.s[members]) / coords.total_length),
                "phi_centroid": float(
                    np.arctan2(np.mean(np.sin(phis)), np.mean(np.cos(phis)))
                ),
            }

        verts = np.asarray(mesh.vertices, dtype=float)
        named = _classify(islands)
        candidates = {}
        for c in sorted(named):
            name = named[c]
            members = islands[c]["members"]
            if name == "caudal":
                upper = np.abs(coords.phi[members]) < 0.5 * np.pi
                groups = [("caudal_upper", members[upper]),
                          ("caudal_lower", members[~upper])]
            else:
                groups = [(name, members)]
            for gname, gmem in groups:
                if len(gmem) < int(min_island):
                    continue
                candidates.setdefault(gname, []).append(gmem)

        demoted = []
        for gname in sorted(candidates):
            clusters = _merge_touching(candidates[gname], coords)
            keep = clusters[0]
            if len(clusters) > 1:
                order = sorted(
                    range(len(clusters)),
                    key=lambda i: (_prior_mismatch(gname, clusters[i], coords),
                                   -len(clusters[i]),
                                   int(coords.station[clusters[i]].min())),
                )
                keep = clusters[order[0]]
                losers = [clusters[i] for i in order[1:]]
                warnings.warn(
                    "detect_fins: %d disjoint islands classify as %r and their "
                    "station ranges do not touch, so they are not one fin: "
                    "keeping the one closest to the anatomical prior (%s) and "
                    "demoting %s to 'unassigned_island_*' entries whose vertices "
                    "stay labelled 'body'.  An unexpected lump, a fin far from "
                    "its prior, or a flipped up vector all look like this."
                    % (len(clusters), gname, _island_desc(keep, coords),
                       "; ".join(_island_desc(g, coords) for g in losers)),
                    RuntimeWarning,
                    stacklevel=2,
                )
                demoted.extend((gname, g) for g in losers)
            labels[keep] = gname
            fins[gname] = _island_entry(keep, coords, verts)

        demoted.sort(key=lambda kv: float(np.mean(coords.s[kv[1]])))
        for k, (gname, gmem) in enumerate(demoted):
            entry = _island_entry(gmem, coords, verts)
            entry["unassigned"] = True
            entry["collided_with"] = gname
            fins["unassigned_island_%d" % k] = entry

    if check:
        _warn_missing_fins(fins, margin, floor_frac, min_island)
    detection = FinDetection(labels=labels.astype(str), fins=fins, envelope=env)
    if check:
        check_anatomy(detection)
    return detection


# ---------------------------------------------------------------------------
# Orientation audit: body roll, and the sign of ``up``
# ---------------------------------------------------------------------------

def _theil_sen(x, y):
    """Robust line fit: median of pairwise slopes, median intercept.

    Chosen over least squares because a handful of stations always pick the
    wrong maximum-radius vertex (a near-circular cross-section, a cap, a fin
    root that stayed body-labelled) and those outliers must not tilt the fit.
    """
    i, j = np.triu_indices(len(x), k=1)
    dx = x[j] - x[i]
    ok = np.abs(dx) > 1e-12
    if not ok.any():
        return 0.0, float(np.median(y))
    slope = float(np.median((y[j][ok] - y[i][ok]) / dx[ok]))
    return slope, float(np.median(y - slope * x))


def estimate_roll(mesh, coords, det):
    """Fit the body's roll (torsion about its own axis) through the chart.

    A shark's cross-section is taller than it is wide, so the fattest body
    vertex of a station points along the dorsoventral axis.  (Only vertices
    inside the body envelope count: a fin *root* keeps the ``body`` label by
    design, and would otherwise be the fattest thing at its station.)  Its ``phi`` is
    therefore the chart's estimate of where "up" actually is at that station:
    on an unrolled body it stays at 0 (or pi -- the maximum is an *axis*, not a
    direction, so it is folded into the dorsal band ``(-90, 90]`` degrees and
    unwrapped with period pi), and on a rolled body it drifts linearly with arc
    length.  Whole *stations* touched by a detected fin are dropped, not just the
    fin-labelled vertices: a fin's root row keeps the ``body`` label by design
    (it sits under the detection margin), and it is the fattest thing at its
    station, so leaving those stations in makes the fit track the fins instead
    of the ridge.  The block is padded by ``_ROLL_FIN_STATION_PAD`` stations
    each side for the same reason: the *detected* blade is narrower than the
    fin, and the stations just outside it hold nothing but undetected root.
    Stations whose fattest candidate is too near the axis (the end caps) are
    dropped too, as are both terminal stations, which collect all the material
    overhanging the chart.

    Args:
        mesh: the mesh ``coords`` was charted from (used only to check length).
        coords: ``TubeCoords`` for that mesh.
        det: ``FinDetection`` for the same mesh, for the body mask and envelope.

    Returns:
        ``(slope, r2)``: ``slope`` in radians of roll per unit arc length (so
        the roll across the whole body is ``slope * coords.total_length``), and
        the fit's coefficient of determination.  Fewer than
        ``_ROLL_MIN_STATIONS`` usable stations returns ``(0.0, 0.0)`` -- no
        evidence of roll, not a claim that there is none.  That is the quiet
        answer on purpose: a fit made from the leftovers of a fin-covered body
        is worse than no fit, and this one only ever raises an alarm.
    """
    if len(np.asarray(mesh.vertices)) != len(coords.r):
        raise ValueError("coords do not belong to this mesh (%d vs %d vertices)"
                         % (len(np.asarray(mesh.vertices)), len(coords.r)))
    labels = np.asarray(det.labels)
    body = labels == "body"
    env = np.asarray(det.envelope, dtype=float)
    n_stations = int(coords.n_stations) - 1

    blocked = np.zeros(n_stations, dtype=bool)
    for fin in det.fins.values():
        lo, hi = fin["station_range"]
        blocked[max(0, int(lo) - _ROLL_FIN_STATION_PAD):
                min(n_stations, int(hi) + 1 + _ROLL_FIN_STATION_PAD)] = True
    blocked[0] = blocked[-1] = True

    st = coords.station[body]
    order = np.argsort(st, kind="stable")
    st = st[order]
    r = coords.r[body][order]
    phi = coords.phi[body][order]
    s = coords.s[body][order]
    bounds = np.searchsorted(st, np.arange(n_stations + 1))

    ss, psi = [], []
    for i in range(n_stations):
        if blocked[i]:
            continue
        lo, hi = bounds[i], bounds[i + 1]
        if hi - lo < _ROLL_MIN_PER_STATION:
            continue
        if not np.isfinite(env[i]) or env[i] <= 0.0:
            continue
        band = np.flatnonzero(r[lo:hi] <= (1.0 + _ROLL_BAND_FRAC) * env[i])
        if len(band) < _ROLL_MIN_PER_STATION:
            continue
        k = lo + int(band[np.argmax(r[lo:hi][band])])
        if r[k] < _ROLL_MIN_RADIUS_FRAC * env[i]:
            continue
        ss.append(s[k])
        psi.append(phi[k])
    if len(ss) < _ROLL_MIN_STATIONS:
        return 0.0, 0.0

    ss = np.asarray(ss, dtype=float)
    psi = np.asarray(psi, dtype=float)
    # Fold the dorsoventral axis into the dorsal band, then undo the pi jumps a
    # progressive roll makes as it crosses the band edge.
    psi = ((psi + 0.5 * np.pi) % np.pi) - 0.5 * np.pi
    psi = np.unwrap(psi, period=np.pi)

    slope, intercept = _theil_sen(ss, psi)
    resid = psi - (slope * ss + intercept)
    ss_tot = float(((psi - psi.mean()) ** 2).sum())
    r2 = 0.0 if ss_tot <= 1e-18 else float(1.0 - (resid ** 2).sum() / ss_tot)
    return float(slope), r2


def _warn_if_rolled(mesh, centerline, frames):
    """Roll check shared by ``debend`` and the CLI.  Returns ``(slope, r2)``."""
    coords = tube_coords(mesh, centerline, frames)
    det = detect_fins(mesh, coords, check=False)
    slope, r2 = estimate_roll(mesh, coords, det)
    total = slope * coords.total_length
    if abs(total) > _ROLL_WARN_RAD:
        warnings.warn(
            "estimate_roll: the body is rolled about its own axis by %.1f deg "
            "end to end (%.3f rad per unit s, r2 %.2f), past the %.2f rad "
            "threshold.  De-bending preserves (r, phi) by contract, so it will "
            "straighten this body and leave it just as rolled; the fin priors "
            "and the joint schema both assume an unrolled body, so fins will "
            "drift across phi sectors and can be mis-named.  Fix it upstream: "
            "pre-unroll the mesh before charting, or supply a per-station up "
            "(a twisting frame field), which this module does not implement."
            % (np.rad2deg(total), slope, r2, _ROLL_WARN_RAD),
            RuntimeWarning,
            stacklevel=3,
        )
    return slope, r2


def check_anatomy(det, warn=True):
    """Flag a probably-flipped ``up`` vector from the shape of the detection.

    Two facts about sharks make the dorsoventral sign checkable after the fact:
    the dorsal fin is bigger than the anal fin, and a heterocercal caudal has a
    longer upper lobe than lower.  If the chart's ``up`` is negated, ``phi`` is
    reflected, and detection reports exactly the opposite -- silently, with
    plausible-looking names on every island (left and right swap too, since the
    binormal follows the normal).

    Args:
        det: ``FinDetection`` to audit.
        warn: emit a ``RuntimeWarning`` naming the remedy when anything flags.

    Returns:
        List of flag strings; empty means the detection looks right way up.
        An absent fin is never a flag -- ``_warn_missing_fins`` owns that.
    """
    fins = det.fins

    def span(name):
        lo, hi = fins[name]["s_range"]
        return float(hi - lo)

    flags = []
    if "anal" in fins and "dorsal" in fins:
        if fins["anal"]["n_vertices"] > fins["dorsal"]["n_vertices"]:
            flags.append(
                "the island named 'anal' is bigger than the one named 'dorsal' "
                "(%d vs %d vertices); on a shark the dorsal fin is the bigger"
                % (fins["anal"]["n_vertices"], fins["dorsal"]["n_vertices"])
            )
        elif span("anal") > span("dorsal"):
            flags.append(
                "the island named 'anal' spans more arc length than the one "
                "named 'dorsal' (%.4f vs %.4f)" % (span("anal"), span("dorsal"))
            )
    if "caudal_lower" in fins and "caudal_upper" in fins:
        if span("caudal_lower") > span("caudal_upper"):
            flags.append(
                "'caudal_lower' spans more arc length than 'caudal_upper' "
                "(%.4f vs %.4f); a heterocercal tail has the longer lobe up"
                % (span("caudal_lower"), span("caudal_upper"))
            )
    if flags and warn:
        warnings.warn(
            "check_anatomy: up vector probably flipped -- %s.  Remedy: re-run "
            "with --up negated (up -> -up), or pass --auto-up to let the CLI do "
            "it; dorsal/anal, caudal_upper/caudal_lower and left/right are all "
            "mirrored until you do." % "; ".join(flags),
            RuntimeWarning,
            stacklevel=3,
        )
    return flags


def _chart(mesh, centerline, up, check=True):
    """(frames, coords, detection) for one choice of ``up``.  CLI helper."""
    frames = tube_frames(centerline, up=tuple(up))
    coords = tube_coords(mesh, centerline, frames)
    return frames, coords, detect_fins(mesh, coords, check=check)


def _cli(argv=None):
    """``python mesh3d.py IN.glb [-o OUT.glb]`` -- de-bend any mesh to rest pose.

    Prints the extracted chart's geometry and the detected fins, so the same
    command is the smoke test for a real Meshy GLB.
    """
    import argparse

    ap = argparse.ArgumentParser(description=_cli.__doc__)
    ap.add_argument("mesh")
    ap.add_argument("-o", "--out", default=None, help="write the de-bent mesh here")
    ap.add_argument("-n", "--n-stations", type=int, default=64)
    ap.add_argument("-p", "--voxel-pitch", type=float, default=None)
    ap.add_argument("--core-radius-frac", type=float, default=0.17)
    ap.add_argument("--up", type=float, nargs=3, default=(0.0, 0.0, 1.0),
                    help="dorsal direction of the input mesh (default +Z)")
    ap.add_argument("--auto-up", action="store_true",
                    help="if the detected anatomy says --up is upside down "
                         "(anal bigger than dorsal, or the lower caudal lobe "
                         "longer than the upper), negate it and re-chart once")
    args = ap.parse_args(argv)

    mesh = load_mesh(args.mesh)
    centerline, info = extract_centerline_3d(
        mesh, voxel_pitch=args.voxel_pitch, n_stations=args.n_stations,
        core_radius_frac=args.core_radius_frac,
    )
    print("centerline: %d stations, length %.4f, pitch %.5f, head/tail width "
          "%.4f/%.4f" % (len(centerline), info["length"], info["pitch"],
                         info["head_width"], info["tail_width"]))
    up = tuple(float(x) for x in args.up)
    frames, coords, det = _chart(mesh, centerline, up, check=not args.auto_up)
    if args.auto_up:
        flags = check_anatomy(det, warn=False)
        if flags:
            flipped = tuple(0.0 if x == 0.0 else -x for x in up)  # no -0.0
            frames2, coords2, det2 = _chart(mesh, centerline, flipped, check=False)
            if check_anatomy(det2, warn=False):
                print("--auto-up: anatomy flagged with up=%s (%s), and negating "
                      "it does not clear the flags; keeping the sign as given"
                      % (up, "; ".join(flags)))
            else:
                print("--auto-up: anatomy flagged with up=%s (%s); re-charted "
                      "with up=%s, which clears it" % (up, "; ".join(flags), flipped))
                up, frames, coords = flipped, frames2, coords2
        # Re-run with the audit on so the chosen sign gets the usual warnings.
        det = detect_fins(mesh, coords, check=True)
    print("up (dorsal direction) used: %s" % (up,))
    for name in sorted(det.fins):
        fin = det.fins[name]
        print("  %-13s %4d verts  stations %2d-%2d  insertion %s"
              % (name, fin["n_vertices"], fin["station_range"][0],
                 fin["station_range"][1], np.round(fin["insertion_centroid"], 4)))
    slope, r2 = _warn_if_rolled(mesh, centerline, frames)
    print("body roll: %.1f deg end to end (%.3f rad per unit s, r2 %.2f)"
          % (np.rad2deg(slope * coords.total_length), slope, r2))
    straight, target = debend(mesh, centerline, frames, check_roll=False)
    print("de-bent extents: %s" % np.round(straight.extents, 4))
    if args.out:
        out = straight.copy()
        out.metadata.clear()
        out.export(args.out)
        print("wrote %s" % args.out)
    return straight, target, det


if __name__ == "__main__":  # pragma: no cover
    _cli()
