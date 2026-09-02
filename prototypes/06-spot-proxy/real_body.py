"""The real scanned sevengill body, charted, decimated and posable.

WHAT THIS IS.  Prototype 04's rig run ``results/real_v11`` wrote a skinned GLB
whose BIND POSE is the de-bent (straightened) Meshy scan: 1,013,814 vertices,
1,961,876 faces, extents 0.682 x 0.230 x 0.108 m.  That mesh is far too heavy
to rasterise with prototype 05's pure-numpy renderer, and its skinning /
texture are irrelevant here -- prototype 06 never shows RGB to the matcher, it
shows detections.  So this module does three things and nothing else:

1. **Chart it the way 04 does.**  ``report/centerline.json`` holds the straight
   centerline the de-bend produced (64 stations, chord length 0.4803 m, snout
   at ``+X``).  ``mesh3d.canonical_frames`` are its frames (T = -X, N = +Z
   dorsal, B = +Y = the animal's LEFT), which is exactly what
   ``mesh3d.tube_frames(straight_centerline, up=(0,0,1))`` returns, so
   ``tube_coords`` on that pair reproduces the chart ``texture_identity.
   straighten`` builds.  ``texture_identity.chart_coords(..., normalize=
   "extent")`` then converts ``(s_metres, phi)`` to prototype 05's canonical
   ``(s in [0,1], phi in [-pi,pi))``.  BECAUSE the normalisation is the same
   one that produced ``results/real/identity/chart_skin.png`` -- of which
   ``assets/chart_skin_x4.png`` is a verified 4x nearest upscale -- the base
   skin chart lines up with these vertex coordinates cell for cell.

2. **Decimate by vertex clustering** on a regular grid, carrying the chart
   coordinates rather than re-deriving them: ``s`` and ``r`` by arithmetic
   mean, ``phi`` by CIRCULAR mean (a cell straddling the ventral seam averages
   to the seam, not to the dorsal midline), ``is_fin`` by ``any``.

3. **Pose it** with prototype 05's planar-bend model
   (``make_dataset.pose_vertices``: ``kappa(u) = amp*cos(2*pi*wave*u+phase)``),
   swept through ``mesh3d.tube_to_points`` on rotation-minimising
   ``mesh3d.tube_frames`` of the bent centreline.

REST POSITIONS ARE RECONSTRUCTED, NOT AVERAGED.  A cluster's mean position and
``tube_to_points`` of its mean ``(s, r, phi)`` differ by a fraction of a cell.
Keeping the averaged position would make ``pose(amp=0)`` a near-identity rather
than an identity, and every downstream statement about the pose being an
isometry in the chart would be approximate for no reason.  So the stored rest
vertices ARE ``tube_to_points(s, r, phi)`` on the straight centreline; the
distance to the naive cluster mean is measured and reported in ``meta``.

THE STRETCH CAVEAT.  Fins ride the chart: their vertices are swept by the same
``(s, r, phi) -> position`` map as the body, so on the inside of a bend a blade
at radius ``r`` is compressed by ``1 - kappa*r`` and on the outside stretched by
``1 + kappa*r``.  Prototype 04 fixed exactly this for its de-bend by carrying
fin islands rigidly (``mesh3d.map_mesh(rigid_fins=True)``); this module does
NOT, because prototype 06 frames the HEAD and FOREBODY (``s <~ 0.35``) where
the only fin is the pectoral leading edge and the bends used are gentle
(``amp <= 0.35``).  ``is_fin`` is carried so a caller can measure or mask it;
:func:`fin_stretch` reports the worst-case factor for a given pose.

Run ``python real_body.py --cells 1.5,2,2.5,4,6 --json out.json`` to build
the caches and print the decimation table.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from typing import NamedTuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROTOTYPES = os.path.dirname(_HERE)
for _d in (os.path.join(_PROTOTYPES, "02-centerline-chart"),
           os.path.join(_PROTOTYPES, "04-sevengill-rig")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import mesh3d  # noqa: E402  (path shim above)
import texture_identity  # noqa: E402

__all__ = [
    "RealBody",
    "DEFAULT_GLB",
    "DEFAULT_REPORT",
    "DEFAULT_CELL_MM",
    "load_rest_mesh",
    "rest_centerline",
    "chart_rest",
    "decimate",
    "build",
    "load_cached",
    "pose",
    "bent_centerline",
    "fin_stretch",
]

#: Prototype 04's final real run.  Its bind pose is the straightened scan.
DEFAULT_GLB = os.path.normpath(os.path.join(
    _PROTOTYPES, "04-sevengill-rig", "results", "real_v11", "sevengill_rigged.glb"))
DEFAULT_REPORT = os.path.normpath(os.path.join(
    _PROTOTYPES, "04-sevengill-rig", "results", "real_v11", "report"))
DEFAULT_CACHE_DIR = os.path.join(_HERE, "assets")

#: Grid cell for vertex clustering, millimetres.  Measured on this mesh
#: (1,013,814 verts / 1,961,876 faces):
#:
#:   1.5 mm -> 75,731 / 152,467   2.0 mm -> 43,304 / 87,366
#:   2.5 mm -> 28,007 /  56,551   4.0 mm -> 10,933 / 22,212
#:   6.0 mm ->  4,875 /   9,966
#:
#: 1.5 mm is the default because that is where the PECTORAL RIM stops being a
#: sawtooth (the blade is thinner than a cell over its last few millimetres, so
#: clustering merges its two faces and quantises the rim to the grid).  It
#: costs 5.1 s of rasterisation per 2016x1512 frame, well inside budget --
#: see ``synth_render.bench`` and the README's timing table.
DEFAULT_CELL_MM = 1.5

#: Stations used to sweep a posed centreline.  512 keeps the per-corner turn
#: (which displaces a vertex at radius r by r*turn) under 0.1 mm for the pose
#: amplitudes this prototype uses.
POSE_STATIONS = 512


class RealBody(NamedTuple):
    """The decimated, charted rest body.

    vertices: ``(V, 3)`` float64 rest positions, ``tube_to_points`` of
        ``(s_m, r, phi)`` on :attr:`centerline` (snout ``+X``, dorsal ``+Z``,
        animal's left ``+Y``).
    faces: ``(F, 3)`` int64.
    s: ``(V,)`` canonical chart arc length in ``[0, 1]``, 0 = snout tip,
        1 = caudal terminus (prototype 05's convention).
    phi: ``(V,)`` in ``[-pi, pi)``, 0 = dorsal midline, ``+pi/2`` = animal's
        LEFT flank, ``+-pi`` = ventral midline.
    r: ``(V,)`` perpendicular distance from the axis, metres.
    s_m: ``(V,)`` arc length along the station chart, metres; this is what
        ``mesh3d.tube_to_points`` consumes and it runs OUTSIDE ``[0, L]``
        (the snout cap and the caudal lobe overhang the chart).
    is_fin: ``(V,)`` bool, True on a ``mesh3d.detect_fins`` blade.
    centerline: ``(64, 3)`` the straight rest centreline.
    total_length: chord length of ``centerline``, metres.
    s_raw_range: ``(lo, hi)`` the ``s_m`` extent the ``[0, 1]`` normalisation
        was fitted to; ``s = (s_m - lo) / (hi - lo)``.
    meta: dict of provenance and measured decimation numbers.
    """

    vertices: np.ndarray
    faces: np.ndarray
    s: np.ndarray
    phi: np.ndarray
    r: np.ndarray
    s_m: np.ndarray
    is_fin: np.ndarray
    centerline: np.ndarray
    total_length: float
    s_raw_range: tuple
    meta: dict


# ---------------------------------------------------------------------------
# 1. Load and chart the full-resolution rest mesh
# ---------------------------------------------------------------------------

def load_rest_mesh(glb=DEFAULT_GLB):
    """The rigged GLB's bind pose as one ``trimesh.Trimesh``.

    ``force="scene"`` then concatenate: the file holds a single geometry, so
    the concatenation is lossless, and the skin (joints/weights) is simply not
    read -- the bind pose IS the straightened mesh prototype 04 rigged.
    ``process=False`` keeps the vertex order the report's arrays refer to.
    """
    import trimesh

    scene = trimesh.load(str(glb), force="scene", process=False)
    geoms = list(scene.geometry.values())
    if not geoms:
        raise ValueError("%s holds no geometry" % (glb,))
    mesh = geoms[0] if len(geoms) == 1 else trimesh.util.concatenate(geoms)
    return mesh


def rest_centerline(report_dir=DEFAULT_REPORT, n_stations=64, length=None):
    """``(centerline, frames)`` for the straightened rest pose.

    Read from ``report/centerline.json`` when it is there (that is the exact
    polyline the de-bend produced); otherwise reconstructed from
    ``mesh3d.straight_centerline``'s rule -- snout at ``+S/2`` on ``+X``,
    uniform stations -- which needs the chart length.
    """
    path = os.path.join(str(report_dir), "centerline.json")
    if os.path.exists(path):
        with open(path) as fh:
            doc = json.load(fh)
        cl = np.asarray(doc["straight_centerline"], dtype=float)
    else:
        if length is None:
            raise ValueError("no %s; pass length= to reconstruct" % path)
        s = np.linspace(0.0, float(length), int(n_stations))
        cl = np.column_stack([0.5 * s[-1] - s, np.zeros_like(s), np.zeros_like(s)])
    return cl, mesh3d.canonical_frames(len(cl))


def chart_rest(mesh, centerline, frames, detect_fins=True):
    """``(coords, vertex_s, vertex_phi, is_fin)`` for the full-resolution mesh.

    ``coords`` is ``mesh3d.tube_coords`` (default 8x upsample, i.e. the chart
    that 04's ``debend``/``detect_fins`` run on).  ``vertex_s``/``vertex_phi``
    are ``texture_identity.chart_coords(normalize="extent")`` -- the SAME
    normalisation that produced ``assets/chart_skin_x4.png``.
    """
    coords = mesh3d.tube_coords(mesh, centerline, frames)
    vs, vphi = texture_identity.chart_coords(coords, normalize="extent")
    if not detect_fins:
        return coords, vs, vphi, np.zeros(len(vs), dtype=bool)
    with warnings.catch_warnings():
        # Island demotions and the roll estimate are prototype 04 diagnostics
        # already recorded in results/real_v11/rig_run.log; nothing here binds
        # a fin by name, only "is this vertex on a blade".
        warnings.simplefilter("ignore", RuntimeWarning)
        det = mesh3d.detect_fins(mesh, coords, check=False)
    labels = np.asarray(det.labels)
    return coords, vs, vphi, labels != "body"


# ---------------------------------------------------------------------------
# 2. Decimation by vertex clustering
# ---------------------------------------------------------------------------

def _circular_mean(values, group_index, n_groups):
    """Per-group circular mean of angles, in ``[-pi, pi)``."""
    cos = np.bincount(group_index, weights=np.cos(values), minlength=n_groups)
    sin = np.bincount(group_index, weights=np.sin(values), minlength=n_groups)
    out = np.arctan2(sin, cos)
    return (out + math.pi) % (2.0 * math.pi) - math.pi


def _group_mean(values, group_index, n_groups, counts):
    return np.bincount(group_index, weights=values, minlength=n_groups) / counts


def decimate(vertices, faces, s_m, r, phi, is_fin, cell):
    """Cluster vertices on a regular ``cell``-sized grid; returns a dict.

    The cluster of a vertex is ``floor(v / cell)``; every occupied cell becomes
    one output vertex carrying the cell's mean ``s_m`` and ``r``, its CIRCULAR
    mean ``phi``, the mean position, and ``is_fin = any``.  Faces whose three
    vertices no longer land in three distinct cells are dropped (degenerate),
    as are duplicates of an already-emitted triangle (two source triangles can
    collapse onto the same output triple).  The first surviving occurrence
    keeps its original winding, so face normals stay consistent with the
    source mesh.

    Returns ``{"vertices", "faces", "s_m", "r", "phi", "is_fin", "counts",
    "n_dropped_degenerate", "n_dropped_duplicate"}``.
    """
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    cell = float(cell)
    if cell <= 0.0:
        raise ValueError("cell must be positive, got %r" % (cell,))

    keys = np.floor(v / cell).astype(np.int64)
    _, group, counts = np.unique(keys, axis=0, return_inverse=True,
                                 return_counts=True)
    group = np.asarray(group, dtype=np.int64).ravel()
    n_groups = int(counts.size)
    counts = counts.astype(float)

    out_v = np.column_stack([
        _group_mean(v[:, k], group, n_groups, counts) for k in range(3)])
    out_s = _group_mean(np.asarray(s_m, dtype=float), group, n_groups, counts)
    out_r = _group_mean(np.asarray(r, dtype=float), group, n_groups, counts)
    out_phi = _circular_mean(np.asarray(phi, dtype=float), group, n_groups)
    out_fin = np.zeros(n_groups, dtype=bool)
    np.logical_or.at(out_fin, group, np.asarray(is_fin, dtype=bool))

    tri = group[f]
    ok = ((tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2])
          & (tri[:, 0] != tri[:, 2]))
    n_degenerate = int((~ok).sum())
    tri = tri[ok]
    key = np.sort(tri, axis=1)
    _, first = np.unique(key, axis=0, return_index=True)
    first.sort()
    n_duplicate = int(len(tri) - len(first))
    tri = np.ascontiguousarray(tri[first])

    return {
        "vertices": out_v, "faces": tri, "s_m": out_s, "r": out_r,
        "phi": out_phi, "is_fin": out_fin, "counts": counts,
        "n_dropped_degenerate": n_degenerate, "n_dropped_duplicate": n_duplicate,
    }


# ---------------------------------------------------------------------------
# 3. Build / cache
# ---------------------------------------------------------------------------

def _cache_path(cell_mm, cache_dir=DEFAULT_CACHE_DIR):
    return os.path.join(str(cache_dir), "real_body_%gmm.npz" % float(cell_mm))


def build(cell_mm=DEFAULT_CELL_MM, glb=DEFAULT_GLB, report_dir=DEFAULT_REPORT,
          cache_dir=DEFAULT_CACHE_DIR, cache=True, report=False,
          _charted=None):
    """Load, chart, decimate and cache; returns a :class:`RealBody`.

    ``_charted`` is an internal hand-off so that a caller building several cell
    sizes charts the 1 M-vertex mesh once: pass the tuple
    ``(mesh, centerline, frames, coords, vertex_s, vertex_phi, is_fin)``.
    """
    t_all = time.time()
    if _charted is None:
        t0 = time.time()
        mesh = load_rest_mesh(glb)
        t_load = time.time() - t0
        centerline, frames = rest_centerline(report_dir)
        t0 = time.time()
        coords, vs, vphi, is_fin = chart_rest(mesh, centerline, frames)
        t_chart = time.time() - t0
    else:
        mesh, centerline, frames, coords, vs, vphi, is_fin = _charted
        # None, not NaN: this dict is serialised to JSON, and NaN is not JSON.
        t_load = t_chart = None

    lo, hi = float(coords.s.min()), float(coords.s.max())
    t0 = time.time()
    dec = decimate(np.asarray(mesh.vertices, dtype=float),
                   np.asarray(mesh.faces, dtype=np.int64),
                   coords.s, coords.r, coords.phi, is_fin,
                   cell=float(cell_mm) / 1000.0)
    t_dec = time.time() - t0

    # Rest positions from the chart, not the cluster mean -- see the module
    # docstring.  Reconstruct on the SAME (centerline, frames) pair the chart
    # was measured on, with upsample 1 (a straight polyline has no corners, so
    # densifying it changes nothing).
    rest = _points_from_chart(dec["s_m"], dec["r"], dec["phi"],
                              centerline, frames)
    drift = np.linalg.norm(rest - dec["vertices"], axis=1)

    s_norm = (dec["s_m"] - lo) / (hi - lo)
    meta = {
        "glb": str(glb),
        "cell_mm": float(cell_mm),
        "n_vertices_source": int(len(mesh.vertices)),
        "n_faces_source": int(len(mesh.faces)),
        "n_vertices": int(len(rest)),
        "n_faces": int(len(dec["faces"])),
        "vertex_ratio": float(len(rest) / float(len(mesh.vertices))),
        "face_ratio": float(len(dec["faces"]) / float(len(mesh.faces))),
        "n_dropped_degenerate": dec["n_dropped_degenerate"],
        "n_dropped_duplicate": dec["n_dropped_duplicate"],
        "n_fin_vertices": int(dec["is_fin"].sum()),
        "cluster_drift_mm": {
            "mean": float(drift.mean() * 1000.0),
            "p95": float(np.percentile(drift, 95) * 1000.0),
            "max": float(drift.max() * 1000.0),
        },
        "s_raw_range": [lo, hi],
        "total_length_m": float(coords.total_length),
        "seconds": {"load": t_load, "chart": t_chart, "decimate": t_dec,
                    "total": time.time() - t_all},
    }
    body = RealBody(
        vertices=rest, faces=dec["faces"], s=s_norm, phi=dec["phi"],
        r=dec["r"], s_m=dec["s_m"], is_fin=dec["is_fin"],
        centerline=np.asarray(centerline, dtype=float),
        total_length=float(coords.total_length), s_raw_range=(lo, hi),
        meta=meta,
    )
    if cache:
        os.makedirs(str(cache_dir), exist_ok=True)
        path = _cache_path(cell_mm, cache_dir)
        np.savez_compressed(
            path, vertices=body.vertices, faces=body.faces, s=body.s,
            phi=body.phi, r=body.r, s_m=body.s_m, is_fin=body.is_fin,
            centerline=body.centerline,
            total_length=np.array(body.total_length),
            s_raw_range=np.array(body.s_raw_range),
            meta=np.array(json.dumps(meta)))
        meta["cache"] = path
    if report:
        print("decimate %.2f mm: %d verts (%.2f%%), %d faces (%.2f%%), "
              "%d degenerate, %d duplicate, drift mean %.3f mm max %.3f mm, %.1fs"
              % (cell_mm, meta["n_vertices"], 100.0 * meta["vertex_ratio"],
                 meta["n_faces"], 100.0 * meta["face_ratio"],
                 meta["n_dropped_degenerate"], meta["n_dropped_duplicate"],
                 meta["cluster_drift_mm"]["mean"],
                 meta["cluster_drift_mm"]["max"], t_dec))
    return body


def load_cached(cell_mm=DEFAULT_CELL_MM, cache_dir=DEFAULT_CACHE_DIR, **kw):
    """The cached :class:`RealBody` for ``cell_mm``, building it if absent."""
    path = _cache_path(cell_mm, cache_dir)
    if not os.path.exists(path):
        return build(cell_mm=cell_mm, cache_dir=cache_dir, **kw)
    with np.load(path, allow_pickle=False) as z:
        return RealBody(
            vertices=z["vertices"], faces=z["faces"], s=z["s"], phi=z["phi"],
            r=z["r"], s_m=z["s_m"], is_fin=z["is_fin"],
            centerline=z["centerline"], total_length=float(z["total_length"]),
            s_raw_range=tuple(z["s_raw_range"].tolist()),
            meta=json.loads(str(z["meta"])))


# ---------------------------------------------------------------------------
# 4. Pose
# ---------------------------------------------------------------------------

def _tube_coords_on(s_m, r, phi, centerline):
    """A ``mesh3d.TubeCoords`` addressing ``centerline``'s station polyline.

    ``upsample=1``: the station polyline IS the chart here, so ``station`` and
    ``segment`` coincide and ``tube_to_points`` needs no dense re-derivation.
    ``s_m`` outside ``[0, L]`` is fine -- it clamps to the terminal segment and
    extrapolates, which is what carries the snout cap and the caudal overhang.
    """
    cum = mesh3d.arc_length(np.asarray(centerline, dtype=float))
    st = np.clip(np.searchsorted(cum, np.asarray(s_m, dtype=float), side="right") - 1,
                 0, len(cum) - 2).astype(np.int64)
    return mesh3d.TubeCoords(
        s=np.asarray(s_m, dtype=float), r=np.asarray(r, dtype=float),
        phi=np.asarray(phi, dtype=float), station=st,
        total_length=float(cum[-1]), n_stations=len(cum),
        segment=st, upsample=1)


def _points_from_chart(s_m, r, phi, centerline, frames=None):
    coords = _tube_coords_on(s_m, r, phi, centerline)
    return mesh3d.tube_to_points(coords, centerline, frames)


def bent_centerline(length, amp=0.0, wave=0.5, phase=0.0, yaw_deg=0.0,
                    n_stations=POSE_STATIONS):
    """Prototype 05's planar bend as an arc-length-exact station polyline.

    ``kappa(u) = amp * cos(2*pi*wave*u + phase)`` is heading turned per unit
    arc-length FRACTION (``make_dataset.PoseParams``); the heading is its
    mean-removed integral plus ``yaw_deg``.  Segments are built from the
    MIDPOINT heading and are all exactly ``length / (n_stations - 1)`` long, so
    the polyline's chord length is ``length`` to machine precision and ``s``
    measured on the straight rest chart transfers unchanged (the pose is an
    isometry in ``s``, which is what makes ``(s, phi)`` pose-invariant).

    At ``amp = 0`` and ``yaw_deg = 0`` this is the rest axis: snout at
    ``+length/2`` on ``+X``, tangent ``-X``, matching
    ``mesh3d.straight_centerline`` / ``canonical_frames``.
    """
    n = int(n_stations)
    if n < 2:
        raise ValueError("n_stations must be >= 2")
    u = np.linspace(0.0, 1.0, n)
    kappa = float(amp) * np.cos(2.0 * math.pi * float(wave) * u + float(phase))
    theta = np.concatenate([[0.0], np.cumsum(
        0.5 * (kappa[1:] + kappa[:-1]) * np.diff(u))])
    theta = theta - theta.mean() + math.radians(float(yaw_deg))
    # Heading pi at rest: tangent -X, so N=+Z stays dorsal and B=+Y the
    # animal's left, matching mesh3d.canonical_frames.
    heading = theta + math.pi
    mid = 0.5 * (heading[1:] + heading[:-1])
    step = float(length) / (n - 1)
    seg = np.column_stack([np.cos(mid), np.sin(mid), np.zeros(n - 1)]) * step
    pts = np.concatenate([np.zeros((1, 3)), np.cumsum(seg, axis=0)], axis=0)
    pts = pts - pts.mean(axis=0)
    # Put the snout end (s = 0) where the rest axis has it, so a posed body
    # and the rest body share an origin and a scale.
    return pts + np.array([0.0, 0.0, 0.0])


def pose(body, amp=0.0, wave=0.5, phase=0.0, yaw_deg=0.0,
         n_stations=POSE_STATIONS):
    """Bent vertices ``(V, 3)`` for ``body`` under a planar lateral bend.

    The bend is swept with ``mesh3d.tube_to_points`` on
    ``mesh3d.tube_frames`` of :func:`bent_centerline`; for a planar curve
    seeded with ``up=+Z`` the rotation-minimising normal stays ``+Z``, so
    ``phi = 0`` is dorsal at every station and the sweep is a pure lateral
    bend.  ``(s, phi)`` -- and therefore every chart ground-truth map -- are
    unchanged by the pose.

    ``amp = 0, yaw_deg = 0`` reproduces ``body.vertices`` to machine precision
    up to the rest axis, which is the same axis (see :func:`bent_centerline`).
    See the module docstring for the FIN STRETCH caveat.
    """
    cl = bent_centerline(body.total_length, amp=amp, wave=wave, phase=phase,
                         yaw_deg=yaw_deg, n_stations=n_stations)
    frames = mesh3d.tube_frames(cl, up=(0.0, 0.0, 1.0))
    return _points_from_chart(body.s_m, body.r, body.phi, cl, frames)


def fin_stretch(body, amp=0.0):
    """Worst-case axial stretch factor ``1 + |kappa|*r`` over fin vertices.

    ``kappa`` here is curvature in 1/metre: the pose's ``amp`` is radians per
    unit arc-length fraction, so ``kappa_metric = amp / total_length``.
    """
    if not body.is_fin.any():
        return 1.0
    k = abs(float(amp)) / max(float(body.total_length), 1e-12)
    return float(1.0 + k * float(body.r[body.is_fin].max()))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--glb", default=DEFAULT_GLB)
    ap.add_argument("--report-dir", default=DEFAULT_REPORT)
    ap.add_argument("--cells", default="2.5,4,6",
                    help="comma-separated cell sizes in mm")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--json", default=None, help="write the table here")
    args = ap.parse_args(argv)

    t0 = time.time()
    mesh = load_rest_mesh(args.glb)
    t_load = time.time() - t0
    centerline, frames = rest_centerline(args.report_dir)
    t0 = time.time()
    coords, vs, vphi, is_fin = chart_rest(mesh, centerline, frames)
    t_chart = time.time() - t0
    print("source: %d verts, %d faces; load %.1fs, chart %.1fs; "
          "s_raw [%.4f, %.4f] m over chart %.4f m; %d fin vertices"
          % (len(mesh.vertices), len(mesh.faces), t_load, t_chart,
             coords.s.min(), coords.s.max(), coords.total_length,
             int(is_fin.sum())))
    charted = (mesh, centerline, frames, coords, vs, vphi, is_fin)
    rows = []
    for cell in [float(c) for c in args.cells.split(",")]:
        body = build(cell_mm=cell, cache_dir=args.cache_dir,
                     cache=not args.no_cache, report=True, _charted=charted)
        rows.append(body.meta)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=2, sort_keys=True)
        print("wrote", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
