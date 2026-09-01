"""Measure the linear-blend-skinning artifact at the peak of the ``escape`` clip.

README section 7 claims LBS "collapses volume where two bones meet at a large angle"
and that "at ``escape`` peak curvature the peduncle pinches".  That paragraph is a
DISCLOSURE, so it has to carry numbers and it has to name the right place.  This
script produces both, on the same synthetic sevengill ``demo.py`` uses and through the
same ``rig_sevengill.run_pipeline`` code path, so the figures in the README are
reproducible with one command::

    python scripts/measure_lbs_artifact.py              # the shipped, capped C-start
    python scripts/measure_lbs_artifact.py --uncapped   # the old 6 /BL caricature

What it reports, all at the single frame of maximum total spine turn:

* **worst trunk edge ratio** -- ``min(posed length / rest length)`` over every mesh
  edge whose BOTH endpoints are labelled ``body``.  Fin edges are excluded: a fin is
  two bones and a blade, its stretching is a different artifact.
* **count of trunk edges shrinking by more than 40%** -- how widespread the collapse
  is, not just how deep.
* **worst trunk face area ratio** -- the 2D version of the same thing; a candy-wrapper
  pinch shows up in area before it shows up in any single edge.
* **where the worst 100 edges are** -- each edge midpoint is projected onto the rest
  spine polyline and attributed to the spine SEGMENT it lands on, so the output names
  the joints between which the collapse happens instead of asserting "the peduncle".

Every number is a property of LBS plus this weight map, not of the mesh: dual-quaternion
skinning would remove it and glTF cannot express it (README section 7).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import mesh3d  # noqa: E402
import motion  # noqa: E402
import rig  # noqa: E402
import rig_sevengill  # noqa: E402
import synth  # noqa: E402

SEED = 0
TURN_DEG = 120.0
N_STATIONS = 64
SHRINK_THRESHOLD = 0.40
N_WORST = 100


def unique_edges(faces):
    """(E, 2) sorted unique undirected edges of a triangle mesh."""
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    return np.unique(np.sort(e, axis=1), axis=0)


def triangle_area(v, faces):
    a = v[faces[:, 1]] - v[faces[:, 0]]
    b = v[faces[:, 2]] - v[faces[:, 0]]
    return 0.5 * np.linalg.norm(np.cross(a, b), axis=1)


def _head_vertices(verts, skeleton, fraction=0.12):
    """Body vertices forward of ``fraction`` of the rest spine's arc length."""
    spine_idx = np.asarray(skeleton.spine_indices, dtype=int)
    spine_pos = np.asarray(skeleton.joints, dtype=float)[spine_idx]
    s = rig._arc_length(spine_pos)
    k, t = rig._project_on_polyline(np.asarray(verts, dtype=float), spine_pos)
    s_v = s[k] + t * (s[k + 1] - s[k])
    return s_v <= float(fraction) * s[-1]


def spine_segment_names(skeleton, points):
    """Name of the rest-spine segment each point projects onto."""
    spine_idx = np.asarray(skeleton.spine_indices, dtype=int)
    spine_pos = np.asarray(skeleton.joints, dtype=float)[spine_idx]
    k, _ = rig._project_on_polyline(np.asarray(points, dtype=float), spine_pos)
    return np.asarray([
        "%s -> %s" % (skeleton.names[int(spine_idx[i])],
                      skeleton.names[int(spine_idx[i + 1])])
        for i in k
    ])


def build(seed=SEED, n_stations=N_STATIONS, verbose=False):
    """The demo's mesh, de-bent, rigged -- ``rig_sevengill.run_pipeline``, no GLB."""
    straight_truth = synth.make_sevengill(seed=seed)
    total_length = float(
        mesh3d.arc_length(np.asarray(straight_truth.metadata["centerline"], dtype=float))[-1]
    )
    bent, _ = synth.bend(
        straight_truth, synth.c_curve(total_length, TURN_DEG, n_stations)
    )
    return rig_sevengill.run_pipeline(
        bent, out=None, motions=("escape",), n_stations=n_stations,
        seed=seed, validate=False, verbose=verbose,
    )


def measure(result, escape=None):
    """The whole report as a dict of plain numbers and strings."""
    mesh = result.straight_mesh
    verts = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    labels = np.asarray(result.detection.labels).astype(str)
    skeleton = result.skeleton
    weights = np.asarray(result.weights, dtype=float)
    body_length = float(result.centerline_info["length"])

    mskel = motion.MotionSkeleton.from_skeleton(skeleton, body_length=body_length)
    clip = motion.make_clip(mskel, "escape", escape=escape)
    escape = clip.meta["escape"]

    # the frame of maximum total spine turn
    yaw = np.asarray(clip.meta["spine_yaw"], dtype=float)
    total_turn = np.degrees(np.abs(yaw.sum(axis=1)))
    peak = int(np.argmax(total_turn))
    posed = rig.lbs(verts, weights, skeleton, rig.quat_to_rotmat(clip.quats[peak]))

    is_body = labels == "body"
    edges = unique_edges(faces)
    trunk = edges[is_body[edges[:, 0]] & is_body[edges[:, 1]]]
    rest_len = np.linalg.norm(verts[trunk[:, 0]] - verts[trunk[:, 1]], axis=1)
    posed_len = np.linalg.norm(posed[trunk[:, 0]] - posed[trunk[:, 1]], axis=1)
    ratio = posed_len / np.maximum(rest_len, 1e-15)

    body_faces = faces[is_body[faces].all(axis=1)]
    rest_area = triangle_area(verts, body_faces)
    posed_area = triangle_area(posed, body_faces)
    keep = rest_area > 1e-15
    area_ratio = posed_area[keep] / rest_area[keep]

    # self-contact: how close does the posed caudal lobe come to the posed head?
    # (README section 7: "a C-start bends the body far enough that the tail can
    # intersect the head.  Nothing detects or resolves that.")
    caudal = np.zeros(len(verts), dtype=bool)
    for name in result.detection.fins:
        if name.startswith("caudal"):
            caudal |= labels == name
    head = _head_vertices(verts, skeleton)
    if caudal.any() and head.any():
        d = np.linalg.norm(
            posed[caudal][:, None, :] - posed[head][None, :, :], axis=2
        )
        caudal_head_bl = float(d.min()) / body_length
    else:
        caudal_head_bl = float("nan")

    order = np.argsort(ratio)[:N_WORST]
    mids = 0.5 * (verts[trunk[order, 0]] + verts[trunk[order, 1]])
    seg = spine_segment_names(skeleton, mids)
    names, counts = np.unique(seg, return_counts=True)
    cluster = sorted(zip(names.tolist(), counts.tolist()), key=lambda kv: -kv[1])

    worst = int(order[0])
    worst_seg = spine_segment_names(
        skeleton, 0.5 * (verts[trunk[worst, 0]] + verts[trunk[worst, 1]])[None, :]
    )[0]

    return {
        "n_vertices": int(len(verts)),
        "n_faces": int(len(faces)),
        "body_length": body_length,
        "peak_frame": peak,
        "peak_time_s": float(clip.times[peak]),
        "escape_peak_curvature_per_bl": float(escape.peak_curvature_per_bl),
        "total_spine_turn_deg": float(total_turn[peak]),
        "closure_bl": motion.escape_closure_bl(escape, s_j=mskel.spine_fractions),
        "caudal_to_head_bl": caudal_head_bl,
        "n_trunk_edges": int(len(trunk)),
        "worst_edge_ratio": float(ratio[worst]),
        "worst_edge_segment": worst_seg,
        "n_edges_shrinking_over_40pct": int((ratio < 1.0 - SHRINK_THRESHOLD).sum()),
        "n_body_faces": int(len(body_faces)),
        "worst_area_ratio": float(area_ratio.min()),
        "worst_100_clusters": cluster,
    }


def report(rows):
    out = []
    add = out.append
    for title, r in rows:
        add("=" * 74)
        add("LBS artifact at the escape peak -- %s" % title)
        add("=" * 74)
        add("mesh          %d verts, %d faces (the demo's synthetic sevengill, de-bent)"
            % (r["n_vertices"], r["n_faces"]))
        add("BL            %.4f world units (chart length)" % r["body_length"])
        add("peak frame    %d at t = %.3f s" % (r["peak_frame"], r["peak_time_s"]))
        add("escape        peak curvature %.3f /BL, total spine turn %.1f deg"
            % (r["escape_peak_curvature_per_bl"], r["total_spine_turn_deg"]))
        add("self-contact  last spine joint to snout %.3f BL; posed caudal lobe to "
            "posed head %.3f BL" % (r["closure_bl"], r["caudal_to_head_bl"]))
        add("")
        add("trunk edges (both endpoints labelled 'body'): %d" % r["n_trunk_edges"])
        add("  worst edge ratio (posed/rest)      %.4f   at %s"
            % (r["worst_edge_ratio"], r["worst_edge_segment"]))
        add("  edges shrinking by more than 40%%   %d  (%.3f%% of trunk edges)"
            % (r["n_edges_shrinking_over_40pct"],
               100.0 * r["n_edges_shrinking_over_40pct"] / max(r["n_trunk_edges"], 1)))
        add("  worst face area ratio              %.4f   (%d body faces)"
            % (r["worst_area_ratio"], r["n_body_faces"]))
        add("  the worst %d edges sit in:" % N_WORST)
        for name, count in r["worst_100_clusters"]:
            add("      %-52s %3d" % (name, count))
        add("")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--uncapped", action="store_true",
                    help="also measure the pre-cap %.1f /BL caricature"
                         % motion.ESCAPE_PEAK_CURVATURE_UNCAPPED_PER_BL)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("-n", "--n-stations", type=int, default=N_STATIONS)
    args = ap.parse_args(argv)

    result = build(seed=args.seed, n_stations=args.n_stations)
    rows = [("shipped default (capped to %.0f deg of total turn)"
             % motion.ESCAPE_MAX_TOTAL_TURN_DEG, measure(result))]
    if args.uncapped:
        rows.append((
            "UNCAPPED %.1f /BL, for contrast"
            % motion.ESCAPE_PEAK_CURVATURE_UNCAPPED_PER_BL,
            measure(result, escape=motion.EscapeParams(
                peak_curvature_per_bl=motion.ESCAPE_PEAK_CURVATURE_UNCAPPED_PER_BL)),
        ))
    text = report(rows)
    print(text)
    return text


if __name__ == "__main__":  # pragma: no cover
    main()
