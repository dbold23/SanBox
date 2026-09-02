"""Behavioural tests for the procedural sevengill and the forward bend."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest
import trimesh
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import mesh3d
import synth

from conftest import N_STATIONS

_VALIDATOR_DIR = (
    "/tmp/claude-0/-home-user-SanBox/44444952-aca5-58be-98ef-2fd60cfa4cb2/"
    "scratchpad/node_modules"
)


def test_single_connected_surface(straight):
    """Fins are welded, not floating: one mesh-graph component, as on a scan."""
    e = np.asarray(straight.edges_unique)
    from scipy.sparse import coo_matrix

    n = len(straight.vertices)
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    n_comp, _ = connected_components(g, directed=False)
    assert n_comp == 1
    assert len(np.unique(np.asarray(straight.faces))) == n     # nothing orphaned


def test_the_solid_fins_close_into_one_manifold_solid(straight, bare):
    """Every fin is a closed shell welded into the skin, not a sheet stuck on it.

    ``is_watertight`` is False on the mesh as built, and for one reason only:
    the body carries a duplicated column at the UV seam so the texture wraps
    without a degenerate parameterisation, and those pairs are topologically
    apart though geometrically together.  Merge them by position -- which is
    what a consumer that does not need the UVs would do -- and the mesh is a
    closed, consistently wound, genus-0 solid with no boundary and no
    non-manifold edge.  The count of coincident pairs is the seam and nothing
    else: two fins sit on the seam itself and their slits pull it open, so the
    stations they cover no longer coincide.
    """
    v = np.asarray(straight.vertices, dtype=float)
    n_stations = 112

    pairs = cKDTree(v).query_pairs(1e-9)
    seam_stations = n_stations - sum(
        f["station_range"][1] - f["station_range"][0] + 1
        for name, f in straight.metadata["fins"].items()
        if abs(f["phi_root"]) < 1e-9
    )
    assert len(pairs) == seam_stations
    assert not straight.is_watertight                       # ...and that is why

    solid = trimesh.Trimesh(vertices=v, faces=np.asarray(straight.faces),
                            process=True)
    assert solid.is_watertight
    assert solid.is_winding_consistent
    assert solid.euler_number == 2                          # a sphere, no handles
    assert solid.volume > 0.0

    e = np.sort(np.asarray(solid.faces)[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    assert set(counts.tolist()) == {2}      # no boundary edge, none shared by 3

    hollow = trimesh.Trimesh(vertices=np.asarray(bare.vertices),
                             faces=np.asarray(bare.faces), process=True)
    assert solid.volume > hollow.volume     # the fins enclose real volume


def test_fins_have_a_real_cross_section(straight):
    """Measured, not requested: a NACA-00xx section 10-14% of the local chord
    thick at the root, a nose facet several times the trailing-edge facet, and
    nothing anywhere of zero thickness."""
    report = synth.fin_section_report(straight)
    assert sorted(report) == sorted(set(synth.LABELS) - {"body"})
    floor = 2.0 * synth.FIN_MIN_HALF_THICKNESS * straight.metadata["total_length"]
    for name, r in sorted(report.items()):
        print("\n%-14s chord %.5f  root %.5f (%.1f%% c)  LE %.2f%% c  TE %.2f%% c"
              "  tip %.5f  min %.6f"
              % (name, r["root_chord"], r["root_thickness"],
                 100.0 * r["root_ratio"], 100.0 * r["le_closure"],
                 100.0 * r["te_closure"], r["tip_thickness"], r["min_thickness"]))
        assert 0.10 <= r["root_ratio"] <= 0.14
        # Nowhere zero.  The floor is enforced on the lofted sections; the root
        # lips sit on the curved body, so their straight-line separation runs a
        # hair under the arc the builder opened.
        assert r["min_thickness"] > 0.0
        assert r["min_thickness"] >= 0.9 * floor
        assert r["te_closure"] > 0.0                        # thin, not closed
        assert r["le_closure"] > 3.0 * r["te_closure"]      # round nose, thin tail
        assert 0.0 < r["tip_thickness"] < 0.5 * r["root_thickness"]


def test_fin_vertex_labels_still_mark_exactly_the_blade(straight):
    """Ground truth follows the new vertices: both sides of every section carry
    the fin's name, and the root lips -- which are body-grid vertices sitting at
    body radius -- stay ``body``, as the detector's contract expects."""
    truth = np.asarray(straight.metadata["vertex_labels"])
    for name, fin in straight.metadata["fins"].items():
        pairs = np.asarray(fin["section_pairs"])
        assert fin["volumetric"] is True
        assert set(truth[pairs[0].ravel()]) == {"body"}      # the root lips
        assert set(truth[pairs[1:].ravel()]) == {name}       # the blade
        assert (truth == name).sum() == pairs[1:].size


def test_a_fin_that_cannot_have_its_own_slit_falls_back_to_a_sheet():
    """Two fins cannot open the same strip of skin.  Only a hand-built spec can
    ask for it -- ``test_mesh3d`` stacks a second caudal lobe on the first to
    test island merging -- and the later fin is then built as the old
    zero-thickness sheet and says so in its metadata, rather than cutting a
    second slit through the first fin's lips and tearing the surface."""
    saved = synth.FIN_SPECS
    try:
        synth.FIN_SPECS = dict(
            saved, caudal_upper_b=(0.94, 1.00, 0.0, 0.100, 0.240, 0.55, 10)
        )
        mesh = synth.make_sevengill()
    finally:
        synth.FIN_SPECS = saved

    fins = mesh.metadata["fins"]
    assert fins["caudal_upper_b"]["volumetric"] is False
    assert fins["caudal_upper_b"]["section_pairs"] is None
    assert all(f["volumetric"] for k, f in fins.items() if k != "caudal_upper_b")
    assert "caudal_upper_b" not in synth.fin_section_report(mesh)
    assert len(np.unique(np.asarray(mesh.faces))) == len(mesh.vertices)


def test_solid_fins_false_rebuilds_the_old_sheets(straight):
    """The A/B control the README's before/after numbers are measured against."""
    sheets = synth.make_sevengill(solid_fins=False)
    assert not any(f["volumetric"] for f in sheets.metadata["fins"].values())
    assert synth.fin_section_report(sheets) == {}
    assert sorted(sheets.metadata["fins"]) == sorted(straight.metadata["fins"])
    assert len(sheets.vertices) < len(straight.vertices)


def test_canonical_orientation(straight):
    """Snout +X, tail -X, dorsal +Z, animal's left +Y."""
    v = np.asarray(straight.vertices)
    labels = straight.metadata["vertex_labels"]
    assert v[np.argmax(v[:, 0])][0] > 0.49          # snout tip near +L/2
    assert v[labels == "caudal_upper"][:, 0].min() < -0.45
    assert v[labels == "dorsal"][:, 2].mean() > 0.05
    assert v[labels == "anal"][:, 2].mean() < -0.03
    assert v[labels == "pectoral_L"][:, 1].mean() > 0.05
    assert v[labels == "pectoral_R"][:, 1].mean() < -0.05
    # Heterocercal: the upper lobe runs much further posterior than the lower
    # one and reaches higher -- length, not just height, is the diagnostic.
    up = v[labels == "caudal_upper"]
    lo = v[labels == "caudal_lower"]
    assert up[:, 0].min() < lo[:, 0].min() - 0.10
    assert up[:, 2].max() > abs(lo[:, 2].min())


def test_anatomy_binding(straight):
    """Seven gill slits, one dorsal, dorsal over the pelvics, no second dorsal."""
    meta = straight.metadata
    assert len(meta["gill_u"]) == 7
    fins = meta["fins"]
    assert sorted(fins) == sorted(set(synth.LABELS) - {"body"})
    assert "dorsal_2" not in fins and sum(k.startswith("dorsal") for k in fins) == 1
    # Single dorsal sits over/behind the pelvics and behind the pectorals.
    assert fins["pectoral_L"]["u1"] < fins["pelvic_L"]["u0"]
    assert fins["dorsal"]["u0"] < fins["pelvic_L"]["u1"]      # overlapping
    assert fins["dorsal"]["u0"] > fins["pelvic_L"]["u0"] - 0.1
    assert fins["anal"]["u0"] > fins["pelvic_L"]["u0"]
    assert fins["caudal_upper"]["u0"] > fins["anal"]["u1"]


def test_deterministic(straight):
    other = synth.make_sevengill()
    assert np.array_equal(np.asarray(straight.vertices), np.asarray(other.vertices))
    assert np.array_equal(straight.visual.uv, other.visual.uv)
    assert np.array_equal(straight.faces, other.faces)


def test_without_fins(bare, straight):
    assert bare.metadata["fins"] == {}
    assert set(bare.metadata["vertex_labels"]) == {"body"}
    assert len(bare.vertices) < len(straight.vertices)


def test_curve_arc_length_and_turn():
    for maker, turn in ((synth.c_curve, 120.0), (synth.s_curve, 90.0)):
        c = maker(0.8, turn, 200)
        assert mesh3d.arc_length(c)[-1] == pytest.approx(0.8, rel=1e-3)
    c = synth.c_curve(0.8, 120.0, 400)
    t0, t1 = c[1] - c[0], c[-1] - c[-2]
    t0 /= np.linalg.norm(t0)
    t1 /= np.linalg.norm(t1)
    assert np.degrees(np.arccos(np.clip(t0 @ t1, -1, 1))) == pytest.approx(120.0, abs=1.0)
    # The S-curve reverses handedness at the join: net turn ~ 0.
    s = synth.s_curve(0.8, 90.0, 400)
    t0, t1 = s[1] - s[0], s[-1] - s[-2]
    t0 /= np.linalg.norm(t0)
    t1 /= np.linalg.norm(t1)
    assert np.degrees(np.arccos(np.clip(t0 @ t1, -1, 1))) < 15.0


def test_bend_preserves_topology_and_girth(straight, bent):
    mesh, info = bent
    assert np.array_equal(mesh.faces, straight.faces)
    assert np.array_equal(mesh.visual.uv, straight.visual.uv)
    assert len(mesh.vertices) == len(straight.vertices)
    assert not np.allclose(mesh.vertices, straight.vertices)
    # A tube-coordinate bend is an isometry in s and preserves r exactly -- for
    # every vertex whose foot lands on the same dense segment before and after.
    # It re-lands on a neighbouring dense segment for a few percent of vertices,
    # all of them at large r near a corner of the dense polyline, where the
    # nearest-segment rule is genuinely ambiguous; there the recovered s shifts
    # by up to r * (turn per dense segment) = 0.20 * 120 deg / (63 * 8) ~ 8e-4
    # BL.  That bound, not exactness, is what the chart guarantees on a bent
    # centerline -- eight times tighter than the station polyline's.
    # Fins are not charted: each island rides its insertion frame as a rigid
    # plate (``mesh3d.map_mesh``), so the chart contract is checked on the body
    # and rigidity on the blades.
    a = mesh3d.tube_coords(straight, info["source_centerline"], info["source_frames"])
    b = mesh3d.tube_coords(mesh, info["centerline"], info["frames"])
    w = np.zeros(len(a.s))
    for members, weights, *_ in mesh.metadata["rigid_islands"]:
        w[members] = weights
    body = w == 0
    same = (a.segment == b.segment) & body
    assert same.sum() > 0.95 * body.sum()
    assert np.allclose(a.r[same], b.r[same], atol=1e-9)
    assert np.allclose(a.s[same], b.s[same], atol=1e-9)
    assert np.abs(a.r[body] - b.r[body]).max() < 1e-3
    assert np.abs(a.s[body] - b.s[body]).max() < 1e-3
    sv, bv = np.asarray(straight.vertices), np.asarray(mesh.vertices)
    for members, weights, *_ in mesh.metadata["rigid_islands"]:
        blade = members[weights >= 1.0]
        if len(blade) < 2:
            continue
        ref = blade[0]
        d0 = np.linalg.norm(sv[blade] - sv[ref], axis=1)
        d1 = np.linalg.norm(bv[blade] - bv[ref], axis=1)
        assert np.allclose(d0, d1, atol=1e-9)          # the blade is a rigid body


def test_bend_is_invertible_on_ground_truth(straight, bent):
    """With the *true* centerline the chart round trip is limited only by the
    nearest-segment ambiguity at corners, which scales as r * (turn per
    segment); at 64 stations over a 120 deg arc that is ~6e-3 BL worst case on a
    fin tip and ~5e-4 BL rms."""
    mesh, info = bent
    back, _ = mesh3d.map_mesh(
        mesh, info["centerline"], info["frames"],
        info["source_centerline"], info["source_frames"],
        records=mesh.metadata["rigid_islands"],
    )
    err = np.linalg.norm(np.asarray(back.vertices) - np.asarray(straight.vertices), axis=1)
    assert np.sqrt((err ** 2).mean()) < 1e-3
    assert err.max() < 1e-2
    # body vertices and fully rigid blade vertices invert exactly; only the
    # root blend band (a position mix of the two transports) is approximate
    w = np.zeros(len(err))
    for members, weights, *_ in mesh.metadata["rigid_islands"]:
        w[members] = weights
    exact = (w == 0) | (w >= 1)
    assert np.median(err[exact]) < 1e-12
    assert err[exact].max() < 1e-3        # nearest-segment re-landing at corners, r * turn per dense segment


def test_export_glb_round_trip(straight):
    tmp = tempfile.mkdtemp()
    try:
        path = synth.export_glb(straight, os.path.join(tmp, "s.glb"))
        back = mesh3d.load_mesh(path, report=False)
        assert back.visual.uv is not None
        assert len(back.faces) == len(straight.faces)
        assert np.allclose(back.extents, straight.extents, atol=1e-5)
    finally:
        shutil.rmtree(tmp)


@pytest.mark.skipif(
    shutil.which("node") is None or not os.path.isdir(_VALIDATOR_DIR),
    reason="node / gltf-validator unavailable",
)
def test_exported_glb_passes_gltf_validator(straight, bent):
    tmp = tempfile.mkdtemp()
    try:
        paths = [
            synth.export_glb(straight, os.path.join(tmp, "straight.glb")),
            synth.export_glb(bent[0], os.path.join(tmp, "bent.glb")),
        ]
        script = (
            "const fs=require('fs');"
            "const V=require('%s/gltf-validator');"
            "(async()=>{const o=[];for(const f of process.argv.slice(1)){"
            "const r=await V.validateBytes(new Uint8Array(fs.readFileSync(f)));"
            "o.push({f,e:r.issues.numErrors,w:r.issues.numWarnings});}"
            "console.log(JSON.stringify(o));})()" % _VALIDATOR_DIR
        )
        out = subprocess.run(
            ["node", "-e", script] + paths, capture_output=True, text=True, check=True
        )
        for rec in json.loads(out.stdout):
            assert rec["e"] == 0, rec
            assert rec["w"] == 0, rec
    finally:
        shutil.rmtree(tmp)
