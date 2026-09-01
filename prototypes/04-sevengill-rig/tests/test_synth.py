"""Behavioural tests for the procedural sevengill and the forward bend."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components

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
    # every vertex whose foot lands on the same segment before and after.  It
    # re-lands on a different segment for ~4% of vertices, all of them at large
    # r near a corner of the bent polyline, where the nearest-segment rule is
    # genuinely ambiguous; there the recovered s shifts by up to r * (turn per
    # segment) = 0.20 * 120 deg / 63 ~ 6e-3 BL.  That bound, not exactness, is
    # what the chart guarantees on a bent centerline.
    a = mesh3d.tube_coords(straight, info["source_centerline"], info["source_frames"])
    b = mesh3d.tube_coords(mesh, info["centerline"], info["frames"])
    same = a.station == b.station
    assert same.mean() > 0.95
    assert np.allclose(a.r[same], b.r[same], atol=1e-9)
    assert np.allclose(a.s[same], b.s[same], atol=1e-9)
    assert np.abs(a.r - b.r).max() < 1e-3
    assert np.abs(a.s - b.s).max() < 7e-3


def test_bend_is_invertible_on_ground_truth(straight, bent):
    """With the *true* centerline the chart round trip is limited only by the
    nearest-segment ambiguity at corners, which scales as r * (turn per
    segment); at 64 stations over a 120 deg arc that is ~6e-3 BL worst case on a
    fin tip and ~5e-4 BL rms."""
    mesh, info = bent
    back = mesh3d.map_points(
        np.asarray(mesh.vertices), info["centerline"], info["frames"],
        info["source_centerline"], info["source_frames"],
    )
    err = np.linalg.norm(back - np.asarray(straight.vertices), axis=1)
    assert np.sqrt((err ** 2).mean()) < 1e-3
    assert err.max() < 1e-2


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
