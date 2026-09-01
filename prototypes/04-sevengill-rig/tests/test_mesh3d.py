"""Behavioural tests for the 3D tube chart.

Tolerances are stated in multiples of the voxel pitch, because that is the one
scale that actually limits the pipeline: everything downstream of
``extract_centerline_3d`` is exact arithmetic, and every measured residual below
tracks the pitch (measured on the C-120 pose: rms error 0.68, 0.57, 0.68 voxels
at pitch 0.0125, 0.0063, 0.0040 BL respectively).
"""

from __future__ import annotations

import collections
import os
import shutil
import tempfile
import warnings

import numpy as np
import pytest
import trimesh

import mesh3d
import synth

from conftest import N_STATIONS, dist_to_curve


# ---------------------------------------------------------------------------
# Chart algebra
# ---------------------------------------------------------------------------

def test_canonical_frames_match_rmf_of_the_straight_axis(straight):
    cl = mesh3d.straight_centerline(straight.metadata["centerline"])
    t, n, b = mesh3d.tube_frames(cl)
    ct, cn, cb = mesh3d.canonical_frames(len(cl))
    assert np.allclose(t, ct) and np.allclose(n, cn) and np.allclose(b, cb)
    # Right-handed, snout +X so the head->tail tangent is -X, dorsal +Z.
    assert np.allclose(np.cross(t, n), b)
    assert cl[0, 0] > cl[-1, 0]


def test_tube_coords_inverse_is_exact(straight, bent):
    """(s, r, phi, station) -> xyz is an exact inverse on the same centerline."""
    for mesh, cl in ((straight, straight.metadata["centerline"]),
                     (bent[0], bent[1]["centerline"])):
        frames = mesh3d.tube_frames(cl)
        c = mesh3d.tube_coords(mesh, cl, frames)
        back = mesh3d.tube_to_points(c, cl, frames)
        assert np.abs(back - np.asarray(mesh.vertices)).max() < 1e-12


def test_tube_coords_are_anatomically_meaningful(straight):
    """On the straight mesh, phi = 0 is the dorsal ridge and s grows toward -X."""
    cl = straight.metadata["centerline"]
    c = mesh3d.tube_coords(straight, cl)
    v = np.asarray(straight.vertices)
    labels = straight.metadata["vertex_labels"]

    order = np.argsort(c.s[labels == "body"])
    xs = v[labels == "body"][order][:, 0]
    assert np.corrcoef(np.arange(len(xs)), xs)[0, 1] < -0.99   # s runs head->tail

    dorsal = labels == "dorsal"
    assert np.abs(c.phi[dorsal]).max() < np.deg2rad(20)
    assert np.abs(np.abs(c.phi[labels == "anal"]) - np.pi).max() < np.deg2rad(20)
    assert c.phi[labels == "pectoral_L"].mean() > 0      # +Y = animal's left
    assert c.phi[labels == "pectoral_R"].mean() < 0
    # Body radius recovered as r; the peduncle is thinner than the girth maximum.
    body_r = c.r[labels == "body"]
    assert body_r.max() == pytest.approx(
        straight.metadata["max_radius"] * straight.metadata["ellipticity"], rel=0.05
    )


def test_map_points_rejects_mismatched_station_counts(straight):
    cl = straight.metadata["centerline"]
    with pytest.raises(ValueError, match="same station count"):
        mesh3d.map_points(
            np.asarray(straight.vertices), cl, mesh3d.tube_frames(cl),
            mesh3d.resample_polyline(cl, len(cl) + 1),
            mesh3d.tube_frames(mesh3d.resample_polyline(cl, len(cl) + 1)),
        )


def test_overhang_beyond_the_chart_is_carried_not_clipped(straight):
    """A caudal lobe sits past the end of the chart; s must exceed the chart
    length rather than being clamped onto the last station, or the heterocercal
    tail would collapse into a disc during a de-bend."""
    cl = straight.metadata["centerline"]
    c = mesh3d.tube_coords(straight, cl)
    labels = straight.metadata["vertex_labels"]
    assert c.s[labels == "caudal_upper"].max() > c.total_length * 1.2
    assert c.s.min() < 0.0                                    # snout tip


# ---------------------------------------------------------------------------
# Centerline extraction
# ---------------------------------------------------------------------------

def test_centerline_matches_ground_truth(bent, extracted):
    _, info_gt = bent
    cl, info = extracted
    d = dist_to_curve(cl, info_gt["centerline"])
    # Sub-voxel: measured max 0.20 pitch, mean 0.07 pitch.
    assert d.max() < 0.5 * info["pitch"]
    assert d.mean() < 0.2 * info["pitch"]


def test_centerline_is_head_first(bent, extracted):
    _, info_gt = bent
    cl, info = extracted
    gt = info_gt["centerline"]
    assert info["head_width"] > info["tail_width"]
    assert np.linalg.norm(cl[0] - gt[0]) < np.linalg.norm(cl[0] - gt[-1])
    assert np.linalg.norm(cl[-1] - gt[-1]) < np.linalg.norm(cl[-1] - gt[0])


def test_centerline_is_arc_length_uniform(extracted):
    cl, _ = extracted
    seg = np.linalg.norm(np.diff(cl, axis=0), axis=1)
    assert seg.std() / seg.mean() < 1e-4
    assert len(cl) == N_STATIONS


def test_fins_do_not_divert_the_centerline(straight, bare):
    """The whole point of the thick-core threshold.

    Control: the bare body tube.  With the threshold on, adding eight fins moves
    the centerline by well under a voxel.  With it off, the medial path escapes
    down the caudal lobe and the centerline is wrong by a quarter of a body
    length -- which is the failure this design exists to prevent.
    """
    with_fins, info = mesh3d.extract_centerline_3d(straight, n_stations=N_STATIONS)
    control, _ = mesh3d.extract_centerline_3d(bare, n_stations=N_STATIONS)
    assert dist_to_curve(with_fins, control).max() < 1.0 * info["pitch"]

    diverted, _ = mesh3d.extract_centerline_3d(
        straight, n_stations=N_STATIONS, core_radius_frac=0.0, core_pitch_mult=0.0
    )
    assert dist_to_curve(diverted, control).max() > 20.0 * info["pitch"]


def test_non_tubular_mesh_warns():
    blob = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    with pytest.warns(RuntimeWarning, match="does not look tubular"):
        mesh3d.extract_centerline_3d(blob, n_stations=32)


# ---------------------------------------------------------------------------
# De-bend / re-bend
# ---------------------------------------------------------------------------

def _round_trip(straight, bent_mesh, n_stations=N_STATIONS):
    cl, info = mesh3d.extract_centerline_3d(bent_mesh, n_stations=n_stations)
    out, target = mesh3d.debend(bent_mesh, cl)
    delta = np.asarray(out.vertices) - np.asarray(straight.vertices)
    # The chart's origin is arbitrary (extraction trims a few percent off each
    # end), so a rigid translation along the axis is not an error; remove it.
    delta -= delta.mean(axis=0)
    return out, target, info, np.linalg.norm(delta, axis=1)


def test_debend_recovers_the_straight_rest_pose(straight, bent):
    out, _, info, err = _round_trip(straight, bent[0])
    labels = straight.metadata["vertex_labels"]
    body = err[labels == "body"]
    pitch = info["pitch"]
    rms = np.sqrt((err ** 2).mean())
    print(
        "\nC-120 de-bend: pitch %.5f BL | all rms %.5f (%.2f px) max %.5f (%.1f px)"
        " | body rms %.5f (%.2f px) max %.5f (%.2f px)"
        % (pitch, rms, rms / pitch, err.max(), err.max() / pitch,
           np.sqrt((body ** 2).mean()), np.sqrt((body ** 2).mean()) / pitch,
           body.max(), body.max() / pitch)
    )
    # Measured: body rms 0.21 px, body max 0.89 px, all rms 0.50 px, max 4.5 px.
    assert np.sqrt((body ** 2).mean()) < 0.5 * pitch
    assert body.max() < 2.0 * pitch
    assert rms < 1.5 * pitch
    # The worst vertex is always the caudal upper-lobe tip: it overhangs the
    # chart by ~0.25 BL and is carried rigidly by the terminal frame, so the
    # residual terminal tangent error is amplified by that lever arm.
    assert err.max() < 12.0 * pitch
    assert labels[int(np.argmax(err))].startswith("caudal")


def test_debend_recovers_an_s_pose(straight, bent_s):
    _, _, info, err = _round_trip(straight, bent_s[0])
    labels = straight.metadata["vertex_labels"]
    body = err[labels == "body"]
    assert np.sqrt((body ** 2).mean()) < 0.5 * info["pitch"]
    assert np.sqrt((err ** 2).mean()) < 1.5 * info["pitch"]


def test_debend_preserves_topology_uv_and_material(straight, bent, extracted):
    mesh, _ = bent
    cl, _ = extracted
    out, target = mesh3d.debend(mesh, cl)
    assert np.array_equal(out.faces, mesh.faces)
    assert np.array_equal(out.visual.uv, mesh.visual.uv)
    assert getattr(out.visual.material, "image", None) is not None
    assert len(out.vertices) == len(mesh.vertices)
    assert not np.shares_memory(out.vertices, mesh.vertices)
    # Straightened onto the +X axis, snout first.
    assert np.allclose(target[:, 1:], 0.0)
    assert target[0, 0] > target[-1, 0]
    v = np.asarray(out.vertices)
    assert v[np.argmax(v[:, 0])][0] > 0.35


def test_rebend_inverts_debend(bent, extracted):
    mesh, _ = bent
    cl, info = extracted
    out, target = mesh3d.debend(mesh, cl)
    back = mesh3d.rebend(out, cl)
    err = np.linalg.norm(np.asarray(back.vertices) - np.asarray(mesh.vertices), axis=1)
    # Exact for the 96% of vertices whose foot lands on the same segment both
    # ways; the rest carry the same nearest-segment corner ambiguity as the
    # forward bend, bounded by r * (turn per segment).
    assert np.median(err) < 1e-12
    assert np.percentile(err, 95) < 0.05 * info["pitch"]
    assert err.max() < 1.5 * info["pitch"]


def test_debend_survives_a_glb_round_trip(bent, extracted):
    mesh, _ = bent
    cl, _ = extracted
    tmp = tempfile.mkdtemp()
    try:
        path = synth.export_glb(mesh, os.path.join(tmp, "bent.glb"))
        loaded = mesh3d.load_mesh(path, report=False)
        cl2, _ = mesh3d.extract_centerline_3d(loaded, n_stations=N_STATIONS)
        out, _ = mesh3d.debend(loaded, cl2)
        assert out.visual.uv is not None
        assert np.array_equal(out.visual.uv, loaded.visual.uv)
        assert np.array_equal(out.faces, loaded.faces)
        assert dist_to_curve(cl2, cl).max() < 0.02
    finally:
        shutil.rmtree(tmp)


# ---------------------------------------------------------------------------
# Fin detection
# ---------------------------------------------------------------------------

def test_fin_islands_match_construction(chart, straight):
    mesh, cl, frames, coords, det = chart
    truth = straight.metadata["vertex_labels"]
    assert sorted(det.fins) == sorted(set(synth.LABELS) - {"body"})
    for name, fin in det.fins.items():
        majority, count = collections.Counter(
            truth[fin["vertex_indices"]]
        ).most_common(1)[0]
        assert majority == name
        assert count == fin["n_vertices"]          # islands are pure, not merely
        assert fin["n_vertices"] >= 40             # majority-correct


def test_fin_detection_has_no_false_positives(chart, straight):
    """A vertex the generator calls body must never be given a fin label."""
    _, _, _, _, det = chart
    truth = straight.metadata["vertex_labels"]
    assert set(det.labels[truth == "body"]) == {"body"}
    # Roots sit at body radius and stay 'body' by design; the blades are found.
    recall = np.mean(det.labels[truth != "body"] != "body")
    assert recall > 0.75


def test_fin_insertions_are_anatomically_ordered(chart):
    _, _, _, coords, det = chart
    s_mid = {k: 0.5 * sum(v["s_range"]) for k, v in det.fins.items()}
    assert s_mid["pectoral_L"] < s_mid["pelvic_L"] < s_mid["caudal_upper"]
    assert s_mid["pectoral_R"] < s_mid["pelvic_R"]
    assert s_mid["pelvic_L"] < s_mid["anal"] < s_mid["caudal_lower"]
    # Single dorsal, far posterior, over the pelvics.
    assert abs(s_mid["dorsal"] - s_mid["pelvic_L"]) < 0.12 * coords.total_length
    assert s_mid["dorsal"] > 0.5 * coords.total_length
    for name in ("pectoral_L", "pelvic_L", "dorsal", "caudal_upper"):
        lo, hi = det.fins[name]["station_range"]
        assert 0 <= lo <= hi < N_STATIONS
        assert det.fins[name]["insertion_centroid"].shape == (3,)


def test_fin_labels_survive_the_debend(chart, straight, extracted):
    """Labels are a property of the chart, not of the pose: detecting on the
    bent mesh and on the de-bent mesh must agree vertex for vertex."""
    mesh, cl, _, _, det_bent = chart
    out, target = mesh3d.debend(mesh, cl)
    coords = mesh3d.tube_coords(out, target)
    det_straight = mesh3d.detect_fins(out, coords)
    assert np.mean(det_bent.labels == det_straight.labels) > 0.99


# ---------------------------------------------------------------------------
# Fin detection, directly (fixtures above go through the bent C-pose; these
# call ``detect_fins`` on the rest pose, where the construction truth is exact)
# ---------------------------------------------------------------------------

def _mesh_with_specs(specs, **kw):
    """Build a sevengill from a replaced ``synth.FIN_SPECS``.

    ``make_sevengill`` reads the module global at call time, so swapping it and
    putting it back is the whole of the fixture.
    """
    saved = synth.FIN_SPECS
    try:
        synth.FIN_SPECS = specs
        return synth.make_sevengill(**kw)
    finally:
        synth.FIN_SPECS = saved


def _mesh_with_extra_fins(extra, **kw):
    """The stock sevengill plus extra welded sheets, keyed by label."""
    specs = dict(synth.FIN_SPECS)
    specs.update(extra)
    return _mesh_with_specs(specs, **kw)


def _chart_of(mesh, n_stations=N_STATIONS):
    """(centerline, coords) from the mesh's own extracted centerline."""
    cl, _ = mesh3d.extract_centerline_3d(mesh, n_stations=n_stations)
    return cl, mesh3d.tube_coords(mesh, cl)


def _quietly(fn, *args, **kw):
    """Call ``fn`` with warnings silenced (a test that is not about them)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kw)


@pytest.fixture(scope="module")
def straight_chart(straight):
    """``detect_fins`` on the rest pose, on the centerline extracted from it."""
    cl, coords = _chart_of(straight)
    return straight, cl, coords, mesh3d.detect_fins(straight, coords)


def test_detect_fins_finds_eight_pure_islands_on_the_rest_pose(straight_chart):
    """The direct contract: eight names, every island 100% one true fin, every
    reported record self-consistent with ``labels``."""
    mesh, _, coords, det = straight_chart
    truth = mesh.metadata["vertex_labels"]
    assert sorted(det.fins) == sorted(set(synth.LABELS) - {"body"})
    for name, fin in det.fins.items():
        idx = np.asarray(fin["vertex_indices"])
        assert set(truth[idx]) == {name}                     # pure, not majority
        assert fin["n_vertices"] == len(idx) >= 40
        assert np.array_equal(np.flatnonzero(det.labels == name), np.sort(idx))
        lo, hi = fin["station_range"]
        assert 0 <= lo <= hi <= N_STATIONS - 2
        assert fin["s_range"][0] <= fin["s_range"][1]
        assert fin["insertion_centroid"].shape == (3,)
        # The insertion is the root end: nearer the axis than the island mean.
        assert (np.mean(coords.r[idx[coords.r[idx] <= np.percentile(coords.r[idx], 25.0)]])
                < np.mean(coords.r[idx]))
    assert set(det.labels[truth == "body"]) == {"body"}      # no false positives
    assert det.envelope.shape == (N_STATIONS - 1,)
    assert np.all(det.envelope > 0)


def test_detect_fins_margin_is_the_recall_knob(straight_chart):
    """``margin`` is the one dial that decides what counts as protruding, so it
    must move recall monotonically and never invent vertices."""
    mesh, _, coords, _ = straight_chart
    truth = mesh.metadata["vertex_labels"]
    counts = []
    for margin in (0.10, 0.30, 0.60):
        det = _quietly(mesh3d.detect_fins, mesh, coords, margin=margin, check=False)
        assert set(det.labels[truth == "body"]) == {"body"}
        counts.append([det.fins.get(n, {"n_vertices": 0})["n_vertices"]
                       for n in ("dorsal", "pectoral_L", "pelvic_L")])
    for loose, mid, tight in zip(*counts):
        assert loose > mid > tight


def test_detect_fins_min_island_drops_specks(straight_chart):
    mesh, _, coords, det = straight_chart
    smallest = min(f["n_vertices"] for f in det.fins.values())
    big = _quietly(mesh3d.detect_fins, mesh, coords,
                   min_island=smallest + 1, check=False)
    assert len(big.fins) < len(det.fins)
    assert all(f["n_vertices"] > smallest for f in big.fins.values())


def test_a_disjoint_lump_is_not_absorbed_into_the_dorsal_fin():
    """REGRESSION (fix G1): a second dorsal-midline island at s = 0.42 used to be
    unioned into ``dorsal`` silently, stretching its station range across half
    the body and dragging the insertion centroid with it.  Now the island that
    matches the anatomical prior keeps the name, the other is demoted, and the
    collision is named in a warning."""
    mesh = _mesh_with_extra_fins(
        {"dorsal_lump": (0.40, 0.44, 0.0, 0.055, 0.010, 0.60, 6)}
    )
    _, coords = _chart_of(mesh)
    with pytest.warns(RuntimeWarning, match="disjoint islands classify as 'dorsal'"):
        det = mesh3d.detect_fins(mesh, coords)

    truth = mesh.metadata["vertex_labels"]
    dorsal = det.fins["dorsal"]
    assert set(truth[dorsal["vertex_indices"]]) == {"dorsal"}
    assert dorsal["station_range"][0] > 0.65 * (N_STATIONS - 1)     # ~43-51
    assert dorsal["s_range"][0] / coords.total_length > 0.65

    demoted = [k for k in det.fins if k.startswith("unassigned_island_")]
    assert len(demoted) == 1
    lump = det.fins[demoted[0]]
    assert lump["unassigned"] is True and lump["collided_with"] == "dorsal"
    assert set(truth[lump["vertex_indices"]]) == {"dorsal_lump"}
    assert 0.38 < 0.5 * sum(lump["s_range"]) / coords.total_length < 0.46
    # Demoted vertices stay 'body': nothing downstream binds them to a fin.
    assert set(det.labels[lump["vertex_indices"]]) == {"body"}


def test_two_caudal_islands_still_merge_into_one_fin():
    """The other half of fix G1: two islands at the *same* place along the body
    are one fin arriving in pieces, and must merge without a word."""
    mesh = _mesh_with_extra_fins(
        {"caudal_upper_b": (0.94, 1.00, 0.0, 0.100, 0.240, 0.55, 10)}
    )
    _, coords = _chart_of(mesh)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)      # a merge is silent
        det = mesh3d.detect_fins(mesh, coords)

    truth = mesh.metadata["vertex_labels"]
    idx = det.fins["caudal_upper"]["vertex_indices"]
    got = collections.Counter(truth[idx])
    assert set(got) == {"caudal_upper", "caudal_upper_b"}
    assert min(got.values()) > 40                # both islands really are in
    assert det.fins["caudal_upper"]["n_vertices"] == sum(got.values())
    assert not [k for k in det.fins if k.startswith("unassigned_island_")]
    assert set(det.labels[idx]) == {"caudal_upper"}


def test_sliver_fins_are_reported_not_silently_dropped():
    """REGRESSION (fix G4): at 0.3x span three blades never clear the envelope.
    Missing fins used to be a silent absence in the dict."""
    specs = {n: s[:3] + (0.3 * s[3],) + s[4:] for n, s in synth.FIN_SPECS.items()}
    mesh = _mesh_with_specs(specs)
    _, coords = _chart_of(mesh)
    with pytest.warns(RuntimeWarning, match="expected fins were not found"):
        det = mesh3d.detect_fins(mesh, coords)
    missing = set(synth.LABELS) - {"body"} - set(det.fins)
    assert missing                                # the point of the fixture
    assert set(det.labels[mesh.metadata["vertex_labels"] == "body"]) == {"body"}


# ---------------------------------------------------------------------------
# Orientation audit: body roll (G2) and the sign of ``up`` (G3)
# ---------------------------------------------------------------------------

def _rolled(mesh, total_rad):
    """Copy of ``mesh`` rolled about its own axis, linearly in arc length.

    Built through the chart itself (phi += k * s), which is exactly the torsion
    ``estimate_roll`` is meant to see and ``debend`` is not meant to remove.
    """
    cl = mesh.metadata["centerline"]
    c = mesh3d.tube_coords(mesh, cl)
    out = mesh.copy()
    out.vertices = mesh3d.tube_to_points(
        c._replace(phi=c.phi + (total_rad / c.total_length) * c.s), cl
    )
    return out


@pytest.fixture(scope="module")
def rolled90(straight):
    """The rest pose with 90 deg of roll from snout to peduncle."""
    return _rolled(straight, 0.5 * np.pi)


def test_estimate_roll_measures_the_roll_and_ignores_an_unrolled_body(
        straight_chart, rolled90):
    mesh, _, coords, det = straight_chart
    slope, r2 = mesh3d.estimate_roll(mesh, coords, det)
    assert abs(slope * coords.total_length) < mesh3d._ROLL_WARN_RAD   # control

    cl_r, coords_r = _chart_of(rolled90)
    det_r = _quietly(mesh3d.detect_fins, rolled90, coords_r, check=False)
    slope_r, r2_r = mesh3d.estimate_roll(rolled90, coords_r, det_r)
    total = np.rad2deg(slope_r * coords_r.total_length)
    print("\nroll: control %.2f deg (r2 %.2f), rolled %.1f deg (r2 %.2f)"
          % (np.rad2deg(slope * coords.total_length), r2, total, r2_r))
    assert total == pytest.approx(90.0, abs=15.0)
    assert r2_r > 0.4


def test_estimate_roll_rejects_coords_from_another_mesh(straight_chart, rolled90):
    mesh, _, coords, det = straight_chart
    with pytest.raises(ValueError, match="do not belong to this mesh"):
        mesh3d.estimate_roll(trimesh.creation.icosphere(subdivisions=2), coords, det)


def test_debend_warns_about_roll_and_straightens_it_no_worse(straight, bent, rolled90):
    """REGRESSION (fix G2): de-bending keeps (r, phi) by contract, so it removes
    the bend and leaves the roll.  That has to be said out loud -- and it must
    not cost accuracy, so the residual stays at the unrolled control's level."""
    bent_rolled, _ = synth.bend(rolled90)
    with pytest.warns(RuntimeWarning, match="rolled about its own axis"):
        _, _, info_r, err_r = _round_trip(rolled90, bent_rolled)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)      # control is silent
        _, _, info, err = _round_trip(straight, bent[0])

    bl = mesh3d.arc_length(straight.metadata["centerline"])[-1]
    rms, rms_r = np.sqrt((err ** 2).mean()), np.sqrt((err_r ** 2).mean())
    print("\nde-bend rms: control %.3f%% BL, rolled %.3f%% BL"
          % (100.0 * rms / bl, 100.0 * rms_r / bl))
    # Measured: control 0.39% BL, rolled 0.40% BL (the verifier's 0.4-0.5%).
    assert rms_r < 0.006 * bl
    assert rms_r < 1.3 * rms
    assert np.sqrt((err_r ** 2).mean()) < 1.5 * info_r["pitch"]


def _dorsal_to_minus_y(mesh):
    """Copy of the canonical pose rotated so dorsal is -Y (and left is +Z).

    Vertex order and ``metadata['vertex_labels']`` are untouched, so the
    construction truth still indexes it.
    """
    out = mesh.copy()
    out.apply_transform(trimesh.transformations.rotation_matrix(0.5 * np.pi,
                                                                [1.0, 0.0, 0.0]))
    return out


@pytest.fixture(scope="module")
def flipped(straight):
    """(mesh with dorsal at -Y, its centerline) -- the wrong-``--up`` case."""
    mesh = _dorsal_to_minus_y(straight)
    cl, _ = mesh3d.extract_centerline_3d(mesh, n_stations=N_STATIONS)
    return mesh, cl


def _detect_with_up(mesh, cl, up, check=True):
    frames = mesh3d.tube_frames(cl, up=up)
    coords = mesh3d.tube_coords(mesh, cl, frames)
    return mesh3d.detect_fins(mesh, coords, check=check)


def test_check_anatomy_catches_a_wrong_sign_up_vector(flipped, straight):
    """REGRESSION (fix G3): dorsal at -Y charted with ``--up 0 1 0`` mirrors the
    whole animal and every island still gets a plausible name."""
    mesh, cl = flipped
    truth = straight.metadata["vertex_labels"]

    with pytest.warns(RuntimeWarning, match="up vector probably flipped"):
        wrong = _detect_with_up(mesh, cl, (0.0, 1.0, 0.0))
    assert mesh3d.check_anatomy(wrong, warn=False)
    # Mirrored, not merely wrong: with phi reflected, every island named
    # 'dorsal' / 'anal' / 'caudal_upper' is some OTHER fin (which one depends on
    # how the G1 no-silent-merge rule resolves the collisions -- the verifier
    # observed the pelvics landing under 'dorsal'), never the true one.
    for name in ("dorsal", "anal", "caudal_upper"):
        got = set(truth[wrong.fins[name]["vertex_indices"]])
        assert got and name not in got, (name, got)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)      # right way up: quiet
        right = _detect_with_up(mesh, cl, (0.0, -1.0, 0.0))
    assert mesh3d.check_anatomy(right, warn=False) == []
    assert sorted(right.fins) == sorted(set(synth.LABELS) - {"body"})
    for name, fin in right.fins.items():
        assert set(truth[fin["vertex_indices"]]) == {name}          # 8/8 correct


def test_cli_auto_up_negates_a_flipped_up_and_says_so(flipped, straight, capsys):
    """The CLI half of fix G3, on a GLB, through the real argv path."""
    mesh, _ = flipped
    tmp = tempfile.mkdtemp()
    try:
        path = synth.export_glb(mesh, os.path.join(tmp, "flipped.glb"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, det = mesh3d._cli(
                [path, "-n", str(N_STATIONS), "--up", "0", "1", "0", "--auto-up"]
            )
    finally:
        shutil.rmtree(tmp)
    out = capsys.readouterr().out
    assert "--auto-up: anatomy flagged" in out
    assert "up (dorsal direction) used: (0.0, -1.0, 0.0)" in out
    assert mesh3d.check_anatomy(det, warn=False) == []
    assert sorted(det.fins) == sorted(set(synth.LABELS) - {"body"})
    # 8/8 correct against construction truth, matched through the GLB by position.
    truth = straight.metadata["vertex_labels"]
    verts = np.asarray(mesh.vertices)
    assert len(verts) == len(det.labels)
    for name, fin in det.fins.items():
        assert set(truth[np.asarray(fin["vertex_indices"])]) == {name}
