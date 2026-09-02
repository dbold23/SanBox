"""End-to-end behavioural tests: the demo, the CLI, and the GLB it writes.

The demo is run ONCE per session into a temporary directory (about 3 s) and every
test here interrogates that one run, so the suite stays fast while still testing
the real pipeline rather than a stub.  Nothing is asserted about internals the
three modules already cover; these tests are about the seams -- does the CLI wire
them together, does the GLB that comes out load and validate, and is the rest
pose that got baked into the bind matrices actually straight.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np
import pygltflib
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import demo  # noqa: E402
import gltf_export  # noqa: E402
import mesh3d  # noqa: E402
import motion  # noqa: E402
import rig  # noqa: E402
import rig_sevengill  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Module A measured, on this pose at this pitch: body RMS 0.20 px, all-vertex RMS
# 0.50 px, body max 0.89 px.  These brackets are those numbers with room for the
# platform's floating point, not aspirations.
RMS_BODY_PX = 0.5
RMS_ALL_PX = 1.0
MAX_BODY_PX = 2.0
DEMO_BUDGET_S = 120.0


@pytest.fixture(scope="session")
def demo_run(tmp_path_factory):
    """One full demo run into a scratch directory, with its wall-clock time."""
    out = str(tmp_path_factory.mktemp("demo"))
    t0 = time.time()
    info = demo.run(out_dir=out, seconds=2.0, fps=30.0)
    info["wall_s"] = time.time() - t0
    return info


# ---------------------------------------------------------------------------
# The demo itself
# ---------------------------------------------------------------------------
def test_demo_completes_well_inside_the_budget(demo_run):
    assert demo_run["wall_s"] < DEMO_BUDGET_S, (
        "demo took %.1f s, budget %.0f s" % (demo_run["wall_s"], DEMO_BUDGET_S)
    )


def test_demo_writes_every_advertised_output(demo_run):
    for key in ("bent_glb", "rest_glb", "rigged_glb"):
        path = demo_run[key]
        assert os.path.isfile(path), "%s missing" % path
        assert os.path.getsize(path) > 1024
    for name in ("centerline.json", "fins.json", "skeleton.json", "weights.json",
                 "contact_strip.png"):
        path = os.path.join(demo_run["report_dir"], name)
        assert os.path.isfile(path), "report/%s missing" % name
        assert os.path.getsize(path) > 0


def test_every_emitted_glb_has_zero_validator_errors(demo_run):
    for key in ("bent_glb", "rest_glb", "rigged_glb"):
        issues = gltf_export.validate_glb(demo_run[key], raise_on_error=False)
        assert issues["numErrors"] == 0, (
            "%s: %s" % (demo_run[key],
                        json.dumps([m for m in issues["messages"]
                                    if m.get("severity") == 0], indent=1))
        )


def test_validator_warnings_are_reported_not_swallowed(demo_run):
    """The brief asks for warnings to be *reported*; this run happens to have none,
    and if that ever changes the message list must still be available."""
    issues = gltf_export.validate_glb(demo_run["rigged_glb"], raise_on_error=False)
    assert "numWarnings" in issues and "messages" in issues
    assert issues["numWarnings"] == 0, (
        "new validator warnings: %s"
        % [m.get("code") for m in issues["messages"] if m.get("severity") == 1]
    )


# ---------------------------------------------------------------------------
# The rest pose that got baked into the rig
# ---------------------------------------------------------------------------
def test_rest_pose_matches_ground_truth_within_module_a_tolerance(demo_run):
    pitch = demo_run["voxel_pitch"]
    assert demo_run["rms_body"] / pitch < RMS_BODY_PX
    assert demo_run["rms_all"] / pitch < RMS_ALL_PX
    assert demo_run["max_body"] / pitch < MAX_BODY_PX


def test_rest_pose_is_actually_straight(demo_run):
    """The point of the whole prototype: the bind pose has no residual C."""
    result = demo_run["result"]
    straight = np.asarray(result.straight_centerline, dtype=float)
    assert np.allclose(straight[:, 1:], 0.0, atol=1e-12)
    # and the input really was strongly bent, or the test proves nothing
    sagitta = np.max(np.linalg.norm(
        result.centerline - rig_sevengill._chord_projection(result.centerline), axis=1))
    assert sagitta / result.centerline_info["length"] > 0.15


def test_topology_uvs_and_texture_survive_the_debend(demo_run):
    assert demo_run["faces_identical"]
    assert demo_run["uv_roundtrip_error"] < 1e-6
    reloaded = mesh3d.load_mesh(demo_run["rigged_glb"], report=False)
    assert len(reloaded.faces) == len(demo_run["result"].straight_mesh.faces)
    assert getattr(reloaded.visual, "uv", None) is not None


def test_fin_labels_are_pure_and_never_steal_body_vertices(demo_run):
    assert demo_run["mislabelled_body"] == 0
    assert demo_run["fin_purity_min"] == 1.0


# ---------------------------------------------------------------------------
# The GLB's own structure
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def rigged_gltf(demo_run):
    return pygltflib.GLTF2().load(demo_run["rigged_glb"])


def test_rigged_glb_has_the_expected_joint_count(demo_run, rigged_gltf):
    skeleton = demo_run["result"].skeleton
    expected = rig.NUM_SPINE_JOINTS + 2 * len(skeleton.fins)
    assert skeleton.num_joints == expected
    assert len(rigged_gltf.skins) == 1
    assert len(rigged_gltf.skins[0].joints) == expected
    assert len(rigged_gltf.nodes) == expected + 1          # + the mesh node
    names = [rigged_gltf.nodes[j].name for j in rigged_gltf.skins[0].joints]
    assert names[:rig.NUM_SPINE_JOINTS] == list(rig.SPINE_JOINTS)


def test_rigged_glb_animation_names(demo_run, rigged_gltf):
    got = [a.name for a in rigged_gltf.animations]
    assert got == list(demo.DEMO_MOTIONS) + [rig_sevengill.AS_SCANNED_NAME]


def test_animation_channel_counts_are_predictable(demo_run, rigged_gltf):
    """J rotation channels per clip, plus ONE translation channel for as_scanned.

    REGRESSION (fix M4): as_scanned used to write a translation channel on all J
    joints, J - 1 of them a constant restating the node's own rest offset once per
    frame.  Only the root actually translates, so only the root gets a channel.
    """
    n = demo_run["result"].skeleton.num_joints
    root = int(demo_run["result"].skeleton.spine_indices[0])
    for anim in rigged_gltf.animations:
        is_as_scanned = anim.name == rig_sevengill.AS_SCANNED_NAME
        assert len(anim.channels) == (n + 1 if is_as_scanned else n), anim.name
        assert len(anim.samplers) == len(anim.channels), anim.name
        paths = set(c.target.path for c in anim.channels)
        assert paths == ({"rotation", "translation"} if is_as_scanned else {"rotation"})
        moving = [c.target.node for c in anim.channels if c.target.path == "translation"]
        assert moving == ([root] if is_as_scanned else [])


def test_bind_pose_nodes_are_unrotated(rigged_gltf):
    """Identity bind rotations are what make rig.forward_kinematics == the glTF
    skinning matrix; a non-identity node rotation here would silently double-apply."""
    for node in rigged_gltf.nodes:
        if node.rotation is not None:
            assert node.rotation == [0.0, 0.0, 0.0, 1.0]


# ---------------------------------------------------------------------------
# as_scanned
# ---------------------------------------------------------------------------
def test_as_scanned_starts_at_rest_and_reaches_the_scan_pose(demo_run):
    clip = demo_run["result"].clips[rig_sevengill.AS_SCANNED_NAME]
    assert np.allclose(clip.quats[0], np.array([0.0, 0.0, 0.0, 1.0]))
    assert not clip.loop
    # the final pose is a real bend, not a no-op
    angle = 2.0 * np.arccos(np.clip(np.abs(clip.quats[-1][:, 3]), -1.0, 1.0))
    assert np.degrees(angle).max() > 5.0


def test_as_scanned_puts_the_spine_back_on_the_scanned_centerline(demo_run):
    err = demo_run["as_scanned_joint_error_max"]
    assert err / demo_run["voxel_pitch"] < 1.0, "%.5f world units" % err


def test_as_scanned_skinned_surface_lands_on_the_scan(demo_run):
    """LBS cannot reproduce the chart exactly (it blends two rigid transforms per
    body vertex where the chart transports a continuous frame), so this is a
    loose behavioural bound, not a round-trip identity."""
    rel = demo_run["as_scanned_surface_rms"] / demo_run["total_length"]
    assert rel < 0.02, "%.4f BL" % rel


def test_fin_joints_stay_at_identity_in_as_scanned(demo_run):
    """The chart measures the body, not the fins; a fin's scan pose is whatever the
    body carries it to."""
    result = demo_run["result"]
    clip = result.clips[rig_sevengill.AS_SCANNED_NAME]
    sk = result.skeleton
    fin_idx = [j for j in range(sk.num_joints) if sk.kinds[j] != "spine"]
    assert fin_idx
    assert np.allclose(clip.quats[:, fin_idx, :],
                       np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-12)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------
def test_report_json_is_wellformed_and_carries_the_named_facts(demo_run):
    d = demo_run["report_dir"]
    centerline = json.load(open(os.path.join(d, "centerline.json")))
    assert len(centerline["scanned_centerline"]) == centerline["n_stations"]
    assert len(centerline["station_radius"]) == centerline["n_stations"]
    assert centerline["voxel_pitch"] > 0 and centerline["sagitta"] > 0

    fins = json.load(open(os.path.join(d, "fins.json")))
    assert sum(fins["label_counts"].values()) == fins["n_vertices"]
    assert set(fins["fins"]) == set(demo_run["result"].detection.fins)
    for name, entry in fins["fins"].items():
        lo, hi = entry["s_fraction_range"]
        assert lo <= hi
        # The chart stops at the PEDUNCLE by design (module A: the thick-core
        # threshold keeps the medial path out of the fins), so caudal material
        # legitimately reports s > 1 and is transported rigidly by the terminal
        # frame.  Everything else must live inside the chart.
        assert -0.05 <= lo, name
        if not name.startswith("caudal"):
            assert hi <= 1.0, name
    assert any(entry["s_fraction_range"][1] > 1.0
               for name, entry in fins["fins"].items() if name.startswith("caudal"))

    skeleton = json.load(open(os.path.join(d, "skeleton.json")))
    assert skeleton["num_joints"] == len(skeleton["names"]) == len(skeleton["parents"])
    assert all(p < j for j, p in enumerate(skeleton["parents"]))
    assert set(skeleton["clips"]) == set(demo_run["result"].clips)
    assert set(skeleton["fin_parents"].values()) <= set(rig.SPINE_JOINTS)

    weights = json.load(open(os.path.join(d, "weights.json")))
    assert weights["shape"] == [len(demo_run["result"].straight_mesh.vertices),
                                skeleton["num_joints"]]
    assert weights["max_influences"] <= 4
    assert abs(weights["row_sum_min"] - 1.0) < 1e-9
    assert weights["unweighted_joints"] == []


def test_report_states_the_units_of_the_as_scanned_joint_error(demo_run):
    """REGRESSION (fix M4): a bare 'spine_joint_error' number is unreadable.

    It is a length in WORLD units, and the two places the README quotes joint error
    and de-bend error normalise by different things unless it says so.  The report now
    carries the unit explicitly and the same vector pre-divided by the chart length.
    """
    skeleton = json.load(open(os.path.join(demo_run["report_dir"], "skeleton.json")))
    centerline = json.load(open(os.path.join(demo_run["report_dir"], "centerline.json")))
    as_scanned = skeleton["as_scanned"]
    assert "world units" in as_scanned["spine_joint_error_units"]
    assert "length" in as_scanned["spine_joint_error_units"]
    err = np.asarray(as_scanned["spine_joint_error"], dtype=float)
    err_bl = np.asarray(as_scanned["spine_joint_error_bl"], dtype=float)
    assert len(err) == len(err_bl) == rig.NUM_SPINE_JOINTS
    np.testing.assert_allclose(err_bl, err / centerline["length"], rtol=1e-12)


# ---------------------------------------------------------------------------
# the fin-base seam, end to end (fix M2)
# ---------------------------------------------------------------------------
def _seam_edges(faces, labels):
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    e = np.unique(
        np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0), axis=1),
        axis=0,
    )
    is_fin = np.asarray(labels) != "body"
    return e[is_fin[e[:, 0]] != is_fin[e[:, 1]]]


def _max_seam_change_bl(result, weights, clip):
    verts = np.asarray(result.straight_mesh.vertices, dtype=float)
    seam = _seam_edges(result.straight_mesh.faces,
                       np.asarray(result.detection.labels).astype(str))
    rest = np.linalg.norm(verts[seam[:, 0]] - verts[seam[:, 1]], axis=1)
    worst = 0.0
    for q in clip.quats:
        posed = rig.lbs(verts, weights, result.skeleton, rig.quat_to_rotmat(q))
        now = np.linalg.norm(posed[seam[:, 0]] - posed[seam[:, 1]], axis=1)
        worst = max(worst, float(np.abs(now - rest).max()))
    return 100.0 * worst / float(result.centerline_info["length"]), len(seam)


def test_fin_base_seam_edges_stay_continuous_under_the_cruise_clip(demo_run):
    """REGRESSION (fix M2), measured on the real pipeline output.

    A fin island was 100% fin root while the body vertices one ring outside it were
    100% spine, so under a clip the mesh edges crossing that boundary changed length
    by up to 1.94% BL -- a visible tear at every fin base.  The fin-root weight now
    ramps out over ``rig.DEFAULT_FIN_BLEND_RINGS`` rings and the same edges move by
    0.45% BL.  The gate is 1.0% BL.

    (Both numbers moved when the stand-in's fins became solid: the seam is now
    the boundary of a two-sided blade, so it is 346 edges rather than 174 and
    they sit slightly deeper in the body.  The control -- weights with the
    blend off -- reads 1.94% BL instead of 2.13%, so its gate is 1.5%.)
    """
    result = demo_run["result"]
    clip = result.clips["cruise"]
    blended, n_seam = _max_seam_change_bl(result, result.weights, clip)
    assert n_seam > 50, "the demo mesh must actually have fin-base seam edges"
    assert blended < 1.0, "max |dL| over seam edges = %.4f %%BL" % blended

    hard = rig.compute_weights(
        result.straight_mesh.vertices, result.detection.labels, result.skeleton,
    )
    hard_pct, _ = _max_seam_change_bl(result, hard, clip)
    assert hard_pct > 1.5, "the un-blended seam must still be the bad case (%.4f %%BL)" % hard_pct
    assert blended < 0.5 * hard_pct


def test_pipeline_weights_carry_the_fin_base_blend(demo_run):
    """The blend is on by default in the pipeline, not just available in rig.py."""
    result = demo_run["result"]
    labels = np.asarray(result.detection.labels).astype(str)
    ring, owner = rig.fin_seam_rings(
        result.straight_mesh.faces, labels, len(labels), rig.DEFAULT_FIN_BLEND_RINGS
    )
    inner = np.nonzero(ring == 1)[0]
    assert len(inner) > 0
    for v in inner[:50]:
        root = int(result.skeleton.fins[str(owner[v])][0])
        assert result.weights[v, root] > 0.0
    np.testing.assert_allclose(result.weights.sum(axis=1), 1.0, atol=1e-9)
    assert (result.weights > 0).sum(axis=1).max() <= 4


def test_contact_strip_is_a_readable_png(demo_run):
    from PIL import Image
    img = Image.open(os.path.join(demo_run["report_dir"], "contact_strip.png"))
    assert img.mode == "RGB"
    assert img.size[0] > 1000 and img.size[1] > 400
    # not a blank canvas
    assert len(img.convert("RGB").getcolors(maxcolors=1 << 20)) > 50


# ---------------------------------------------------------------------------
# The CLI as a process boundary
# ---------------------------------------------------------------------------
def test_cli_runs_argv_end_to_end(demo_run, tmp_path):
    out = str(tmp_path / "cli.glb")
    report = str(tmp_path / "rep")
    result = rig_sevengill.main([
        "--glb", demo_run["bent_glb"], "--out", out,
        "--motion", "cruise,glide,rest", "--fps", "24", "--seconds", "2",
        "--voxel-pitch", "AUTO", "--keep-bent", "--report", report,
        "-n", "48", "--quiet",
    ])
    assert os.path.isfile(out)
    assert set(result.clips) == {"cruise", "glide", "rest", rig_sevengill.AS_SCANNED_NAME}
    assert all(c.fps == 24.0 for c in result.clips.values())
    assert len(result.centerline) == 48
    assert result.issues["numErrors"] == 0
    assert os.path.isfile(os.path.join(report, "contact_strip.png"))


def test_cli_rejects_an_unknown_motion(tmp_path, demo_run):
    with pytest.raises(SystemExit):
        rig_sevengill.main([
            "--glb", demo_run["bent_glb"], "--out", str(tmp_path / "x.glb"),
            "--motion", "cruise,backflip", "--quiet",
        ])


def test_cli_help_lists_only_implemented_modes():
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "rig_sevengill.py"), "--help"],
        capture_output=True, timeout=120,
    )
    assert proc.returncode == 0
    text = proc.stdout.decode("utf-8")
    for mode in motion.MODES:
        if motion.MODE_CONFIG[mode]["implemented"]:
            assert mode in text
    assert "breach" not in text and "strike" not in text


def test_unimplemented_modes_surface_their_reason(demo_run):
    mskel = motion.MotionSkeleton.from_skeleton(demo_run["result"].skeleton)
    for mode in ("breach", "strike"):
        with pytest.raises(NotImplementedError):
            rig_sevengill.clip_for_mode(mskel, mode)


def test_looping_clip_length_is_a_whole_number_of_tail_beats(demo_run):
    """``--seconds`` is nominal: motion.make_clip refuses a loop that pops, so the
    CLI rounds to whole periods rather than crashing on a fractional request."""
    mskel = motion.MotionSkeleton.from_skeleton(demo_run["result"].skeleton, fps=30.0)
    clip = rig_sevengill.clip_for_mode(mskel, "cruise", fps=30.0, seconds=4.0)
    period = motion.params_for_mode("cruise").period_s
    assert clip.loop
    assert abs(clip.duration_s / period - round(clip.duration_s / period)) < 1e-9
    assert np.allclose(clip.quats[0], clip.quats[-1])


def test_pipeline_is_deterministic(demo_run):
    a = rig_sevengill.run_pipeline(demo_run["bent_glb"], out=None, motions=("cruise",),
                                   n_stations=32, verbose=False)
    b = rig_sevengill.run_pipeline(demo_run["bent_glb"], out=None, motions=("cruise",),
                                   n_stations=32, verbose=False)
    assert np.array_equal(a.centerline, b.centerline)
    assert np.array_equal(a.straight_mesh.vertices, b.straight_mesh.vertices)
    assert np.array_equal(a.weights, b.weights)
    assert np.array_equal(a.clips["cruise"].quats, b.clips["cruise"].quats)
