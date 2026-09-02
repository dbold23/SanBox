"""Contract tests for prototype 06's body, chart and renderer.

The four the brief names, plus the invariants they depend on:

* the ALBEDO ROUND TRIP -- a single spot stamped at ``(s, phi)`` in the chart
  lands on the image pixels whose chart ground truth is ``(s, phi)``;
* PHI PERIODICITY at the ventral seam -- ``phi = +pi - eps`` and
  ``phi = -pi + eps`` sample the same chart neighbourhood;
* DECIMATION preserves the ``s`` range and drops the vertex count;
* ``pose(amp=0)`` reproduces the rest pose.

Most tests run on a synthetic tube built here, so they need neither the 175 MB
rigged GLB nor its decimation cache.  The two that check the REAL body are
skipped when ``assets/real_body_*.npz`` has not been built.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_P06 = os.path.dirname(_HERE)
if _P06 not in sys.path:
    sys.path.insert(0, _P06)

import real_body  # noqa: E402
import synth_render as S  # noqa: E402

import mesh3d  # noqa: E402  (put on the path by real_body)
import pattern  # noqa: E402  (put on the path by synth_render)
import render  # noqa: E402


# ---------------------------------------------------------------------------
# A small tube that satisfies the RealBody contract
# ---------------------------------------------------------------------------

def make_tube(n_stations=64, n_around=64, length=0.5, radius=0.04):
    """A closed cylinder charted exactly like :class:`real_body.RealBody`."""
    s_m = np.linspace(0.0, length, n_stations)
    phi = -math.pi + (np.arange(n_around) + 0.5) * (2.0 * math.pi / n_around)
    S_m, PHI = np.meshgrid(s_m, phi, indexing="ij")
    s_flat = S_m.ravel()
    phi_flat = PHI.ravel()
    r_flat = np.full(s_flat.shape, float(radius))

    cl = np.column_stack([0.5 * length - s_m, np.zeros(n_stations),
                          np.zeros(n_stations)])
    frames = mesh3d.canonical_frames(n_stations)
    verts = real_body._points_from_chart(s_flat, r_flat, phi_flat, cl, frames)

    faces = []
    for i in range(n_stations - 1):
        for j in range(n_around):
            a = i * n_around + j
            b = i * n_around + (j + 1) % n_around
            c = (i + 1) * n_around + j
            d = (i + 1) * n_around + (j + 1) % n_around
            faces.append([a, c, b])
            faces.append([b, c, d])
    faces = np.asarray(faces, dtype=np.int64)

    return real_body.RealBody(
        vertices=verts, faces=faces, s=s_flat / length, phi=phi_flat,
        r=r_flat, s_m=s_flat, is_fin=np.zeros(len(s_flat), dtype=bool),
        centerline=cl, total_length=float(length), s_raw_range=(0.0, length),
        meta={"synthetic": True})


@pytest.fixture(scope="module")
def tube():
    return make_tube()


@pytest.fixture(scope="module")
def config():
    return S.load_config()


def _cached_real_body():
    for cell in (1.5, 2.0, 2.5, 4.0, 6.0):
        if os.path.exists(real_body._cache_path(cell)):
            return real_body.load_cached(cell)
    return None


# ---------------------------------------------------------------------------
# 1. Albedo sampling round trip
# ---------------------------------------------------------------------------

def test_sample_chart_hits_the_stamped_cell():
    """A value written at chart cell (j, i) is read back at that cell's (s, phi)."""
    h, w = 64, 128
    chart = np.zeros((h, w))
    s_axis, phi_axis = pattern.chart_axes((h, w))
    j, i = 20, 77
    chart[j, i] = 1.0
    assert S.sample_chart(chart, s_axis[i], phi_axis[j]) == pytest.approx(1.0)
    # ...and nowhere else: two cells away in either axis reads back zero.
    assert S.sample_chart(chart, s_axis[i + 2], phi_axis[j]) == pytest.approx(0.0)
    assert S.sample_chart(chart, s_axis[i], phi_axis[j + 2]) == pytest.approx(0.0)


def test_sample_chart_is_bilinear_between_cells():
    h, w = 32, 64
    chart = np.zeros((h, w))
    s_axis, phi_axis = pattern.chart_axes((h, w))
    chart[10, 30] = 1.0
    mid_s = 0.5 * (s_axis[30] + s_axis[31])
    assert S.sample_chart(chart, mid_s, phi_axis[10]) == pytest.approx(0.5)


def test_single_spot_round_trips_through_the_render(tube, config):
    """A spot stamped at (s0, phi0) darkens the pixels whose chart GT is (s0, phi0).

    The whole prototype rests on this: the matcher is handed detections made on
    pixels, and their chart coordinates have to be the ones the pattern was
    written in.  Both directions are checked -- the darkest pixel is at the
    right chart coordinate, AND the pixels at that chart coordinate are dark.
    """
    res = (192, 256)
    # phi0 is NEGATIVE: the camera below sits at -Y, so it sees the
    # animal's RIGHT flank, which is phi < 0 (B = +Y is the LEFT).
    s0, phi0, radius = 0.42, -1.10, 0.02
    h, w = 128, 512
    darkness = np.zeros((h, w))
    s_axis, phi_axis = pattern.chart_axes((h, w))
    dS = s_axis[None, :] - s0
    dP = pattern.wrap_phi(phi_axis[:, None] - phi0) * 0.085
    darkness[np.hypot(dS, dP) <= radius] = 1.0

    skin = np.ones((h, w, 3))
    albedo = S.albedo_chart(skin, darkness, 0.9)

    verts = real_body.pose(tube)
    camera = render.Camera(
        eye=np.array([0.0, -0.45, 0.05]), target=np.array([0.0, 0.0, 0.0]),
        up=(0.0, 0.0, 1.0), resolution=res, kind="pinhole", fov_y_deg=45.0)
    light = render.DirectionalLight(direction=(0.0, 1.0, -0.3), ambient=1.0,
                                    intensity=0.0)
    inst = render.Instance(vertices=verts, faces=tube.faces, color=(1.0, 1.0, 1.0),
                           vertex_s=tube.s, vertex_phi=tube.phi)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        out = render.render([inst], camera, light=light, exclusion=None,
                            background=(0.0, 0.0, 0.0), shadows=False)
    draw = S.SceneDraw(
        pose={}, side="L", camera={}, light={"ambient": 1.0, "intensity": 0.0},
        specular={"strength": 0.0, "shininess": 30.0}, background={},
        occluders=[], degrade={})
    cfg = S.load_config(overrides={"skin": {"tone_dorsal": 1.0, "tone_ventral": 1.0,
                                            "fin_tone_from_normal": False}})
    rgb = S.shade_subject(out, camera, light, draw, albedo, cfg)

    vis = out["visible_skin"]
    assert vis.sum() > 500
    lum = rgb[..., 0]
    dark = vis & (lum < 0.5)
    assert dark.sum() > 20, "the stamped spot did not reach any pixel"

    # forward: every dark pixel's chart coordinate is inside the stamped disc
    d = np.hypot(out["chart_s"][dark] - s0,
                 pattern.wrap_phi(out["chart_phi"][dark] - phi0) * 0.085)
    assert d.max() <= radius * 1.35, "dark pixels outside the stamped disc"

    # backward: pixels well inside the disc are dark
    dall = np.full(vis.shape, np.inf)
    dall[vis] = np.hypot(out["chart_s"][vis] - s0,
                         pattern.wrap_phi(out["chart_phi"][vis] - phi0) * 0.085)
    inside = vis & (dall < 0.6 * radius)
    assert inside.sum() > 10
    assert (lum[inside] < 0.5).mean() > 0.95


# ---------------------------------------------------------------------------
# 2. Phi periodicity at the ventral seam
# ---------------------------------------------------------------------------

def test_sample_chart_wraps_at_the_ventral_seam():
    """``phi`` rows are periodic: the seam blends, it does not clamp."""
    h, w = 64, 128
    chart = np.zeros((h, w))
    chart[0, 40] = 1.0        # ONLY the row at phi = -pi + half a cell
    s_axis, phi_axis = pattern.chart_axes((h, w))
    assert S.sample_chart(chart, s_axis[40], phi_axis[0]) == pytest.approx(1.0)
    # phi = +-pi sits exactly halfway between row h-1 (0.0) and row 0 (1.0);
    # a CLAMPING lookup would return 0.0 there, which is the bug this catches.
    assert S.sample_chart(chart, s_axis[40], math.pi) == pytest.approx(0.5)
    assert S.sample_chart(chart, s_axis[40], -math.pi) == pytest.approx(0.5)
    # A clamping lookup would give 1.0 at one end and 0.0 just past it.
    assert (S.sample_chart(chart, s_axis[40], math.pi + 1e-9)
            == pytest.approx(S.sample_chart(chart, s_axis[40], -math.pi + 1e-9)))


def test_sample_chart_phi_is_2pi_periodic():
    rng = np.random.default_rng(0)
    chart = rng.random((32, 64))
    s = rng.random(200)
    phi = rng.uniform(-math.pi, math.pi, 200)
    a = S.sample_chart(chart, s, phi)
    b = S.sample_chart(chart, s, phi + 2.0 * math.pi)
    c = S.sample_chart(chart, s, phi - 4.0 * math.pi)
    assert np.allclose(a, b)
    assert np.allclose(a, c)


def test_circular_mean_averages_across_the_seam():
    """Decimation's phi average must not send a seam cell to the dorsal midline."""
    phi = np.array([math.pi - 0.1, -math.pi + 0.1])
    got = real_body._circular_mean(phi, np.zeros(2, dtype=np.int64), 1)[0]
    assert abs(abs(got) - math.pi) < 1e-9
    naive = float(phi.mean())
    assert abs(naive) < 0.2      # the bug this guards against


def test_decimation_keeps_the_seam(tube):
    """Clustering a tube keeps every phi in [-pi, pi) and covers the seam."""
    dec = real_body.decimate(tube.vertices, tube.faces, tube.s_m, tube.r,
                             tube.phi, tube.is_fin, cell=0.01)
    assert dec["phi"].min() >= -math.pi - 1e-9
    assert dec["phi"].max() < math.pi + 1e-9
    # The tube's own vertices stop half a cell short of the seam (+-3.0925 rad)
    # and a cluster averages a couple of them, so 2.9 is the honest bound here;
    # the real body, whose seam column sits ON +-pi, reaches 3.14 (checked in
    # test_real_body_cache_is_sane).
    assert (np.abs(dec["phi"]) > 2.9).any(), "no cell near the ventral seam"


# ---------------------------------------------------------------------------
# 3. Decimation
# ---------------------------------------------------------------------------

def test_decimate_drops_vertices_and_preserves_the_s_range(tube):
    for cell in (0.008, 0.016, 0.03):
        dec = real_body.decimate(tube.vertices, tube.faces, tube.s_m, tube.r,
                                 tube.phi, tube.is_fin, cell=cell)
        assert len(dec["vertices"]) < len(tube.vertices)
        assert len(dec["faces"]) < len(tube.faces)
        # A cluster mean cannot leave the source range, and with cells this
        # much smaller than the body it cannot shrink it by more than a cell.
        assert dec["s_m"].min() >= tube.s_m.min() - 1e-12
        assert dec["s_m"].max() <= tube.s_m.max() + 1e-12
        assert dec["s_m"].min() - tube.s_m.min() < cell
        assert tube.s_m.max() - dec["s_m"].max() < cell
        assert (dec["faces"][:, 0] != dec["faces"][:, 1]).all()
        assert (dec["faces"][:, 1] != dec["faces"][:, 2]).all()
        assert (dec["faces"][:, 0] != dec["faces"][:, 2]).all()


def test_decimate_is_monotone_in_cell_size(tube):
    counts = [len(real_body.decimate(tube.vertices, tube.faces, tube.s_m, tube.r,
                                     tube.phi, tube.is_fin, cell=c)["vertices"])
              for c in (0.008, 0.016, 0.03)]
    assert counts[0] > counts[1] > counts[2]


def test_decimate_faces_are_unique(tube):
    dec = real_body.decimate(tube.vertices, tube.faces, tube.s_m, tube.r,
                             tube.phi, tube.is_fin, cell=0.012)
    key = np.sort(dec["faces"], axis=1)
    assert len(np.unique(key, axis=0)) == len(key)


def test_real_body_cache_is_sane():
    body = _cached_real_body()
    if body is None:
        pytest.skip("no assets/real_body_*.npz; run `python real_body.py`")
    assert body.meta["n_vertices"] < body.meta["n_vertices_source"]
    assert body.meta["n_faces"] < body.meta["n_faces_source"]
    assert 0.0 <= body.s.min() < 0.01
    assert 0.99 < body.s.max() <= 1.0
    assert body.phi.min() >= -math.pi and body.phi.max() < math.pi + 1e-9
    assert (np.abs(body.phi) > 3.0).any(), "no vertex on the ventral seam"
    assert body.is_fin.any() and not body.is_fin.all()
    assert body.faces.max() < len(body.vertices)


# ---------------------------------------------------------------------------
# 4. Pose
# ---------------------------------------------------------------------------

def test_pose_zero_amplitude_reproduces_the_rest_pose(tube):
    got = real_body.pose(tube, amp=0.0)
    assert np.abs(got - tube.vertices).max() < 1e-12


def test_pose_zero_amplitude_reproduces_the_real_rest_pose():
    body = _cached_real_body()
    if body is None:
        pytest.skip("no assets/real_body_*.npz; run `python real_body.py`")
    got = real_body.pose(body, amp=0.0)
    assert np.abs(got - body.vertices).max() < 1e-9


def test_pose_preserves_arc_length_and_is_rigid_under_yaw(tube):
    """A bend moves material but keeps the chart: it is an isometry in s."""
    bent = real_body.pose(tube, amp=0.4, wave=0.75, phase=0.6)
    cl = real_body.bent_centerline(tube.total_length, amp=0.4, wave=0.75, phase=0.6)
    assert mesh3d.arc_length(cl)[-1] == pytest.approx(tube.total_length, rel=1e-12)
    # radius from the axis is preserved station by station
    coords = mesh3d.tube_coords(bent, cl, mesh3d.tube_frames(cl))
    assert np.abs(coords.r - tube.r).max() < 2e-4
    # a pure yaw is a rigid rotation about Z of the rest pose
    yawed = real_body.pose(tube, amp=0.0, yaw_deg=30.0)
    a = math.radians(30.0)
    rot = np.array([[math.cos(a), -math.sin(a), 0.0],
                    [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    assert np.abs(yawed - tube.vertices @ rot.T).max() < 1e-9


def test_pose_changes_geometry_when_amp_is_nonzero(tube):
    # amp is radians of heading per unit arc-length FRACTION; with wave=0.5 and
    # phase=pi/2 the curvature does not change sign, so the body takes a C of
    # about 0.76 rad end to end.
    bent = real_body.pose(tube, amp=1.2, wave=0.5, phase=0.5 * math.pi)
    assert np.abs(bent - tube.vertices).max() > 0.01


def test_fin_stretch_reports_the_bend_factor():
    body = make_tube()
    body = body._replace(is_fin=body.r > 0.0)
    assert real_body.fin_stretch(body, amp=0.0) == pytest.approx(1.0)
    got = real_body.fin_stretch(body, amp=0.4)
    assert got == pytest.approx(1.0 + 0.4 / body.total_length * body.r.max())


# ---------------------------------------------------------------------------
# Charts, tone, scene
# ---------------------------------------------------------------------------

def test_skin_chart_shape_and_range(config):
    chart, stats = S.skin_chart((128, 256), config)
    assert chart.shape == (128, 256, 3)
    assert chart.min() >= 0.0 and chart.max() <= 1.0
    assert 0.0 <= stats["unobserved_frac"] < 0.2


def test_tone_multiplier_is_monotone_from_dorsal_to_ventral(config):
    phi = np.linspace(0.0, math.pi, 64)
    tone = S.tone_multiplier(phi, config)
    assert np.all(np.diff(tone) >= -1e-12)
    assert tone[0] == pytest.approx(config["skin"]["tone_dorsal"])
    assert tone[-1] == pytest.approx(config["skin"]["tone_ventral"])
    assert np.allclose(S.tone_multiplier(-phi, config), tone)


def test_albedo_chart_darkens_only_where_the_pattern_is():
    skin = np.full((16, 32, 3), 0.6)
    darkness = np.zeros((16, 32))
    darkness[4, 7] = 1.0
    alb = S.albedo_chart(skin, darkness, 0.9)
    assert alb[4, 7, 0] == pytest.approx(0.6 * 0.1)
    assert alb[0, 0, 0] == pytest.approx(0.6)


def test_eye_mask_lands_on_the_measured_eye():
    body = _cached_real_body()
    if body is None or not os.path.exists(S.DEFAULT_EYE_JSON):
        pytest.skip("no cached body / eye_patch.json")
    mask = S.eye_chart_mask((256, 1024), body, radius_m=0.009)
    # The peak is a cell centre near, not exactly on, the eye centre.
    assert mask is not None and mask.max() > 0.9
    h, w = mask.shape
    ys, xs = np.nonzero(mask > 0.5)
    s = (xs + 0.5) / w
    phi = -math.pi + (ys + 0.5) * (2.0 * math.pi / h)
    assert 0.0 < s.min() and s.max() < 0.10
    assert (np.abs(phi) > 1.0).all()          # lateral, not dorsal or ventral
    assert (phi > 0).any() and (phi < 0).any()  # both eyes


def test_draw_scene_is_deterministic_in_the_seed(config):
    a = S.draw_scene(np.random.default_rng([1, 2, 3]), config)
    b = S.draw_scene(np.random.default_rng([1, 2, 3]), config)
    assert a.pose == b.pose and a.camera == b.camera and a.light == b.light
    assert a.occluders == b.occluders and a.degrade == b.degrade


def test_config_overrides_are_deep(config):
    cfg = S.load_config(overrides={"pattern": {"n_spots": 7}})
    assert cfg["pattern"]["n_spots"] == 7
    assert cfg["pattern"]["min_sep"] == S.DEFAULT_CONFIG["pattern"]["min_sep"]
    assert S.DEFAULT_CONFIG["pattern"]["n_spots"] != 7   # not mutated


def test_frame_camera_fits_the_requested_width(tube, config):
    draw = S.draw_scene(np.random.default_rng([4]), config)
    draw = draw._replace(camera=dict(draw.camera, s_frame_max=0.6,
                                     width_frac=0.85, roll_deg=0.0,
                                     elevation_deg=10.0, azimuth_deg=0.0))
    camera, dist, target, direction = S.frame_camera(tube.vertices, tube.s, draw)
    px, _, pz = camera.project(tube.vertices[tube.s <= 0.6])
    ok = np.isfinite(px) & (pz > camera.near)
    extent = float(px[ok].max() - px[ok].min())
    assert extent == pytest.approx(0.85 * camera.width, rel=0.02)
    assert dist > 0.0


# ---------------------------------------------------------------------------
# Spot ground truth
# ---------------------------------------------------------------------------

def _fake_out(shape, s_field, phi_field, visible):
    return {"visible_skin": visible, "chart_s": s_field, "chart_phi": phi_field}


def test_spot_ground_truth_finds_a_visible_spot():
    h, w = 40, 60
    s_field = np.tile(np.linspace(0.0, 1.0, w), (h, 1))
    phi_field = np.tile(np.linspace(-1.0, 1.0, h)[:, None], (1, w))
    visible = np.ones((h, w), dtype=bool)
    spots = np.zeros(2, dtype=[("id", "i4"), ("s", "f8"), ("phi", "f8"),
                               ("radius", "f8"), ("rendered_darkness", "f8")])
    spots["id"] = [0, 1]
    spots["s"] = [0.5, 0.5]
    spots["phi"] = [0.0, 0.0]
    spots["radius"] = [0.05, 0.05]
    spots["rendered_darkness"] = [0.8, 0.0]     # the second was never stamped
    rows = S.spot_ground_truth(_fake_out((h, w), s_field, phi_field, visible),
                               spots, 0.085)
    assert rows[0]["visible"] and rows[1]["visible"] is False
    assert rows[0]["cx"] == pytest.approx((w - 1) * 0.5, abs=1.0)
    assert rows[0]["radius_px"] > 0.0
    assert rows[1]["cx"] is None


def test_spot_ground_truth_rejects_an_off_body_spot():
    h, w = 20, 20
    s_field = np.full((h, w), 0.10)
    phi_field = np.zeros((h, w))
    visible = np.ones((h, w), dtype=bool)
    spots = np.zeros(1, dtype=[("id", "i4"), ("s", "f8"), ("phi", "f8"),
                               ("radius", "f8"), ("rendered_darkness", "f8")])
    spots["s"] = [0.90]
    spots["radius"] = [0.01]
    spots["rendered_darkness"] = [0.9]
    rows = S.spot_ground_truth(_fake_out((h, w), s_field, phi_field, visible),
                               spots, 0.085)
    assert rows[0]["visible"] is False and rows[0]["cx"] is None
