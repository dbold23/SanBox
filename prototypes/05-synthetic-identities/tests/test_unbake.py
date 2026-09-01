"""Tests for the photograph -> canonical chart path (unbake.py).

The fixture renderer is analytic, so every pixel's true (s, phi) is known and
these tests measure unbake's error rather than a rasteriser's.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import bake
import fixtures
import unbake

NOMINAL_R = unbake.NOMINAL_RADIUS
GRAZING_SIN = 0.40          # |sin phi| below this is the grazing zone
S_MARGIN = (0.08, 0.92)     # the medial axis retracts from both tips


def _scene(side="L", n_spots=45, seed=3, **render_kw):
    chart, spots = fixtures.make_test_chart(
        n_s=160, n_phi=320, n_spots=n_spots, seed=seed)
    rgb, mask, info = fixtures.render_lateral_tube(
        chart, n_px=(200, 640), side=side, seed=1, **render_kw)
    return chart, spots, rgb, mask, info


def _dorsal_point(mask):
    """An image point on the dorsal side, standing in for an annotated landmark.

    In the fixture render dorsal is image-up, so a point a quarter of the way
    down the silhouette at mid-body is dorsal.  On real data this is Schema S1's
    ``dorsal_fin_origin`` or ``gill_slit_1_dorsal_origin`` annotation.
    """
    col = mask.shape[1] // 2
    rows = np.flatnonzero(mask[:, col])
    return (float(col), float(rows.min() + 0.25 * (rows.max() - rows.min())))


def _visible_spots(spots, side):
    sgn = 1 if side == "L" else -1
    keep = (
        (np.sin(np.abs(spots[:, 1])) > GRAZING_SIN)
        & (spots[:, 1] * sgn > 0)
        & (spots[:, 0] > S_MARGIN[0])
        & (spots[:, 0] < S_MARGIN[1])
    )
    return spots[keep]


def _recall(gt, det, tol=0.03):
    if len(gt) == 0:
        return 1.0, 0
    if len(det) == 0:
        return 0.0, 0
    d = np.hypot(
        gt[:, None, 0] - det[None, :, 0],
        bake.wrap_to_pi(gt[:, None, 1] - det[None, :, 1]) * NOMINAL_R,
    )
    hit = d.min(axis=1) < tol
    return float(hit.mean()), int(hit.sum())


# ---------------------------------------------------------------------------
# the headline requirement: spots come back where they went in
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("side", ["L", "R"])
def test_spot_centroid_recall_outside_grazing_zones(side):
    chart, spots, rgb, mask, info = _scene(side=side)
    res = unbake.photo_to_chart(
        rgb, mask, side=side, chart_shape=(128, 256),
        dorsal_point=_dorsal_point(mask))
    det = fixtures.detect_chart_spots(res.chart)
    gt = _visible_spots(spots, side)
    recall, n_hit = _recall(gt, det)
    print("side %s: %d/%d source spots recovered (recall %.3f), %d detections"
          % (side, n_hit, len(gt), recall, len(det)))
    assert len(gt) >= 10, "fixture must present enough non-grazing spots to score"
    assert recall > 0.8


def test_recovered_chart_correlates_with_the_source_chart():
    """Not just spot centroids: the whole visible sheet must line up."""
    chart, spots, rgb, mask, info = _scene(side="L", n_spots=60, seed=11)
    res = unbake.photo_to_chart(
        rgb, mask, side="L", chart_shape=(128, 256),
        dorsal_point=_dorsal_point(mask))
    n_s, n_phi = res.chart.shape
    S, P = np.meshgrid(*bake.chart_axes(n_s, n_phi), indexing="ij")
    src = bake.sample_chart(chart, S, P)
    ok = (
        np.isfinite(res.chart)
        & (res.confidence > 0.5)
        & (S > S_MARGIN[0]) & (S < S_MARGIN[1])
    )
    corr = float(np.corrcoef(src[ok], res.chart[ok])[0, 1])
    print("chart correlation on the confident visible half: %.4f over %d cells"
          % (corr, ok.sum()))
    assert ok.sum() > 2000
    assert corr > 0.7


def test_confidence_is_the_foreshortening_factor():
    """Confidence must peak on the flank and vanish at both silhouette edges."""
    chart, spots, rgb, mask, info = _scene()
    res = unbake.photo_to_chart(
        rgb, mask, side="L", chart_shape=(128, 256),
        dorsal_point=_dorsal_point(mask))
    n_s, n_phi = res.confidence.shape
    _, phi_ax = bake.chart_axes(n_s, n_phi)
    mid = slice(int(0.3 * n_s), int(0.7 * n_s))
    prof = res.confidence[mid].mean(axis=0)

    flank = np.argmin(np.abs(phi_ax - np.pi / 2))
    dorsal = np.argmin(np.abs(phi_ax - 0.0))
    ventral = np.argmin(np.abs(np.abs(phi_ax) - np.pi))
    far = np.argmin(np.abs(phi_ax + np.pi / 2))
    print("confidence: flank %.3f, dorsal %.3f, ventral %.3f, far side %.3f"
          % (prof[flank], prof[dorsal], prof[ventral], prof[far]))
    assert prof[flank] > 0.8
    assert prof[dorsal] < 0.35 and prof[ventral] < 0.35
    assert prof[far] < 1e-6           # the far half is never observed
    # and the peak really is at the flank
    assert abs(phi_ax[int(np.argmax(prof))] - np.pi / 2) < 0.4


def test_far_half_of_the_girth_is_not_invented():
    """Nothing may appear on the unseen flank -- except one cell of bleed.

    A left-side view observes phi in (0, pi).  The only negative-phi cells that
    may light up are the one or two immediately below -pi, which the bilinear
    splat of a genuinely observed ventral-midline sample reaches ACROSS the
    wrap -- that is the phi axis being periodic, not invention.  Everything
    from the dorsal midline round to the far flank must be untouched.
    """
    chart, spots, rgb, mask, info = _scene(side="L")
    res = unbake.photo_to_chart(
        rgb, mask, side="L", chart_shape=(96, 192),
        dorsal_point=_dorsal_point(mask), exclusion_mask=None, landmarks=None)
    n_s, n_phi = res.chart.shape
    _, phi_ax = bake.chart_axes(n_s, n_phi)
    cell = 2.0 * np.pi / n_phi
    far = (phi_ax < -0.2) & (phi_ax > -np.pi + 2 * cell)
    assert far.sum() > n_phi // 3
    assert not np.isfinite(res.chart[:, far]).any()
    assert np.all(res.confidence[:, far] == 0.0)
    # and the wrap bleed really is confined to the seam cells
    seam = phi_ax <= -np.pi + 2 * cell
    assert float(np.nanmax(res.confidence[:, seam])) < 0.05


# ---------------------------------------------------------------------------
# the dorso-ventral sign, which is silent and total when wrong
# ---------------------------------------------------------------------------

def test_dorsal_point_gives_the_exact_sign_on_both_sides():
    signs = {}
    for side in ("L", "R"):
        chart, spots, rgb, mask, info = _scene(side=side)
        res = unbake.photo_to_chart(
            rgb, mask, side=side, chart_shape=(96, 192),
            dorsal_point=_dorsal_point(mask))
        signs[side] = res.dorsal_sign
        assert "dorsal_point" in res.notes[0]
    # the R render is the horizontal mirror of the L one, so the chart's left
    # normal flips and the dorsal sign must flip with it
    assert signs["L"] == -signs["R"]
    print("dorsal_sign L=%+d, R=%+d" % (signs["L"], signs["R"]))


def test_countershading_heuristic_agrees_when_the_cue_is_real():
    chart, spots, rgb, mask, info = _scene(shading=0.05, countershading=0.45)
    truth = unbake.photo_to_chart(
        rgb, mask, side="L", chart_shape=(96, 192),
        dorsal_point=_dorsal_point(mask)).dorsal_sign
    guessed = unbake.photo_to_chart(rgb, mask, side="L", chart_shape=(96, 192))
    assert guessed.dorsal_sign == truth
    assert "inferred from countershading" in guessed.notes[0]
    print(guessed.notes[0])


def test_countershading_heuristic_refuses_when_the_key_light_cancels_it():
    """Countershading exists to cancel overhead lighting -- so this cue dies.

    At the fixture defaults (an overhead key light plus a cosine countershading
    term) the two flank halves land within ~1% of each other and the heuristic
    must say it does not know, rather than silently mirroring the chart.
    """
    chart, spots, rgb, mask, info = _scene()
    res = unbake.photo_to_chart(rgb, mask, side="L", chart_shape=(96, 192))
    assert "UNRELIABLE" in res.notes[0]
    print(res.notes[0])


# ---------------------------------------------------------------------------
# exclusions, anchored to Schema S1
# ---------------------------------------------------------------------------

def test_schema_landmarks_load_and_carry_the_expected_types():
    kp = unbake.load_schema_landmarks()
    assert kp["eye_center"]["id"] == 2 and kp["eye_center"]["type"] == "I"
    assert kp["mouth_rictus"]["id"] == 3 and kp["mouth_rictus"]["type"] == "I"
    assert kp["gill_slit_7_dorsal_origin"]["id"] == 7        # SEVEN slits
    assert len(kp) == 30


def test_unknown_landmark_names_are_rejected_against_the_schema():
    with pytest.raises(ValueError, match="Schema S1"):
        unbake.eye_mouth_exclusion((32, 64), {"eyeball": (0.05, 0.9)})


def test_eye_and_mouth_exclusion_geometry():
    shape = (128, 256)
    landmarks = {"eye_center": (0.06, 1.0), "mouth_rictus": (0.11, 2.6)}
    mask, notes = unbake.eye_mouth_exclusion(shape, landmarks)
    S, P = unbake._chart_meshgrid(shape)

    assert mask[np.argmin(np.abs(S[:, 0] - 0.06)),
                np.argmin(np.abs(P[0] - 1.0))]          # the eye itself
    assert mask[np.argmin(np.abs(S[:, 0] - 0.05)),
                np.argmin(np.abs(P[0] - np.pi))]        # ventral head = mouth
    # the dorsolateral head freckle patch (Schema S1 head_patch_bounds) survives
    patch = (S > 0.02) & (S < 0.10) & (np.abs(P) < 1.2)
    assert not mask[patch & (np.abs(P) > 0.2) & (S > 0.075)].any()
    # nothing posterior is touched
    assert not mask[S > 0.2].any()
    assert 0.0 < mask.mean() < 0.10
    print("exclusion covers %.2f%% of the chart" % (100 * mask.mean()))


def test_exclusions_zero_the_confidence_and_void_the_chart():
    chart, spots, rgb, mask, info = _scene()
    landmarks = {"eye_center": (0.06, 1.0), "mouth_rictus": (0.11, 2.6)}
    shape = (128, 256)
    excl, _ = unbake.eye_mouth_exclusion(shape, landmarks)
    res = unbake.photo_to_chart(
        rgb, mask, side="L", chart_shape=shape,
        dorsal_point=_dorsal_point(mask), exclusion_mask=excl)
    assert np.all(res.confidence[excl] == 0.0)
    assert not np.isfinite(res.chart[excl]).any()
    assert np.isfinite(res.chart[~excl]).any()


def test_no_hook_and_no_landmarks_means_a_loud_note_not_a_silent_pass(monkeypatch):
    """The degraded path, forced: ``sys.modules['pattern'] = None`` makes the
    lazy ``import pattern`` raise, which is how an absent module P looks."""
    monkeypatch.setitem(sys.modules, "pattern", None)
    _, notes = unbake.resolve_exclusion_mask((16, 32))
    assert any("NO exclusion applied" in n for n in notes)


def test_fallback_exclusion_is_used_when_module_p_is_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "pattern", None)
    landmarks = {"eye_center": (0.06, 1.0), "mouth_rictus": (0.11, 2.6)}
    m, notes = unbake.resolve_exclusion_mask((64, 128), landmarks=landmarks)
    assert "eye_mouth_exclusion" in notes[0]
    assert m.any()


def test_module_p_exclusion_mask_takes_precedence(monkeypatch):
    stub = types.ModuleType("pattern")
    called = {}

    def exclusion_mask(chart_shape, landmarks=None, axis_order="s_major"):
        called["shape"] = chart_shape
        called["axis_order"] = axis_order
        m = np.zeros(chart_shape, dtype=bool)
        m[:4] = True
        return m

    stub.exclusion_mask = exclusion_mask
    monkeypatch.setitem(sys.modules, "pattern", stub)
    m, notes = unbake.resolve_exclusion_mask((16, 32), landmarks=None)
    assert called["shape"] == (16, 32)
    assert called["axis_order"] == "s_major"      # our layout, stated not guessed
    assert m[:4].all() and not m[4:].any()
    assert "pattern.exclusion_mask" in notes[0]


def test_the_chart_exclusion_mask_alias_is_also_probed(monkeypatch):
    """Module P ships the hook as ``chart_exclusion_mask`` on purpose; the
    activation of that integration lives here, so it must be exercised here."""
    stub = types.ModuleType("pattern")

    def chart_exclusion_mask(chart_shape, landmarks=None, axis_order="s_major"):
        m = np.zeros(chart_shape, dtype=bool)
        m[-3:] = True
        return m

    stub.chart_exclusion_mask = chart_exclusion_mask
    monkeypatch.setitem(sys.modules, "pattern", stub)
    m, notes = unbake.resolve_exclusion_mask((16, 32))
    assert m[-3:].all() and not m[:-3].any()
    assert "chart_exclusion_mask" in notes[0]


def test_a_wrongly_shaped_hook_result_is_refused_not_transposed(monkeypatch):
    stub = types.ModuleType("pattern")
    stub.exclusion_mask = lambda shape, lm=None, axis_order="s_major": np.zeros(
        (shape[1], shape[0]), dtype=bool)
    monkeypatch.setitem(sys.modules, "pattern", stub)
    m, notes = unbake.resolve_exclusion_mask((16, 32))
    assert not m.any()
    assert "expected" in notes[0] and "s-major" in notes[0]


# ---------------------------------------------------------------------------
# the handoff to module P
# ---------------------------------------------------------------------------

def test_copy_from_photo_states_the_layout_and_semantics_explicitly(monkeypatch):
    """The handshake must never rely on module P's auto-detection.

    ``axis_order='auto'`` is undecidable on a square chart and
    ``chart_semantics='auto'`` is a mean test; both are fine defaults for a
    human at a prompt and unacceptable for a pipeline, so unbake states them.
    """
    chart, spots, rgb, mask, info = _scene()
    stub = types.ModuleType("pattern")
    seen = {}

    def copy_from_chart(chart_img, **kw):
        seen["chart"] = chart_img
        seen.update(kw)
        return "individual:%s" % kw.get("identity")

    stub.copy_from_chart = copy_from_chart
    monkeypatch.setitem(sys.modules, "pattern", stub)
    out, res = unbake.copy_from_photo(
        rgb, mask, side="L", identity="sg0007",
        chart_shape=(96, 192), dorsal_point=_dorsal_point(mask))
    assert out == "individual:sg0007"
    assert seen["axis_order"] == "s_major"
    assert seen["chart_semantics"] == "albedo"
    assert seen["chart"].shape == (96, 192)
    assert seen["confidence"].shape == (96, 192)
    assert np.isfinite(seen["chart"]).all()          # NaNs never cross the API
    assert np.isfinite(seen["confidence"]).all()
    assert float(seen["confidence"].max()) > 0.8
    assert isinstance(res, unbake.UnbakeResult)


def test_copy_from_photo_without_module_p_names_the_fallback(monkeypatch):
    chart, spots, rgb, mask, info = _scene()
    monkeypatch.setitem(sys.modules, "pattern", None)
    with pytest.raises(RuntimeError, match="photo_to_chart"):
        unbake.copy_from_photo(rgb, mask, side="L", chart_shape=(48, 96),
                               dorsal_point=_dorsal_point(mask))


# ---------------------------------------------------------------------------
# normalisation and plumbing
# ---------------------------------------------------------------------------

def test_normalisation_removes_shading_and_countershading():
    chart, spots, rgb, mask, info = _scene(n_spots=60, seed=13)
    kw = dict(side="L", chart_shape=(128, 256))
    dp = _dorsal_point(mask)
    raw = unbake.photo_to_chart(rgb, mask, normalize=False, dorsal_point=dp, **kw)
    norm = unbake.photo_to_chart(rgb, mask, normalize=True, dorsal_point=dp, **kw)
    n_s, n_phi = raw.chart.shape
    S, P = np.meshgrid(*bake.chart_axes(n_s, n_phi), indexing="ij")
    src = bake.sample_chart(chart, S, P)
    ok = np.isfinite(raw.chart) & np.isfinite(norm.chart) & (raw.confidence > 0.5)
    c_raw = float(np.corrcoef(src[ok], raw.chart[ok])[0, 1])
    c_norm = float(np.corrcoef(src[ok], norm.chart[ok])[0, 1])
    print("correlation to source chart: raw %.4f, normalised %.4f" % (c_raw, c_norm))
    assert c_norm > c_raw
    # a normalised chart is an albedo multiplier: unmarked skin sits at ~1
    skin = ok & (src > 0.95)
    assert 0.93 < float(np.median(norm.chart[skin])) < 1.07


def test_local_half_width_tracks_the_rendered_radius():
    chart, spots, rgb, mask, info = _scene()
    res = unbake.photo_to_chart(rgb, mask, side="L", chart_shape=(64, 128),
                                dorsal_point=_dorsal_point(mask))
    s_st, r_st = info["radius_stations"]
    truth = np.interp(np.linspace(0.0, 1.0, len(res.radius_px)), s_st, r_st)
    truth_px = truth * info["px_per_unit"]
    mid = slice(len(truth_px) // 5, 4 * len(truth_px) // 5)
    rel = np.abs(res.radius_px[mid] - truth_px[mid]) / truth_px[mid]
    print("local half-width relative error over the mid body: mean %.3f max %.3f"
          % (rel.mean(), rel.max()))
    assert rel.mean() < 0.06


def test_side_must_be_L_or_R():
    chart, spots, rgb, mask, info = _scene()
    with pytest.raises(ValueError, match="side must be"):
        unbake.photo_to_chart(rgb, mask, side="left")


def test_photo_to_chart_is_deterministic():
    chart, spots, rgb, mask, info = _scene()
    kw = dict(side="L", chart_shape=(64, 128), dorsal_point=_dorsal_point(mask))
    a = unbake.photo_to_chart(rgb, mask, **kw)
    b = unbake.photo_to_chart(rgb, mask, **kw)
    assert np.array_equal(np.nan_to_num(a.chart, nan=-1),
                          np.nan_to_num(b.chart, nan=-1))
    assert np.array_equal(a.confidence, b.confidence)


def test_grayscale_and_uint8_inputs_are_accepted():
    chart, spots, rgb, mask, info = _scene()
    dp = _dorsal_point(mask)
    gray = bake.luminance(rgb)
    a = unbake.photo_to_chart(gray, mask, side="L", chart_shape=(64, 128),
                              dorsal_point=dp)
    b = unbake.photo_to_chart((rgb * 255).astype(np.uint8), mask, side="L",
                              chart_shape=(64, 128), dorsal_point=dp)
    assert np.isfinite(a.chart).any() and np.isfinite(b.chart).any()
    ok = np.isfinite(a.chart) & np.isfinite(b.chart)
    assert float(np.corrcoef(a.chart[ok], b.chart[ok])[0, 1]) > 0.98


# ---------------------------------------------------------------------------
# end to end across the module boundary, with the real pattern module
# ---------------------------------------------------------------------------

def test_a_real_generated_individual_survives_render_and_copy_back():
    """pattern -> chart -> lateral photo -> unbake -> pattern, closed loop.

    This is the owner's "copy a real individual" path with a KNOWN answer
    substituted for the real animal: module P generates an individual, the
    fixture renders a lateral photograph of it, unbake recovers a chart, and
    module P fits a fresh spot table to that chart.  The fitted spots must land
    on the generated ones wherever the view actually saw the skin.
    """
    pattern = pytest.importorskip("pattern")
    ind = pattern.randomize(seed=21)
    dark, src_spots = pattern.render_chart(ind, resolution=(192, 384))
    chart = bake.from_pattern_chart(dark)               # (n_s, n_phi) multiplier

    rgb, mask, info = fixtures.render_lateral_tube(
        chart, n_px=(220, 700), side="L", seed=2)
    got, res = unbake.copy_from_photo(
        rgb, mask, side="L", identity="sg_test",
        chart_shape=(160, 320), dorsal_point=_dorsal_point(mask))

    assert got.identity == "sg_test"
    assert got.provenance["axis_order"] == "s_major"
    assert got.provenance["semantics"] in ("albedo", "auto")
    fitted = got.spots
    assert len(fitted) > 5

    # score only the spots this view could actually resolve
    visible = src_spots[
        (np.sin(np.abs(src_spots["phi"])) > GRAZING_SIN)
        & (src_spots["phi"] > 0)
        & (src_spots["s"] > S_MARGIN[0]) & (src_spots["s"] < S_MARGIN[1])
        & (src_spots["rendered_darkness"] > 0.35)
    ]
    gt = np.column_stack([visible["s"], visible["phi"]])
    det = np.column_stack([fitted["s"], fitted["phi"]])
    recall, n_hit = _recall(gt, det, tol=0.035)
    print("closed loop: %d/%d generated spots refitted (recall %.3f) "
          "from %d fitted spots" % (n_hit, len(gt), recall, len(det)))
    assert len(gt) >= 8
    assert recall > 0.8


def test_the_pattern_module_exclusion_hook_is_actually_used():
    """The gill-slit arbitration, live: module P's mask must reach the chart."""
    pattern = pytest.importorskip("pattern")
    shape = (128, 256)
    chart, spots, rgb, mask, info = _scene()
    res = unbake.photo_to_chart(
        rgb, mask, side="L", chart_shape=shape,
        dorsal_point=_dorsal_point(mask))
    assert any("pattern." in n for n in res.notes)
    hook = getattr(pattern, "exclusion_mask", None) or pattern.chart_exclusion_mask
    excl = np.asarray(hook(shape, None, axis_order="s_major"), dtype=bool)
    assert excl.shape == shape
    assert np.all(res.confidence[excl] == 0.0)
    assert not np.isfinite(res.chart[excl]).any()
    print("module P exclusion covers %.1f%% of the chart" % (100 * excl.mean()))
