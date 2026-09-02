"""Tests for the calibration driver and the four render defects it fixes.

``calibrate.py`` is bookkeeping around three modules that already have tests
(``synth_render``, ``synth_features``, ``compare_features``), so what is tested
here is the bookkeeping -- a config dict round-trips through the override
merge, the objective is exactly 0 when the two sides are the same sample, and a
grid record is JSON-safe and carries the numbers the report quotes -- plus one
regression test per defect the bridge run found, because those are the changes
that would otherwise silently come back:

(a) ``frame_camera`` put the eye BELOW the animal for a positive elevation, so
    every frame looked up at the countershaded ventrum.
(b) the spot field faded out from |phi| ~ 1.3 rad, leaving the camera-facing
    flank bare, where a real sevengill is spotted to the ventral transition.
(c) ``s_target`` sat behind the frame's own left edge, so the snout was clipped
    on every render (bbox_width_frac 1.00 against a real median of 0.913).
(d) ``skin.tone_ventral`` above 1 clipped to white on the pectoral leading
    edge, drawing a hard white band across the fin root.

Run with the MAIN checkout venv (python 3.9)::

    "/Volumes/External Dive 2TB/projects/marine-cv/7Gill/.venv/bin/python" \\
        -W ignore -m pytest prototypes/06-spot-proxy/tests/test_calibrate.py -q

Nothing here renders a frame or loads a model; the whole file is sub-second
apart from the two ``pattern`` placement tests, which draw a few hundred spots.
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_P06 = os.path.dirname(_HERE)
if _P06 not in sys.path:
    sys.path.insert(0, _P06)

import calibrate as C  # noqa: E402
import compare_features as CF  # noqa: E402
import synth_render as SR  # noqa: E402

import exclusions  # noqa: E402  (prototype 05, on sys.path via synth_render)
import pattern  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _tube(n_s=160, n_phi=16, length=0.68, radius=0.055):
    """A straight tube: vertices, their chart ``s``.  Snout at ``+X``, ``s=0``.

    ``frame_camera`` only needs positions and ``s``; the real body is charted
    the same way (``s`` runs from the snout backwards, dorsal is ``+Z``).
    """
    s = np.linspace(0.0, 1.0, n_s)
    phi = np.linspace(-math.pi, math.pi, n_phi, endpoint=False)
    S, P = np.meshgrid(s, phi, indexing="ij")
    x = length * (0.5 - S)                      # s increases toward -X
    y = radius * np.sin(P)
    z = radius * np.cos(P)                      # phi = 0 is dorsal (+Z)
    return (np.column_stack([x.ravel(), y.ravel(), z.ravel()]),
            S.ravel().copy())


def _draw(**camera):
    """A ``SceneDraw`` carrying only what ``frame_camera`` reads."""
    cam = {"elevation_deg": 0.0, "azimuth_deg": 0.0, "roll_deg": 0.0,
           "width_frac": 0.85, "s_frame_max": 0.40, "s_target": 0.18,
           "fov_y_deg": 44.0, "resolution": [378, 504]}
    cam.update(camera)
    return SR.SceneDraw(pose={}, side="L", camera=cam, light={}, specular={},
                        background={}, occluders=[], degrade={})


def _side(per_image, pooled, n_records=10):
    """A ``compare_features.load_side``-shaped dict from plain arrays."""
    return {
        "path": "<test>", "n_records": n_records, "n_ok": n_records,
        "n_no_body": 0,
        "per_image": {t: {k: np.asarray(v, dtype=np.float64)
                          for k, v in per_image.items()} for t in CF.THRESHOLDS},
        "pooled": {t: {k: np.asarray(v, dtype=np.float64)
                       for k, v in pooled.items()} for t in CF.THRESHOLDS},
    }


def _identical_sides(seed=0, n=40):
    rng = np.random.default_rng(seed)
    per_image = {k: rng.normal(1.0, 0.2, size=n) for k in CF.PER_IMAGE_FEATURES}
    pooled = {k: rng.normal(0.03, 0.01, size=400) for k in CF.POOLED_FEATURES}
    return _side(per_image, pooled, n), _side(copy.deepcopy(per_image),
                                              copy.deepcopy(pooled), n)


# --------------------------------------------------------------------------- #
# 1. a config dict round-trips                                                 #
# --------------------------------------------------------------------------- #
def test_merge_overrides_is_a_deep_merge_that_mutates_nothing():
    a = {"camera": {"elevation_deg": [0.0, 35.0], "roll_deg": [-30.0, 30.0]}}
    b = {"camera": {"elevation_deg": [0.0, 20.0]}, "pattern": {"n_spots": 1400}}
    a_before = copy.deepcopy(a)
    b_before = copy.deepcopy(b)

    out = C.merge_overrides(a, b)

    assert out["camera"]["elevation_deg"] == [0.0, 20.0]     # b wins
    assert out["camera"]["roll_deg"] == [-30.0, 30.0]        # a survives
    assert out["pattern"]["n_spots"] == 1400
    assert a == a_before and b == b_before                   # inputs untouched
    out["camera"]["roll_deg"].append(99.0)                   # deep copy, not alias
    assert a["camera"]["roll_deg"] == [-30.0, 30.0]


def test_merge_overrides_of_nothing_is_the_shipped_default():
    assert C.config_from(C.merge_overrides(None, {})) == SR.load_config()


def test_config_from_touches_one_leaf_and_leaves_its_siblings_alone():
    ov = {"pattern": {"radius_median": 0.0051}}
    cfg = C.config_from(ov)

    assert cfg["pattern"]["radius_median"] == 0.0051
    for key, value in SR.DEFAULT_CONFIG["pattern"].items():
        if key != "radius_median":
            assert cfg["pattern"][key] == value
    assert cfg["camera"] == SR.DEFAULT_CONFIG["camera"]
    # and the module default itself is untouched
    assert SR.DEFAULT_CONFIG["pattern"]["radius_median"] == 0.0030


def test_config_from_survives_a_json_round_trip():
    """The winning override is written to best.json and read back by --config."""
    ov = {"camera": {"elevation_deg": [0.0, 22.0], "s_target": [0.12, 0.20]},
          "pattern": {"n_spots": 1400, "edge_softness": 0.55}}
    back = json.loads(json.dumps(ov))
    assert C.config_from(back) == C.config_from(ov)
    assert json.loads(json.dumps(C.config_from(ov))) == C.config_from(ov)


# --------------------------------------------------------------------------- #
# 2. the objective is 0 on identical summaries                                 #
# --------------------------------------------------------------------------- #
def test_objective_is_exactly_zero_when_both_sides_are_the_same_sample():
    real, synth = _identical_sides()
    summary = CF.compare(real, synth)

    assert CF.objective(summary) == 0.0
    assert CF.objective(summary, threshold="0.40") == 0.0
    assert CF.geometry_objective(summary) == 0.0
    for thr in ("0.25", "0.40", "0.50"):
        block = summary["thresholds"][thr]
        for group in ("per_image", "pooled"):
            for key, entry in block[group].items():
                assert entry["ks_D"] == 0.0, (thr, group, key)


def test_record_row_reports_that_zero_and_stays_json_safe():
    real, synth = _identical_sides()
    summary = CF.compare(real, synth)
    det_summary = {"pooled": {"0.25": {"precision": 1.0, "recall": 0.5,
                                       "n_det": 10, "n_gt": 20, "tp": 10,
                                       "fp": 0}}}
    row = C.record_row("zero", {"pattern": {"n_spots": 7}}, summary,
                       det_summary, {"n_frames": 4}, corpus="<none>")

    assert row["objective"] == 0.0
    assert row["objective_040"] == 0.0
    assert row["geometry_objective"] == 0.0
    assert row["name"] == "zero"
    assert row["overrides"] == {"pattern": {"n_spots": 7}}
    assert row["n_frames"] == 4
    assert row["detector"]["0.25"]["precision"] == 1.0
    assert row["body"] == {"n_records": 0, "n_with_body": 0, "n_no_body": 0,
                           "body_conf_q": [], "body_conf_median": None}
    assert json.loads(json.dumps(row, default=float))["objective"] == 0.0


def test_record_row_ks_block_holds_every_feature_at_every_threshold():
    real, synth = _identical_sides()
    row = C.record_row("k", {}, CF.compare(real, synth), {"pooled": {}},
                       {"n_frames": 1}, corpus="<none>")

    assert set(row["ks"]) == {"0.25", "0.40", "0.50"}
    for block in row["ks"].values():
        assert set(block["per_image"]) == set(CF.PER_IMAGE_FEATURES)
        assert set(block["pooled"]) == set(CF.POOLED_FEATURES)
        assert all(isinstance(v, float) for v in block["per_image"].values())


def test_a_shifted_sample_scores_above_zero_so_the_zero_is_not_vacuous():
    real, synth = _identical_sides()
    for t in CF.THRESHOLDS:
        synth["pooled"][t]["size"] = synth["pooled"][t]["size"] + 0.20
    summary = CF.compare(real, synth)

    # one of the three pooled features is fully separated (the shift is 20
    # sigma), the other two and all four per-image features are still the same
    # sample, so the objective is exactly 0.5 * (1/3) by its own definition.
    assert summary["thresholds"]["0.25"]["pooled"]["size"]["ks_D"] == 1.0
    assert summary["thresholds"]["0.25"]["pooled"]["nn"]["ks_D"] == 0.0
    detail = CF.objective(summary, detail=True)
    assert detail["per_image_mean_D"] == 0.0
    assert detail["pooled_mean_D"] == pytest.approx(1.0 / 3.0)
    assert CF.objective(summary) == pytest.approx(0.5 / 3.0)


def test_table_sorts_by_objective_and_prints_one_row_per_config():
    real, synth = _identical_sides()
    good = C.record_row("good", {}, CF.compare(real, synth), {"pooled": {}},
                        {"n_frames": 2}, corpus="<none>")
    for t in CF.THRESHOLDS:
        synth["pooled"][t]["size"] = synth["pooled"][t]["size"] + 0.05
    bad = C.record_row("bad", {}, CF.compare(real, synth), {"pooled": {}},
                       {"n_frames": 2}, corpus="<none>")

    lines = C.table([bad, good, {"name": "boom", "error": "nope"}]).splitlines()

    assert len(lines) == 4                       # header, rule, two configs
    assert lines[2].startswith("| good ")        # lower objective first
    assert lines[3].startswith("| bad ")
    assert "boom" not in "\n".join(lines)        # errored configs are not rows


# --------------------------------------------------------------------------- #
# 3. defect (a): the camera elevation sign                                     #
# --------------------------------------------------------------------------- #
def test_elevation_puts_the_eye_above_the_animal():
    """A POSITIVE elevation must raise the eye in +Z (dorsal), not lower it.

    The pre-fix code negated the Rodrigues angle, so every frame with a
    positive ``elevation_deg`` looked UP at the ventrum -- the measured cause
    of "spots only in a dorsal band near the silhouette", because the ventrum
    is where countershading takes the spots away.
    """
    verts, s = _tube()
    for elev in (10.0, 30.0, 50.0):
        camera, _, target, direction = SR.frame_camera(verts, s, _draw(
            elevation_deg=elev))
        assert direction[2] > 0.0, elev
        assert camera.eye[2] > target[2], elev
        assert direction[2] == pytest.approx(math.sin(math.radians(elev)),
                                             abs=1e-6)


def test_zero_elevation_is_a_level_lateral_view_on_both_sides():
    verts, s = _tube()
    left = SR.frame_camera(verts, s, _draw(elevation_deg=0.0))[3]
    draw_r = _draw(elevation_deg=0.0)
    draw_r = draw_r._replace(side="R")
    right = SR.frame_camera(verts, s, draw_r)[3]

    assert left[2] == pytest.approx(0.0, abs=1e-9)
    assert right[2] == pytest.approx(0.0, abs=1e-9)
    assert left[1] * right[1] < 0.0              # opposite flanks
    assert np.allclose(left, -right, atol=1e-9)


def test_the_before_override_replays_the_pre_fix_camera_frame_for_frame():
    """``[0, -50]`` negates each draw; ``[-50, 0]`` would only mirror the sample.

    ``synth_render._u`` is ``low + (high - low) * u`` (byte-identical to
    ``rng.uniform`` on a forward range), so a reversed range flips the sign of
    the value drawn from the same variate.  That is what makes the BEFORE
    baseline the same frames the old code rendered rather than a fresh draw.
    """
    assert C.BEFORE_OVERRIDE["camera"]["elevation_deg"] == [0.0, -50.0]

    fwd = np.random.default_rng(7)
    rev = np.random.default_rng(7)
    for _ in range(8):
        a = SR._u(fwd, [0.0, 50.0])
        b = SR._u(rev, [0.0, -50.0])
        assert b == -a

    ref = np.random.default_rng(7)
    chk = np.random.default_rng(7)
    for _ in range(8):
        assert SR._u(ref, [3.0, 50.0]) == chk.uniform(3.0, 50.0)


def test_span_takes_a_scalar_or_a_range():
    """``camera.s_target`` was a bare number; an old config.json must still load."""
    rng = np.random.default_rng(0)
    assert SR._span(rng, 0.25) == 0.25
    assert SR._span(rng, (0.25, 0.25)) == 0.25
    values = [SR._span(rng, [0.10, 0.20]) for _ in range(64)]
    assert all(0.10 <= v <= 0.20 for v in values)
    assert len(set(values)) > 1


# --------------------------------------------------------------------------- #
# 4. defect (b): spots must reach the flank                                    #
# --------------------------------------------------------------------------- #
def test_spot_countershading_keeps_the_whole_flank_dark():
    """Full amplitude out to |phi| = 2.3 rad, fading only near the ventrum.

    The pre-fix onset was 1.30 rad -- 75 deg off the dorsal midline -- so the
    flank the camera actually sees was already half faded.
    """
    p = SR.DEFAULT_CONFIG["pattern"]
    w = exclusions.countershading_weight_at(
        np.array([0.0, 1.0, 1.6, 2.0, 2.3, 2.7, math.pi]),
        phi_onset=p["cs_phi_onset"], phi_full=p["cs_phi_full"],
        floor=p["cs_floor"])

    assert p["cs_phi_onset"] >= 2.3
    assert w[:5] == pytest.approx(np.ones(5), abs=1e-12)   # flank at full dark
    assert w[5] < 0.85                                     # fading by 2.7
    assert w[6] == pytest.approx(p["cs_floor"], abs=1e-12)  # ventral midline


def test_the_placement_prior_no_longer_starves_the_flank():
    """``dorsal_exponent`` 0.80 thinned |phi| = 2.3 to 24% of the dorsal rate."""
    cfg = SR.load_config()
    params = pattern.PatternParams(
        dorsal_exponent=cfg["pattern"]["dorsal_exponent"],
        countershading={"phi_onset": cfg["pattern"]["cs_phi_onset"],
                        "phi_full": cfg["pattern"]["cs_phi_full"],
                        "floor": cfg["pattern"]["cs_floor"]})
    old = params.replace(dorsal_exponent=0.80,
                         countershading={"phi_onset": 1.30, "phi_full": 2.85,
                                         "floor": 0.05})

    phi = np.array([2.3])
    new_w = float(pattern._density_weight(phi, params)[0]
                  / pattern._density_weight(np.zeros(1), params)[0])
    old_w = float(pattern._density_weight(phi, old)[0]
                  / pattern._density_weight(np.zeros(1), old)[0])

    assert old_w < 0.25
    assert new_w > 0.55
    assert new_w > 2.0 * old_w


def test_pattern_context_applies_every_pattern_knob_including_the_globals():
    """``pattern._EDGE_SOFTNESS`` is a module constant, set as an attribute."""
    before = pattern._EDGE_SOFTNESS
    try:
        cfg = SR.load_config(overrides={"pattern": {
            "n_spots": 40, "n_common": 3, "radius_median": 0.0051,
            "radius_log_sigma": 0.61, "min_sep": 0.0158, "darkness_mean": 0.72,
            "darkness_sigma": 0.26, "darkness_min": 0.25, "darkness_max": 0.99,
            "common_darkness": 0.31, "dorsal_exponent": 0.29,
            "ecc_sigma": 0.55, "ecc_max": 3.2, "cs_phi_onset": 2.31,
            "cs_phi_full": 3.06, "cs_floor": 0.07, "edge_softness": 0.55}})
        ctx = SR.pattern_context(cfg)

        assert pattern._EDGE_SOFTNESS == 0.55
        p = ctx.params
        assert (p.n_spots_target, p.n_common) == (40, 3)
        assert p.radius_median == 0.0051 and p.radius_log_sigma == 0.61
        assert p.min_sep == 0.0158
        assert (p.darkness_mean, p.darkness_sigma) == (0.72, 0.26)
        assert (p.darkness_min, p.darkness_max) == (0.25, 0.99)
        assert p.common_darkness == 0.31
        assert p.dorsal_exponent == 0.29
        assert (p.ecc_sigma, p.ecc_max) == (0.55, 3.2)
        assert p.countershading == {"phi_onset": 2.31, "phi_full": 3.06,
                                    "floor": 0.07}
        # drop_regions removes the gill-slit SCORING mask from PLACEMENT only
        assert "gill_slits" not in {r.name for r in ctx.regions}
    finally:
        pattern._EDGE_SOFTNESS = before


def test_edge_softness_widens_the_stamp_ramp():
    """A softer edge must spread the same spot over more chart pixels."""
    before = pattern._EDGE_SOFTNESS
    try:
        totals = []
        for softness in (0.25, 0.85):
            pattern._EDGE_SOFTNESS = softness
            img = np.zeros((128, 256))
            s_axis, phi_axis = pattern.chart_axes((128, 256))
            pattern._stamp_ellipse(img, 0.5, 0.0, 0.02, 0.02, 0.0, 0.9,
                                   s_axis, phi_axis, 0.085)
            totals.append((float((img > 0).sum()), float((img > 0.85).sum()),
                           float(img.sum())))
        # the SUPPORT is the same either way -- coverage is
        # clip((1 - dn) / softness, 0, 1) and dies at dn = 1 regardless -- so
        # what softness moves is how much of that support reaches full
        # amplitude, which is exactly the hard-edged look the real spots lack.
        assert totals[1][0] == totals[0][0]      # same pixels touched
        assert totals[1][1] < 0.5 * totals[0][1]  # far fewer at full darkness
        assert totals[1][2] < totals[0][2]       # and less ink overall
    finally:
        pattern._EDGE_SOFTNESS = before


# --------------------------------------------------------------------------- #
# 5. defect (c): the snout has to land inside the frame                        #
# --------------------------------------------------------------------------- #
def _snout_margin(s_target, s_frame_max, width_frac):
    """Signed image-fraction from the left edge to the snout.

    The framed set is ``s <= s_frame_max`` and its projected horizontal extent
    is ``width_frac`` of the image, centred on ``s_target``.  The snout (s = 0)
    therefore sits ``s_target / s_frame_max * width_frac`` of a width left of
    centre, so it is inside the frame when that is under 0.5.
    """
    return 0.5 - (s_target / s_frame_max) * width_frac


def test_the_shipped_framing_spans_both_clipped_and_unclipped_snouts():
    """The real corpus is a MIXTURE: bbox_width_frac median 0.913, 21% at 1.00.

    So the shipped ranges must not put the snout inside on every draw either;
    what they must not do is clip it on every draw, which is what the pre-fix
    values did.
    """
    cam = SR.DEFAULT_CONFIG["camera"]
    best = _snout_margin(cam["s_target"][0], cam["s_frame_max"][1],
                         cam["width_frac"][0])
    worst = _snout_margin(cam["s_target"][1], cam["s_frame_max"][0],
                          cam["width_frac"][1])
    assert best > 0.0, "no draw in the shipped range keeps the snout in frame"
    assert worst < 0.0, "every draw keeps it in; the real corpus clips 21%"

    rng = np.random.default_rng(0)
    inside = np.mean([_snout_margin(SR._span(rng, cam["s_target"]),
                                    SR._u(rng, cam["s_frame_max"]),
                                    SR._u(rng, cam["width_frac"])) > 0.0
                      for _ in range(4000)])
    assert 0.25 < inside < 0.95, inside


def test_the_pre_fix_framing_clipped_the_snout_on_every_draw():
    """s_target 0.25 against s_frame_max 0.26-0.38 at width_frac 0.80-0.95."""
    best = _snout_margin(0.25, 0.38, 0.80)
    assert best < 0.0


def test_frame_camera_keeps_the_snout_in_frame_for_the_shipped_ranges():
    """Not arithmetic this time -- project the tube and look at the pixels."""
    verts, s = _tube()
    cam = SR.DEFAULT_CONFIG["camera"]
    draw = _draw(s_target=cam["s_target"][0], s_frame_max=cam["s_frame_max"][1],
                 width_frac=cam["width_frac"][0], elevation_deg=12.0)
    camera, _, _, _ = SR.frame_camera(verts, s, draw)

    px, py, pz = camera.project(verts[s < 0.01])
    ok = np.isfinite(px) & (pz > camera.near)
    assert ok.any()
    assert px[ok].min() > 0.0
    assert px[ok].max() < camera.resolution[1]


# --------------------------------------------------------------------------- #
# 6. defect (d): the ventral tone must not clip to white                       #
# --------------------------------------------------------------------------- #
def test_the_ventral_tone_multiplier_never_exceeds_one():
    """A multiplier above 1 blows out on the pectoral's rounded leading edge.

    ``fin_tone_from_normal`` takes a fin pixel's tone angle from
    ``arccos(n_z)``, which sweeps the whole 0..pi ramp within a few pixels of
    the blade edge, so any value above 1 there clips to white and draws a hard
    band along the fin root.
    """
    cfg = SR.load_config()
    tone = SR.tone_multiplier(np.linspace(0.0, math.pi, 257), cfg)

    assert tone.max() <= 1.0
    assert cfg["skin"]["tone_ventral"] <= 1.0
    assert tone[0] == pytest.approx(cfg["skin"]["tone_dorsal"])
    assert tone[-1] == pytest.approx(cfg["skin"]["tone_ventral"], rel=1e-6)
    assert np.all(np.diff(tone) >= -1e-12)       # monotone dorsal -> ventral


def test_the_pre_fix_tone_did_clip():
    cfg = SR.load_config(overrides={"skin": {"tone_ventral": 1.18,
                                             "tone_dorsal": 0.27}})
    assert SR.tone_multiplier(np.array([math.pi]), cfg)[0] > 1.0


def test_the_skin_tint_is_brown_purple_not_neutral():
    """R above B above G: the mauve-brown cast of the real photographs."""
    r, g, b = SR.DEFAULT_CONFIG["skin"]["tint"]
    assert r > b > g


# --------------------------------------------------------------------------- #
# 7. the driver's own plumbing                                                 #
# --------------------------------------------------------------------------- #
def test_stage_job_builders_produce_runnable_kwargs():
    jobs = C.stage1_jobs({"camera": {"fov_y_deg": 40.0}}, root="/tmp/x", seed=3,
                         n_individuals=2, sightings=1)

    assert len(jobs) == len(C.STAGE1)
    assert {j["name"] for j in jobs} == {n for n, _ in C.STAGE1}
    for job in jobs:
        assert job["overrides"]["camera"]["fov_y_deg"] == 40.0   # base applied
        assert job["seed"] == 3 and job["n_individuals"] == 2
    # the candidate wins over the base where they collide
    named = {j["name"]: j["overrides"] for j in jobs}
    assert named["s05_elev_low"]["camera"]["elevation_deg"] == [0.0, 22.0]
    assert named["s00_fixed"]["camera"] == {"fov_y_deg": 40.0}


def test_stage2_candidates_only_touch_pattern_skin_background_or_occluders():
    for name, ov in C.STAGE2:
        assert set(ov) <= {"pattern", "skin", "background", "occluders"}, name


def test_stage1_candidates_only_touch_the_camera():
    for name, ov in C.STAGE1:
        assert set(ov) <= {"camera"}, name


@pytest.mark.skipif(not os.path.exists(C.REAL_DETECTIONS),
                    reason="results/real/detections.jsonl not built")
def test_real_side_caches_and_reloads_identically(tmp_path):
    cache = str(tmp_path / "real_side.pkl")
    first = C.real_side(C.REAL_DETECTIONS, cache)
    assert os.path.exists(cache)
    second = C.real_side(C.REAL_DETECTIONS, cache)

    assert first["n_records"] == second["n_records"]
    assert first["n_ok"] == second["n_ok"]
    for t in CF.THRESHOLDS:
        for k in CF.PER_IMAGE_FEATURES:
            a, b = first["per_image"][t][k], second["per_image"][t][k]
            assert np.array_equal(a, b, equal_nan=True)
        for k in CF.POOLED_FEATURES:
            assert np.array_equal(first["pooled"][t][k], second["pooled"][t][k])
