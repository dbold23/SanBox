"""Tests for occluders and water (nuisance.py).

Contracts under test:

  * a kelp occluder REDUCES the identity-mask area, and the occlusion mask
    covers EXACTLY that reduction -- the identity mask after equals the
    identity mask before minus the occlusion mask, as a set;
  * a second shark in the foreground behaves the same way and is never
    mistaken for the subject;
  * turbidity reduces contrast monotonically with distance, and the limit at
    infinite range is the veiling light;
  * every nuisance is a SEEDED PARAMETER OBJECT: same seed, same pixels;
    different seed, different pixels;
  * nuisance never edits the ground-truth masks.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import module_r_fixtures as F
import nuisance
import render


def _luma(rgb):
    return np.asarray(rgb, dtype=np.float64) @ np.array([0.2126, 0.7152, 0.0722])


# ---------------------------------------------------------------------------
# kelp
# ---------------------------------------------------------------------------

def test_kelp_ribbon_is_a_thin_double_sided_occluder():
    p = nuisance.KelpParams(n_segments=8, length=1.0, width=0.1)
    blade = nuisance.kelp_ribbon((0.0, 0.0, 0.0), (0, 0, 1), (1, 0, 0), p)
    assert blade.role == "occluder"
    assert blade.double_sided and not blade.casts_shadow
    assert not blade.has_chart                      # an occluder has no identity
    assert len(blade.faces) == 2 * p.n_segments
    v = blade.vertices
    assert v[:, 2].max() - v[:, 2].min() == pytest.approx(p.length, abs=1e-9)
    # every quad has area: the taper floor keeps the tip from collapsing
    n = p.n_segments + 1
    w = v[n:] - v[:n]                       # the width vector at each station
    widths = np.linalg.norm(w, axis=1)
    assert widths.min() > 0.2 * p.width
    assert widths.max() <= p.width + 1e-12
    # the twist really twists: the width vector turns along the blade
    cos = (w[0] @ w[-1]) / (widths[0] * widths[-1])
    assert cos < 0.9


def test_kelp_params_reject_a_curtain_behind_the_subject():
    with pytest.raises(ValueError):
        nuisance.KelpParams(depth_frac=1.0)
    with pytest.raises(ValueError):
        nuisance.OccluderPlacement(depth_frac=1.5)


def test_kelp_curtain_is_deterministic_under_seed():
    _, inst = F.subject()
    cam = F.side_camera(inst, resolution=(128, 128))
    a = nuisance.kelp_curtain(cam, seed=11)
    b = nuisance.kelp_curtain(cam, seed=11)
    c = nuisance.kelp_curtain(cam, seed=12)
    assert len(a) == len(b) == len(c) == nuisance.KelpParams().n_blades
    for x, y in zip(a, b):
        assert np.array_equal(x.vertices, y.vertices)
    assert not np.allclose(a[0].vertices, c[0].vertices)


def test_kelp_sits_between_the_camera_and_the_subject():
    _, inst = F.subject()
    cam = F.side_camera(inst, resolution=(128, 128))
    blades = nuisance.kelp_curtain(cam, nuisance.KelpParams(depth_frac=0.4), seed=2)
    body_z = cam.world_to_camera(inst.vertices)[:, 2]
    for blade in blades:
        z = cam.world_to_camera(blade.vertices)[:, 2]
        assert z.max() < body_z.max()
        assert z.mean() < body_z.mean()


def test_kelp_reduces_the_identity_mask_and_occlusion_covers_the_reduction():
    _, inst = F.subject()
    cam = F.side_camera(inst, resolution=(256, 256))
    light = F.key_light()
    clear = render.render([inst], cam, light=light, exclusion="auto")
    blades = nuisance.kelp_curtain(cam, nuisance.KelpParams(n_blades=7), seed=5)
    occl = render.render([inst] + blades, cam, light=light, exclusion="auto")

    lost = int(clear["identity"].sum()) - int(occl["identity"].sum())
    print("kelp: identity %d -> %d (-%d px), occlusion %d px"
          % (clear["identity"].sum(), occl["identity"].sum(), lost,
             occl["occlusion"].sum()))
    assert lost > 100                                    # it actually occludes
    # EXACTLY the reduction: no other mask moved, and every lost identity
    # pixel is an occluded one.
    assert np.array_equal(occl["identity"], clear["identity"] & ~occl["occlusion"])
    assert int((clear["identity"] & occl["occlusion"]).sum()) == lost
    # Kelp casts no shadow by default, so no BODY pixel changed shadow state.
    # (The full shadow mask does change: the blades have attached shadow of
    # their own, and they are covered geometry too.)
    skin = occl["visible_skin"]
    assert np.array_equal(occl["shadow"] & skin, clear["shadow"] & skin)
    # ... and the blades' own pixels are never identity pixels
    assert not (occl["identity"] & (occl["instance"] > 0)).any()


def test_kelp_that_casts_shadow_also_darkens_the_body():
    _, inst = F.subject()
    cam = F.side_camera(inst, resolution=(192, 192))
    light = F.key_light(direction=(0.0, 0.0, -1.0))
    params = nuisance.KelpParams(n_blades=8, depth_frac=0.5, casts_shadow=True,
                                 length=2.0, width=0.25)
    blades = nuisance.kelp_curtain(cam, params, seed=4)
    clear = render.render([inst], cam, light=light, exclusion=None)
    dappled = render.render([inst] + blades, cam, light=light, exclusion=None)
    assert dappled["cast_shadow"].sum() > 0
    assert dappled["shadow"].sum() > clear["shadow"].sum()


# ---------------------------------------------------------------------------
# a second shark as an occluder
# ---------------------------------------------------------------------------

def test_second_shark_occludes_without_becoming_the_subject():
    _, inst = F.subject()
    cam = F.side_camera(inst, resolution=(256, 256))
    light = F.key_light()
    clear = render.render([inst], cam, light=light, exclusion="auto")
    other = nuisance.place_occluder(
        inst, cam, nuisance.OccluderPlacement(depth_frac=0.55, offset_x=0.15,
                                              offset_y=0.0, yaw_deg=20.0,
                                              scale=0.9))
    assert other.role == "occluder"
    assert other.has_chart                     # it keeps its own parameterisation
    out = render.render([inst, other], cam, light=light, exclusion="auto")
    print("second shark: identity %d -> %d, occlusion %d px"
          % (clear["identity"].sum(), out["identity"].sum(),
             out["occlusion"].sum()))
    assert out["occlusion"].sum() > 200
    assert out["identity"].sum() < clear["identity"].sum()
    assert np.array_equal(out["identity"], clear["identity"] & ~out["occlusion"])
    # the occluder's pixels carry no chart GT and no identity
    on_other = out["instance"] == 1
    assert on_other.any()
    assert np.isnan(out["chart_s"][on_other]).all()
    assert not (out["identity"] & on_other).any()


def test_occluder_placement_sample_is_seeded():
    a = nuisance.OccluderPlacement.sample(3)
    b = nuisance.OccluderPlacement.sample(3)
    c = nuisance.OccluderPlacement.sample(4)
    assert repr(a) == repr(b) != repr(c)


# ---------------------------------------------------------------------------
# turbidity
# ---------------------------------------------------------------------------

def test_attenuation_from_visibility_uses_the_two_percent_convention():
    c = nuisance.attenuation_per_metre(4.5)
    assert c == pytest.approx(-np.log(0.02) / 4.5, rel=1e-12)
    assert c == pytest.approx(0.8693, abs=1e-3)
    # clearer water attenuates less
    assert nuisance.attenuation_per_metre(12.0) < nuisance.attenuation_per_metre(3.0)
    with pytest.raises(ValueError):
        nuisance.attenuation_per_metre(0.0)


def test_evidence_brackets_are_recorded_and_the_default_sits_inside_them():
    lo, hi = nuisance.VISIBILITY_M_BRACKET
    assert lo < nuisance.VISIBILITY_M_TYPICAL < hi
    assert nuisance.VISIBILITY_M_BEST_BRACKET[0] > hi
    assert "SECONDARY" in nuisance.__doc__ and "SCCOOS" in nuisance.__doc__
    # the tint is normalised so retinting does not change overall visibility
    assert float(np.mean(nuisance.CHANNEL_ATTENUATION_RATIOS)) == pytest.approx(1.0, abs=1e-9)


def test_turbidity_monotonically_reduces_contrast_with_distance():
    patch = np.zeros((2, 2, 3))
    patch[0] = 0.85
    patch[1] = 0.15
    params = nuisance.TurbidityParams(visibility_m=4.5)
    ranges = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 4.5, 6.0, 9.0])
    contrast, green = [], []
    for d in ranges:
        out = nuisance.apply_turbidity(patch, np.full((2, 2), d), params)
        contrast.append(float(_luma(out[0, 0]) - _luma(out[1, 0])))
        green.append(float(out[0, 0, 1] - out[1, 0, 1]))
    contrast = np.array(contrast)
    print("luma contrast vs range:", np.round(contrast, 4).tolist())
    assert (np.diff(contrast) < 0).all()
    assert (contrast > 0).all()
    # ... and per channel it is exactly Beer-Lambert, with no free parameter
    cg = params.broadband_c * params.channel_ratios[1]
    assert np.allclose(green, 0.70 * np.exp(-cg * ranges), rtol=1e-9)


def test_turbidity_limit_is_the_veiling_light_and_red_dies_first():
    rgb = np.full((1, 3, 3), 0.9)
    params = nuisance.TurbidityParams(visibility_m=4.0)
    far = nuisance.apply_turbidity(rgb, np.full((1, 3), np.inf), params)
    assert np.allclose(far[0, 0], params.veiling)
    near = nuisance.apply_turbidity(rgb, np.zeros((1, 3)), params)
    assert np.allclose(near, rgb)
    mid = nuisance.apply_turbidity(rgb, np.full((1, 3), 3.0), params)
    lost = 0.9 - mid[0, 0]
    assert lost[0] > lost[2] > lost[1]        # red attenuates fastest, green least
    with pytest.raises(ValueError):
        nuisance.apply_turbidity(rgb, np.zeros((5, 5)), params)


def test_turbidity_on_a_real_render_washes_the_animal_toward_the_water():
    _, inst = F.subject()
    near = F.side_camera(inst, resolution=(128, 128), distance=1.5)
    far = F.side_camera(inst, resolution=(128, 128), distance=5.0)
    params = nuisance.TurbidityParams(visibility_m=4.5)
    spread = []
    for cam in (near, far):
        out = render.render([inst], cam, light=F.key_light(), exclusion=None)
        fogged = nuisance.apply_turbidity(out["rgb"], out["depth"], params)
        body = _luma(fogged[out["visible_skin"]])
        spread.append(float(body.max() - body.min()))
    print("body luma spread: near %.4f, far %.4f" % tuple(spread))
    assert spread[1] < spread[0] * 0.6


def test_turbidity_params_sample_is_seeded_and_inside_the_bracket():
    a = nuisance.TurbidityParams.sample(9)
    b = nuisance.TurbidityParams.sample(9)
    assert a.visibility_m == b.visibility_m
    lo, hi = nuisance.VISIBILITY_M_BRACKET
    assert lo <= a.visibility_m <= hi
    assert nuisance.TurbidityParams.sample(10).visibility_m != a.visibility_m


# ---------------------------------------------------------------------------
# caustics, jitter, blur
# ---------------------------------------------------------------------------

def test_caustic_field_is_seeded_low_frequency_and_bounded():
    p = nuisance.CausticParams(n_waves=4)
    a = nuisance.caustic_field((64, 64), p, seed=1)
    b = nuisance.caustic_field((64, 64), p, seed=1)
    c = nuisance.caustic_field((64, 64), p, seed=2)
    assert np.array_equal(a, b) and not np.allclose(a, c)
    assert a.min() >= -1.0 - 1e-9 and a.max() <= 1.0 + 1e-9
    assert abs(float(a.mean())) < 0.25
    # Low frequency: an adjacent-pixel step is a few percent of the field's
    # own range, i.e. the pattern is many pixels wide, not per-pixel noise.
    step = float(np.abs(np.diff(a, axis=1)).mean())
    print("caustic mean adjacent step %.4f of range %.3f" % (step, np.ptp(a)))
    assert step < 0.05 * float(np.ptp(a))
    # and it flickers in time
    assert not np.allclose(a, nuisance.caustic_field((64, 64), p, seed=1, time=1.0))


def test_caustics_are_multiplicative_and_can_be_restricted_to_lit_skin():
    rgb = np.full((32, 32, 3), 0.5)
    mask = np.zeros((32, 32), dtype=bool)
    mask[:16] = True
    p = nuisance.CausticParams(contrast=0.2)
    out = nuisance.apply_caustics(rgb, p, seed=3, mask=mask)
    assert np.allclose(out[16:], rgb[16:])              # untouched outside
    assert not np.allclose(out[:16], rgb[:16])
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert abs(float(out[:16].mean() - 0.5)) < 0.05     # no net brightening


def test_camera_jitter_is_seeded_and_structure_preserving():
    _, inst = F.subject()
    cam = F.side_camera(inst, resolution=(96, 128))
    p = nuisance.CameraJitterParams()
    a = nuisance.jitter_camera(cam, p, seed=6)
    b = nuisance.jitter_camera(cam, p, seed=6)
    c = nuisance.jitter_camera(cam, p, seed=7)
    assert np.array_equal(a.eye, b.eye) and a.roll_deg == b.roll_deg
    assert not np.array_equal(a.eye, c.eye)
    assert a.resolution == cam.resolution and a.kind == cam.kind
    assert not np.array_equal(a.eye, cam.eye)
    assert a.roll_deg != cam.roll_deg
    still = nuisance.jitter_camera(
        cam, nuisance.CameraJitterParams(translate=0.0, aim=0.0, roll_deg=0.0),
        seed=6)
    assert np.allclose(still.eye, cam.eye) and still.roll_deg == cam.roll_deg
    # jitter really moves the image
    out_a = render.render([inst], cam, exclusion=None)
    out_b = render.render([inst], a, exclusion=None)
    assert not np.array_equal(out_a["visible_skin"], out_b["visible_skin"])


def test_motion_blur_spreads_a_point_along_its_angle_and_conserves_light():
    img = np.zeros((21, 21, 3))
    img[10, 10] = 1.0
    p = nuisance.MotionBlurParams(length_px=9, angle_deg=0.0)
    out = nuisance.motion_blur(img, p)
    assert out.sum() == pytest.approx(img.sum(), rel=1e-9)
    row = out[10, :, 0]
    col = out[:, 10, 0]
    assert (row > 0).sum() > (col > 0).sum()            # smeared horizontally
    vert = nuisance.motion_blur(img, nuisance.MotionBlurParams(9, 90.0))
    assert (vert[:, 10, 0] > 0).sum() > (vert[10, :, 0] > 0).sum()
    assert np.array_equal(nuisance.motion_blur(img, nuisance.MotionBlurParams(1)), img)


def test_average_frames_is_the_honest_motion_blur():
    a = np.zeros((4, 4, 3))
    b = np.ones((4, 4, 3))
    assert np.allclose(nuisance.average_frames([a, b]), 0.5)
    with pytest.raises(ValueError):
        nuisance.average_frames(a)


# ---------------------------------------------------------------------------
# the whole water column
# ---------------------------------------------------------------------------

def test_apply_water_changes_pixels_but_never_the_ground_truth():
    _, inst = F.subject()
    cam = F.side_camera(inst, resolution=(160, 160))
    blades = nuisance.kelp_curtain(cam, seed=8)
    out = render.render([inst] + blades, cam, light=F.key_light(), exclusion="auto")
    params = nuisance.WaterParams.sample(21)
    wet = nuisance.apply_water(out, params)

    assert not np.allclose(wet["rgb"], out["rgb"])
    for key in ("identity", "visible_skin", "shadow", "occlusion", "exclusion",
                "chart_s", "chart_phi", "depth", "instance"):
        x, y = np.asarray(out[key]), np.asarray(wet[key])
        assert np.array_equal(x, y, equal_nan=x.dtype.kind == "f"), key
    assert wet["meta"]["nuisance"]["seed"] == 21
    assert "caustics" in wet["meta"]["nuisance"]["order"]
    # the far background has become the veiling light
    bg = ~out["coverage"]
    assert np.allclose(wet["rgb"][bg], params.turbidity.veiling, atol=1e-6)
    # deterministic
    again = nuisance.apply_water(out, params)
    assert np.array_equal(wet["rgb"], again["rgb"])
    assert not np.allclose(nuisance.apply_water(out, params, seed=22)["rgb"],
                           wet["rgb"])


def test_a_full_nuisance_frame_still_fits_the_two_second_budget():
    _, inst = F.subject(n_stations=64, n_around=40)
    cam = F.side_camera(inst, resolution=(512, 512))
    light = F.key_light()
    scene = ([inst]
             + nuisance.kelp_curtain(cam, nuisance.KelpParams(n_blades=6), seed=1)
             + [nuisance.place_occluder(inst, cam,
                                        nuisance.OccluderPlacement.sample(2))])
    render.render(scene, cam, light=light, exclusion="auto")      # warm up
    t0 = time.time()
    out = render.render(scene, cam, light=light, exclusion="auto")
    wet = nuisance.apply_water(out, nuisance.WaterParams.sample(3))
    dt = time.time() - t0
    print("512x512 kelp + second shark + water: %.3f s" % dt)
    assert dt < 2.0
    assert wet["identity"].sum() > 0
