"""Tests for the software renderer (render.py).

The contracts under test, in the order the module docstring states them:

  * the mask algebra -- identity = visible_skin AND NOT (exclusion, shadow,
    occlusion) -- holds as a SET IDENTITY, not approximately;
  * shadows: with no light occluder the shadow mask is EXACTLY the attached
    set (``ndotl <= 0``); a blocker between light and body produces a
    non-empty cast shadow, and only on lit-facing pixels;
  * the chart ground truth really is ground truth: at the pixel a known
    vertex projects into, ``chart_s``/``chart_phi`` reproduce that vertex's
    own ``(s, phi)`` to interpolation tolerance;
  * determinism, and the < 2 s/frame budget at 512.

Tolerances are MEASUREMENTS: the numbers asserted sit a margin below what the
reporting tests print.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import module_r_fixtures as F
import render


# ---------------------------------------------------------------------------
# camera + rasteriser basics
# ---------------------------------------------------------------------------

def test_camera_projects_its_own_target_to_the_frame_centre():
    for kind in ("ortho", "pinhole"):
        cam = render.Camera(eye=(0.0, 3.0, 0.0), target=(0.0, 0.0, 0.0),
                            resolution=(128, 256), kind=kind, ortho_height=2.0)
        px, py, pz = cam.project(cam.target)
        assert px == pytest.approx(0.5 * 256 - 0.5, abs=1e-9)
        assert py == pytest.approx(0.5 * 128 - 0.5, abs=1e-9)
        assert pz == pytest.approx(3.0, rel=1e-12)


def test_camera_basis_is_orthonormal_and_roll_rotates_the_image():
    cam = render.Camera(eye=(1.0, 4.0, 0.5), target=(0.0, 0.0, 0.0),
                        resolution=(64, 64), roll_deg=17.0)
    B = np.stack([cam.right, cam.up, cam.forward])
    assert np.abs(B @ B.T - np.eye(3)).max() < 1e-12
    level = cam.replace(roll_deg=0.0)
    p = cam.target + level.up * 0.3        # straight "up" in the LEVEL frame
    cx, cy = 0.5 * 64 - 0.5, 0.5 * 64 - 0.5
    px0, py0, _ = level.project(p)
    px1, py1, _ = cam.project(p)
    r0 = np.hypot(px0 - cx, py0 - cy)
    r1 = np.hypot(px1 - cx, py1 - cy)
    assert r1 == pytest.approx(r0, rel=1e-9)          # a roll is a rotation ...
    ang = np.degrees(np.arctan2(px1 - cx, -(py1 - cy)))
    assert ang == pytest.approx(17.0, abs=1e-6)       # ... by exactly 17 deg


def test_depth_is_camera_space_z_and_background_is_infinite():
    _, inst = F.subject()
    cam = F.side_camera(inst, resolution=(96, 96), distance=2.5)
    out = render.render([inst], cam, exclusion=None)
    d = out["depth"]
    assert np.isinf(d[~out["coverage"]]).all()
    body = d[out["coverage"]]
    # the animal's nearest surface is one body radius closer than its axis
    assert body.min() == pytest.approx(2.5 - 0.16, abs=0.02)
    assert body.max() < 2.5 + 0.16 + 1e-9


def test_pinhole_foreshortens_and_ortho_does_not():
    tube, inst = F.subject()
    v = inst.vertices
    near = v[np.argmin(np.linalg.norm(v - np.array([0.0, 3.0, 0.0]), axis=1))]
    ortho = F.side_camera(inst, resolution=(128, 128), kind="ortho")
    pin = F.side_camera(inst, resolution=(128, 128), kind="pinhole")
    # the same world offset subtends more pixels when it is nearer the pinhole
    def span(cam, base):
        a = cam.project(base)[0]
        b = cam.project(base + np.array([0.2, 0.0, 0.0]))[0]
        return abs(b - a)
    assert span(ortho, near) == pytest.approx(span(ortho, near + np.array([0.0, -0.3, 0.0])), rel=1e-9)
    assert span(pin, near) > span(pin, near + np.array([0.0, -0.3, 0.0])) * 1.05


def test_instance_mask_indexes_the_instance_list():
    _, inst = F.subject()
    cam = F.side_camera(inst, resolution=(128, 128))
    blocker = F.light_blocker(inst, F.key_light(), offset=-1.0)   # toward camera
    out = render.render([inst, blocker], cam, light=F.key_light(), exclusion=None)
    ids = set(np.unique(out["instance"]).tolist())
    assert ids <= {-1, 0, 1}
    assert (out["instance"] == 0).any() and (out["instance"] == 1).any()
    assert (out["instance"][~out["coverage"]] == -1).all()
    assert np.allclose(out["rgb"][~out["coverage"]], render.BACKGROUND_RGB)


def test_texture_is_sampled_not_invented():
    """A flat-coloured texture must come back as that colour (ambient-only)."""
    tube, inst = F.subject(textured=False)
    tex = np.zeros((16, 32, 3))
    tex[..., 0] = 0.8
    inst = render.Instance.from_uv_tube(tube, texture=tex, name="flat")
    cam = F.side_camera(inst, resolution=(96, 96))
    light = render.DirectionalLight(direction=(0.0, -1.0, 0.0), intensity=0.0,
                                    ambient=0.5)
    out = render.render([inst], cam, light=light, shadows=False, exclusion=None)
    body = out["rgb"][out["visible_skin"]]
    assert np.allclose(body[:, 0], 0.4, atol=1e-9)      # 0.8 albedo * 0.5 ambient
    assert np.allclose(body[:, 1:], 0.0, atol=1e-9)


def test_sample_texture_clamps_and_survives_nan():
    tex = np.linspace(0.0, 1.0, 8)[None, :, None] * np.ones((4, 1, 3))
    got = render.sample_texture(tex, np.array([[-5.0, 0.5], [5.0, 0.5],
                                               [np.nan, np.nan]]))
    assert got[0, 0] == pytest.approx(tex[0, 0, 0])
    assert got[1, 0] == pytest.approx(tex[0, -1, 0])
    assert np.allclose(got[2], 0.5)


# ---------------------------------------------------------------------------
# chart ground truth
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["ortho", "pinhole"])
def test_chart_gt_at_projected_vertices_matches_the_vertex(kind):
    tube, inst = F.subject(n_stations=48, n_around=32)
    cam = F.side_camera(inst, resolution=(384, 384), kind=kind)
    out = render.render([inst], cam, exclusion=None)
    px, py, _ = cam.project(inst.vertices)
    xi = np.rint(px).astype(int)
    yi = np.rint(py).astype(int)
    keep = F.front_facing_vertices(inst, cam)
    keep = keep[(xi[keep] >= 0) & (xi[keep] < 384) & (yi[keep] >= 0) & (yi[keep] < 384)]
    got_s = out["chart_s"][yi[keep], xi[keep]]
    got_p = out["chart_phi"][yi[keep], xi[keep]]
    ok = np.isfinite(got_s) & np.isfinite(got_p)
    assert ok.all() and ok.sum() > 300
    ds = np.abs(got_s[ok] - inst.vertex_s[keep][ok])
    dp = np.abs((got_p[ok] - inst.vertex_phi[keep][ok] + np.pi) % (2 * np.pi) - np.pi)
    ring_step = 2 * np.pi / 32          # one face of the fixture, in phi
    # MEASURED: ortho max|ds| 0.0015 / max|dphi| 0.129; pinhole 0.0019 / 0.098.
    print("chart GT (%s): n=%d  max|ds|=%.5f  max|dphi|=%.5f (ring step %.3f)"
          % (kind, ok.sum(), ds.max(), dp.max(), ring_step))
    assert ds.max() < 0.004             # ~1.5 px worth of s at this framing
    assert dp.max() < ring_step         # under one face of the tube


def test_chart_maps_are_nan_off_the_body_and_in_range_on_it():
    _, inst = F.subject()
    cam = F.side_camera(inst, resolution=(128, 128))
    out = render.render([inst], cam, exclusion=None)
    off = ~out["coverage"]
    assert np.isnan(out["chart_s"][off]).all()
    assert np.isnan(out["chart_phi"][off]).all()
    on = out["visible_skin"]
    assert (out["chart_s"][on] >= -1e-9).all() and (out["chart_s"][on] <= 1 + 1e-9).all()
    assert (np.abs(out["chart_phi"][on]) <= np.pi + 1e-9).all()
    # the LEFT-flank camera sees the animal's left: phi > 0 dominates
    assert np.nanmedian(out["chart_phi"][on]) > 1.0


def test_occluder_without_chart_coords_contributes_no_chart_gt():
    _, inst = F.subject()
    cam = F.side_camera(inst, resolution=(96, 96))
    blk = F.light_blocker(inst, F.key_light(), offset=-1.0)
    out = render.render([inst, blk], cam, light=F.key_light(), exclusion=None)
    on_blocker = out["instance"] == 1
    assert on_blocker.any()
    assert np.isnan(out["chart_s"][on_blocker]).all()


# ---------------------------------------------------------------------------
# shadows
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("direction", [(0.0, 0.0, -1.0), (0.3, 0.4, -1.0),
                                       (-0.8, 0.2, -0.5)])
def test_without_a_light_occluder_shadow_is_exactly_the_attached_set(direction):
    """The convex fixture body must self-shadow NOWHERE.

    This is the acne test: a naive depth comparison marks thousands of lit
    pixels as cast-shadowed at grazing angles.  See the bias constants in
    render.py.
    """
    _, inst = F.subject()
    cam = F.side_camera(inst, resolution=(256, 256))
    light = render.DirectionalLight(direction=direction, ambient=0.2)
    out = render.render([inst], cam, light=light, exclusion=None)
    attached = out["coverage"] & (out["ndotl"] <= 0.0)
    print("false cast-shadow pixels: %d" % out["cast_shadow"].sum())
    assert out["cast_shadow"].sum() == 0
    assert np.array_equal(out["shadow"], attached)
    assert attached.sum() > 0


def test_a_blocker_between_light_and_body_casts_a_shadow_on_the_lit_side():
    _, inst = F.subject()
    cam = F.side_camera(inst, resolution=(256, 256))
    light = F.key_light(direction=(0.0, 0.0, -1.0))
    clear = render.render([inst], cam, light=light, exclusion=None)
    blk = F.light_blocker(inst, light, offset=0.6, half=0.3)
    shaded = render.render([inst, blk], cam, light=light, exclusion=None)
    cast = shaded["cast_shadow"]
    print("cast-shadow pixels: %d of %d body pixels"
          % (cast.sum(), shaded["visible_skin"].sum()))
    assert cast.sum() > 200
    # cast shadow only ever lands on lit-facing geometry ...
    assert (shaded["ndotl"][cast] > 0.0).all()
    # ... it is a strict addition to the attached set ...
    assert np.array_equal(shaded["attached_shadow"], clear["attached_shadow"])
    assert (shaded["shadow"] & ~clear["shadow"]).sum() == cast.sum()
    # ... and shadowed skin is darker than the same skin was when lit
    was_lit = cast & clear["visible_skin"]
    assert (shaded["rgb"][was_lit].sum(axis=1) < clear["rgb"][was_lit].sum(axis=1)).all()
    # the blocker is out of the camera's way: no occlusion, only shadow
    assert shaded["occlusion"].sum() == 0


def test_shadows_can_be_switched_off():
    _, inst = F.subject()
    cam = F.side_camera(inst, resolution=(128, 128))
    light = F.key_light(direction=(0.0, 0.0, -1.0))
    blk = F.light_blocker(inst, light, offset=0.6, half=0.3)
    out = render.render([inst, blk], cam, light=light, shadows=False, exclusion=None)
    assert out["cast_shadow"].sum() == 0
    assert np.array_equal(out["shadow"], out["attached_shadow"])


def test_a_non_casting_instance_makes_no_shadow():
    _, inst = F.subject()
    cam = F.side_camera(inst, resolution=(128, 128))
    light = F.key_light(direction=(0.0, 0.0, -1.0))
    blk = F.light_blocker(inst, light, offset=0.6, half=0.3)
    blk.casts_shadow = False
    out = render.render([inst, blk], cam, light=light, exclusion=None)
    assert out["cast_shadow"].sum() == 0


# ---------------------------------------------------------------------------
# occlusion + the mask algebra
# ---------------------------------------------------------------------------

def test_a_foreground_occluder_hides_skin_and_is_recorded():
    _, inst = F.subject()
    cam = F.side_camera(inst, resolution=(192, 192))
    light = F.key_light()
    clear = render.render([inst], cam, light=light, exclusion=None)
    # a quad parked between the camera and the animal
    centre = inst.vertices.mean(axis=0)
    pos = cam.eye + cam.forward * (0.5 * np.linalg.norm(centre - cam.eye))
    v = np.stack([pos - 0.25 * cam.right - 0.25 * cam.up,
                  pos + 0.25 * cam.right - 0.25 * cam.up,
                  pos + 0.25 * cam.right + 0.25 * cam.up,
                  pos - 0.25 * cam.right + 0.25 * cam.up])
    quad = render.Instance(v, np.array([[0, 1, 2], [0, 2, 3]]), color=(0.1, 0.1, 0.1),
                           role="occluder", double_sided=True, casts_shadow=False,
                           name="card")
    out = render.render([inst, quad], cam, light=light, exclusion=None)
    assert out["occlusion"].sum() > 100
    assert not (out["occlusion"] & out["visible_skin"]).any()
    assert np.array_equal(out["occlusion"], clear["visible_skin"] & ~out["visible_skin"])
    assert np.array_equal(out["identity"], clear["identity"] & ~out["occlusion"])


def test_identity_mask_algebra_holds_exactly():
    _, inst = F.subject()
    cam = F.side_camera(inst, resolution=(256, 256))
    light = F.key_light(direction=(0.0, 0.0, -1.0))
    blk = F.light_blocker(inst, light, offset=0.6, half=0.3)
    out = render.render([inst, blk], cam, light=light, exclusion="auto")
    ident = out["identity"]
    assert ident.sum() > 500
    assert (ident & ~out["visible_skin"]).sum() == 0        # subset of visible skin
    for other in ("shadow", "occlusion", "exclusion"):
        assert (ident & out[other]).sum() == 0, other
    # and it is exactly the difference, not merely a subset
    expected = (out["visible_skin"] & np.isfinite(out["chart_s"])
                & ~out["shadow"] & ~out["occlusion"] & ~out["exclusion"])
    assert np.array_equal(ident, expected)


def test_exclusion_is_pulled_through_the_chart_not_the_image():
    """Excluded pixels must be exactly the ones whose (s, phi) is excluded."""
    _, inst = F.subject()
    cam = F.side_camera(inst, resolution=(256, 256))
    out = render.render([inst], cam, light=F.key_light(), exclusion="auto")
    assert out["meta"]["exclusion_source"].startswith("exclusions.")
    exc = out["exclusion"]
    assert exc.sum() > 0
    assert not (exc & ~out["visible_skin"]).any()
    mask, order, _ = render.resolve_exclusion_chart("auto")
    sel = out["visible_skin"] & np.isfinite(out["chart_s"])
    direct = render.sample_chart_mask(mask, out["chart_s"][sel],
                                      out["chart_phi"][sel], axis_order=order)
    assert np.array_equal(exc[sel], direct)
    # the eye is anterior: excluded pixels cluster at low s
    print("excluded body pixels: %d, median s = %.3f"
          % (exc.sum(), float(np.nanmedian(out["chart_s"][exc]))))
    assert float(np.nanmedian(out["chart_s"][exc])) < 0.35


def test_explicit_exclusion_chart_in_either_axis_order():
    _, inst = F.subject()
    cam = F.side_camera(inst, resolution=(128, 128))
    phi_major = np.zeros((64, 128), dtype=bool)
    phi_major[:, :32] = True                       # anterior quarter of the body
    a = render.render([inst], cam, exclusion=phi_major)
    b = render.render([inst], cam, exclusion=(phi_major.T, "s_major"))
    assert np.array_equal(a["exclusion"], b["exclusion"])
    assert a["exclusion"].sum() > 0
    assert float(np.nanmax(a["chart_s"][a["exclusion"]])) < 0.26


def test_missing_exclusion_module_degrades_to_no_exclusion(monkeypatch):
    """Module R must not be blocked on module P being importable."""
    import builtins
    real = builtins.__import__

    def fake(name, *a, **kw):
        if name == "exclusions":
            raise ImportError("simulated: module P absent")
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake)
    mask, order, note = render.resolve_exclusion_chart("auto")
    assert mask is None and note.startswith("unavailable")
    _, inst = F.subject()
    cam = F.side_camera(inst, resolution=(64, 64))
    out = render.render([inst], cam, exclusion="auto")
    assert out["exclusion"].sum() == 0
    assert out["identity"].sum() > 0


def test_sample_chart_mask_is_nearest_wraps_phi_and_kills_nan():
    m = np.zeros((4, 4), dtype=bool)
    m[0, 1] = True                                  # phi bin 0 (phi ~ -pi), s bin 1
    got = render.sample_chart_mask(m, np.array([0.3, 0.3, 0.3, np.nan]),
                                   np.array([-np.pi + 0.1, np.pi + 0.1,
                                             0.0, 0.0]))
    # phi bin 0 is [-pi, -pi/2); pi + 0.1 wraps into it, 0.0 does not
    assert got.tolist() == [True, True, False, False]
    with pytest.raises(ValueError):
        render.sample_chart_mask(m, 0.5, 0.0, axis_order="nonsense")


# ---------------------------------------------------------------------------
# determinism + budget
# ---------------------------------------------------------------------------

def test_render_is_deterministic():
    _, inst = F.subject(seed=7)
    cam = F.side_camera(inst, resolution=(160, 160))
    light = F.key_light()
    a = render.render([inst], cam, light=light, exclusion="auto")
    b = render.render([inst], cam, light=light, exclusion="auto")
    for key in render.OUTPUT_KEYS:
        x, y = np.asarray(a[key]), np.asarray(b[key])
        assert np.array_equal(x, y, equal_nan=x.dtype.kind == "f"), key


def test_a_512_frame_renders_in_under_two_seconds():
    _, inst = F.subject(n_stations=64, n_around=40)
    cam = F.side_camera(inst, resolution=(512, 512))
    light = F.key_light()
    render.render([inst], cam, light=light, exclusion="auto")     # warm imports
    t0 = time.time()
    out = render.render([inst], cam, light=light, exclusion="auto")
    dt = time.time() - t0
    print("512x512 frame: %.3f s (%d identity px)" % (dt, out["identity"].sum()))
    assert dt < 2.0
    assert out["rgb"].shape == (512, 512, 3)


def test_transform_instance_preserves_the_surface_parameterisation():
    _, inst = F.subject()
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    moved = render.transform_instance(inst, rotation=rot,
                                      translation=(0.5, 0.0, 0.0), scale=0.5)
    assert np.array_equal(moved.vertex_s, inst.vertex_s)
    assert np.array_equal(moved.vertex_phi, inst.vertex_phi)
    assert moved.vertices.mean(axis=0) == pytest.approx(
        inst.vertices.mean(axis=0) + np.array([0.5, 0.0, 0.0]), abs=1e-9)
    span = lambda v: (v.max(axis=0) - v.min(axis=0))
    assert np.sort(span(moved.vertices))[-1] == pytest.approx(
        0.5 * np.sort(span(inst.vertices))[-1], rel=1e-9)


def test_instance_validation_rejects_broken_inputs():
    v = np.zeros((4, 3))
    f = np.array([[0, 1, 2]])
    with pytest.raises(ValueError):
        render.Instance(v, np.array([[0, 1, 9]]))
    with pytest.raises(ValueError):
        render.Instance(v, f, role="decoy")
    with pytest.raises(ValueError):
        render.Instance(v, f, uv=np.zeros((3, 2)))
    with pytest.raises(ValueError):
        render.Instance(v, f, texture=np.zeros((4, 4, 3)))      # texture, no uv
    with pytest.raises(ValueError):
        render.Instance(v, f, vertex_s=np.zeros(3), vertex_phi=np.zeros(3))


# ---------------------------------------------------------------------------
# the bridge to the downstream dataset contract (melops_data LTWH bboxes)
# ---------------------------------------------------------------------------

def test_mask_bbox_is_ltwh_in_pixel_edges_and_can_be_made_relative():
    m = np.zeros((10, 12), dtype=bool)
    m[3:6, 4:9] = True
    box = render.mask_bbox_ltwh(m)
    assert box == [4.0, 3.0, 5.0, 3.0]
    assert render.mask_bbox_ltwh(m, relative_to=[4.0, 3.0, 5.0, 3.0]) == [0.0, 0.0, 5.0, 3.0]
    single = np.zeros((4, 4), dtype=bool)
    single[1, 2] = True
    assert render.mask_bbox_ltwh(single) == [2.0, 1.0, 1.0, 1.0]
    assert render.mask_bbox_ltwh(np.zeros((3, 3), dtype=bool)) is None


def test_chart_span_cuts_the_head_crop_in_arc_length_not_pixels():
    _, inst = F.subject()
    cam = F.side_camera(inst, resolution=(160, 320))
    out = render.render([inst], cam, light=F.key_light(), exclusion="auto")
    skin = out["visible_skin"]
    cut = 0.22                       # stand-in for a Schema S1 station
    head = render.chart_span_mask(out["chart_s"], 0.0, cut, within=skin)
    tail = render.chart_span_mask(out["chart_s"], cut, 1.0 + 1e-9, within=skin)
    assert head.any() and tail.any()
    assert not (head & tail).any()
    assert np.array_equal(head | tail, skin & np.isfinite(out["chart_s"]))
    body_box = render.mask_bbox_ltwh(skin)
    head_box = render.mask_bbox_ltwh(head, relative_to=body_box)
    assert 0.0 <= head_box[0] and head_box[2] <= body_box[2]
    assert head_box[2] < body_box[2]          # the head is a strict sub-crop
