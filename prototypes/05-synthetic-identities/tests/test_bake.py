"""Tests for chart <-> mesh UV texture (bake.py) on the synthetic UV tube.

Every tolerance here is a MEASUREMENT, not a wish: the numbers printed by the
reporting tests are the prototype's evidence that the round trip works, and the
assertions sit a comfortable margin below what was measured.
"""

from __future__ import annotations

import numpy as np
import pytest

import bake
import fixtures


# ---------------------------------------------------------------------------
# fixture sanity: the ground truth must actually be ground truth
# ---------------------------------------------------------------------------

def test_uv_tube_vertices_match_their_chart_coords():
    tube = fixtures.make_uv_tube(n_stations=32, n_around=24)
    rebuilt = fixtures.tube_surface_points(tube, tube.vertex_s, tube.vertex_phi)
    assert np.abs(rebuilt - np.asarray(tube.mesh.vertices)).max() < 1e-9


def test_uv_tube_layout_is_clean():
    tube = fixtures.make_uv_tube(n_stations=20, n_around=16)
    uv = np.asarray(tube.mesh.visual.uv)
    assert uv.min() >= 0.0 and uv.max() <= 1.0
    n_st, n_col = tube.grid_shape
    grid = np.asarray(tube.mesh.vertices).reshape(n_st, n_col, 3)
    # the duplicated seam column is geometrically identical, UV-distinct
    assert np.abs(grid[:, 0] - grid[:, -1]).max() < 1e-12
    uvg = uv.reshape(n_st, n_col, 2)
    assert np.allclose(uvg[:, 0, 1], 0.0) and np.allclose(uvg[:, -1, 1], 1.0)
    # every face is monotone and narrow in v: no face straddles the atlas border
    faces = np.asarray(tube.mesh.faces)
    fv = uv[faces][..., 1]
    assert (fv.max(axis=1) - fv.min(axis=1)).max() <= 1.0 / 16 + 1e-9


def test_bent_tube_is_still_a_valid_chart():
    tube = fixtures.make_uv_tube(n_stations=40, n_around=24, bend=1.0)
    rebuilt = fixtures.tube_surface_points(tube, tube.vertex_s, tube.vertex_phi)
    assert np.abs(rebuilt - np.asarray(tube.mesh.vertices)).max() < 1e-9
    assert np.ptp(tube.centerline[:, 1]) > 0.05      # it really is bent


# ---------------------------------------------------------------------------
# chart-space primitives
# ---------------------------------------------------------------------------

def test_sample_chart_wraps_in_phi_and_clamps_in_s():
    rng = np.random.default_rng(0)
    chart = rng.random((16, 32))
    eps = 1e-7
    a = bake.sample_chart(chart, 0.5, np.pi - eps)
    b = bake.sample_chart(chart, 0.5, -np.pi + eps)
    assert abs(float(a) - float(b)) < 1e-4          # continuous across +-pi
    # s clamps rather than wraps
    lo = bake.sample_chart(chart, -5.0, 0.3)
    assert np.isclose(float(lo), float(bake.sample_chart(chart, 0.0, 0.3)))


def test_sample_chart_reproduces_cell_centres():
    rng = np.random.default_rng(1)
    chart = rng.random((12, 20))
    s_ax, phi_ax = bake.chart_axes(12, 20)
    S, P = np.meshgrid(s_ax, phi_ax, indexing="ij")
    assert np.abs(bake.sample_chart(chart, S, P) - chart).max() < 1e-12


def test_splat_is_the_adjoint_of_sample():
    """<A x, y> == <x, A^T y> for the bilinear sample/splat pair."""
    rng = np.random.default_rng(2)
    n_s, n_phi = 10, 16
    x = rng.random((n_s, n_phi))
    s = rng.uniform(0.0, 1.0, 500)
    phi = rng.uniform(-np.pi, np.pi, 500)
    y = rng.random(500)
    lhs = float(np.dot(bake.sample_chart(x, s, phi), y))
    acc, _ = bake.splat_to_chart(s, phi, y, n_s, n_phi)
    rhs = float(np.sum(x * acc))
    assert abs(lhs - rhs) < 1e-9 * max(1.0, abs(lhs))


def test_darkness_and_multiplier_semantics_agree():
    tube = fixtures.make_uv_tube(n_stations=24, n_around=20)
    chart_mult, _ = fixtures.make_test_chart(n_s=48, n_phi=96, n_spots=12, seed=4)
    darkness = 1.0 - chart_mult
    kw = dict(tex_size=96, base_albedo=None, delight=False)
    a = bake.bake_chart_to_texture(
        tube.mesh, tube.vertex_s, tube.vertex_phi, chart_mult, **kw)
    b = bake.bake_chart_to_texture(
        tube.mesh, tube.vertex_s, tube.vertex_phi, darkness,
        chart_semantics="darkness", amplitude=1.0, **kw)
    assert np.abs(a - b).max() < 1e-6


# ---------------------------------------------------------------------------
# per-texel (s, phi) and the phi seam
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seam_phi", [np.pi, -2.0, 0.4])
def test_texel_chart_coords_match_the_analytic_parameterisation(seam_phi):
    """The interpolated (s, phi) must equal the tube's exact UV parameterisation.

    The analytic answer is read straight off the UV convention -- u = s,
    v = fraction of the way round from the seam -- so this is an independent
    check, not a restatement of the implementation.
    """
    tube = fixtures.make_uv_tube(n_stations=48, n_around=32, seam_phi=seam_phi)
    tex = 128
    s_tex, phi_tex, raster = bake.texel_chart_coords(
        tube.mesh, tube.vertex_s, tube.vertex_phi, tex)
    cov = raster.covered
    u = (np.arange(tex) + 0.5) / tex
    v = (np.arange(tex) + 0.5) / tex
    U, V = np.meshgrid(u, v)                      # U varies along columns
    s_true = U
    phi_true = bake.wrap_to_pi(seam_phi + V * 2.0 * np.pi)

    assert np.abs(s_tex[cov] - s_true[cov]).max() < 1.0 / 32
    dphi = np.abs(bake.wrap_to_pi(phi_tex[cov] - phi_true[cov]))
    assert dphi.max() < 2.0 * np.pi / 32          # under one face of girth
    print("seam_phi=%.2f  max|ds|=%.2e  max|dphi|=%.2e rad"
          % (seam_phi, np.abs(s_tex[cov] - s_true[cov]).max(), dphi.max()))


def test_seam_crossing_faces_produce_no_phi_discontinuity():
    """A face whose stored phi corners straddle +-pi must not tear.

    With the atlas seam at pi the +-pi wrap of the stored phi lands exactly on
    the atlas border and no interior face ever sees it.  Moving the seam to
    -2.0 rad puts that 2 pi jump on an INTERIOR atlas column -- the raw corner
    values there really do differ by ~6.25 rad -- so a naive barycentric
    interpolation of phi would sweep the whole animal across two texels.
    """
    tube = fixtures.make_seam_offset_tube(
        seam_phi=-2.0, n_stations=48, n_around=32)
    tex = 160
    _, phi_tex, raster = bake.texel_chart_coords(
        tube.mesh, tube.vertex_s, tube.vertex_phi, tex)
    col = phi_tex[:, tex // 2]

    # the raw values contain the jump ...
    assert np.abs(np.diff(col)).max() > 5.0
    # ... but every step is one uniform texel of girth once wrapped
    steps = np.abs(bake.wrap_to_pi(np.diff(col)))
    expected = 2.0 * np.pi / tex
    assert np.abs(steps - expected).max() < 1e-9
    print("seam-offset tube: max wrapped step %.6f rad, expected %.6f"
          % (steps.max(), expected))


def test_texels_across_the_atlas_seam_agree():
    """The first and last atlas rows are one texel apart ON THE ANIMAL."""
    tube = fixtures.make_seam_offset_tube(
        seam_phi=-2.0, n_stations=48, n_around=32)
    chart, _ = fixtures.make_test_chart(n_s=96, n_phi=192, n_spots=40, seed=5)
    tex_size = 192
    tex = bake.bake_chart_to_texture(
        tube.mesh, tube.vertex_s, tube.vertex_phi, chart, tex_size,
        base_albedo=None, delight=False, gutter=0)
    rgb = tex[..., 0]
    across_seam = np.abs(rgb[0] - rgb[-1]).mean()
    typical = np.abs(np.diff(rgb, axis=0)).mean()
    assert across_seam < 2.0 * typical + 1e-6
    print("across-seam mean |d| = %.5f, typical adjacent-row mean |d| = %.5f"
          % (across_seam, typical))


# ---------------------------------------------------------------------------
# the round trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seam_phi", [np.pi, -2.0])
def test_bake_then_unbake_texture_recovers_the_chart(seam_phi):
    tube = fixtures.make_uv_tube(n_stations=64, n_around=48, seam_phi=seam_phi)
    chart, _ = fixtures.make_test_chart(n_s=96, n_phi=192, n_spots=50, seed=6)
    tex = bake.bake_chart_to_texture(
        tube.mesh, tube.vertex_s, tube.vertex_phi, chart, 256,
        base_albedo=None, delight=False)
    back = bake.mesh_texture_to_chart(
        tube.mesh, tex, tube.vertex_s, tube.vertex_phi, chart_shape=(96, 192))
    got = back.mean(axis=2)
    ok = np.isfinite(got)
    assert ok.mean() > 0.99
    corr = float(np.corrcoef(chart[ok], got[ok])[0, 1])
    rmse = float(np.sqrt(np.mean((chart[ok] - got[ok]) ** 2)))
    print("seam_phi=%.2f round trip: correlation %.4f, RMSE %.4f, coverage %.3f"
          % (seam_phi, corr, rmse, ok.mean()))
    assert corr > 0.90
    assert rmse < 0.10


def test_round_trip_has_no_seam_localised_error():
    """Round-trip error must not concentrate near the phi wrap."""
    tube = fixtures.make_seam_offset_tube(
        seam_phi=-2.0, n_stations=64, n_around=48)
    chart, _ = fixtures.make_test_chart(n_s=96, n_phi=192, n_spots=50, seed=7)
    tex = bake.bake_chart_to_texture(
        tube.mesh, tube.vertex_s, tube.vertex_phi, chart, 256,
        base_albedo=None, delight=False)
    back = bake.mesh_texture_to_chart(
        tube.mesh, tex, tube.vertex_s, tube.vertex_phi,
        chart_shape=(96, 192)).mean(axis=2)
    err = np.abs(np.where(np.isfinite(back), back, chart) - chart)
    n_phi = err.shape[1]
    band = np.zeros(n_phi, dtype=bool)
    band[:3] = band[-3:] = True                  # the +-pi columns of the chart
    assert err[:, band].mean() < 2.0 * err[:, ~band].mean() + 1e-6
    print("wrap-column mean err %.5f vs elsewhere %.5f"
          % (err[:, band].mean(), err[:, ~band].mean()))


def test_bake_is_deterministic():
    tube = fixtures.make_uv_tube(n_stations=32, n_around=24)
    chart, _ = fixtures.make_test_chart(n_s=48, n_phi=96, n_spots=20, seed=8)
    args = (tube.mesh, tube.vertex_s, tube.vertex_phi, chart, 128)
    a = bake.bake_chart_to_texture(*args, base_albedo=(0.4, 0.4, 0.35))
    b = bake.bake_chart_to_texture(*args, base_albedo=(0.4, 0.4, 0.35))
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# de-lighting
# ---------------------------------------------------------------------------

def _shaded_case(tex_size=256):
    """A clean albedo, a smooth shading field, and the texture that mixes them."""
    tube = fixtures.make_uv_tube(n_stations=64, n_around=48)
    chart, _ = fixtures.make_test_chart(n_s=96, n_phi=192, n_spots=50, seed=9)
    s_tex, phi_tex, raster = bake.texel_chart_coords(
        tube.mesh, tube.vertex_s, tube.vertex_phi, tex_size)
    cov = raster.covered
    s0, p0 = np.nan_to_num(s_tex), np.nan_to_num(phi_tex)
    # a dorsal key light (cos phi) plus a fore-aft falloff: the shape real
    # baked-in lighting takes on a horizontal animal
    shade = np.clip(0.55 + 0.27 * np.cos(p0) + 0.25 * s0, 0.2, 1.4)
    clean = np.full(cov.shape + (3,), 0.5)
    dirty = np.clip(clean * shade[..., None], 0.0, 1.0)
    ideal = np.clip(clean * bake.sample_chart(chart, s0, p0)[..., None], 0.0, 1.0)
    return tube, chart, tex_size, cov, dirty, ideal


def test_delighting_removes_a_low_frequency_gradient():
    tube, chart, tex_size, cov, dirty, ideal = _shaded_case()
    target = ideal[cov].ravel()

    def corr(**kw):
        tex = bake.bake_chart_to_texture(
            tube.mesh, tube.vertex_s, tube.vertex_phi, chart, tex_size,
            base_albedo=dirty, **kw)
        return float(np.corrcoef(tex[..., :3][cov].ravel(), target)[0, 1])

    off = corr(delight=False)
    blur = corr(delight=True, delight_method="blur")
    basis = corr(delight=True, delight_method="basis")
    print("corr to clean pattern: no de-light %.4f, blur %.4f, basis %.4f"
          % (off, blur, basis))
    assert blur > off                       # a blur helps ...
    assert basis > blur                     # ... a phi-harmonic basis helps far more
    assert basis > 0.95


def test_delighting_preserves_the_spots():
    """De-lighting must flatten the shading without eating spot contrast."""
    tube, chart, tex_size, cov, dirty, ideal = _shaded_case()
    tex = bake.bake_chart_to_texture(
        tube.mesh, tube.vertex_s, tube.vertex_phi, chart, tex_size,
        base_albedo=dirty, delight=True)
    got = bake.mesh_texture_to_chart(
        tube.mesh, tex, tube.vertex_s, tube.vertex_phi,
        chart_shape=(96, 192)).mean(axis=2)
    ok = np.isfinite(got)
    # spot cells are the ones the source chart marks as dark
    spot = ok & (chart < 0.6)
    skin = ok & (chart > 0.95)
    contrast = float(got[skin].mean() - got[spot].mean()) / max(float(got[skin].mean()), 1e-9)
    expected = float(chart[skin].mean() - chart[spot].mean()) / float(chart[skin].mean())
    print("relative spot contrast after de-lighting %.3f (source %.3f)"
          % (contrast, expected))
    assert contrast > 0.6 * expected


def test_delighting_flattens_countershading_too_and_says_so():
    """The documented limit, asserted so it cannot regress silently.

    Countershading is low-frequency ALBEDO.  The de-lighter cannot tell it from
    low-frequency SHADING and removes it.  That is the designed separation of
    layers (species tone is re-applied after the bake), and this test pins the
    behaviour rather than pretending otherwise.
    """
    tube = fixtures.make_uv_tube(n_stations=48, n_around=32)
    tex_size = 160
    s_tex, phi_tex, raster = bake.texel_chart_coords(
        tube.mesh, tube.vertex_s, tube.vertex_phi, tex_size)
    cov = raster.covered
    counter = 1.0 - 0.35 * (1.0 + np.cos(np.nan_to_num(phi_tex))) / 2.0
    albedo = np.clip(np.full(cov.shape + (3,), 0.5) * counter[..., None], 0, 1)
    flat = np.ones((16, 32))
    tex = bake.bake_chart_to_texture(
        tube.mesh, tube.vertex_s, tube.vertex_phi, flat, tex_size,
        base_albedo=albedo, delight=True)
    lum = bake.luminance(tex[..., :3])[cov]
    spread_in = float(np.ptp(bake.luminance(albedo)[cov]))
    spread_out = float(np.ptp(lum))
    print("dorso-ventral luminance spread: %.4f in, %.4f out" % (spread_in, spread_out))
    assert spread_out < 0.15 * spread_in


# ---------------------------------------------------------------------------
# contracts and failure modes
# ---------------------------------------------------------------------------

def test_missing_uv_is_a_clear_error():
    import trimesh
    tube = fixtures.make_uv_tube(n_stations=8, n_around=8)
    bare = trimesh.Trimesh(vertices=tube.mesh.vertices, faces=tube.mesh.faces,
                           process=False)
    bare.visual = trimesh.visual.ColorVisuals(bare)
    with pytest.raises(ValueError, match="UV"):
        bake.bake_chart_to_texture(bare, tube.vertex_s, tube.vertex_phi,
                                   np.ones((8, 8)), 32)


def test_wrong_length_vertex_arrays_are_rejected():
    tube = fixtures.make_uv_tube(n_stations=8, n_around=8)
    with pytest.raises(ValueError, match="per-vertex"):
        bake.bake_chart_to_texture(tube.mesh, tube.vertex_s[:-1],
                                   tube.vertex_phi, np.ones((8, 8)), 32)


def test_alpha_is_true_coverage_and_gutter_does_not_touch_it():
    tube = fixtures.make_uv_tube(n_stations=24, n_around=16)
    chart = np.ones((16, 32))
    a = bake.bake_chart_to_texture(tube.mesh, tube.vertex_s, tube.vertex_phi,
                                   chart, 96, gutter=0)
    b = bake.bake_chart_to_texture(tube.mesh, tube.vertex_s, tube.vertex_phi,
                                   chart, 96, gutter=3)
    assert np.array_equal(a[..., 3], b[..., 3])


# ---------------------------------------------------------------------------
# interop with the pattern generator (module P) -- the contract, exercised
# ---------------------------------------------------------------------------

def test_chart_convention_matches_the_pattern_modules_exactly():
    """bake and exclusions must agree cell-for-cell on what (s, phi) mean.

    They disagree only on array LAYOUT -- ``(n_s, n_phi)`` here,
    ``(H_phi, W_s)`` there.  If the cell centres ever drift apart, every
    baked pattern shifts by half a cell and nothing else in the prototype
    would notice, so it is pinned here.
    """
    exclusions = pytest.importorskip("exclusions")
    n_s, n_phi = 96, 192
    s_mine, phi_mine = bake.chart_axes(n_s, n_phi)
    s_theirs, phi_theirs = exclusions.chart_axes((n_phi, n_s))
    assert np.allclose(s_mine, s_theirs)
    assert np.allclose(phi_mine, phi_theirs)
    assert "0=dorsal midline" in exclusions.CHART_CONVENTION
    assert "LEFT flank" in exclusions.CHART_CONVENTION


def test_pattern_chart_adapters_round_trip():
    chart, _ = fixtures.make_test_chart(n_s=64, n_phi=128, n_spots=15, seed=14)
    dark = bake.to_pattern_chart(chart)
    assert dark.shape == (128, 64)                 # (H_phi, W_s)
    assert np.abs(bake.from_pattern_chart(dark) - chart).max() < 1e-12


def test_a_real_pattern_chart_bakes_to_the_right_place_on_the_mesh():
    """pattern.render_chart -> from_pattern_chart -> bake -> read back.

    End to end across the module boundary, checking the thing a transpose bug
    would break: that a spot ends up at the (s, phi) the generator put it at.
    """
    pattern = pytest.importorskip("pattern")
    ind = pattern.randomize(seed=5)
    dark, spots = pattern.render_chart(ind, resolution=(192, 384))
    assert dark.shape == (192, 384)
    assert len(spots) > 10

    mult = bake.from_pattern_chart(dark)
    assert mult.shape == (384, 192)                # s-major, as bake wants
    tube = fixtures.make_uv_tube(n_stations=64, n_around=48)
    tex = bake.bake_chart_to_texture(
        tube.mesh, tube.vertex_s, tube.vertex_phi, mult, 256,
        base_albedo=None, delight=False)
    back = bake.mesh_texture_to_chart(
        tube.mesh, tex, tube.vertex_s, tube.vertex_phi,
        chart_shape=(384, 192)).mean(axis=2)

    ok = np.isfinite(back)
    corr = float(np.corrcoef(mult[ok], back[ok])[0, 1])
    print("pattern -> bake -> chart correlation %.4f" % corr)
    assert corr > 0.85

    # every rendered spot must sit on a dark texel of the recovered chart
    strong = spots[spots["rendered_darkness"] > 0.35]
    assert len(strong) > 5
    vals = bake.sample_chart(np.nan_to_num(back, nan=1.0),
                            strong["s"], strong["phi"])
    skin = float(np.median(back[ok]))
    hit = float(np.mean(vals < skin - 0.05))
    print("%.2f of %d strong spots land on darkened texels (skin level %.3f)"
          % (hit, len(strong), skin))
    assert hit > 0.8


def test_baking_a_darkness_chart_as_a_multiplier_is_warned_about():
    """The inversion that would otherwise reach the dataset in silence."""
    tube = fixtures.make_uv_tube(n_stations=24, n_around=16)
    chart, _ = fixtures.make_test_chart(n_s=48, n_phi=96, n_spots=15, seed=15)
    darkness = 1.0 - chart
    with pytest.warns(RuntimeWarning, match="darkness map"):
        bake.bake_chart_to_texture(
            tube.mesh, tube.vertex_s, tube.vertex_phi, darkness, 64,
            chart_semantics="multiplier")
    with pytest.warns(RuntimeWarning, match="albedo"):
        bake.bake_chart_to_texture(
            tube.mesh, tube.vertex_s, tube.vertex_phi, chart, 64,
            chart_semantics="darkness")


def test_uncovered_texels_never_come_out_black_or_nan():
    """Gutter texels take the nearest covered colour even when the base albedo is NaN there."""
    import numpy as np
    import bake
    import fixtures

    mesh, vs, vphi = fixtures.make_uv_tube(24, 16, 1.0, seed=0)[:3]
    tex_size = (64, 64)
    base = np.full(tex_size + (3,), 0.6)
    base[:, :3, :] = np.nan                       # an off-atlas strip with undefined albedo
    chart = np.zeros((32, 64))                    # darkness 0 = no marks
    tex = bake.bake_chart_to_texture(mesh, vs, vphi, chart, tex_size,
                                     base_albedo=base, delight=False,
                                     chart_semantics="darkness")
    tex = np.asarray(tex)
    rgb, alpha = tex[..., :3], tex[..., 3]
    assert np.isfinite(rgb).all()
    gutters = alpha == 0
    if gutters.any():
        assert rgb[gutters].min() > 0.3, "gutter texels must inherit skin colour, not black"
