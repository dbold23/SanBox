"""Behavioural tests for the texture -> identity bridge between 04 and 05.

The pipeline is run three times for the whole module -- on
``synth.make_sevengill(textured=True)`` (whose procedural skin stands in for
the Meshy photo texture), unmodified on the C-curved
``demo/sevengill_synthetic_bent.glb``, and on a 120-degree bend of the *same*
mesh the first run used (the controlled pose-invariance comparison).  All three
are module-scoped, so the suite pays for the centreline extraction, the bakes
and the validator once each rather than per test.

Tolerances are stated as RATIOS, not absolutes, wherever the quantity is a
property of the fixture texture rather than of the code: ``synth.py`` is a
fixture and is allowed to change, and a test that pins "89 spots" fails for the
wrong reason when it does.  For orientation, measured on this repository at the
time of writing (seed 0, 256-texel texture, 240 x 128 chart, 7150-vertex mesh):

    low-frequency swing        1.132 -> 0.0019   (de-lighting)
    spot-scale contrast        0.0466 -> 0.0363  (chart, real -> de-lit)
    spot-scale contrast        0.0363 -> 0.0009  (de-lit -> flattened skin)
    individual #0              55 spots from 61 connected components
    render -> re-fit           36 spots, recoverable count 36
    similarity to a resight    0.973 / 0.913 / 0.841 / 0.770 over 3 years
    similarity to a random     -0.009 .. 0.028
    straight vs C-120 bend     same identity, similarity > 0.6
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest
from scipy import ndimage

import gltf_export
import mesh3d
import synth
import texture_identity as ti

from texture_identity import bake, drift, exclusions, pattern

BENT_GLB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "demo", "sevengill_synthetic_bent.glb",
)

TEX_SIZE = 256          # the procedural texture's native size: no resampling
N_RESIGHTS = 2
N_RANDOM = 2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def run_synth(tmp_path_factory):
    """The full pipeline on the synthetic sevengill, GLBs validated."""
    out = tmp_path_factory.mktemp("identity_synth")
    return ti.run(glb=None, out_dir=str(out), n_resights=N_RESIGHTS,
                  years=3.0, n_random=N_RANDOM, seed=0, tex_size=TEX_SIZE,
                  validate=True, report=False)


@pytest.fixture(scope="module")
def run_bent(tmp_path_factory):
    """The same pipeline, unmodified, on the C-curved demo GLB.

    This is the file on disk, whatever built it: the point of this fixture is
    that a real C-posed GLB goes in and a validated catalogue comes out with no
    argument changed.  It is deliberately NOT used to compare identities with
    ``run_synth`` -- the demo GLB is a build artefact that can lag ``synth.py``,
    and comparing identities across two different meshes tests the artefact's
    age, not this pipeline.  ``run_bent_synth`` does that comparison instead.
    """
    if not os.path.exists(BENT_GLB):
        pytest.skip("demo/sevengill_synthetic_bent.glb has not been built")
    out = tmp_path_factory.mktemp("identity_bent")
    return ti.run(glb=BENT_GLB, out_dir=str(out), n_resights=1, years=3.0,
                  n_random=1, seed=0, tex_size=TEX_SIZE, validate=True,
                  report=False)


@pytest.fixture(scope="module")
def run_bent_synth(tmp_path_factory):
    """The pipeline on a C-120 bend of the SAME mesh ``run_synth`` used.

    Same vertices, same atlas, same texture -- only the pose differs -- so a
    difference in the recovered identity can only come from the de-bend.
    """
    straight = synth.make_sevengill(textured=True, seed=0)
    bent, _ = synth.bend(
        straight,
        synth.c_curve(float(straight.metadata["total_length"]), 120.0, 64),
    )
    out = tmp_path_factory.mktemp("identity_bent_synth")
    return ti.run(glb=bent, out_dir=str(out), n_resights=0, years=1.0,
                  n_random=0, seed=0, tex_size=TEX_SIZE, validate=False,
                  report=False)


@pytest.fixture(scope="module")
def straightened(run_synth):
    return run_synth["straightened"]


# ---------------------------------------------------------------------------
# The 04 -> 05 convention bridge
# ---------------------------------------------------------------------------

def test_chart_coords_send_the_ventral_seam_from_plus_pi_to_minus_pi():
    """04 hands out (-pi, pi]; 05 wants [-pi, pi).  The seam is the whole bug."""
    phi_04 = np.array([0.0, math.pi / 2, -math.pi / 2, math.pi,
                       math.pi - 1e-9, -math.pi + 1e-9])
    coords = mesh3d.TubeCoords(
        s=np.linspace(0.0, 1.0, len(phi_04)), r=np.ones(len(phi_04)),
        phi=phi_04, station=np.zeros(len(phi_04), dtype=np.int64),
        total_length=1.0, n_stations=2,
    )
    _, phi_05 = ti.chart_coords(coords)
    assert np.all(phi_05 >= -math.pi) and np.all(phi_05 < math.pi)
    assert phi_05[3] == pytest.approx(-math.pi)          # the disputed value
    assert np.allclose(phi_05[[0, 1, 2]], phi_04[[0, 1, 2]])
    assert phi_05[4] == pytest.approx(math.pi - 1e-9)    # just inside: unchanged


def test_chart_coords_produce_exactly_05s_convention(straightened):
    s, phi = straightened.vertex_s, straightened.vertex_phi
    assert s.min() == pytest.approx(0.0) and s.max() == pytest.approx(1.0)
    assert phi.min() >= -math.pi and phi.max() < math.pi
    assert len(s) == len(straightened.mesh.vertices)


def test_chart_s_runs_snout_to_tail(straightened):
    """s = 0 is the snout tip (+X, the straight pose's head) and 1 the tail."""
    v = np.asarray(straightened.mesh.vertices)
    s = straightened.vertex_s
    assert v[np.argmin(s), 0] > v[np.argmax(s), 0]
    # 04's straight pose is snout +X, so s must anti-correlate with x.
    assert np.corrcoef(s, v[:, 0])[0, 1] < -0.95


def test_chart_phi_zero_is_dorsal_and_plus_half_pi_is_left(straightened):
    """phi = 0 -> +Z, phi = +pi/2 -> +Y, checked on mid-body vertices only."""
    v = np.asarray(straightened.mesh.vertices)
    s, phi = straightened.vertex_s, straightened.vertex_phi
    body = (s > 0.35) & (s < 0.55) & (straightened.coords.r > 1e-4)
    r = straightened.coords.r
    dorsal = body & (np.abs(phi) < 0.15)
    left = body & (np.abs(phi - math.pi / 2) < 0.15)
    ventral = body & (np.abs(np.abs(phi) - math.pi) < 0.15)
    assert dorsal.any() and left.any() and ventral.any()
    assert np.all(v[dorsal, 2] > 0.6 * r[dorsal])
    assert np.all(v[left, 1] > 0.6 * r[left])
    assert np.all(v[ventral, 2] < -0.6 * r[ventral])


def test_chart_normalisation_extent_keeps_the_caudal_overhang(straightened):
    """``normalize='chart'`` folds the overhang onto the ends; extent does not."""
    s_extent, _ = ti.chart_coords(straightened.coords, normalize="extent")
    s_chart, _ = ti.chart_coords(straightened.coords, normalize="chart")
    # The centreline stops at the peduncle, so raw s leaves [0, L].
    assert straightened.coords.s.max() > straightened.coords.total_length
    piled = float(np.mean(s_chart >= 1.0 - 1e-12))
    assert piled > 0.01, "the caudal fin should clip onto s = 1"
    assert float(np.mean(s_extent >= 1.0 - 1e-12)) < piled / 10.0


def test_default_chart_shape_is_isotropic_and_s_major():
    n_s, n_phi = ti.default_chart_shape()
    assert (n_phi, n_s) == pattern.isotropic_resolution(ti.CHART_H_PHI)
    assert n_s > n_phi, "copy_from_chart is told s_major; keep the shape honest"


def test_exclusion_mask_layouts_are_transposes():
    shape = ti.default_chart_shape()
    bake_layout = ti.exclusion_mask(shape, layout="bake")
    pattern_layout = ti.exclusion_mask(shape, layout="pattern")
    assert bake_layout.shape == shape
    assert pattern_layout.shape == (shape[1], shape[0])
    assert np.array_equal(bake_layout, pattern_layout.T)
    assert 0.0 < bake_layout.mean() < 0.5
    with pytest.raises(ValueError):
        ti.exclusion_mask(shape, layout="uv")


# ---------------------------------------------------------------------------
# Reading the real texture off the surface
# ---------------------------------------------------------------------------

def test_texture_image_reads_both_trimesh_material_classes():
    simple = synth.make_sevengill(textured=True)      # SimpleMaterial.image
    a = ti.texture_image(simple)
    assert a.ndim == 3 and a.shape[2] == 3
    assert 0.0 <= a.min() and a.max() <= 1.0
    if os.path.exists(BENT_GLB):
        pbr = mesh3d.load_mesh(BENT_GLB, report=False)   # PBRMaterial
        b = ti.texture_image(pbr)
        assert b.shape == a.shape


def test_texture_image_caps_the_working_resolution():
    mesh = synth.make_sevengill(textured=True)
    small = ti.texture_image(mesh, tex_size=64)
    assert max(small.shape[:2]) == 64
    assert ti.texture_image(mesh, tex_size=4096).shape[:2] == (256, 256)


def test_texture_image_refuses_an_untextured_mesh():
    bare = synth.make_sevengill(textured=False)
    with pytest.raises(ValueError, match="no base-colour texture"):
        ti.texture_image(bare)


def test_the_real_texture_survives_the_debend(straightened):
    """De-bend moves vertices only: UVs and the image are the scan's own."""
    original = synth.make_sevengill(textured=True)
    assert len(straightened.mesh.vertices) == len(original.vertices)
    assert np.array_equal(straightened.mesh.faces, original.faces)
    assert np.allclose(np.asarray(straightened.mesh.visual.uv),
                       np.asarray(original.visual.uv))
    assert np.allclose(straightened.texture, ti.texture_image(original,
                                                              tex_size=TEX_SIZE))


def test_chart_read_covers_the_body(run_synth):
    """Most cells are measured; the rest are honestly NaN, not invented.

    Fin texels are dropped from the read, so the cells a blade covered come
    back unobserved rather than carrying the blade's albedo -- which is why
    this floor is 0.75 and not 1.
    """
    chart = run_synth["chart_real"]
    assert chart.shape == tuple(run_synth["chart_shape"]) + (3,)
    finite = np.isfinite(chart[..., 0])
    assert finite.mean() > 0.75
    assert run_synth["summary"]["chart_finite_frac"] > 0.75


def test_body_texel_alpha_drops_the_fins_and_nothing_else(straightened):
    delit = ti.delight_texture(straightened.mesh, straightened.vertex_s,
                               straightened.vertex_phi, straightened.texture)
    alpha = ti.body_texel_alpha(straightened.mesh, straightened.fins,
                                delit.raster)
    covered = delit.raster.covered
    assert set(np.unique(alpha)) <= {0.0, 1.0}
    assert not alpha[~covered].any(), "uncovered texels cannot be body"
    kept = alpha[covered].mean()
    assert 0.5 < kept < 1.0, "fins exist and are a minority of the atlas"
    # No fins detected -> nothing excluded.
    everything = ti.body_texel_alpha(straightened.mesh, None, delit.raster)
    assert np.array_equal(everything.astype(bool), covered)


# ---------------------------------------------------------------------------
# De-lighting
# ---------------------------------------------------------------------------

def test_delighting_removes_the_lowfrequency_term(run_synth):
    stats = run_synth["delit"].stats
    assert stats["lowfreq_swing_before"] > 0.4
    assert stats["lowfreq_swing_after"] < 0.05
    assert stats["lowfreq_swing_after"] < 0.1 * stats["lowfreq_swing_before"]
    assert stats["gain_clipped_frac"] < 0.05


def test_delighting_keeps_the_identity_layer(run_synth):
    """Shading out, freckles in: that is the whole point of the divide."""
    real = ti.highfreq_contrast(run_synth["chart_real"])
    delit = ti.highfreq_contrast(run_synth["chart_delighted"])
    assert real > 0.01
    # Measured 0.0466 -> 0.0363 (78 % kept).  The floor is deliberately well
    # below that: what must not happen is the divide eating the spots, and it
    # would have to lose a third of them before this fires.
    assert delit > 0.6 * real


def test_the_skin_base_has_the_identity_layer_flattened_out(run_synth):
    """A random individual must not inherit the real animal's freckles."""
    delit = ti.highfreq_contrast(run_synth["chart_delighted"])
    skin = ti.highfreq_contrast(run_synth["chart_skin"])
    assert skin < 0.15 * delit


def test_delight_uses_bakes_own_path(straightened):
    """delight_texture is bake's de-light, not a re-implementation of it."""
    mesh, s, phi = straightened.mesh, straightened.vertex_s, straightened.vertex_phi
    delit = ti.delight_texture(mesh, s, phi, straightened.texture)
    _, dbg = bake.bake_chart_to_texture(
        mesh, s, phi, np.ones((8, 16)), straightened.texture.shape[:2],
        base_albedo=straightened.texture, delight=True,
        chart_semantics="multiplier", return_debug=True,
    )
    assert np.allclose(delit.albedo, dbg["albedo"])
    assert np.allclose(delit.gain, dbg["gain"])
    assert np.all(delit.gain >= 0.25 - 1e-9) and np.all(delit.gain <= 4.0 + 1e-9)


# ---------------------------------------------------------------------------
# Individual #0
# ---------------------------------------------------------------------------

def test_individual0_is_a_copy_of_the_real_skin(run_synth):
    ind0 = run_synth["individual0"]
    assert ind0.identity == "individual0"
    assert ind0.provenance["origin"] == "copy_from_chart"
    assert ind0.provenance["axis_order"] == "s_major"
    assert ind0.provenance["semantics"] == "albedo"
    assert ind0.seed == -1, "a copied individual was not drawn from a seed"
    assert len(ind0) > 20


def test_individual0_spot_count_matches_an_independent_segmentation(run_synth):
    """Count the chart's dark blobs directly and compare with the fitted table.

    ``copy_from_chart`` labels with a wrap across the ventral seam; this
    reference does not, so the two can differ by the number of marks that
    straddle ``phi = +-pi``.  The tolerance covers that and nothing else.
    """
    ind0 = run_synth["individual0"]
    chart = bake.luminance(np.nan_to_num(run_synth["chart_delighted"], nan=1.0))
    mask = ti.exclusion_mask(run_synth["chart_shape"])
    binary = ((1.0 - chart) >= ind0.provenance["threshold"]) & ~mask
    labels, n = ndimage.label(binary)
    if n:
        sizes = np.bincount(labels.ravel())[1:]
        reference = int((sizes >= ind0.provenance["min_area_px"]).sum())
    else:
        reference = 0
    assert reference > 20
    assert abs(len(ind0) - reference) <= max(3, 0.15 * reference)


def test_individual0_round_trips_through_render_and_refit(run_synth):
    """Re-fitting a rendered individual recovers the RECOVERABLE spots.

    ``recoverable_spot_count`` is 05's honest denominator: a mark faded below
    the segmentation threshold by countershading or region signal is not
    recoverable in principle, so it is not counted as a loss here.
    """
    rt = run_synth["summary"]["individual0"]["round_trip"]
    assert rt["recoverable"] > 10
    assert abs(rt["refit"] - rt["recoverable"]) <= max(2, 0.15 * rt["recoverable"])
    assert rt["refit"] <= rt["n_spots"]


def test_no_individual_has_a_spot_in_an_exclusion_region(run_synth):
    """Eyes, nares, mouth and the seven gill slits carry no identity."""
    everyone = ([run_synth["individual0"]] + list(run_synth["resights"])
                + list(run_synth["randoms"]))
    regions = run_synth["context"].regions
    assert regions
    for ind in everyone:
        assert ind.regions == regions
        inside = exclusions.regions_contain(regions, ind.spots["s"],
                                            ind.spots["phi"])
        assert not np.any(inside), "%s put %d spots in an exclusion region" % (
            ind.identity, int(np.sum(inside)))


def test_rendered_charts_are_blank_inside_the_exclusion_mask(run_synth):
    """Enforced at render time too, not only at fit time."""
    mask = ti.exclusion_mask(run_synth["chart_shape"], layout="pattern")
    for entry in run_synth["baked"]:
        assert float(np.abs(entry["chart"][mask]).max()) == 0.0


def test_individual0_json_carries_the_spot_table(run_synth):
    with open(run_synth["paths"]["individual0_json"]) as fh:
        blob = json.load(fh)
    ind0 = run_synth["individual0"]
    assert blob["identity"] == "individual0"
    assert blob["n_spots"] == len(ind0) == len(blob["spots"])
    assert blob["provenance"]["origin"] == "copy_from_chart"
    assert blob["chart_convention"] == exclusions.CHART_CONVENTION
    row = blob["spots"][0]
    assert set(row) == {"id", "s", "phi", "radius", "eccentricity", "angle",
                        "darkness", "birth_date"}
    assert row["s"] == pytest.approx(float(ind0.spots["s"][0]))
    assert -math.pi <= row["phi"] < math.pi


def test_fit_individual_pins_the_ambiguous_conventions(run_synth):
    """Auto-detection would read this chart as a darkness map and invert it."""
    chart = run_synth["chart_delighted"]
    lum = bake.luminance(chart)
    # copy_from_chart's "auto" reads the mean of the FINITE cells and calls
    # anything below 0.5 a darkness map -- which would invert this chart.
    assert lum[np.isfinite(lum)].mean() < 0.5, (
        "auto semantics would guess 'darkness' here")
    ind = ti.fit_individual(chart, context=run_synth["context"])
    assert ind.provenance["semantics"] == "albedo"
    assert len(ind) == len(run_synth["individual0"])


# ---------------------------------------------------------------------------
# Resightings and random individuals
# ---------------------------------------------------------------------------

def test_resights_are_the_same_animal_later(run_synth):
    ind0 = run_synth["individual0"]
    resights = run_synth["resights"]
    assert len(resights) == N_RESIGHTS
    previous = pattern.as_date(ind0.date)
    for r in resights:
        assert len(r) == len(ind0)
        assert np.array_equal(r.spots["id"], ind0.spots["id"])
        assert np.array_equal(r.spots["birth_date"], ind0.spots["birth_date"])
        assert pattern.as_date(r.date) > previous
        previous = pattern.as_date(r.date)
    assert resights[-1].length_cm > ind0.length_cm      # von Bertalanffy growth


def test_resight_similarity_beats_random_similarity(run_synth):
    ind0 = run_synth["individual0"]
    near = min(drift.similarity(ind0, r) for r in run_synth["resights"])
    far = max(drift.similarity(ind0, r) for r in run_synth["randoms"])
    assert near > 0.5
    assert far < 0.25
    assert near > 3.0 * max(far, 1e-6)


def test_random_individuals_are_matched_in_density_but_not_in_pattern(run_synth):
    ind0 = run_synth["individual0"]
    randoms = run_synth["randoms"]
    assert len(randoms) == N_RANDOM
    for r in randoms:
        assert r.provenance["origin"] == "randomize"
        assert abs(len(r) - len(ind0)) <= max(4, 0.2 * len(ind0))
    assert drift.similarity(randoms[0], randoms[1]) < 0.25


def test_resight_series_rejects_a_negative_count(run_synth):
    with pytest.raises(ValueError):
        ti.resight_series(run_synth["individual0"], -1, 1.0)


# ---------------------------------------------------------------------------
# The GLBs
# ---------------------------------------------------------------------------

def _glb_paths(result):
    return [entry["glb"] for entry in result["baked"]]


def test_every_emitted_glb_validates_with_zero_errors(run_synth):
    paths = _glb_paths(run_synth)
    assert len(paths) == 1 + N_RESIGHTS + N_RANDOM
    for path in paths:
        issues = gltf_export.validate_glb(path, raise_on_error=False)
        assert issues["numErrors"] == 0, (path, issues["messages"])


def test_glbs_keep_the_geometry_and_the_atlas(run_synth, straightened):
    """Same body, different skin: that is what makes the corpus comparable."""
    uv = np.asarray(straightened.mesh.visual.uv, dtype=float)
    for path in _glb_paths(run_synth):
        back = mesh3d.load_mesh(path, report=False)
        assert len(back.vertices) == len(straightened.mesh.vertices)
        assert len(back.faces) == len(straightened.mesh.faces)
        assert np.allclose(np.asarray(back.vertices),
                           np.asarray(straightened.mesh.vertices), atol=1e-6)
        assert np.array_equal(np.asarray(back.faces),
                              np.asarray(straightened.mesh.faces))
        back_uv = np.asarray(back.visual.uv, dtype=float)
        assert back_uv.shape == uv.shape
        assert np.abs(back_uv - uv).max() < ti.UV_ROUND_TRIP_TOL


def test_each_glb_carries_its_own_texture(run_synth):
    images = {}
    for path in _glb_paths(run_synth):
        back = mesh3d.load_mesh(path, report=False)
        arr = ti.texture_image(back)
        images[os.path.basename(path)] = arr
    names = sorted(images)
    assert len({im.tobytes() for im in images.values()}) == len(names)
    ind0 = images["individual0.glb"]
    # A resighting is a small perturbation; an unrelated animal is not.
    d_resight = np.abs(ind0 - images["resight_01.glb"]).mean()
    d_random = np.abs(ind0 - images["random_01.glb"]).mean()
    assert d_resight < d_random


def test_a_baked_texture_reads_back_as_the_identity_it_encodes(run_synth,
                                                              straightened):
    """Bake -> GLB -> load -> chart puts the marks back where they started."""
    entry = run_synth["by_name"]["individual0"]
    back = mesh3d.load_mesh(entry["glb"], report=False)
    chart = ti.read_chart(back, ti.texture_image(back), straightened.vertex_s,
                          straightened.vertex_phi,
                          chart_shape=run_synth["chart_shape"])
    darkness = 1.0 - bake.luminance(np.nan_to_num(chart, nan=1.0))
    rendered = entry["chart"].T                      # (n_s, n_phi) darkness
    mask = ti.exclusion_mask(run_synth["chart_shape"])
    hot = (rendered > 0.5) & ~mask
    cold = (rendered <= 0.0) & ~mask
    assert hot.sum() > 50
    assert darkness[hot].mean() > darkness[cold].mean() + 0.05


def test_write_textured_glb_rejects_an_invalid_file(tmp_path, straightened):
    """The validator is wired in, not decorative."""
    path = tmp_path / "broken.glb"
    ti.write_textured_glb(straightened.mesh,
                          np.zeros(straightened.texture.shape[:2] + (4,)),
                          str(path), validate=True)
    assert path.exists()
    with open(str(path), "r+b") as fh:      # corrupt the container header
        fh.seek(0)
        fh.write(b"XXXX")
    with pytest.raises(gltf_export.GltfValidationError):
        gltf_export.validate_glb(str(path))


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def test_run_writes_every_documented_output(run_synth):
    paths = run_synth["paths"]
    for key in ("chart_real", "chart_delighted", "chart_skin",
                "individual0_json", "contact_sheet", "summary",
                "texture_real", "texture_delighted", "texture_skin"):
        assert os.path.exists(paths[key]), key
    for k in range(N_RESIGHTS):
        assert os.path.exists(paths["resight_%02d_glb" % (k + 1)])
    for k in range(N_RANDOM):
        assert os.path.exists(paths["random_%02d_glb" % (k + 1)])


def test_contact_sheet_has_one_row_per_stage(run_synth):
    from PIL import Image

    sheet = Image.open(run_synth["paths"]["contact_sheet"])
    assert sheet.mode == "RGB"
    # real | de-lit | individual #0 | 2 resights | 2 randoms
    rows = 3 + min(2, N_RESIGHTS) + min(2, N_RANDOM)
    assert sheet.size[1] == 8 + rows * (18 + 200 + 8)
    assert sheet.size[0] > 600


def test_contact_sheet_rejects_an_empty_panel_list(tmp_path):
    with pytest.raises(ValueError):
        ti.contact_sheet([], str(tmp_path / "sheet.png"))


def test_chart_to_image_transposes_and_inverts_darkness():
    chart = np.zeros((6, 4))
    chart[1, 2] = 1.0
    albedo = ti.chart_to_image(chart, semantics="albedo")
    darkness = ti.chart_to_image(chart, semantics="darkness")
    assert albedo.shape == (4, 6, 3)
    assert albedo[2, 1, 0] == 255 and albedo[0, 0, 0] == 0
    assert darkness[2, 1, 0] == 0 and darkness[0, 0, 0] == 255
    with pytest.raises(ValueError):
        ti.chart_to_image(chart, semantics="multiplier")


def test_summary_json_is_serialisable_and_complete(run_synth):
    with open(run_synth["paths"]["summary"]) as fh:
        blob = json.load(fh)
    assert blob["chart_shape_bake"] == list(run_synth["chart_shape"])
    assert blob["individual0"]["n_spots"] == len(run_synth["individual0"])
    assert len(blob["resights"]) == N_RESIGHTS
    assert len(blob["randoms"]) == N_RANDOM
    assert blob["delight"]["lowfreq_swing_before"] > blob["delight"][
        "lowfreq_swing_after"]
    assert blob["straighten"]["fins_found"]


def test_side_view_render_shows_the_body(run_synth, straightened):
    frame = ti.render_side(straightened.mesh, straightened.texture,
                           straightened.vertex_s, straightened.vertex_phi)
    assert frame.shape == ti.PREVIEW_RESOLUTION + (3,)
    assert frame.dtype == np.uint8
    background = frame[0, 0].astype(int)
    covered = np.abs(frame.astype(int) - background).sum(axis=2) > 30
    assert 0.05 < covered.mean() < 0.6


# ---------------------------------------------------------------------------
# The C-curved input
# ---------------------------------------------------------------------------

def test_pipeline_runs_unmodified_on_the_bent_demo_glb(run_bent):
    assert os.path.exists(run_bent["paths"]["contact_sheet"])
    assert os.path.exists(run_bent["paths"]["individual0_json"])
    for path in _glb_paths(run_bent):
        issues = gltf_export.validate_glb(path, raise_on_error=False)
        assert issues["numErrors"] == 0, (path, issues["messages"])
    st = run_bent["straightened"]
    assert st.info["fins_found"], "fin detection ran on the de-bent pose"
    assert st.vertex_s.min() == pytest.approx(0.0)
    assert st.vertex_s.max() == pytest.approx(1.0)


def test_the_bent_input_recovers_the_same_individual(run_synth, run_bent_synth):
    """The C-curve is a pose, not an identity: de-bending must remove it.

    Same mesh, same atlas, same texture, bent by 120 degrees -- so the only
    thing between the two runs is ``debend``, and the identity read off the
    skin has to survive it.
    """
    a, b = run_synth["individual0"], run_bent_synth["individual0"]
    assert len(a) > 20 and len(b) > 20
    assert abs(len(a) - len(b)) <= max(4, 0.2 * len(a))
    assert drift.similarity(a, b) > 0.6
    assert (run_bent_synth["delit"].stats["lowfreq_swing_after"]
            < 0.1 * run_bent_synth["delit"].stats["lowfreq_swing_before"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_main_runs_end_to_end(tmp_path):
    out = tmp_path / "cli"
    result = ti.main([
        "--out", str(out), "--n-resights", "1", "--n-random", "0",
        "--tex-size", "128", "--chart-h-phi", "64", "--no-validate", "--quiet",
    ])
    assert result["chart_shape"] == ti.default_chart_shape(64)
    assert len(result["resights"]) == 1 and result["randoms"] == []
    assert os.path.exists(str(out / "contact_sheet.png"))
    assert os.path.exists(str(out / "resight_01.glb"))
    assert result["summary"]["texture_size"] == [128, 128]


def test_p05_shim_reports_a_missing_prototype():
    with pytest.raises(ImportError, match="prototype 05 not found"):
        ti._add_p05_to_path("/nonexistent/prototype-05")
