"""Contract tests for chart-space identity patterns (randomize / render / copy)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from module_p_fixtures import *  # noqa: F401,F403  (see its docstring)
from module_p_fixtures import TEST_RESOLUTION

import exclusions as E
import pattern as P


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_generation_is_deterministic_under_seed(regions):
    a = P.Individual.generate(7, regions=regions)
    b = P.Individual.generate(7, regions=regions)
    assert np.array_equal(a.spots, b.spots)
    assert a.spots.dtype == P.SPOT_DTYPE
    c = P.Individual.generate(8, regions=regions)
    assert not np.array_equal(a.spots["s"], c.spots["s"])


def test_rendering_is_deterministic(individual):
    img1, tab1 = P.render_chart(individual, TEST_RESOLUTION)
    img2, tab2 = P.render_chart(individual, TEST_RESOLUTION)
    assert np.array_equal(img1, img2)
    assert np.array_equal(tab1, tab2)


def test_randomize_alias(regions):
    a = P.randomize(7, regions=regions)
    b = P.Individual.generate(7, regions=regions)
    assert np.array_equal(a.spots, b.spots)


# ---------------------------------------------------------------------------
# Spot field properties
# ---------------------------------------------------------------------------


def test_target_spot_count_is_reached(individual):
    assert len(individual) == individual.params.n_spots_target
    assert individual.provenance["realised_spots"] == len(individual)


def test_minimum_separation_is_honoured(individual):
    d = P.nearest_neighbour_spacing(individual.spots,
                                    individual.params.phi_scale)
    assert d.min() >= individual.params.min_sep - 1e-12


def test_density_is_dorsal_heavy(individual):
    phi = np.abs(individual.spots["phi"])
    assert (phi < math.pi / 2).mean() > 0.6


def test_spot_attributes_are_in_documented_ranges(individual):
    p = individual.params
    assert np.all(individual.spots["radius"] > 0)
    assert np.all(individual.spots["eccentricity"] >= 1.0)
    assert np.all(individual.spots["eccentricity"] <= p.ecc_max)
    assert np.all(individual.spots["darkness"] >= p.darkness_min)
    assert np.all(individual.spots["darkness"] <= p.darkness_max)
    assert np.all(np.abs(individual.spots["phi"]) <= math.pi)


def test_spacing_reported_in_chart_units_and_cm(individual):
    chart = individual.spot_spacing()
    cm = individual.spot_spacing("cm")
    assert cm == pytest.approx(chart * individual.length_cm)
    with pytest.raises(ValueError):
        individual.spot_spacing("furlongs")


# ---------------------------------------------------------------------------
# Exclusions
# ---------------------------------------------------------------------------


def test_no_spot_is_placed_inside_an_exclusion_region(individual):
    inside = E.regions_contain(individual.regions, individual.spots["s"],
                               individual.spots["phi"])
    assert not inside.any()


def test_mask_zeroes_the_rendered_pattern_everywhere_it_is_true(individual):
    """The binding contract: True mask pixel => no identity signal, ever."""
    img, _ = P.render_chart(individual, TEST_RESOLUTION)
    mask = E.mask_from_regions(individual.regions, TEST_RESOLUTION)
    assert mask.any()
    assert np.all(img[mask] == 0.0)
    assert img[~mask].max() > 0.1  # and the rest is not blank


def test_mask_holds_even_for_spots_forced_into_an_excluded_region(individual):
    """A pattern that came from elsewhere cannot leak into the eye."""
    tampered = individual.copy()
    tampered.spots["s"][:20] = 0.06   # eye_center station
    tampered.spots["phi"][:20] = 1.20
    img, _ = P.render_chart(tampered, TEST_RESOLUTION)
    mask = E.mask_from_regions(tampered.regions, TEST_RESOLUTION)
    assert np.all(img[mask] == 0.0)


def test_hard_ventral_exclusion_flows_through_generation(schema, stations):
    regions = E.exclusion_regions(schema, stations=stations,
                                  ventral_hard_exclude_phi=2.2)
    ind = P.Individual.generate(5, regions=regions)
    assert np.all(np.abs(ind.spots["phi"]) < 2.2 + 1e-9)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_render_range_and_shape(individual):
    img, tab = P.render_chart(individual, TEST_RESOLUTION)
    assert img.shape == TEST_RESOLUTION
    assert img.dtype == np.float64
    assert img.min() >= 0.0 and img.max() <= 1.0
    assert len(tab) == len(individual)
    assert "rendered_darkness" in tab.dtype.names
    for name in ("id", "s", "phi", "radius", "darkness", "birth_date"):
        assert name in tab.dtype.names


def test_render_resolution_independence_of_marked_area(individual):
    """Coarse and fine renders agree on how much of the body is marked."""
    lo, _ = P.render_chart(individual, (96, 192))
    hi, _ = P.render_chart(individual, (288, 576))
    assert abs((lo > 0.1).mean() - (hi > 0.1).mean()) < 0.02


def test_marks_wrap_across_the_ventral_seam(regions):
    """A mark at phi = pi must appear on BOTH edge rows, not be clipped."""
    ind = P.Individual.generate(3, regions=())
    spots = P.empty_spots(1)
    spots[0] = (0, 0.5, math.pi, 0.02, 1.0, 0.0, 0.9,
                np.datetime64("2019-01-01", "D"))
    seam = ind.copy(spots=spots, regions=())
    img, _ = P.render_chart(seam, TEST_RESOLUTION,
                            apply_countershading=False)
    assert img[0].max() > 0.3
    assert img[-1].max() > 0.3


def test_countershading_dims_the_ventrum(regions):
    ind = P.Individual.generate(9, regions=regions)
    on, tab_on = P.render_chart(ind, TEST_RESOLUTION)
    off, tab_off = P.render_chart(ind, TEST_RESOLUTION,
                                  apply_countershading=False)
    ventral = np.abs(ind.spots["phi"]) > 2.6
    if ventral.any():
        assert np.all(tab_on["rendered_darkness"][ventral]
                      < tab_off["rendered_darkness"][ventral])
    assert on.sum() < off.sum()


# ---------------------------------------------------------------------------
# Region identity-signal knobs (the make_synthetic contract)
# ---------------------------------------------------------------------------


def test_region_signal_knobs_zero_the_head_signal(regions, stations):
    params = P.PatternParams(head_signal=0.0, flank_signal=1.0,
                             tail_signal=1.0,
                             head_s_max=stations["gill_slit_7_dorsal_origin"],
                             flank_s_max=stations["precaudal_pit"])
    ind = P.Individual.generate(11, params=params, regions=regions)
    img, tab = P.render_chart(ind, TEST_RESOLUTION)
    head = tab["s"] <= params.head_s_max
    assert head.any() and (~head).any()
    assert np.all(tab["rendered_darkness"][head] == 0.0)
    assert tab["rendered_darkness"][~head].max() > 0.1
    # a flank spot may overlap the boundary by up to its own radius, so the
    # image assertion uses a one-spot margin inside the head band
    s_axis, _ = E.chart_axes(TEST_RESOLUTION)
    head_cols = s_axis <= params.head_s_max - 0.02
    assert img[:, head_cols].max() == 0.0


def test_region_signal_knobs_are_independent(regions, stations):
    base = P.PatternParams(head_s_max=stations["gill_slit_7_dorsal_origin"],
                           flank_s_max=stations["precaudal_pit"])
    full = P.Individual.generate(12, params=base, regions=regions)
    tail_off = P.Individual.generate(
        12, params=base.replace(tail_signal=0.0), regions=regions)
    _, t_full = P.render_chart(full, TEST_RESOLUTION)
    _, t_off = P.render_chart(tail_off, TEST_RESOLUTION)
    tail = t_full["s"] > base.flank_s_max
    assert tail.any()
    assert np.all(t_off["rendered_darkness"][tail] == 0.0)
    assert np.allclose(t_off["rendered_darkness"][~tail],
                       t_full["rendered_darkness"][~tail])


def test_common_texture_is_shared_and_carries_no_identity(regions, stations):
    """A zero-signal region is textured but uninformative (make_synthetic)."""
    params = P.PatternParams(head_signal=0.0, n_common=40,
                             head_s_max=stations["gill_slit_7_dorsal_origin"],
                             flank_s_max=stations["precaudal_pit"])
    a = P.Individual.generate(21, params=params, regions=regions)
    b = P.Individual.generate(22, params=params, regions=regions)
    s_axis, _ = E.chart_axes(TEST_RESOLUTION)
    head_cols = s_axis <= params.head_s_max - 0.02
    img_a, _ = P.render_chart(a, TEST_RESOLUTION)
    img_b, _ = P.render_chart(b, TEST_RESOLUTION)
    ha, hb = img_a[:, head_cols], img_b[:, head_cols]
    assert ha.max() > 0.05                     # textured
    assert np.array_equal(ha, hb)              # identical across individuals


# ---------------------------------------------------------------------------
# Scars
# ---------------------------------------------------------------------------


def test_scar_healing_curve_matches_the_manta_bracket():
    sc = P.Scar(0, 0.5, 0.5, 0.04, 0.008, 0.0, 0.5, "2020-01-01")
    assert P.scar_visibility(sc, "2019-12-31") == 0.0
    assert P.scar_visibility(sc, "2020-01-01") == pytest.approx(1.0)
    # 5% of initial by 295 days is the calibration point
    assert P.scar_visibility(sc, "2020-10-22") == pytest.approx(0.05, abs=0.005)
    # "undetectable within 179 days" (blacktip) -> well under a fifth
    assert P.scar_visibility(sc, "2020-06-28") < 0.2
    days = np.arange(0, 400, 10)
    vis = [P.scar_visibility(sc, np.datetime64("2020-01-01", "D")
                             + np.timedelta64(int(d), "D")) for d in days]
    assert np.all(np.diff(vis) < 0)


def test_persistent_scars_stabilise_into_residue():
    sc = P.Scar(0, 0.5, 0.5, 0.04, 0.008, 0.0, 0.5, "2020-01-01",
                persist=True, residue=0.15)
    assert P.scar_visibility(sc, "2023-01-01") == pytest.approx(0.15)


def test_scars_render_and_tabulate(regions):
    params = P.PatternParams(n_spots_target=40, n_scars=5)
    ind = P.Individual.generate(31, params=params, regions=regions,
                                date="2020-01-01")
    assert len(ind.scars) == 5
    tab = P.scar_table(ind)
    assert np.all(tab["visibility"] == 1.0)
    fresh, _ = P.render_chart(ind, TEST_RESOLUTION, date="2020-01-01")
    healed, _ = P.render_chart(ind, TEST_RESOLUTION, date="2021-06-01")
    assert fresh.sum() > healed.sum()
    bare, _ = P.render_chart(ind, TEST_RESOLUTION, include_scars=False)
    assert bare.sum() < fresh.sum()


# ---------------------------------------------------------------------------
# COPY: chart image -> spot table
# ---------------------------------------------------------------------------


def test_copy_from_chart_round_trips_the_recoverable_spots(individual):
    img, tab = P.render_chart(individual, (256, 512))
    copied = P.copy_from_chart(img, params=individual.params,
                               regions=individual.regions,
                               date=str(individual.date))
    recoverable = P.recoverable_spot_count(tab, copied.provenance["threshold"])
    assert recoverable > 0.5 * len(individual)
    assert abs(len(copied) - recoverable) <= 0.10 * recoverable
    assert copied.provenance["origin"] == "copy_from_chart"


def test_copy_from_chart_recovers_positions_and_re_renders(individual):
    img, _ = P.render_chart(individual, (256, 512),
                            apply_countershading=False)
    copied = P.copy_from_chart(img, params=individual.params,
                               regions=individual.regions,
                               date=str(individual.date))
    # every fitted spot sits on top of a real one
    ds = copied.spots["s"][:, None] - individual.spots["s"][None, :]
    dp = E.wrap_phi(copied.spots["phi"][:, None]
                    - individual.spots["phi"][None, :])
    d = np.sqrt(ds ** 2 + (dp * individual.params.phi_scale) ** 2)
    nearest = d.min(axis=1)
    assert np.median(nearest) < 0.3 * individual.params.min_sep
    # and the re-render correlates strongly with the source chart
    re_img, _ = P.render_chart(copied, (256, 512),
                               apply_countershading=False)
    a = img.ravel() - img.mean()
    b = re_img.ravel() - re_img.mean()
    ncc = float(a.dot(b) / math.sqrt(a.dot(a) * b.dot(b)))
    assert ncc > 0.75


def test_copy_from_chart_recovers_size_and_shape():
    spots = P.empty_spots(2)
    spots[0] = (0, 0.30, 0.4, 0.012, 1.0, 0.0, 0.8,
                np.datetime64("2020-01-01", "D"))
    spots[1] = (1, 0.60, -0.4, 0.012, 2.0, 0.0, 0.8,
                np.datetime64("2020-01-01", "D"))
    src = P.Individual("truth", 0, P.PatternParams(), spots, regions=())
    img, _ = P.render_chart(src, (256, 512), apply_countershading=False)
    copied = P.copy_from_chart(img, params=src.params)
    assert len(copied) == 2
    order = np.argsort(copied.spots["s"])
    got = copied.spots[order]
    # thresholding cuts inside the soft edge, so the fitted radius is
    # documented to run ~10-15% small (see copy_from_chart's KNOWN BIAS)
    assert got["radius"] == pytest.approx(np.array([0.012, 0.012]), rel=0.25)
    assert np.all(got["radius"] < 0.012)
    corrected = P.copy_from_chart(img, params=src.params, radius_gain=1.0 / 0.875)
    assert corrected.spots["radius"] == pytest.approx(
        np.array([0.012, 0.012]), rel=0.08)
    assert got["eccentricity"][1] > got["eccentricity"][0] + 0.4


def test_copy_from_chart_treats_the_seam_as_one_mark():
    spots = P.empty_spots(1)
    spots[0] = (0, 0.5, math.pi, 0.015, 1.0, 0.0, 0.9,
                np.datetime64("2020-01-01", "D"))
    src = P.Individual("seam", 0, P.PatternParams(), spots, regions=())
    img, _ = P.render_chart(src, (256, 512), apply_countershading=False)
    copied = P.copy_from_chart(img, params=src.params)
    assert len(copied) == 1
    assert abs(abs(copied.spots["phi"][0]) - math.pi) < 0.05


def test_copy_from_chart_ignores_masked_regions(individual):
    img, _ = P.render_chart(individual, (256, 512))
    mask = np.zeros(img.shape, dtype=bool)
    mask[:, :img.shape[1] // 2] = True     # blind the whole anterior half
    copied = P.copy_from_chart(img, mask=mask, params=individual.params)
    assert len(copied) > 0
    assert copied.spots["s"].min() >= 0.5 - 1e-9


def test_copy_from_chart_rejects_non_2d():
    with pytest.raises(ValueError):
        P.copy_from_chart(np.zeros((4, 4, 3)))


def test_isotropic_resolution_is_square_in_the_scaled_metric():
    h, w = P.isotropic_resolution(192, phi_scale=0.085)
    assert abs((1.0 / w) - (2 * math.pi / h) * 0.085) < 1e-4


# ---------------------------------------------------------------------------
# Interop with the photo -> chart module (bake.py / unbake.py)
# ---------------------------------------------------------------------------


def test_copy_from_chart_accepts_the_transposed_s_major_layout(individual):
    img, _ = P.render_chart(individual, (192, 384))
    a = P.copy_from_chart(img, params=individual.params,
                          regions=individual.regions)
    b = P.copy_from_chart(img.T, params=individual.params,
                          regions=individual.regions)
    assert a.provenance["axis_order"] == "phi_major"
    assert b.provenance["axis_order"] == "s_major"
    assert len(a) == len(b)
    assert np.allclose(np.sort(a.spots["s"]), np.sort(b.spots["s"]))
    forced = P.copy_from_chart(img.T, params=individual.params,
                               regions=individual.regions,
                               axis_order="s_major")
    assert np.allclose(np.sort(forced.spots["s"]), np.sort(a.spots["s"]))
    with pytest.raises(ValueError):
        P.copy_from_chart(img, axis_order="sideways")


def test_copy_from_chart_accepts_an_albedo_multiplier_chart(individual):
    img, _ = P.render_chart(individual, (192, 384))
    dark = P.copy_from_chart(img, params=individual.params,
                             regions=individual.regions)
    albedo = P.copy_from_chart(1.0 - img, params=individual.params,
                               regions=individual.regions)
    assert dark.provenance["semantics"] == "darkness"
    assert albedo.provenance["semantics"] == "albedo"
    assert len(dark) == len(albedo)
    with pytest.raises(ValueError):
        P.copy_from_chart(img, chart_semantics="luminance")


def test_copy_from_chart_honours_a_confidence_map(individual):
    img, _ = P.render_chart(individual, (192, 384))
    conf = np.ones(img.shape)
    conf[:, : img.shape[1] // 2] = 0.0      # anterior half unobserved
    copied = P.copy_from_chart(img, params=individual.params,
                               regions=individual.regions, confidence=conf)
    assert len(copied) > 0
    assert copied.spots["s"].min() >= 0.5 - 1e-9
    assert copied.provenance["min_confidence"] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        P.copy_from_chart(img, confidence=np.ones((3, 3)))


def test_exclusion_mask_adapter_is_s_major_and_uses_supplied_landmarks():
    shape = (384, 192)          # (n_s, n_phi), the bake/unbake layout
    m = P.chart_exclusion_mask(shape)
    assert m.shape == shape and m.dtype == np.bool_
    assert 0.02 < m.mean() < 0.40
    # phi-major request is the transpose
    assert np.array_equal(P.chart_exclusion_mask((192, 384), axis_order="phi_major"),
                          m.T)
    # a supplied landmark moves the region it belongs to
    moved = P.chart_exclusion_mask(shape, {"eye_center": (0.30, 1.20)})
    s_axis, _ = E.chart_axes((192, 384))
    near = np.abs(s_axis - 0.30) < 0.015
    assert moved.T[:, near].any()
    assert not m.T[:, near].any()


def test_exclusion_mask_adapter_keeps_the_bare_name_unbound():
    """unbake resolves ``chart_exclusion_mask``; ``exclusion_mask`` stays free.

    That bare name is what unbake's "module P is absent" fallback tests bind
    against, so module P does not claim it. See the note above
    ``chart_exclusion_mask``.
    """
    assert hasattr(P, "chart_exclusion_mask")
    assert not hasattr(P, "exclusion_mask")


def test_exclusion_mask_adapter_records_the_gill_slit_disagreement():
    """unbake.eye_mouth_exclusion keeps the gill slits; module P drops them."""
    assert P.EXCLUSION_MASK_INCLUDE_GILL_SLITS is True
    with_gills = P.chart_exclusion_mask((384, 192))
    without = P.chart_exclusion_mask((384, 192), include_gill_slits=False)
    assert with_gills.mean() > without.mean()
    assert np.all(with_gills[without])
