"""Contract tests for chart space and the exclusion mask."""

from __future__ import annotations

import math

import numpy as np
import pytest
from module_p_fixtures import *  # noqa: F401,F403  (see its docstring)
from module_p_fixtures import TEST_RESOLUTION

import exclusions as E


# ---------------------------------------------------------------------------
# Chart convention
# ---------------------------------------------------------------------------


def test_chart_axes_convention():
    s, phi = E.chart_axes((8, 4))
    assert s.shape == (4,) and phi.shape == (8,)
    assert np.allclose(s, [0.125, 0.375, 0.625, 0.875])
    # rows span [-pi, pi) at pixel centres; the dorsal midline (phi = 0) is
    # the boundary between the two middle rows, ventral (+/-pi) at the edges.
    assert phi[0] == pytest.approx(-math.pi + math.pi / 8)
    assert phi[-1] == pytest.approx(math.pi - math.pi / 8)
    assert np.min(np.abs(phi)) == pytest.approx(math.pi / 8)


def test_wrap_phi_is_half_open():
    assert E.wrap_phi(math.pi) == pytest.approx(-math.pi)
    assert E.wrap_phi(-math.pi) == pytest.approx(-math.pi)
    assert E.wrap_phi(3 * math.pi / 2) == pytest.approx(-math.pi / 2)


def test_meshgrid_shape():
    S, PHI = E.chart_meshgrid(TEST_RESOLUTION)
    assert S.shape == PHI.shape == TEST_RESOLUTION


# ---------------------------------------------------------------------------
# Schema parsing
# ---------------------------------------------------------------------------


def test_schema_is_read_not_invented(schema):
    assert schema.name == "keypoints_sevengill_v1"
    assert schema.version == "S1"
    assert len(schema.keypoints) == 30
    assert schema.midline_axis_fractions == (0.125, 0.25, 0.375, 0.5, 0.625,
                                             0.75, 0.875)
    assert schema.axis_origin == "snout_tip"
    assert schema.axis_terminus == "precaudal_pit"
    assert schema.chart_origin == "gill_slit_1_dorsal_origin"
    assert set(schema.posterior_trio) == {"pelvic_origin", "dorsal_fin_origin",
                                          "cloaca"}
    # seven slits, not five: the sequence brackets slits 1..7
    assert "gill_slit_7" in schema.ap_sequence


def test_midline_stations_come_from_the_yaml(schema, stations):
    mid = schema.midline_stations(stations["precaudal_pit"])
    assert len(mid) == 7
    assert mid["midline_04"] == pytest.approx(0.5 * stations["precaudal_pit"])


def test_trunk_tube_span_matches_schema_chart_block(schema, stations):
    lo, hi = schema.trunk_tube_span(stations)
    assert lo == pytest.approx(stations["gill_slit_1_dorsal_origin"])
    assert hi == pytest.approx(stations["precaudal_pit"])
    assert 0.0 < lo < hi < 1.0


# ---------------------------------------------------------------------------
# Station validation: exactly what the yaml asserts, and nothing more
# ---------------------------------------------------------------------------


def test_default_stations_validate(schema, stations):
    E.validate_stations(stations, schema)
    grades = E.station_grades()
    # every provisional number must carry a grade, and most must be UNVERIFIED
    for name in stations:
        if name.startswith("midline_"):
            continue
        assert name in grades, name
    assert "[UNVERIFIED]" in grades["pectoral_origin"]
    assert "[DEFINITION]" in grades["snout_tip"]


def test_validate_rejects_ap_sequence_violation(schema, stations):
    bad = dict(stations)
    bad["pectoral_origin"] = 0.10  # anterior to gill slit 7: forbidden
    with pytest.raises(ValueError) as exc:
        E.validate_stations(bad, schema)
    assert "ordered_ap_sequence" in str(exc.value)


def test_validate_does_not_order_the_posterior_trio(schema, stations):
    """The schema forbids enforcing an order among pelvic/dorsal/cloaca."""
    for order in ((0.55, 0.60, 0.565), (0.60, 0.55, 0.62), (0.62, 0.58, 0.56)):
        alt = dict(stations)
        alt["pelvic_origin"], alt["dorsal_fin_origin"], alt["cloaca"] = order
        E.validate_stations(alt, schema)  # must not raise for ANY permutation


def test_validate_rejects_trio_outside_its_bracket(schema, stations):
    bad = dict(stations)
    bad["cloaca"] = 0.20  # anterior to pectoral_origin: the one bracket asserted
    with pytest.raises(ValueError):
        E.validate_stations(bad, schema)


def test_validate_rejects_non_keypoints_and_out_of_range(schema, stations):
    with pytest.raises(ValueError):
        E.validate_stations(dict(stations, second_dorsal_origin=0.5), schema)
    with pytest.raises(ValueError):
        E.validate_stations(dict(stations, eye_center=1.4), schema)
    with pytest.raises(ValueError):
        E.validate_stations(dict(stations, snout_tip=0.05), schema)


# ---------------------------------------------------------------------------
# Regions and the mask
# ---------------------------------------------------------------------------


def test_region_phi_containment_wraps_at_the_ventral_seam():
    r = E.Region("ventral", 0.0, 1.0, math.pi, 0.4)
    assert bool(r.contains(np.array(0.5), np.array(math.pi - 0.1)))
    assert bool(r.contains(np.array(0.5), np.array(-math.pi + 0.1)))
    assert not bool(r.contains(np.array(0.5), np.array(0.0)))


def test_mask_is_boolean_deterministic_and_shaped(schema_path):
    m1 = E.build_exclusion_mask(schema_path, TEST_RESOLUTION)
    m2 = E.build_exclusion_mask(schema_path, TEST_RESOLUTION)
    assert m1.dtype == np.bool_
    assert m1.shape == TEST_RESOLUTION
    assert np.array_equal(m1, m2)
    assert 0.02 < m1.mean() < 0.40  # excludes a real but minority area


def test_mask_agrees_with_analytic_containment(regions):
    mask = E.mask_from_regions(regions, (48, 96))
    S, PHI = E.chart_meshgrid((48, 96))
    assert np.array_equal(mask, E.regions_contain(regions, S, PHI))


def test_named_anatomy_is_excluded(regions, stations):
    named = dict((r.name, r) for r in regions)
    assert set(named) >= {"eye_left", "eye_right", "naris_left", "naris_right",
                          "mouth_jaw", "gill_slits"}
    inside = [
        (stations["eye_center"], 1.20),                       # left eye
        (stations["eye_center"], -1.20),                      # right eye
        (stations["naris_anterior_margin"], 2.10),            # left naris
        (0.05, math.pi),                                      # mouth, ventral
        (0.18, 1.6),                                          # a gill slit
        (0.18, math.pi),                                      # throat, slit 1-7
    ]
    for s, phi in inside:
        assert bool(E.regions_contain(regions, np.array(s), np.array(phi))), (s, phi)


def test_la_jolla_freckle_patch_is_not_excluded(regions, schema, stations):
    """The head patch between the nares and gill slit 1 must survive.

    The yaml's chart block names ``[naris_anterior_margin,
    gill_slit_1_dorsal_origin]`` as the head patch the operating La Jolla
    programme matches on; excluding it would delete the head arm of the
    ablation.
    """
    lo, hi = schema.head_patch_bounds
    s_mid = 0.5 * (stations[lo] + stations[hi])
    for phi in (0.0, 0.5, -0.5, 0.9, -0.9):
        assert not bool(E.regions_contain(regions, np.array(s_mid),
                                          np.array(phi))), phi


def test_mid_flank_is_free(regions):
    for s in (0.35, 0.5, 0.65):
        for phi in (0.0, 1.0, -1.0):
            assert not bool(E.regions_contain(regions, np.array(s),
                                              np.array(phi)))


def test_fin_insertions_are_opt_in(schema, stations):
    base = E.exclusion_regions(schema, stations=stations)
    fins = E.exclusion_regions(schema, stations=stations,
                               include_fin_insertions=True)
    assert len(fins) > len(base)
    names = set(r.name for r in fins)
    assert {"dorsal_insertion", "anal_insertion",
            "pectoral_insertion_left"} <= names
    m0 = E.mask_from_regions(base, TEST_RESOLUTION)
    m1 = E.mask_from_regions(fins, TEST_RESOLUTION)
    assert m1.mean() > m0.mean()
    assert np.all(m1[m0])  # opt-in regions only ever ADD


def test_hard_ventral_band_is_opt_in(schema, stations):
    base = E.mask_from_regions(
        E.exclusion_regions(schema, stations=stations), TEST_RESOLUTION)
    hard = E.mask_from_regions(
        E.exclusion_regions(schema, stations=stations,
                            ventral_hard_exclude_phi=2.4), TEST_RESOLUTION)
    assert hard.mean() > base.mean()
    _, PHI = E.chart_meshgrid(TEST_RESOLUTION)
    assert np.all(hard[np.abs(PHI) >= 2.5])
    with pytest.raises(ValueError):
        E.exclusion_regions(schema, stations=stations,
                            ventral_hard_exclude_phi=4.0)


# ---------------------------------------------------------------------------
# Countershading prior
# ---------------------------------------------------------------------------


def test_countershading_is_monotone_dorsal_to_ventral():
    phi = np.linspace(0.0, math.pi, 200)
    w = E.countershading_weight_at(phi)
    assert w[0] == pytest.approx(1.0)
    assert w[-1] == pytest.approx(E.COUNTERSHADING_DEFAULTS["floor"])
    assert np.all(np.diff(w) <= 1e-12)
    assert np.allclose(E.countershading_weight_at(phi),
                       E.countershading_weight_at(-phi))


def test_countershading_floor_parameterises_a_bare_ventrum():
    w = E.countershading_weight(TEST_RESOLUTION, floor=0.0)
    assert w.min() == pytest.approx(0.0)
    assert w.max() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        E.countershading_weight_at(0.0, phi_onset=2.0, phi_full=1.0)
