"""Contract tests for resighting drift, growth and chart-space similarity."""

from __future__ import annotations

import math

import numpy as np
import pytest
from module_p_fixtures import *  # noqa: F401,F403  (see its docstring)
from module_p_fixtures import TEST_RESOLUTION

import drift as D
import exclusions as E
import pattern as P

T0 = np.datetime64("2020-01-01", "D")


def _later(days):
    return T0 + np.timedelta64(int(days), "D")


@pytest.fixture(scope="module")
def base(regions):
    return P.Individual.generate(101, regions=regions, date=str(T0),
                                 length_cm=250.0)


# ---------------------------------------------------------------------------
# Determinism and bookkeeping
# ---------------------------------------------------------------------------


def test_resight_is_deterministic_under_seed(base):
    a = D.resight(base, T0, _later(365),
                  rng=np.random.default_rng([1, 2, 3]))
    b = D.resight(base, T0, _later(365),
                  rng=np.random.default_rng([1, 2, 3]))
    assert np.array_equal(a.spots, b.spots)
    assert [s.id for s in a.scars] == [s.id for s in b.scars]


def test_resight_default_rng_is_reproducible(base):
    a = D.resight(base, T0, _later(180))
    b = D.resight(base, T0, _later(180))
    assert np.array_equal(a.spots, b.spots)


def test_resight_records_provenance(base):
    out = D.resight(base, T0, _later(365))
    assert out.provenance["origin"] == "resight"
    assert out.provenance["elapsed_days"] == 365.0
    assert out.provenance["growth_ratio"] > 1.0
    assert out.identity == base.identity
    assert out.date == _later(365)


def test_resight_refuses_a_date_mismatch_or_time_reversal(base):
    with pytest.raises(ValueError):
        D.resight(base, _later(10), _later(20))
    with pytest.raises(ValueError):
        D.resight(base, T0, _later(-10))


def test_zero_interval_is_an_identity_control(base):
    same = D.resight(base, T0, T0)
    assert len(same) == len(base)
    assert np.allclose(same.spots["s"], base.spots["s"])
    assert np.allclose(same.spots["phi"], base.spots["phi"])
    assert D.similarity(base, same, resolution=TEST_RESOLUTION) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Growth: spacing, not count
# ---------------------------------------------------------------------------


def test_spot_count_is_constant_under_growth(base):
    fast = D.VonBertalanffyGrowth(l_inf_cm=400.0, k_per_year=0.5)
    for days in (365, 730, 1460):
        out = D.resight(base, T0, _later(days), growth_model=fast)
        assert len(out) == len(base)
        assert np.array_equal(out.spots["id"], base.spots["id"])
        assert np.array_equal(out.spots["birth_date"], base.spots["birth_date"])
        assert out.length_cm > base.length_cm


def test_growth_scales_absolute_spacing_by_the_length_ratio(base):
    """Spots are a fixed cell population: growth spreads them, in cm."""
    fast = D.VonBertalanffyGrowth(l_inf_cm=400.0, k_per_year=0.5)
    still = D.DriftParams(jitter_rate=0.0, radius_jitter_rate=0.0,
                          darkness_jitter_rate=0.0, scar_rate_per_year=0.0)
    out = D.resight(base, T0, _later(1460), growth_model=fast, params=still)
    ratio = out.length_cm / base.length_cm
    assert ratio > 1.10
    assert (out.spot_spacing("cm") / base.spot_spacing("cm")
            == pytest.approx(ratio, rel=0.05))
    # in NORMALISED chart coordinates the pattern barely moves: only the
    # allometric terms act, isotropic growth is invisible there
    assert (out.spot_spacing() / base.spot_spacing()
            == pytest.approx(1.0, abs=0.05))


def test_isotropic_growth_moves_nothing_in_chart_space(base):
    still = D.DriftParams(jitter_rate=0.0, radius_jitter_rate=0.0,
                          darkness_jitter_rate=0.0, scar_rate_per_year=0.0,
                          head_allometry=0.0, phi_allometry=0.0)
    out = D.resight(base, T0, _later(730),
                    growth_model=D.VonBertalanffyGrowth(), params=still)
    assert np.allclose(out.spots["s"], base.spots["s"])
    assert np.allclose(out.spots["phi"], base.spots["phi"])
    assert out.length_cm > base.length_cm


def test_allometry_shortens_the_head_fraction(base):
    still = D.DriftParams(jitter_rate=0.0, radius_jitter_rate=0.0,
                          darkness_jitter_rate=0.0, scar_rate_per_year=0.0)
    fast = D.VonBertalanffyGrowth(l_inf_cm=400.0, k_per_year=0.5)
    out = D.resight(base, T0, _later(1460), growth_model=fast, params=still)
    moved = out.spots["s"] - base.spots["s"]
    assert np.all(moved <= 1e-12)          # everything shifts head-ward
    # a 33% length gain (a deliberately extreme model) moves marks ~1.5% of
    # the axis; under the DEFAULT growth curve the shift is ~10x smaller
    assert np.mean(np.abs(moved)) < 0.03
    # the allometric map itself is monotone, so on a body with no exclusion
    # regions the antero-posterior ORDER of the marks is preserved. (With
    # exclusions it need not be: a step that would land a mark in the gill
    # band is rejected, and that mark stays put while its neighbour moves.)
    free = P.Individual.generate(101, regions=(), date=str(T0))
    free_out = D.resight(free, T0, _later(1460), growth_model=fast,
                         params=still)
    order = np.argsort(free.spots["s"], kind="stable")
    assert np.all(np.diff(free_out.spots["s"][order]) >= -1e-12)
    gentle = D.resight(base, T0, _later(730), params=still)
    assert np.mean(np.abs(gentle.spots["s"] - base.spots["s"])) < 0.002


def test_growth_models(base):
    assert D.NoGrowth().length_ratio(250.0, 5.0) == 1.0
    assert D.LinearGrowth(10.0).length_ratio(250.0, 1.0) == pytest.approx(1.04)
    vb = D.VonBertalanffyGrowth(l_inf_cm=290.0, k_per_year=0.06)
    assert vb.length_ratio(250.0, 1.0) == pytest.approx(1.0093, abs=1e-3)
    assert vb.grown_length(250.0, 1.0) > 250.0
    with pytest.raises(ValueError):
        vb.length_ratio(0.0, 1.0)


# ---------------------------------------------------------------------------
# Exclusions survive drift
# ---------------------------------------------------------------------------


def test_drift_never_pushes_a_mark_into_an_excluded_region(base):
    loud = D.DriftParams(jitter_rate=0.05)
    out = D.resight(base, T0, _later(730), params=loud)
    assert not E.regions_contain(out.regions, out.spots["s"],
                                 out.spots["phi"]).any()
    img, _ = P.render_chart(out, TEST_RESOLUTION)
    mask = E.mask_from_regions(out.regions, TEST_RESOLUTION)
    assert np.all(img[mask] == 0.0)


def test_new_scars_avoid_excluded_regions(base):
    scarry = D.DriftParams(scar_rate_per_year=20.0)
    out = D.resight(base, T0, _later(730), params=scarry)
    assert len(out.scars) > 10
    for sc in out.scars:
        assert not bool(E.regions_contain(out.regions, np.array(sc.s),
                                          np.array(sc.phi)))


# ---------------------------------------------------------------------------
# Scars
# ---------------------------------------------------------------------------


def test_scars_appear_over_time_and_heal(base):
    scarry = D.DriftParams(scar_rate_per_year=6.0, scar_persist_fraction=0.0)
    out = D.resight(base, T0, _later(365), params=scarry)
    assert len(out.scars) >= 1
    tab_now = P.scar_table(out, _later(365))
    tab_later = P.scar_table(out, _later(365 + 365))
    assert np.all(tab_later["visibility"] < tab_now["visibility"])
    # no scar is born before or after its interval
    assert np.all(tab_now["birth_date"] > T0)
    assert np.all(tab_now["birth_date"] <= _later(365))
    # non-persistent marks are effectively gone at ~6 months (blacktip)
    aged = P.scar_table(out, _later(365 + 179))
    assert aged["visibility"].max() < 0.2


def test_a_documented_fraction_of_scars_persists(base):
    scarry = D.DriftParams(scar_rate_per_year=200.0,
                           scar_persist_fraction=0.25, scar_residue=0.15)
    out = D.resight(base, T0, _later(365), params=scarry)
    tab = P.scar_table(out, _later(365 + 4 * 365))
    persist = tab["persist"]
    assert 0.15 < persist.mean() < 0.40
    assert np.all(tab["visibility"][persist] == pytest.approx(0.15))
    assert np.all(tab["visibility"][~persist] < 1e-6)


def test_scar_pruning_is_opt_in(base):
    keep = D.DriftParams(scar_rate_per_year=20.0, scar_persist_fraction=0.0)
    prune = keep.replace(scar_prune_visibility=0.02)
    a = D.resight(base, T0, _later(1460),
                  rng=np.random.default_rng([9, 9]), params=keep)
    b = D.resight(base, T0, _later(1460),
                  rng=np.random.default_rng([9, 9]), params=prune)
    assert len(b.scars) < len(a.scars)


# ---------------------------------------------------------------------------
# Similarity decay -- the Melops calibration
# ---------------------------------------------------------------------------


def test_similarity_is_one_for_identical_charts(base):
    assert D.similarity(base, base, resolution=TEST_RESOLUTION) == pytest.approx(1.0)


def test_similarity_separates_individuals_from_resightings(regions):
    a = P.Individual.generate(201, regions=regions, date=str(T0))
    b = P.Individual.generate(202, regions=regions, date=str(T0))
    between = D.similarity(a, b, resolution=TEST_RESOLUTION)
    within = D.similarity(a, D.resight(a, T0, _later(730)),
                          resolution=TEST_RESOLUTION)
    assert abs(between) < 0.15
    assert within > 0.6
    assert within > between + 0.4


@pytest.mark.parametrize("seed", [301, 302, 303])
def test_similarity_decays_monotonically_with_elapsed_time(regions, seed):
    ind = P.Individual.generate(seed, regions=regions, date=str(T0))
    curve = D.similarity_curve(ind, elapsed_days=(0, 30, 180, 365, 730),
                               seed=seed, resolution=TEST_RESOLUTION)
    days = [d for d, _ in curve]
    sims = [v for _, v in curve]
    assert days == [0, 30, 180, 365, 730]
    assert sims[0] == pytest.approx(1.0)
    assert all(sims[i] > sims[i + 1] for i in range(len(sims) - 1)), sims


def test_decay_ratio_sits_in_the_melops_calibration_band(regions):
    """Direction and rough magnitude of 0.605 -> 0.474, not a fit.

    The Melops number is an ArcFace cosine between two PHOTOGRAPHS; this is a
    noiseless chart NCC, so only the RATIO is comparable (see drift's module
    docstring). ``DEFAULT_JITTER_RATE`` was calibrated to
    ``MELOPS_RATIO = 0.784``; the band here is deliberately wide.
    """
    near, far = [], []
    for i in range(4):
        ind = P.Individual.generate(400 + i, regions=regions, date=str(T0))
        for days, bucket in ((30, near), (730, far)):
            rng = np.random.default_rng([i, days])
            bucket.append(D.similarity(
                ind, D.resight(ind, T0, _later(days), rng=rng),
                resolution=TEST_RESOLUTION))
    ratio = float(np.mean(far)) / float(np.mean(near))
    assert D.MELOPS_RATIO == pytest.approx(0.7835, abs=1e-3)
    assert 0.65 < ratio < 0.92, ratio


def test_similarity_ignores_excluded_pixels(base):
    """Signal added inside an excluded region cannot change the score."""
    other = D.resight(base, T0, _later(365))
    ref = D.similarity(base, other, resolution=TEST_RESOLUTION)
    assert 0.0 < ref <= 1.0
    extra = P.empty_spots(len(other) + 6)
    extra[:len(other)] = other.spots
    for k in range(6):
        extra[len(other) + k] = (1000 + k, 0.06, 1.20 + 0.02 * k, 0.006, 1.0,
                                 0.0, 1.0, np.datetime64("2019-01-01", "D"))
    tampered = other.copy(spots=extra)
    assert D.similarity(base, tampered,
                        resolution=TEST_RESOLUTION) == pytest.approx(ref)


def test_contrast_fades_but_does_not_vanish(base):
    still = D.DriftParams(jitter_rate=0.0, radius_jitter_rate=0.0,
                          darkness_jitter_rate=0.0, scar_rate_per_year=0.0)
    out = D.resight(base, T0, _later(3650), params=still,
                    growth_model=D.NoGrowth())
    assert out.spots["darkness"].mean() < base.spots["darkness"].mean()
    assert (out.spots["darkness"].mean()
            > still.contrast_floor * 0.95 * base.spots["darkness"].mean())


def test_melanism_patch_changes_area_at_the_white_shark_rate(base):
    patch = P.MelanismPatch(0.5, 0.3, 0.04, 0.25)
    ind = base.copy(melanism=patch)
    out = D.resight(ind, T0, _later(274), rng=np.random.default_rng([5]))
    assert out.melanism is not None
    change = abs(math.log((out.melanism.radius / patch.radius) ** 2))
    assert 0.0 < change < 1.5   # 33% area change in 9 months is |ln| = 0.40


def test_drift_params_replace_and_validation():
    dp = D.DriftParams()
    assert dp.replace(jitter_rate=0.01).jitter_rate == 0.01
    assert dp.jitter_rate == D.DEFAULT_JITTER_RATE
    with pytest.raises(TypeError):
        dp.replace(not_a_field=1)
    assert "jitter_rate" in dp.as_dict()
