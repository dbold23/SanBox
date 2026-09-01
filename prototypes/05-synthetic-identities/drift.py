"""Resighting drift: the same animal, photographed again months or years later.

``resight(individual, t0, t1, growth_model, rng)`` returns a new
:class:`pattern.Individual` for date ``t1``. Five mechanisms act, each with
its own evidence bracket:

1. GROWTH spreads a FIXED cell population. Spots are laid down once and
   spread as the animal grows, so growth scales spot SPACING and never spot
   COUNT [derived from the Melops campaign's size-assortative matching result
   -- ``prototypes/01-melops-ablation/results/CAMPAIGN.md`` reports a
   size-assortativity index of 0.338 that training never moved, i.e. apparent
   similarity tracks body size; label this derivation, it is not a measured
   sevengill result].

   THE SUBTLETY THAT MATTERS IN CHART SPACE. ``s`` is a NORMALISED
   arc-length fraction and ``phi`` a normalised angle, so PURELY ISOTROPIC
   growth moves nothing in the chart -- it only multiplies the physical
   spacing in centimetres. Chart-space motion comes from ALLOMETRY: the head
   shortens as a fraction of total length with growth, and the trunk deepens.
   So this module scales absolute spacing with the length ratio (visible via
   ``Individual.spot_spacing('cm')``) and moves chart coordinates only
   through two small allometric terms, both [UNVERIFIED]:
   ``s' = s ** (1 + head_allometry * ln r)`` (order-preserving, fixes both
   ends) and ``phi' = phi + phi_allometry * ln r * sin phi`` (fixes the
   dorsal and ventral midlines, moves the flanks).

2. JITTER. Each mark random-walks in the chart: displacement sigma grows as
   ``jitter_rate * sqrt(elapsed_years)`` (Brownian, isotropic in the scaled
   metric). See the calibration note on :data:`DEFAULT_JITTER_RATE`.

3. CONTRAST FADE toward a floor, plus per-mark darkness jitter. Pigmentation
   is not permanent in a shark: a white-shark melanistic islet lost 33% of
   its area in 9 months and the newly melanised region did not match the
   original pattern [SEARCH, docs/sevengill-canonical-reid Q0b].

4. SCARS appear (Poisson in time), heal on the reef-manta negative
   exponential (5% of initial length by 295 days), and a documented fraction
   stabilise into permanent residue [SEARCH; see ``pattern.SCAR_TAU_DAYS``].
   ~22.7% of a Ningaloo manta catalogue carried a scar at all -- a plausible
   prior for what fraction of sevengills carry a usable scar [SEARCH].

5. MELANISM PATCH (optional, off by default): a large low-contrast patch
   whose area changes at the white-shark rate.

CALIBRATION TARGET, AND WHAT IT IS NOT. The Melops campaign measured
true-mate similarity falling 0.605 -> 0.474 between the <30-day and 2+-year
bins. That is an ArcFace EMBEDDING cosine between two PHOTOGRAPHS, so its
level is set mostly by photographic nuisance (pose, turbidity, exposure) that
this module does not model; a noiseless chart NCC of an individual against
itself at zero elapsed time is 1.0 by construction. The defensible target is
therefore the RATIO, ``0.474 / 0.605 = 0.784`` over ~2 years, and the
direction. :data:`DEFAULT_JITTER_RATE` was chosen by
:func:`calibrate_jitter_rate` to hit that ratio; it is a calibration, not a
fit, and nothing here should be reported as "matching Melops".
"""

from __future__ import annotations

import math
import numpy as np

from exclusions import wrap_phi, mask_from_regions
from pattern import (  # noqa: F401  (Individual/PatternParams re-exported)
    Individual,
    MelanismPatch,
    PatternParams,
    Scar,
    SPOT_DTYPE,
    as_date,
    days_between,
    render_chart,
    scar_visibility,
)
from exclusions import regions_contain

__all__ = [
    "DriftParams",
    "GrowthModel",
    "VonBertalanffyGrowth",
    "LinearGrowth",
    "NoGrowth",
    "DEFAULT_JITTER_RATE",
    "MELOPS_SIMILARITY_NEAR",
    "MELOPS_SIMILARITY_FAR",
    "MELOPS_RATIO",
    "resight",
    "similarity",
    "similarity_curve",
    "calibrate_jitter_rate",
]

DAYS_PER_YEAR = 365.25

# The Melops campaign's measured true-mate similarity bins [PRIMARY to this
# repository: prototypes/01-melops-ablation/results/CAMPAIGN.md, point 3].
MELOPS_SIMILARITY_NEAR = 0.605   # < 30 days
MELOPS_SIMILARITY_FAR = 0.474    # 2+ years
MELOPS_RATIO = MELOPS_SIMILARITY_FAR / MELOPS_SIMILARITY_NEAR  # 0.7835

# Chart-space Brownian jitter rate, in s-units per sqrt(year).
#
# HOW THIS DEFAULT WAS CHOSEN: calibrate_jitter_rate() bisects on the rate
# until the mean chart NCC ratio similarity(t=730 d) / similarity(t=30 d),
# over 6 seeded individuals at 192x384 with default PatternParams, equals
# MELOPS_RATIO = 0.784. Re-run it after ANY change to the default spot size,
# density or rendering:
#     python -c "import drift; print(drift.calibrate_jitter_rate())"
# The bisection above returned rate 0.001244 -> ratio 0.786 (target 0.784) on
# 2026-09-01 with the defaults in this file. 0.00124 s-units/sqrt(yr) is
# 0.31 cm/sqrt(yr) at 250 cm TL: a mark wanders ~0.44 cm in two years, about
# a third of its own 1.4 cm radius. Contrast fade and new scars carry the
# rest of the decay.
# [CALIBRATED against a ratio, not measured. No sevengill mark-displacement
# measurement exists; the campaign's own note is that no quantified per-year
# mark-change rate exists for ANY shark species.]
DEFAULT_JITTER_RATE = 0.001244


class GrowthModel(object):
    """Interface: ``length_ratio(length_cm, dt_years) -> L(t1)/L(t0)``."""

    def length_ratio(self, length_cm, dt_years):
        raise NotImplementedError

    def grown_length(self, length_cm, dt_years):
        return float(length_cm) * self.length_ratio(length_cm, dt_years)


class NoGrowth(GrowthModel):
    """Ratio 1 always. Use when growth must be held out of an experiment."""

    def length_ratio(self, length_cm, dt_years):
        return 1.0


class VonBertalanffyGrowth(GrowthModel):
    """``L(t1) = Linf - (Linf - L0) * exp(-k * dt)``.

    Defaults ``Linf = 290 cm``, ``k = 0.06 / yr`` are [UNVERIFIED]: a
    plausible slow-growing large hexanchiform, chosen so a 250 cm animal
    gains ~2.3 cm in a year. No Notorynchus cepedianus growth curve was
    retrieved in this programme's scans; pass measured parameters when they
    exist. Bracket the defaults as Linf in [270, 310] cm, k in [0.03, 0.10].
    """

    def __init__(self, l_inf_cm=290.0, k_per_year=0.06):
        self.l_inf_cm = float(l_inf_cm)
        self.k_per_year = float(k_per_year)

    def length_ratio(self, length_cm, dt_years):
        l0 = float(length_cm)
        if l0 <= 0:
            raise ValueError("length_cm must be positive")
        l1 = self.l_inf_cm - (self.l_inf_cm - l0) * math.exp(
            -self.k_per_year * float(dt_years))
        return float(l1 / l0)

    def __repr__(self):
        return "VonBertalanffyGrowth(l_inf_cm=%.1f, k_per_year=%.3f)" % (
            self.l_inf_cm, self.k_per_year)


class LinearGrowth(GrowthModel):
    """Constant ``cm_per_year``; a blunt alternative for short intervals."""

    def __init__(self, cm_per_year=3.0):
        self.cm_per_year = float(cm_per_year)

    def length_ratio(self, length_cm, dt_years):
        return float((float(length_cm) + self.cm_per_year * float(dt_years))
                     / float(length_cm))


class DriftParams(object):
    """Rates for :func:`resight`. Times are in years unless named otherwise."""

    __slots__ = (
        "jitter_rate", "radius_jitter_rate", "darkness_jitter_rate",
        "contrast_fade_per_year", "contrast_floor", "head_allometry",
        "phi_allometry", "scar_rate_per_year", "scar_persist_fraction",
        "scar_residue", "scar_prune_visibility", "melanism_prob_per_year",
        "melanism_area_rate", "s_margin",
    )

    def __init__(self, jitter_rate=None, radius_jitter_rate=0.06,
                 darkness_jitter_rate=0.08, contrast_fade_per_year=0.05,
                 contrast_floor=0.55, head_allometry=0.15, phi_allometry=0.60,
                 scar_rate_per_year=0.6, scar_persist_fraction=0.25,
                 scar_residue=0.15, scar_prune_visibility=0.0,
                 melanism_prob_per_year=0.0, melanism_area_rate=None,
                 s_margin=0.02):
        self.jitter_rate = (DEFAULT_JITTER_RATE if jitter_rate is None
                            else float(jitter_rate))
        self.radius_jitter_rate = float(radius_jitter_rate)
        self.darkness_jitter_rate = float(darkness_jitter_rate)
        self.contrast_fade_per_year = float(contrast_fade_per_year)
        self.contrast_floor = float(contrast_floor)
        self.head_allometry = float(head_allometry)
        self.phi_allometry = float(phi_allometry)
        self.scar_rate_per_year = float(scar_rate_per_year)
        self.scar_persist_fraction = float(scar_persist_fraction)
        self.scar_residue = float(scar_residue)
        self.scar_prune_visibility = float(scar_prune_visibility)
        self.melanism_prob_per_year = float(melanism_prob_per_year)
        # 33% area change in 9 months (white shark, Robbins & Fox) ->
        # |ln(0.67)| / 0.75 yr = 0.535 per year of |ln| area change [SEARCH].
        self.melanism_area_rate = (0.535 if melanism_area_rate is None
                                   else float(melanism_area_rate))
        self.s_margin = float(s_margin)

    def replace(self, **kw):
        out = DriftParams()
        for name in self.__slots__:
            setattr(out, name, getattr(self, name))
        for key, value in kw.items():
            if key not in self.__slots__:
                raise TypeError("unknown DriftParams field %r" % key)
            setattr(out, key, value)
        return out

    def as_dict(self):
        return dict((name, getattr(self, name)) for name in self.__slots__)

    def __repr__(self):
        return "DriftParams(jitter_rate=%.5f, scar_rate_per_year=%.2f)" % (
            self.jitter_rate, self.scar_rate_per_year)


# ---------------------------------------------------------------------------
# The drift operator
# ---------------------------------------------------------------------------


def _allometric_map(s, phi, ratio, params):
    """Chart-space displacement caused by non-isotropic growth."""
    if abs(ratio - 1.0) < 1e-12:
        return np.array(s, dtype=np.float64), np.array(phi, dtype=np.float64)
    ln_r = math.log(ratio)
    gamma = 1.0 + params.head_allometry * ln_r
    s_out = np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0) ** gamma
    phi_out = wrap_phi(np.asarray(phi, dtype=np.float64)
                       + params.phi_allometry * ln_r * np.sin(phi))
    return s_out, phi_out


def resight(individual, t0, t1, growth_model=None, rng=None, params=None):
    """Return the same animal as it appears at ``t1``.

    Parameters
    ----------
    individual : pattern.Individual
        The appearance at ``t0``. ``individual.date`` must equal ``t0``
        (a mismatch raises, so a resighting chain cannot silently skip time).
    t0, t1 : date-like
        ``t1 >= t0``. ``t1 == t0`` is a legal no-op resighting (growth ratio
        1, zero jitter, no new scars) -- useful as a control arm.
    growth_model : GrowthModel, optional
        Default :class:`VonBertalanffyGrowth`.
    rng : numpy.random.Generator, optional
        Default ``default_rng([individual.seed, elapsed_days])``, so a
        resighting is reproducible from the individual and the interval alone.
    params : DriftParams, optional

    Guarantees (all covered by tests):
    * ``len(out.spots) == len(individual.spots)`` -- growth and drift never
      create or destroy a spot, and ``out.spots["id"]`` is the same physical
      mark, so a resighting is a correspondence, not just a re-render.
    * no spot lands inside an exclusion region (a jitter step that would put
      one there is rejected and the mark keeps its position).
    * ``out.spots["birth_date"]`` is unchanged; only scars carry new dates.
    """
    params = params or DriftParams()
    growth_model = growth_model or VonBertalanffyGrowth()
    t0, t1 = as_date(t0), as_date(t1)
    if as_date(individual.date) != t0:
        raise ValueError(
            "individual.date is %s but t0 is %s; resight() drifts FROM the "
            "individual's own date" % (individual.date, t0))
    dt_days = days_between(t0, t1)
    if dt_days < 0:
        raise ValueError("t1 must not precede t0")
    dt_years = dt_days / DAYS_PER_YEAR
    if rng is None:
        rng = np.random.default_rng([int(individual.seed) & 0x7FFFFFFF,
                                     int(round(dt_days))])

    ratio = float(growth_model.length_ratio(individual.length_cm, dt_years))
    spots = individual.spots.copy()
    n = len(spots)

    # 1. growth: allometric chart motion (isotropic growth moves nothing here)
    s_new, phi_new = _allometric_map(spots["s"], spots["phi"], ratio, params)

    # 2. Brownian jitter, isotropic in the scaled chart metric
    sigma = params.jitter_rate * math.sqrt(max(dt_years, 0.0))
    if sigma > 0 and n:
        s_new = s_new + rng.normal(0.0, sigma, size=n)
        phi_new = wrap_phi(phi_new
                           + rng.normal(0.0, sigma, size=n)
                           / individual.params.phi_scale)
    s_new = np.clip(s_new, params.s_margin, 1.0 - params.s_margin)

    # reject any step that would land a mark in an excluded region
    if n and individual.regions:
        bad = regions_contain(individual.regions, s_new, phi_new)
        if bad.any():
            s_new[bad] = spots["s"][bad]
            phi_new[bad] = spots["phi"][bad]
    spots["s"] = s_new
    spots["phi"] = phi_new

    # 3. size and contrast
    if n:
        spots["radius"] = spots["radius"] * np.exp(
            rng.normal(0.0, params.radius_jitter_rate * math.sqrt(dt_years),
                       size=n))
        fade = (params.contrast_floor + (1.0 - params.contrast_floor)
                * math.exp(-params.contrast_fade_per_year * dt_years))
        spots["darkness"] = np.clip(
            spots["darkness"] * fade * np.exp(
                rng.normal(0.0, params.darkness_jitter_rate
                           * math.sqrt(dt_years), size=n)),
            0.0, 1.0)

    # 4. scars: keep the old ones (visibility is a render-time function of
    #    age), add Poisson arrivals with dates spread over the interval
    scars = list(individual.scars)
    next_id = (max(sc.id for sc in scars) + 1) if scars else 0
    n_new = int(rng.poisson(params.scar_rate_per_year * dt_years))
    pp = individual.params
    for _ in range(n_new):
        for _attempt in range(200):
            s_c = float(rng.uniform(params.s_margin, 1.0 - params.s_margin))
            phi_c = float(rng.uniform(-math.pi, math.pi))
            if not bool(regions_contain(individual.regions,
                                        np.array(s_c), np.array(phi_c))):
                break
        else:
            continue
        length = float(pp.scar_length_median
                       * math.exp(rng.normal(0.0, pp.scar_length_log_sigma)))
        born = t0 + np.timedelta64(int(rng.integers(1, max(int(dt_days), 1) + 1)), "D")
        scars.append(Scar(
            id=next_id, s=s_c, phi=phi_c, length=length,
            width=length * pp.scar_width_frac,
            angle=float(rng.uniform(-math.pi / 2.0, math.pi / 2.0)),
            darkness=float(np.clip(rng.normal(pp.scar_darkness_mean, 0.12),
                                   0.1, 1.0)),
            birth_date=born,
            persist=bool(rng.random() < params.scar_persist_fraction),
            residue=params.scar_residue,
        ))
        next_id += 1
    if params.scar_prune_visibility > 0:
        scars = [sc for sc in scars
                 if scar_visibility(sc, t1) >= params.scar_prune_visibility]

    # 5. melanism patch (optional)
    melanism = individual.melanism
    if melanism is not None and dt_years > 0:
        factor = math.exp(rng.normal(0.0, params.melanism_area_rate
                                     * math.sqrt(dt_years)) * 0.5)
        melanism = MelanismPatch(melanism.s, melanism.phi,
                                 melanism.radius * factor, melanism.darkness)
    elif (melanism is None and params.melanism_prob_per_year > 0
          and rng.random() < 1.0 - math.exp(-params.melanism_prob_per_year * dt_years)):
        for _attempt in range(200):
            s_c = float(rng.uniform(0.15, 0.85))
            phi_c = float(rng.uniform(-1.2, 1.2))
            if not bool(regions_contain(individual.regions,
                                        np.array(s_c), np.array(phi_c))):
                melanism = MelanismPatch(s_c, phi_c,
                                         float(rng.uniform(0.02, 0.05)),
                                         float(rng.uniform(0.15, 0.35)))
                break

    provenance = dict(individual.provenance)
    provenance.update({
        "origin": "resight",
        "parent_identity": individual.identity,
        "parent_date": str(individual.date),
        "t0": str(t0), "t1": str(t1),
        "elapsed_days": float(dt_days),
        "growth_ratio": ratio,
        "growth_model": repr(growth_model),
        "jitter_sigma": float(sigma),
        "new_scars": int(n_new),
        "drift_params": params.as_dict(),
    })
    return individual.copy(
        spots=spots, scars=tuple(scars), date=t1,
        length_cm=individual.length_cm * ratio, melanism=melanism,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------


def similarity(ind_a, ind_b, resolution=(192, 384), date_a=None, date_b=None,
               mask_excluded=True, render_kwargs=None):
    """Zero-mean normalised cross-correlation of two rendered charts, in [-1, 1].

    This is a CHART-SPACE similarity: both animals are rendered in the same
    canonical (s, phi) frame, so it measures pattern change alone, with no
    pose, lighting or detection nuisance. It is therefore NOT commensurate
    with the Melops embedding cosine -- see the module docstring. Excluded
    pixels are dropped from the correlation (identity signal may not be
    counted where no identity pattern may exist).
    """
    kw = dict(render_kwargs or {})
    img_a, _ = render_chart(ind_a, resolution, date=date_a, **kw)
    img_b, _ = render_chart(ind_b, resolution, date=date_b, **kw)
    if mask_excluded:
        regions = tuple(ind_a.regions) + tuple(ind_b.regions)
        keep = ~mask_from_regions(regions, resolution) if regions else None
    else:
        keep = None
    a = img_a[keep] if keep is not None else img_a.ravel()
    b = img_b[keep] if keep is not None else img_b.ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float(a.dot(a)) * float(b.dot(b)))
    if denom <= 0:
        return 0.0
    return float(a.dot(b) / denom)


def similarity_curve(individual, elapsed_days=(0, 30, 180, 365, 730),
                     growth_model=None, params=None, seed=0,
                     resolution=(192, 384)):
    """Chart NCC of ``individual`` against its own resighting at each interval.

    Returns a list of ``(days, similarity)``. Each interval is drifted from
    the SAME t0 with its own generator seeded from ``(seed, days)``, so the
    points are independent draws of "the animal after d days", not a single
    trajectory -- which is what a resighting-interval curve means.
    """
    t0 = individual.date
    out = []
    for d in elapsed_days:
        rng = np.random.default_rng([int(seed), int(d), 0xD21F7])
        later = resight(individual, t0, t0 + np.timedelta64(int(d), "D"),
                        growth_model=growth_model, rng=rng, params=params)
        out.append((int(d), similarity(individual, later,
                                       resolution=resolution)))
    return out


def calibrate_jitter_rate(target_ratio=None, n_individuals=6,
                          near_days=30, far_days=730, resolution=(192, 384),
                          lo=0.0002, hi=0.02, tol=0.004, max_iter=18,
                          seed=0, pattern_params=None, drift_params=None,
                          verbose=False):
    """Bisect the Brownian jitter rate onto a chart-NCC decay RATIO.

    Returns ``(rate, achieved_ratio)``. The objective is
    ``mean similarity(far_days) / mean similarity(near_days)`` over
    ``n_individuals`` seeded animals, and the target defaults to
    :data:`MELOPS_RATIO` = 0.784, the Melops true-mate decay 0.605 -> 0.474.

    This function is how :data:`DEFAULT_JITTER_RATE` was chosen, and it is
    the only thing that licenses that constant. It is a CALIBRATION to a
    ratio measured on a different species with a different similarity
    function; it is not a fit to sevengill data, and there is no sevengill
    data to fit.
    """
    from pattern import randomize  # local import keeps module import cheap

    target = MELOPS_RATIO if target_ratio is None else float(target_ratio)
    inds = [randomize(1000 + i, params=pattern_params)
            for i in range(int(n_individuals))]

    def ratio_for(rate):
        dp = (drift_params or DriftParams()).replace(jitter_rate=float(rate))
        near, far = [], []
        for i, ind in enumerate(inds):
            for days, bucket in ((near_days, near), (far_days, far)):
                rng = np.random.default_rng([int(seed), i, int(days)])
                later = resight(ind, ind.date,
                                ind.date + np.timedelta64(int(days), "D"),
                                rng=rng, params=dp)
                bucket.append(similarity(ind, later, resolution=resolution))
        return float(np.mean(far)) / float(np.mean(near))

    r_lo, r_hi = ratio_for(lo), ratio_for(hi)
    if verbose:
        print("bracket: rate %.5f -> ratio %.4f | rate %.5f -> ratio %.4f"
              % (lo, r_lo, hi, r_hi))
    if not (r_hi <= target <= r_lo):
        raise ValueError(
            "target ratio %.4f not bracketed by [%.4f, %.4f]; widen lo/hi"
            % (target, r_hi, r_lo))
    best = (lo, r_lo)
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        r_mid = ratio_for(mid)
        if verbose:
            print("  rate %.5f -> ratio %.4f" % (mid, r_mid))
        best = (mid, r_mid)
        if abs(r_mid - target) <= tol:
            break
        if r_mid > target:
            lo = mid
        else:
            hi = mid
    return best
