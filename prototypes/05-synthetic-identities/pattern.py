"""Chart-space identity patterns for synthetic sevengills.

Three of the four owner-requested verbs live here:

* RANDOMIZE  -- :func:`randomize` / :meth:`Individual.generate` draw a fresh
  blue-noise speckle field for a new individual from a seed;
* COPY       -- :func:`copy_from_chart` fits a spot table to a chart image
  that came from somewhere else (a real animal's extracted pattern), so a
  real individual can be replayed on the synthetic body;
* RENDER     -- :func:`render_chart` rasterises either of the above to a
  ``(H_phi, W_s)`` darkness image plus the exact ground-truth spot table.

(The fourth verb, resighting drift, is :mod:`drift`.)

Coordinates are the canonical chart space defined in :mod:`exclusions` --
``s`` = arc-length fraction snout(0) -> caudal(1), ``phi`` = circumferential
angle, 0 dorsal, +pi/2 the animal's LEFT flank, +/-pi ventral. Nothing here
touches a mesh UV atlas; baking to UV is a later step that needs per-vertex
(s, phi) (prototype 04's ``mesh3d.tube_coords`` for real meshes).

THE CHART METRIC. Distances mix an arc-length fraction with an angle, so a
single scale factor ``phi_scale`` converts radians to s-units:
``dist = hypot(ds, phi_scale * wrap(dphi))``. The default 0.085 says one
radian of girth is 0.085 of the body axis, i.e. a girth of
``2*pi*0.085 = 0.53`` of total length -- a plausible stout-bodied
hexanchiform proportion [UNVERIFIED, DERIVED from gross form; no measured
girth/TL ratio for Notorynchus cepedianus was retrieved]. Every length
parameter below (spot radius, minimum separation, jitter) is in these
s-units, so ``value * length_cm`` is centimetres on the animal.

WHAT IS EXTENDED, NOT DUPLICATED. The spot-field idea is
``prototypes/02-centerline-chart/strain_demo.py`` (seeded procedural spots in
body-frame (s, r)) lifted from a flat half-width chart into a closed
circumferential chart, with size/eccentricity/darkness distributions and an
exclusion mask added. The per-region identity-signal knobs
(``head_signal`` / ``flank_signal`` / ``tail_signal``) are the direct
generalisation of ``melops_data.make_synthetic``'s ``head_signal`` /
``body_signal``: amplitude 0 in a region strips the identity signal there
while an optional shared confounder texture stays, so a head-vs-flank
ablation can be built synthetically.

EVIDENCE for the appearance model (grade each before citing):
* countershading, dark speckling on a grey-brown dorsum, seven gill slits,
  single posterior dorsal -- species description [SEARCH];
* spot patterns are machine-matchable in an elasmobranch over 496 days
  (blue-spotted ribbontail ray, 90.3% I3S) -- so a spot field is the right
  identity primitive [SEARCH, docs/sevengill-canonical-reid Q0b];
* pigmentation is NOT permanent in a shark (white-shark melanistic islet
  -33% area in 9 months) -- so patterns must be allowed to drift [SEARCH];
* spot SIZE, DARKNESS and ECCENTRICITY distributions are [UNVERIFIED]: no
  measured sevengill speckle morphometry was retrieved. The defaults are
  round numbers chosen to look right at 250 cm TL and are parameters, not
  findings.
"""

from __future__ import annotations

import math
import warnings
import numpy as np
from scipy import ndimage

# Region, chart_meshgrid and countershading_weight are re-exported for sibling
# modules that want the chart primitives without a second import.
from exclusions import (  # noqa: F401  (deliberate re-exports)
    CHART_CONVENTION,
    TWO_PI,
    Region,
    chart_axes,
    chart_meshgrid,
    countershading_weight,
    countershading_weight_at,
    default_stations,
    exclusion_regions,
    load_schema,
    mask_from_regions,
    regions_contain,
    wrap_phi,
)

__all__ = [
    "SPOT_DTYPE",
    "PatternParams",
    "Scar",
    "Individual",
    "randomize",
    "render_chart",
    "copy_from_chart",
    "chart_exclusion_mask",
    "EXCLUSION_MASK_INCLUDE_GILL_SLITS",
    "scar_table",
    "scar_visibility",
    "isotropic_resolution",
    "nearest_neighbour_spacing",
    "recoverable_spot_count",
    "empty_spots",
    "DEFAULT_SCHEMA_PATH",
]

DEFAULT_SCHEMA_PATH = (
    "/home/user/SanBox/phase1b/p0-sevengill-schema/keypoints_sevengill_v1.yaml"
)

SPOT_DTYPE = np.dtype([
    ("id", "i4"),
    ("s", "f8"),
    ("phi", "f8"),
    ("radius", "f8"),          # s-units, equivalent-circle radius
    ("eccentricity", "f8"),    # major/minor axis ratio, >= 1
    ("angle", "f8"),           # major-axis orientation in the scaled chart
    ("darkness", "f8"),        # intrinsic, before region signal/countershading
    ("birth_date", "M8[D]"),
])

_RENDERED_DTYPE = np.dtype(SPOT_DTYPE.descr + [("rendered_darkness", "f8")])


def empty_spots(n=0):
    """An all-zero spot table of length ``n`` with :data:`SPOT_DTYPE`."""
    out = np.zeros(int(n), dtype=SPOT_DTYPE)
    out["birth_date"] = np.datetime64("1970-01-01", "D")
    return out


def as_date(value):
    """Coerce ``str`` / ``datetime.date`` / ``datetime64`` to ``datetime64[D]``."""
    return np.datetime64(value, "D")


def days_between(t0, t1):
    """Signed elapsed days between two dates, as a float."""
    return float((as_date(t1) - as_date(t0)) / np.timedelta64(1, "D"))


def isotropic_resolution(h_phi, phi_scale=0.085):
    """``(H_phi, W_s)`` whose pixels are square in the scaled chart metric.

    A chart pixel is ``1/W`` wide in s and ``(2*pi/H) * phi_scale`` tall in
    s-equivalent units; equal when ``W = H / (2*pi*phi_scale)``.
    """
    h = int(h_phi)
    return (h, int(round(h / (TWO_PI * float(phi_scale)))))


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


class PatternParams(object):
    """Generative parameters for one individual's speckle field.

    All lengths are in chart s-units (multiply by ``length_cm`` for cm).

    Parameters
    ----------
    n_spots_target : int
        Number of identity spots to place. The realised count can be lower if
        ``min_sep`` is too large for the admissible area; :class:`Individual`
        records both. Spot COUNT is a fixed cell population -- growth spreads
        the spots, it never adds any (see :mod:`drift`).
    phi_scale : float
        Radians -> s-units conversion for the chart metric (see module doc).
    min_sep : float
        Blue-noise / Poisson-disc minimum centre separation in the scaled
        metric. 0.030 s-units = 7.5 cm at 250 cm TL [UNVERIFIED].
    radius_median, radius_log_sigma : float
        Lognormal equivalent-circle radius. 0.0055 s-units = 1.4 cm radius
        (~2.8 cm spots) at 250 cm TL [UNVERIFIED].
    ecc_sigma, ecc_max : float
        Eccentricity is ``1 + |N(0, ecc_sigma)|`` clipped at ``ecc_max``.
    darkness_mean, darkness_sigma, darkness_min, darkness_max : float
        Truncated-normal intrinsic darkness in [0, 1].
    dorsal_exponent : float
        Density prior ``((1 + cos phi)/2) ** dorsal_exponent``: dorsal-heavy,
        falling off ventrally, on top of the countershading weight.
    countershading : dict or None
        Keyword overrides for :func:`exclusions.countershading_weight_at`.
    head_signal, flank_signal, tail_signal : float
        Per-region identity amplitude, mirroring
        ``melops_data.make_synthetic(head_signal=, body_signal=)``. Multiplies
        rendered darkness; 0 removes the identity signal from that region.
    head_s_max, flank_s_max : float
        Region boundaries in s. Defaults come from the station table
        (``gill_slit_7_dorsal_origin`` and ``precaudal_pit``), so the head
        region is the rigid anterior block the Phase-1B primary arm uses.
    n_common, common_darkness, common_seed : int, float, int
        Optional shared (non-identity) confounder texture, identical for every
        individual. Off by default; turn it on so that a zero-signal region is
        textured but uninformative, as in ``make_synthetic``.
    n_scars, scar_* : initial scar load at generation time.
    """

    __slots__ = (
        "n_spots_target", "phi_scale", "min_sep", "radius_median",
        "radius_log_sigma", "ecc_sigma", "ecc_max", "darkness_mean",
        "darkness_sigma", "darkness_min", "darkness_max", "dorsal_exponent",
        "countershading", "head_signal", "flank_signal", "tail_signal",
        "head_s_max", "flank_s_max", "n_common", "common_darkness",
        "common_seed", "n_scars", "scar_length_median", "scar_length_log_sigma",
        "scar_width_frac", "scar_darkness_mean", "scar_persist_fraction",
        "s_margin", "max_attempts",
    )

    def __init__(self, n_spots_target=240, phi_scale=0.085, min_sep=0.030,
                 radius_median=0.0055, radius_log_sigma=0.35, ecc_sigma=0.28,
                 ecc_max=2.4, darkness_mean=0.55, darkness_sigma=0.15,
                 darkness_min=0.15, darkness_max=1.0, dorsal_exponent=0.8,
                 countershading=None, head_signal=1.0, flank_signal=1.0,
                 tail_signal=1.0, head_s_max=None, flank_s_max=None,
                 n_common=0, common_darkness=0.18, common_seed=20250901,
                 n_scars=0, scar_length_median=0.035,
                 scar_length_log_sigma=0.45, scar_width_frac=0.18,
                 scar_darkness_mean=0.45, scar_persist_fraction=0.25,
                 s_margin=0.02, max_attempts=40000):
        self.n_spots_target = int(n_spots_target)
        self.phi_scale = float(phi_scale)
        self.min_sep = float(min_sep)
        self.radius_median = float(radius_median)
        self.radius_log_sigma = float(radius_log_sigma)
        self.ecc_sigma = float(ecc_sigma)
        self.ecc_max = float(ecc_max)
        self.darkness_mean = float(darkness_mean)
        self.darkness_sigma = float(darkness_sigma)
        self.darkness_min = float(darkness_min)
        self.darkness_max = float(darkness_max)
        self.dorsal_exponent = float(dorsal_exponent)
        self.countershading = dict(countershading or {})
        self.head_signal = float(head_signal)
        self.flank_signal = float(flank_signal)
        self.tail_signal = float(tail_signal)
        self.head_s_max = head_s_max
        self.flank_s_max = flank_s_max
        self.n_common = int(n_common)
        self.common_darkness = float(common_darkness)
        self.common_seed = int(common_seed)
        self.n_scars = int(n_scars)
        self.scar_length_median = float(scar_length_median)
        self.scar_length_log_sigma = float(scar_length_log_sigma)
        self.scar_width_frac = float(scar_width_frac)
        self.scar_darkness_mean = float(scar_darkness_mean)
        self.scar_persist_fraction = float(scar_persist_fraction)
        self.s_margin = float(s_margin)
        self.max_attempts = int(max_attempts)

    def replace(self, **kw):
        """A copy with the named fields overridden."""
        out = PatternParams()
        for name in self.__slots__:
            setattr(out, name, getattr(self, name))
        for key, value in kw.items():
            if key not in self.__slots__:
                raise TypeError("unknown PatternParams field %r" % key)
            setattr(out, key, value)
        return out

    def as_dict(self):
        return dict((name, getattr(self, name)) for name in self.__slots__)

    def __repr__(self):
        return "PatternParams(n_spots_target=%d, min_sep=%.3f, signals=(%.2f, %.2f, %.2f))" % (
            self.n_spots_target, self.min_sep, self.head_signal,
            self.flank_signal, self.tail_signal,
        )


class Scar(object):
    """An elongated mark with a creation date and a healing history.

    ``length`` and ``width`` are in scaled chart units (s-units); ``angle`` is
    the major-axis orientation in the scaled chart. ``persist`` marks the
    fraction of marks that stabilise into permanent residue rather than
    healing away -- the reef-manta result (healed to 5% of initial length by
    295 days, then the residue "stabilised into a distinctive pattern still
    present >3 years later") [SEARCH].
    """

    __slots__ = ("id", "s", "phi", "length", "width", "angle", "darkness",
                 "birth_date", "persist", "residue")

    def __init__(self, id, s, phi, length, width, angle, darkness, birth_date,
                 persist=False, residue=0.15):
        self.id = int(id)
        self.s = float(s)
        self.phi = float(wrap_phi(phi))
        self.length = float(length)
        self.width = float(width)
        self.angle = float(angle)
        self.darkness = float(darkness)
        self.birth_date = as_date(birth_date)
        self.persist = bool(persist)
        self.residue = float(residue)

    def __repr__(self):
        return "Scar(%d, s=%.3f, phi=%.2f, len=%.3f, born=%s, persist=%s)" % (
            self.id, self.s, self.phi, self.length, str(self.birth_date),
            self.persist,
        )


# Healing curve. Reef manta laceration: 5% of initial length by 295 days,
# negative exponential, ~1/3 healed after one month [SEARCH,
# docs/sevengill-canonical-reid/01-evidence-and-answers.md Q0b]. Solving
# exp(-295/tau) = 0.05 gives tau = 295/ln(20) = 98.5 days. Sanity check
# against the blacktip reef shark record ("undetectable within 179 days",
# mating scars healed within a month): exp(-179/98.5) = 0.16, so at ~6 months
# a mark is at 16% of its initial contrast -- consistent with "undetectable"
# in field imagery, and mating-scale marks fade to ~0.74 in a month, which is
# the weakest part of the fit. Treat SCAR_TAU_DAYS as a bracket [60, 130],
# not a constant.
SCAR_TAU_DAYS = 98.5


def scar_visibility(scar, date):
    """Contrast multiplier in [0, 1] for a scar at ``date``.

    Negative-exponential healing with time constant :data:`SCAR_TAU_DAYS`,
    floored at ``scar.residue`` for scars flagged ``persist``. Zero before the
    scar exists.
    """
    age = days_between(scar.birth_date, date)
    if age < 0:
        return 0.0
    v = math.exp(-age / SCAR_TAU_DAYS)
    if scar.persist:
        v = max(v, scar.residue)
    return float(v)


def scar_table(individual, date=None):
    """``(n, )`` structured view of the scars visible at ``date``."""
    date = individual.date if date is None else date
    dtype = np.dtype([("id", "i4"), ("s", "f8"), ("phi", "f8"),
                      ("length", "f8"), ("width", "f8"), ("angle", "f8"),
                      ("darkness", "f8"), ("birth_date", "M8[D]"),
                      ("persist", "?"), ("visibility", "f8")])
    out = np.zeros(len(individual.scars), dtype=dtype)
    for i, sc in enumerate(individual.scars):
        out[i] = (sc.id, sc.s, sc.phi, sc.length, sc.width, sc.angle,
                  sc.darkness, sc.birth_date, sc.persist,
                  scar_visibility(sc, date))
    return out


class MelanismPatch(object):
    """A large low-contrast pigmentation patch that changes area over time.

    Motivated by the white-shark observation of a melanistic islet losing 33%
    of its area in 9 months, with the newly melanised region 10% darker and
    not matching the original pattern [SEARCH]. Off by default.
    """

    __slots__ = ("s", "phi", "radius", "darkness")

    def __init__(self, s, phi, radius, darkness):
        self.s = float(s)
        self.phi = float(wrap_phi(phi))
        self.radius = float(radius)
        self.darkness = float(darkness)

    def __repr__(self):
        return "MelanismPatch(s=%.3f, phi=%.2f, r=%.3f, d=%.2f)" % (
            self.s, self.phi, self.radius, self.darkness)


# ---------------------------------------------------------------------------
# Individual
# ---------------------------------------------------------------------------


class Individual(object):
    """A chart-space identity: a spot table, scars, an exclusion geometry.

    An ``Individual`` is a POSE-FREE, MESH-FREE description of one animal's
    skin at one date. Rendering it (:func:`render_chart`) gives a chart image
    whose every pixel has exact ground-truth (s, phi); drifting it
    (:func:`drift.resight`) gives the same animal at a later date.

    Attributes
    ----------
    identity : str            catalogue identity label
    seed : int                the seed the pattern was drawn from (or -1 for a copy)
    params : PatternParams
    spots : ndarray[SPOT_DTYPE]
    scars : tuple of Scar
    melanism : MelanismPatch or None
    date : datetime64[D]      the date this appearance is valid for
    length_cm : float         total length at ``date`` (drives growth)
    regions : tuple of Region exclusion geometry (analytic, resolution-free)
    provenance : dict         how this object was made
    """

    def __init__(self, identity, seed, params, spots, scars=(), date="2020-01-01",
                 length_cm=250.0, regions=(), melanism=None, provenance=None):
        self.identity = str(identity)
        self.seed = int(seed)
        self.params = params
        self.spots = np.asarray(spots, dtype=SPOT_DTYPE)
        self.scars = tuple(scars)
        self.melanism = melanism
        self.date = as_date(date)
        self.length_cm = float(length_cm)
        self.regions = tuple(regions)
        self.provenance = dict(provenance or {})

    # -- construction -----------------------------------------------------
    @classmethod
    def generate(cls, seed, params=None, identity=None, date="2020-01-01",
                 length_cm=250.0, regions=None, schema_path=None,
                 stations=None):
        """RANDOMIZE: draw a fresh blue-noise speckle field for a new animal.

        Deterministic in ``seed`` alone (given identical ``params``,
        ``regions`` and ``length_cm``): two calls produce byte-identical spot
        tables.

        ``regions`` defaults to :func:`exclusions.exclusion_regions` built from
        ``schema_path`` (default :data:`DEFAULT_SCHEMA_PATH`) -- no spot is
        ever placed inside one.
        """
        params = params or PatternParams()
        if regions is None:
            schema = load_schema(schema_path or DEFAULT_SCHEMA_PATH)
            st = default_stations(schema) if stations is None else stations
            regions = exclusion_regions(schema, stations=st)
            if params.head_s_max is None:
                params = params.replace(
                    head_s_max=float(st["gill_slit_7_dorsal_origin"]))
            if params.flank_s_max is None:
                params = params.replace(flank_s_max=float(st["precaudal_pit"]))
        regions = tuple(regions)
        rng = np.random.default_rng([int(seed), 0x5EA7])
        spots, attempts = _poisson_disc_spots(rng, params, regions, date)
        scars = _initial_scars(rng, params, regions, date)
        identity = identity if identity is not None else "syn%06d" % int(seed)
        return cls(
            identity=identity, seed=seed, params=params, spots=spots,
            scars=scars, date=date, length_cm=length_cm, regions=regions,
            provenance={
                "origin": "randomize",
                "seed": int(seed),
                "requested_spots": params.n_spots_target,
                "realised_spots": int(len(spots)),
                "dart_attempts": int(attempts),
                "chart_convention": CHART_CONVENTION,
            },
        )

    # -- derived ----------------------------------------------------------
    def copy(self, **kw):
        """A shallow copy with the named attributes replaced."""
        fields = dict(
            identity=self.identity, seed=self.seed, params=self.params,
            spots=self.spots.copy(), scars=self.scars, date=self.date,
            length_cm=self.length_cm, regions=self.regions,
            melanism=self.melanism, provenance=dict(self.provenance),
        )
        fields.update(kw)
        return Individual(**fields)

    def region_signal(self, s):
        """Identity amplitude at arc-length ``s`` (head/flank/tail knobs)."""
        p = self.params
        head_max = 0.22 if p.head_s_max is None else float(p.head_s_max)
        flank_max = 0.75 if p.flank_s_max is None else float(p.flank_s_max)
        s = np.asarray(s, dtype=np.float64)
        out = np.full(s.shape, p.tail_signal, dtype=np.float64)
        out = np.where(s <= flank_max, p.flank_signal, out)
        out = np.where(s <= head_max, p.head_signal, out)
        return out

    def spot_spacing(self, units="chart"):
        """Mean nearest-neighbour spot spacing, in chart s-units or cm."""
        d = nearest_neighbour_spacing(self.spots, self.params.phi_scale)
        if d.size == 0:
            return float("nan")
        mean = float(np.mean(d))
        if units == "cm":
            return mean * self.length_cm
        if units != "chart":
            raise ValueError("units must be 'chart' or 'cm'")
        return mean

    def __len__(self):
        return int(len(self.spots))

    def __repr__(self):
        return "Individual(%s, %d spots, %d scars, %s, %.0f cm)" % (
            self.identity, len(self.spots), len(self.scars), str(self.date),
            self.length_cm,
        )


def randomize(seed, params=None, **kw):
    """Alias for :meth:`Individual.generate` (the owner's "randomize" verb)."""
    return Individual.generate(seed, params=params, **kw)


# ---------------------------------------------------------------------------
# Blue-noise sampling
# ---------------------------------------------------------------------------


def _density_weight(phi, params):
    """Placement prior: dorsal-heavy, countershaded ventrum. In [0, 1]."""
    dorsal = ((1.0 + np.cos(phi)) * 0.5) ** params.dorsal_exponent
    return dorsal * countershading_weight_at(phi, **params.countershading)


def _poisson_disc_spots(rng, params, regions, date):
    """Dart-throwing blue noise honouring density, min separation, exclusions.

    Returns ``(spot_table, attempts)``. Rejection is in this order: exclusion
    region -> density prior -> minimum separation, so an excluded region is
    empty by construction and not merely improbable.
    """
    n_target = params.n_spots_target
    sep = params.min_sep
    scale = params.phi_scale
    lo, hi = params.s_margin, 1.0 - params.s_margin
    acc_s = np.empty(n_target, dtype=np.float64)
    acc_p = np.empty(n_target, dtype=np.float64)
    n = 0
    attempts = 0
    batch = max(64, n_target)
    while n < n_target and attempts < params.max_attempts:
        cand_s = rng.uniform(lo, hi, size=batch)
        cand_p = rng.uniform(-math.pi, math.pi, size=batch)
        keep_u = rng.uniform(0.0, 1.0, size=batch)
        attempts += batch
        blocked = regions_contain(regions, cand_s, cand_p)
        ok = (~blocked) & (keep_u < _density_weight(cand_p, params))
        for i in np.nonzero(ok)[0]:
            if n >= n_target:
                break
            s_i, p_i = cand_s[i], cand_p[i]
            if n:
                ds = acc_s[:n] - s_i
                dp = wrap_phi(acc_p[:n] - p_i) * scale
                if np.min(ds * ds + dp * dp) < sep * sep:
                    continue
            acc_s[n] = s_i
            acc_p[n] = p_i
            n += 1
    spots = empty_spots(n)
    spots["id"] = np.arange(n, dtype=np.int32)
    spots["s"] = acc_s[:n]
    spots["phi"] = acc_p[:n]
    spots["radius"] = params.radius_median * np.exp(
        rng.normal(0.0, params.radius_log_sigma, size=n))
    spots["eccentricity"] = np.clip(
        1.0 + np.abs(rng.normal(0.0, params.ecc_sigma, size=n)), 1.0,
        params.ecc_max)
    spots["angle"] = rng.uniform(-math.pi / 2.0, math.pi / 2.0, size=n)
    spots["darkness"] = np.clip(
        rng.normal(params.darkness_mean, params.darkness_sigma, size=n),
        params.darkness_min, params.darkness_max)
    spots["birth_date"] = as_date(date)
    return spots, attempts


def _initial_scars(rng, params, regions, date, start_id=0):
    """Draw ``params.n_scars`` scars, all born on ``date``."""
    out = []
    if params.n_scars <= 0:
        return tuple(out)
    placed = 0
    guard = 0
    while placed < params.n_scars and guard < 2000:
        guard += 1
        s = float(rng.uniform(params.s_margin, 1.0 - params.s_margin))
        phi = float(rng.uniform(-math.pi, math.pi))
        if bool(regions_contain(regions, np.array(s), np.array(phi))):
            continue
        length = float(params.scar_length_median
                       * math.exp(rng.normal(0.0, params.scar_length_log_sigma)))
        out.append(Scar(
            id=start_id + placed, s=s, phi=phi, length=length,
            width=length * params.scar_width_frac,
            angle=float(rng.uniform(-math.pi / 2.0, math.pi / 2.0)),
            darkness=float(np.clip(rng.normal(params.scar_darkness_mean, 0.12),
                                   0.1, 1.0)),
            birth_date=date,
            persist=bool(rng.random() < params.scar_persist_fraction),
        ))
        placed += 1
    return tuple(out)


def recoverable_spot_count(rendered_spots, threshold):
    """How many rendered spots are bright enough for :func:`copy_from_chart`.

    ``copy_from_chart`` segments at a darkness threshold, so a mark whose
    rendered darkness is below that threshold is not recoverable IN
    PRINCIPLE -- countershading has faded it into the pale ventrum. This is
    the honest denominator for a round-trip count check: compare the number
    of fitted spots against ``recoverable_spot_count``, not against the
    generative spot count.
    """
    return int(np.sum(np.asarray(rendered_spots["rendered_darkness"])
                      >= float(threshold)))


def nearest_neighbour_spacing(spots, phi_scale):
    """Per-spot nearest-neighbour distance in the scaled chart metric."""
    n = len(spots)
    if n < 2:
        return np.zeros(0, dtype=np.float64)
    s = np.asarray(spots["s"], dtype=np.float64)
    p = np.asarray(spots["phi"], dtype=np.float64)
    ds = s[:, None] - s[None, :]
    dp = wrap_phi(p[:, None] - p[None, :]) * float(phi_scale)
    d = np.sqrt(ds * ds + dp * dp)
    np.fill_diagonal(d, np.inf)
    return np.min(d, axis=1)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_EDGE_SOFTNESS = 0.25  # fraction of the radius over which coverage ramps to 0


def _stamp_ellipse(image, s0, phi0, a, b, angle, amplitude, s_axis, phi_axis,
                   phi_scale):
    """Composite one soft-edged ellipse into a periodic-in-phi chart image.

    ``a``/``b`` are the semi-axes in the SCALED metric (s-units); ``angle`` is
    the major-axis orientation in that metric. Compositing is
    ``1 - (1 - img) * (1 - cov * amplitude)`` so the result stays in [0, 1]
    and overlapping marks darken rather than clip.
    """
    if amplitude <= 0.0 or a <= 0.0 or b <= 0.0:
        return
    h, w = image.shape
    dphi_px = TWO_PI / h
    reach = max(a, b) * (1.0 + _EDGE_SOFTNESS)
    j0 = int(math.floor((s0 - reach) * w - 1.0))
    j1 = int(math.ceil((s0 + reach) * w + 1.0))
    j0 = max(j0, 0)
    j1 = min(j1, w)
    if j1 <= j0:
        return
    half_rows = int(math.ceil(reach / phi_scale / dphi_px)) + 1
    i_center = int(round((phi0 + math.pi) / dphi_px - 0.5))
    rows = (np.arange(i_center - half_rows, i_center + half_rows + 1) % h)
    cols = np.arange(j0, j1)
    dS = s_axis[cols][None, :] - s0
    dP = wrap_phi(phi_axis[rows][:, None] - phi0) * phi_scale
    ca, sa = math.cos(angle), math.sin(angle)
    u = (dS * ca + dP * sa) / a
    v = (-dS * sa + dP * ca) / b
    dn = np.sqrt(u * u + v * v)
    cov = np.clip((1.0 - dn) / _EDGE_SOFTNESS, 0.0, 1.0)
    if not cov.any():
        return
    patch = image[np.ix_(rows, cols)]
    image[np.ix_(rows, cols)] = 1.0 - (1.0 - patch) * (1.0 - cov * amplitude)


def render_chart(individual, resolution=(256, 512), date=None, mask=None,
                 apply_countershading=True, apply_region_signal=True,
                 include_scars=True, include_common=True):
    """Rasterise an :class:`Individual` to a chart darkness image.

    Returns ``(image, spots)``:

    * ``image`` -- float64 ``(H_phi, W_s)`` in [0, 1], 0 = unmarked skin,
      1 = fully dark. This is a DARKNESS field, not a photograph: shading,
      occlusion and camera nuisance are a later module's job.
    * ``spots`` -- the ground-truth spot table (``id, s, phi, radius,
      eccentricity, angle, darkness, birth_date``) plus ``rendered_darkness``,
      the amplitude actually stamped after region signal and countershading.
      Rows are the individual's spots in order, so ``spots["id"]`` tracks the
      same physical mark across resightings.

    Excluded pixels (eyes, nares, mouth, gill slits, ...) are forced to 0:
    the mask is enforced at render time as well as at sampling time, so a
    drifted or copied pattern can never leak signal into an excluded region.
    """
    h, w = int(resolution[0]), int(resolution[1])
    date = individual.date if date is None else as_date(date)
    params = individual.params
    scale = params.phi_scale
    s_axis, phi_axis = chart_axes((h, w))
    image = np.zeros((h, w), dtype=np.float64)

    if include_common and params.n_common > 0:
        for cs in _common_spots(params, individual.regions, individual.date):
            _stamp_ellipse(image, cs["s"], cs["phi"],
                           cs["radius"] * math.sqrt(cs["eccentricity"]),
                           cs["radius"] / math.sqrt(cs["eccentricity"]),
                           cs["angle"], params.common_darkness,
                           s_axis, phi_axis, scale)

    if individual.melanism is not None:
        m = individual.melanism
        _stamp_ellipse(image, m.s, m.phi, m.radius, m.radius, 0.0, m.darkness,
                       s_axis, phi_axis, scale)

    spots = individual.spots
    signal = (individual.region_signal(spots["s"]) if apply_region_signal
              else np.ones(len(spots)))
    if apply_countershading:
        shade = countershading_weight_at(spots["phi"], **params.countershading)
    else:
        shade = np.ones(len(spots))
    born = spots["birth_date"] <= date
    rendered = np.clip(spots["darkness"] * signal * shade, 0.0, 1.0) * born
    for k in range(len(spots)):
        if rendered[k] <= 0.0:
            continue
        r = spots["radius"][k]
        e = math.sqrt(max(spots["eccentricity"][k], 1.0))
        _stamp_ellipse(image, spots["s"][k], spots["phi"][k], r * e, r / e,
                       spots["angle"][k], rendered[k], s_axis, phi_axis, scale)

    if include_scars:
        for sc in individual.scars:
            vis = scar_visibility(sc, date)
            if vis <= 1e-3:
                continue
            amp = sc.darkness * vis
            if apply_countershading:
                amp *= float(countershading_weight_at(sc.phi,
                                                      **params.countershading))
            _stamp_ellipse(image, sc.s, sc.phi, sc.length * 0.5,
                           max(sc.width * 0.5, 1e-4), sc.angle, amp,
                           s_axis, phi_axis, scale)

    if mask is None:
        mask = mask_from_regions(individual.regions, (h, w))
    if mask is not None:
        image[np.asarray(mask, dtype=bool)] = 0.0

    out = np.zeros(len(spots), dtype=_RENDERED_DTYPE)
    for name in SPOT_DTYPE.names:
        out[name] = spots[name]
    out["rendered_darkness"] = rendered
    return image, out


_COMMON_CACHE = {}


def _common_spots(params, regions, date):
    """Shared non-identity texture, identical across individuals.

    Mirrors ``melops_data.make_synthetic``'s ``common_spots`` layer: a region
    whose identity signal is 0 is still textured, so an ablation that zeroes a
    region measures the loss of IDENTITY, not the loss of all image content.
    """
    key = (params.common_seed, params.n_common, params.min_sep,
           params.phi_scale, tuple(r.name for r in regions))
    cached = _COMMON_CACHE.get(key)
    if cached is None:
        rng = np.random.default_rng([int(params.common_seed), 0xC0FFEE])
        cached, _ = _poisson_disc_spots(
            rng, params.replace(n_spots_target=params.n_common), regions, date)
        _COMMON_CACHE[key] = cached
    return cached


# ---------------------------------------------------------------------------
# COPY: fit a spot table to an existing chart image
# ---------------------------------------------------------------------------


def _otsu(values):
    """Otsu threshold on a 1-D array (256 bins). Returns a float."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return 0.5
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-9:
        return hi
    hist, edges = np.histogram(v, bins=256, range=(lo, hi))
    hist = hist.astype(np.float64)
    p = hist / hist.sum()
    omega = np.cumsum(p)
    centres = 0.5 * (edges[:-1] + edges[1:])
    mu = np.cumsum(p * centres)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom <= 0] = np.nan
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    k = int(np.nanargmax(sigma_b))
    return float(centres[k])


def _label_periodic(binary):
    """Connected components on a chart, wrapping across the ventral seam.

    The chart is periodic in phi, so a mark straddling row 0 / row H-1 is one
    mark. The array is tiled to twice its height, labelled once, and each
    component is kept exactly once -- the copy whose centroid row falls in the
    central window ``[H/2, 3H/2)``.
    """
    h, w = binary.shape
    pad = h // 2
    tiled = np.concatenate([binary[h - pad:], binary, binary[:h - pad]], axis=0)
    lab, n = ndimage.label(tiled, structure=np.ones((3, 3), dtype=int))
    out = []
    if n == 0:
        return out
    objects = ndimage.find_objects(lab)
    for k in range(1, n + 1):
        sl = objects[k - 1]
        sub = lab[sl] == k
        rows = np.nonzero(sub.any(axis=1))[0] + sl[0].start
        centroid_row = float(rows.mean())
        if not (pad <= centroid_row < pad + h):
            continue
        rr, cc = np.nonzero(sub)
        out.append((rr + sl[0].start - pad, cc + sl[1].start))
    return out


def copy_from_chart(pattern_image, mask=None, params=None, threshold=None,
                    identity="copied", date="2020-01-01", length_cm=250.0,
                    regions=(), min_area_px=4, seed=-1, radius_gain=1.0,
                    confidence=None, min_confidence=0.25,
                    chart_semantics="auto", axis_order="auto",
                    max_area_frac=0.02):
    """COPY: fit an :class:`Individual` to an existing chart darkness image.

    ``max_area_frac``: a connected component covering more than this fraction
    of the whole chart is not a spot (a sevengill speckle is well under 0.1%
    of the skin) - it is an unobserved or shadowed region that survived
    thresholding. Such components are dropped with a warning and counted in
    ``provenance["oversized_dropped"]``; ``None`` disables the guard.

    This is the hook for replaying a REAL animal on the synthetic body. The
    photo -> chart extraction (detect the animal, rectify to (s, phi), read
    off the speckle field) is module B's job; this function takes the chart
    image that step produces and turns it into a spot table that
    :func:`render_chart` and :func:`drift.resight` can drive.

    ``pattern_image`` is ``(H_phi, W_s)`` in [0, 1] darkness. Marks are
    segmented by an Otsu threshold (override with ``threshold``), labelled
    with wrap-around across the ventral seam, and each component becomes one
    spot: centroid -> (s, phi), area -> equivalent-circle radius in s-units,
    second moments -> eccentricity and angle, 90th-percentile pixel value ->
    darkness. Components smaller than ``min_area_px`` are discarded as noise,
    and anything inside ``mask`` (or inside ``regions``) is ignored.

    Round-trip fidelity is a SPOT-COUNT tolerance, not an identity: touching
    marks merge into one component and sub-pixel marks vanish. The returned
    ``provenance`` records the threshold and the counts so the loss is
    visible.

    INTEROP WITH THE PHOTO -> CHART MODULE (bake.py / unbake.py). That module
    lays its charts out ``(n_s, n_phi)`` -- the TRANSPOSE of this module's
    ``(H_phi, W_s)`` -- and its charts are ALBEDO MULTIPLIERS (1 = unmarked)
    rather than darkness maps (0 = unmarked). Both are handled:

    * ``axis_order``: ``"phi_major"`` (this module's convention),
      ``"s_major"`` (the bake/unbake convention), or ``"auto"`` (default) --
      auto takes the LONGER axis to be s, which is correct for both
      conventions on any non-square chart, and falls back to ``phi_major``
      with a warning on a square one.
    * ``chart_semantics``: ``"darkness"``, ``"albedo"`` (converted as
      ``1 - chart``), or ``"auto"`` (default) -- auto reads an array whose
      mean exceeds 0.5 as albedo. A speckled sevengill is mostly unmarked
      skin, so the two populations are far apart and the rule is safe; it is
      still recorded in ``provenance["semantics"]`` and overridable.
    * ``confidence``: an optional per-pixel weight in [0, 1] in the SAME
      layout as the chart (unbake emits one). Pixels below
      ``min_confidence`` are treated as unobserved and cannot contribute a
      spot; the rest weight the centroid and moment fits.

    KNOWN BIAS: the fitted radius is systematically ~10-15% SMALL, because
    thresholding cuts a soft-edged mark inside its nominal boundary (at a
    threshold of half the peak darkness the cut is at ~0.87 of the radius for
    the render profile in this module). ``radius_gain`` multiplies every
    fitted radius if a caller wants to correct it; the default 1.0 leaves the
    bias in place and visible rather than papering over it with a constant
    that only holds for one threshold.
    """
    image = np.asarray(pattern_image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("pattern_image must be 2-D (H_phi, W_s)")
    conf = None if confidence is None else np.asarray(confidence,
                                                      dtype=np.float64)
    if conf is not None and conf.shape != image.shape:
        raise ValueError("confidence shape %r != chart shape %r"
                         % (conf.shape, image.shape))

    if axis_order == "auto":
        if image.shape[0] > image.shape[1]:
            resolved_order = "s_major"
        else:
            if image.shape[0] == image.shape[1]:
                warnings.warn(
                    "copy_from_chart(axis_order='auto') cannot tell s from phi "
                    "on a square chart; assuming this module's (H_phi, W_s). "
                    "Pass axis_order explicitly.", RuntimeWarning)
            resolved_order = "phi_major"
    elif axis_order in ("phi_major", "s_major"):
        resolved_order = axis_order
    else:
        raise ValueError("axis_order must be 'auto', 'phi_major' or 's_major'")
    if resolved_order == "s_major":
        image = image.T
        if conf is not None:
            conf = conf.T
        if mask is not None:
            mask = np.asarray(mask).T

    # Unobserved texels arrive as NaN (unbake marks the far half of the girth
    # that way rather than inventing it). They are UNOBSERVED, not unmarked:
    # they are excluded from the semantics decision and then set to "no mark".
    finite = np.isfinite(image)
    if chart_semantics == "auto":
        mean = float(image[finite].mean()) if finite.any() else 0.0
        resolved_semantics = "albedo" if mean > 0.5 else "darkness"
    elif chart_semantics in ("darkness", "albedo"):
        resolved_semantics = chart_semantics
    else:
        raise ValueError("chart_semantics must be 'auto', 'darkness' or 'albedo'")
    if resolved_semantics == "albedo":
        image = 1.0 - image
    if not finite.all():
        image = np.where(finite, image, 0.0)

    h, w = image.shape
    params = params or PatternParams()
    regions = tuple(regions)
    if mask is None and regions:
        mask = mask_from_regions(regions, (h, w))
    work = image.copy()
    if mask is not None:
        work[np.asarray(mask, dtype=bool)] = 0.0
    if conf is not None:
        work[conf < float(min_confidence)] = 0.0

    nonzero = work[work > 1e-6]
    if threshold is None:
        threshold = float(np.clip(_otsu(nonzero), 0.05, 0.90)) if nonzero.size else 0.5
    binary = work >= threshold

    s_axis, phi_axis = chart_axes((h, w))
    scale = params.phi_scale
    px_area = (1.0 / w) * (TWO_PI / h) * scale  # scaled-metric area per pixel

    rows_cols = _label_periodic(binary)
    recs = []
    max_area_px = None if max_area_frac is None else float(max_area_frac) * h * w
    oversized = []
    for rr, cc in rows_cols:
        area_px = int(rr.size)
        if area_px < min_area_px:
            continue
        if max_area_px is not None and area_px > max_area_px:
            oversized.append(area_px)
            continue
        vals = work[rr % h, cc]
        if conf is not None:
            vals = vals * np.clip(conf[rr % h, cc], 0.0, 1.0)
        wsum = float(vals.sum())
        if wsum <= 0:
            continue
        # phi centroid on the circle (rows may straddle the seam pre-unwrap)
        phi_pts = -math.pi + ((rr + 0.5) * (TWO_PI / h))
        ref = phi_pts[0]
        phi_c = float(wrap_phi(ref + np.average(wrap_phi(phi_pts - ref),
                                                weights=vals)))
        s_c = float(np.average(s_axis[cc], weights=vals))
        x = s_axis[cc] - s_c
        y = wrap_phi(phi_pts - phi_c) * scale
        mxx = float(np.average(x * x, weights=vals))
        myy = float(np.average(y * y, weights=vals))
        mxy = float(np.average(x * y, weights=vals))
        tr = mxx + myy
        det = max(mxx * myy - mxy * mxy, 0.0)
        disc = math.sqrt(max(tr * tr / 4.0 - det, 0.0))
        lam1, lam2 = tr / 2.0 + disc, max(tr / 2.0 - disc, 1e-18)
        ecc = float(np.clip(math.sqrt(lam1 / lam2), 1.0, params.ecc_max))
        angle = 0.5 * math.atan2(2.0 * mxy, mxx - myy)
        radius = math.sqrt(area_px * px_area / math.pi) * float(radius_gain)
        darkness = float(np.clip(np.percentile(vals, 90.0), 0.0, 1.0))
        recs.append((s_c, phi_c, radius, ecc, angle, darkness))

    recs.sort(key=lambda r: (r[0], r[1]))
    spots = empty_spots(len(recs))
    if oversized:
        warnings.warn(
            "copy_from_chart dropped %d oversized component(s) (largest %d px = "
            "%.1f%% of the chart; limit %.1f%%) - unobserved or shadowed regions "
            "are not spots" % (len(oversized), max(oversized),
                               100.0 * max(oversized) / float(h * w),
                               100.0 * float(max_area_frac)),
            RuntimeWarning, stacklevel=2)
    for i, (s_c, phi_c, radius, ecc, angle, darkness) in enumerate(recs):
        spots[i] = (i, s_c, phi_c, radius, ecc, angle, darkness, as_date(date))
    return Individual(
        identity=identity, seed=seed, params=params, spots=spots, scars=(),
        date=date, length_cm=length_cm, regions=regions,
        provenance={
            "origin": "copy_from_chart",
            "threshold": float(threshold),
            "components_found": int(len(rows_cols)),
            "spots_kept": int(len(recs)),
            "min_area_px": int(min_area_px),
            "max_area_frac": (None if max_area_frac is None else float(max_area_frac)),
            "oversized_dropped": int(len(oversized)),
            "radius_gain": float(radius_gain),
            "axis_order": resolved_order,
            "semantics": resolved_semantics,
            "min_confidence": (None if conf is None else float(min_confidence)),
            "unobserved_texels": int((~finite).sum()),
            "source_resolution": (int(h), int(w)),
            "chart_convention": CHART_CONVENTION,
        },
    )


# ---------------------------------------------------------------------------
# Adapter for the photo -> chart module (bake.py / unbake.py)
# ---------------------------------------------------------------------------

# THE HOOK IS DELIBERATELY NOT NAMED ``exclusion_mask``.
# ``unbake.resolve_exclusion_mask`` probes module P for the canonical
# exclusion geometry; it accepts BOTH ``pattern.exclusion_mask`` and
# ``pattern.chart_exclusion_mask``, so the integration is live under the
# second name. The bare name is left unbound on purpose: it is the name that
# module's own "module P is absent" fallback tests bind against, and claiming
# it buys nothing.
#
# GILL SLITS -- the disagreement, and how it was settled. Module P's brief
# lists the seven gill slits among the regions that carry no identity;
# ``unbake.eye_mouth_exclusion`` originally kept them, arguing they are the
# chart's re-anchoring contour (Schema S1 ``chart.arc_length_origin:
# gill_slit_1_dorsal_origin``). Settled in favour of EXCLUDING them (see the
# arbitration written into ``unbake.resolve_exclusion_mask``): re-anchoring
# runs on LANDMARKS, not on the identity image, so the anchor costs nothing;
# a slit is a dark aperture EVERY individual has, whose look varies with
# respiration and view angle -- the textbook identity-free shortcut feature;
# and the La Jolla freckle patch that carries the head signal lies anterior
# to the band (Schema S1 ``chart.head_patch_bounds``). This flag remains the
# single place to reverse that if a measurement ever contradicts it.
EXCLUSION_MASK_INCLUDE_GILL_SLITS = True


def chart_exclusion_mask(chart_shape, landmarks=None, schema_path=None,
                         include_gill_slits=None, axis_order="s_major"):
    """Exclusion mask in the bake/unbake layout: bool ``(n_s, n_phi)``.

    This is the function ``unbake.resolve_exclusion_mask`` wants
    (it probes for ``pattern.exclusion_mask(chart_shape, landmarks)`` -- see
    the note above for why the alias is not installed). ``landmarks`` is
    ``{name: (s, phi)}`` in chart coordinates, exactly as that module emits
    it: the ``s`` of each recognised Schema-S1 landmark OVERRIDES the
    provisional station table, so a mask built from a real pose uses measured
    stations and only falls back to :func:`exclusions.default_stations` for
    landmarks the caller did not supply.

    ``axis_order`` defaults to ``"s_major"`` -- the caller's layout, NOT this
    module's -- because this function exists for that caller. Pass
    ``"phi_major"`` for the prototype-05 layout, or just call
    :func:`exclusions.build_exclusion_mask`.

    See :data:`EXCLUSION_MASK_INCLUDE_GILL_SLITS` for the one place where this
    mask and ``unbake.eye_mouth_exclusion`` disagree.
    """
    n_a, n_b = int(chart_shape[0]), int(chart_shape[1])
    if axis_order == "s_major":
        resolution = (n_b, n_a)
    elif axis_order == "phi_major":
        resolution = (n_a, n_b)
    else:
        raise ValueError("axis_order must be 's_major' or 'phi_major'")

    schema = load_schema(schema_path or DEFAULT_SCHEMA_PATH)
    stations = default_stations(schema)
    used = []
    for name, value in dict(landmarks or {}).items():
        if name in stations:
            stations[name] = float(np.asarray(value).ravel()[0])
            used.append(name)
    regions = exclusion_regions(schema, stations=stations)
    keep_gills = (EXCLUSION_MASK_INCLUDE_GILL_SLITS
                  if include_gill_slits is None else bool(include_gill_slits))
    if not keep_gills:
        regions = [r for r in regions if r.name != "gill_slits"]
    mask = mask_from_regions(regions, resolution)
    return mask.T if axis_order == "s_major" else mask
