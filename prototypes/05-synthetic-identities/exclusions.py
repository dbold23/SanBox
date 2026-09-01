"""Canonical chart space and its exclusion regions for the sevengill identity engine.

CHART SPACE (binding for prototype 05; every other module in this prototype
consumes this convention and nothing else):

    s    in [0, 1]   arc-length fraction along the body centerline,
                     s = 0 at ``snout_tip``, s = 1 at the caudal terminus
                     (``caudal_upper_lobe_tip`` station on the body axis).
    phi  in [-pi, pi) circumferential angle,
                     phi =  0      dorsal midline,
                     phi = +pi/2   the animal's LEFT flank,
                     phi = -pi/2   the animal's RIGHT flank,
                     phi = +/-pi   ventral midline (the wrap seam).

A chart image has shape ``(H_phi, W_s)``. Row ``i`` is
``phi = -pi + (i + 0.5) * 2*pi/H`` and column ``j`` is ``s = (j + 0.5)/W``.
Rows are therefore PERIODIC (row 0 and row H-1 are both just below/above the
ventral midline) and the dorsal midline sits at the middle row. Columns are
NOT periodic.

Why chart space and not the mesh UV atlas: a pattern in (s, phi) is
mesh-agnostic, survives any pose the rig can produce, and gives every rendered
pixel exact ground-truth chart coordinates. Baking to a mesh's UV texture is a
separate later step that needs per-vertex (s, phi); for real meshes that comes
from prototype 04's ``mesh3d.tube_coords``.

RELATION TO THE SCHEMA'S OWN CHART BLOCK. ``keypoints_sevengill_v1.yaml``
declares ``chart.arc_length_origin = gill_slit_1_dorsal_origin`` (s = 0) and
``chart.arc_length_terminus = precaudal_pit``: that is the TRUNK TUBE the
Phase-1B rectifier re-anchors on gill contours. Prototype 05's s is the
WHOLE-ANIMAL axis (snout -> caudal) because the synthetic engine must place
head landmarks, gill slits and caudal peduncle in one coordinate. The two are
related by an affine map on s; ``ChartSchema.trunk_tube_span()`` returns the
prototype-05 s interval of the schema's tube so nothing is lost.

STATION FRACTIONS. The yaml supplies (a) the keypoint names and ids, (b) the
seven midline semilandmark axis fractions, (c) the ordered antero-posterior
sequence, and (d) the explicit refusal to order the pelvic/dorsal/cloaca trio.
It does NOT supply arc-length fractions for the anterior anchors -- its own
``open_questions`` says fin stations are provisional and unretrieved
[UNVERIFIED]. So this module reads everything the yaml does contain and keeps
the missing numbers in ``DEFAULT_STATIONS``, each carrying its own grade, and
``validate_stations`` asserts any station table (default or measured) against
the ordering the yaml DOES assert -- and against nothing it does not.
Prototype 04 (or a first annotation batch) can pass a measured table in; that
is the contract, not a hard-coded constant.
"""

from __future__ import annotations

import math
import numpy as np
import yaml

__all__ = [
    "TWO_PI",
    "CHART_CONVENTION",
    "ChartSchema",
    "Region",
    "load_schema",
    "default_stations",
    "station_grades",
    "validate_stations",
    "axis_fraction_to_s",
    "chart_axes",
    "chart_meshgrid",
    "wrap_phi",
    "exclusion_regions",
    "mask_from_regions",
    "regions_contain",
    "build_exclusion_mask",
    "countershading_weight",
    "countershading_weight_at",
]

TWO_PI = 2.0 * math.pi

CHART_CONVENTION = (
    "s in [0,1]: 0=snout_tip, 1=caudal terminus; "
    "phi in [-pi,pi): 0=dorsal midline, +pi/2=animal's LEFT flank, "
    "-pi/2=animal's RIGHT flank, +/-pi=ventral midline. "
    "Chart arrays are (H_phi, W_s); rows periodic, columns not."
)

# ---------------------------------------------------------------------------
# Chart primitives
# ---------------------------------------------------------------------------


def wrap_phi(phi):
    """Wrap an angle (scalar or array) into [-pi, pi)."""
    return (np.asarray(phi, dtype=np.float64) + math.pi) % TWO_PI - math.pi


def chart_axes(resolution):
    """1-D pixel-centre coordinate axes for a chart of shape ``(H_phi, W_s)``.

    Returns ``(s, phi)`` with ``s.shape == (W,)`` and ``phi.shape == (H,)``.
    """
    h, w = int(resolution[0]), int(resolution[1])
    if h < 2 or w < 2:
        raise ValueError("resolution must be at least (2, 2), got %r" % (resolution,))
    s = (np.arange(w, dtype=np.float64) + 0.5) / w
    phi = -math.pi + (np.arange(h, dtype=np.float64) + 0.5) * (TWO_PI / h)
    return s, phi


def chart_meshgrid(resolution):
    """``(S, PHI)`` arrays of shape ``(H_phi, W_s)``, pixel centres."""
    s, phi = chart_axes(resolution)
    return np.meshgrid(s, phi)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class ChartSchema(object):
    """The parts of ``keypoints_sevengill_v1.yaml`` this prototype depends on.

    Attributes are read from the yaml, never invented here:
    ``keypoints`` (name -> id), ``midline_axis_fractions`` (the yaml's
    ``midline_definition.fractions``), ``axis_origin``/``axis_terminus``,
    ``ap_sequence`` (``ordered_ap_sequence.sequence``), ``posterior_trio``
    (``unordered_posterior_trio.members`` -- an ordering among these three is
    forbidden by the schema and this module never enforces one),
    ``chart_origin``/``chart_terminus``/``head_patch_bounds`` from the yaml's
    ``chart`` block.
    """

    def __init__(self, raw, path=None):
        self.path = path
        self.name = raw["schema"]["name"]
        self.version = raw["schema"]["version"]
        self.species = raw["schema"]["species"]
        self.keypoints = dict(
            (kp["name"], int(kp["id"])) for kp in raw["keypoints"]
        )
        mid = raw["midline_definition"]
        self.axis_origin = mid["axis_origin"]
        self.axis_terminus = mid["axis_terminus"]
        self.midline_axis_fractions = tuple(float(f) for f in mid["fractions"])
        seq = raw["ordered_ap_sequence"]
        self.ap_sequence = tuple(str(x) for x in seq["sequence"])
        self.posterior_trio = tuple(
            str(x) for x in seq["unordered_posterior_trio"]["members"]
        )
        chart = raw["chart"]
        self.chart_origin = chart["arc_length_origin"]
        self.chart_terminus = chart["arc_length_terminus"]
        self.head_patch_bounds = tuple(str(x) for x in chart["head_patch_bounds"])

    # -- helpers ----------------------------------------------------------
    def midline_stations(self, precaudal_s):
        """Midline semilandmark stations in prototype-05 s.

        The yaml's midline fractions t are measured on the
        ``snout_tip -> precaudal_pit`` axis; prototype-05 s runs snout ->
        caudal terminus, so ``s = t * precaudal_s``.
        """
        return dict(
            ("midline_%02d" % (k + 1), float(t) * float(precaudal_s))
            for k, t in enumerate(self.midline_axis_fractions)
        )

    def trunk_tube_span(self, stations):
        """(s_lo, s_hi) of the schema's own trunk tube in prototype-05 s."""
        return (
            float(stations[self.chart_origin]),
            float(stations[self.chart_terminus]),
        )

    def __repr__(self):
        return "ChartSchema(%s %s, %d keypoints)" % (
            self.name,
            self.version,
            len(self.keypoints),
        )


def load_schema(schema_yaml_path):
    """Parse ``keypoints_sevengill_v1.yaml`` into a :class:`ChartSchema`."""
    with open(schema_yaml_path, "r") as handle:
        raw = yaml.safe_load(handle)
    return ChartSchema(raw, path=schema_yaml_path)


# ---------------------------------------------------------------------------
# Station table (the numbers the yaml does not carry)
# ---------------------------------------------------------------------------

# s of ``precaudal_pit`` on the snout -> caudal-terminus axis. Everything the
# yaml expresses as an axis fraction t is mapped s = t * _PRECAUDAL_S.
_PRECAUDAL_S = 0.75

_DEFAULT_STATIONS = {
    "snout_tip": 0.000,
    "naris_anterior_margin": 0.035,
    "eye_center": 0.060,
    "spiracle": 0.085,
    "mouth_rictus": 0.100,
    "gill_slit_1_dorsal_origin": 0.140,
    "gill_slit_1_ventral_terminus": 0.140,
    "gill_slit_7_dorsal_origin": 0.220,
    "gill_slit_7_ventral_terminus": 0.220,
    "pectoral_origin": 0.245,
    "pectoral_insertion": 0.300,
    "pelvic_origin": 0.550,
    "cloaca": 0.565,
    "dorsal_fin_origin": 0.600,
    "dorsal_fin_insertion": 0.660,
    "anal_fin_origin": 0.640,
    "anal_fin_insertion": 0.700,
    "precaudal_pit": _PRECAUDAL_S,
    "caudal_subterminal_notch": 0.930,
    "caudal_upper_lobe_tip": 1.000,
}

_STATION_GRADES = {
    "snout_tip": "[DEFINITION] s = 0 by the chart convention; the yaml makes "
    "snout_tip the midline axis origin (midline_definition.axis_origin).",
    "caudal_upper_lobe_tip": "[DEFINITION] s = 1 by the chart convention "
    "(total_length_proxy in the yaml is [0, 21]).",
    "precaudal_pit": "[UNVERIFIED] precaudal length as a fraction of the "
    "total-length proxy. The yaml makes precaudal_pit the midline axis "
    "terminus but gives no fraction of TL. 0.75 is a plausible elongate-"
    "hexanchiform value, not a measurement; override with a measured table.",
}
_PROVISIONAL_GRADE = (
    "[UNVERIFIED] provisional station. The yaml carries no arc-length fraction "
    "for this landmark and its open_questions block states that fin stations "
    "are provisional and were never retrieved for Notorynchus cepedianus. "
    "Ordering (not value) is checked against ordered_ap_sequence."
)
for _name in _DEFAULT_STATIONS:
    _STATION_GRADES.setdefault(_name, _PROVISIONAL_GRADE)


def default_stations(schema=None):
    """A mutable copy of the provisional station table, s in [0, 1].

    Every value except ``snout_tip`` and ``caudal_upper_lobe_tip`` is
    [UNVERIFIED] -- see :func:`station_grades`. If ``schema`` is given the
    seven midline semilandmarks are ADDED from the yaml's own axis fractions
    (those numbers are schema-sourced, not invented here).
    """
    out = dict(_DEFAULT_STATIONS)
    if schema is not None:
        out.update(schema.midline_stations(out["precaudal_pit"]))
    return out


def station_grades():
    """name -> evidence grade string for :func:`default_stations`."""
    return dict(_STATION_GRADES)


def axis_fraction_to_s(t, stations=None):
    """Map a yaml midline axis fraction t (snout->precaudal_pit) to chart s."""
    stations = stations or _DEFAULT_STATIONS
    return np.asarray(t, dtype=np.float64) * float(stations["precaudal_pit"])


def validate_stations(stations, schema):
    """Assert a station table against everything the yaml asserts, and no more.

    Checks, in order:

    1. every station name is a keypoint name in the schema;
    2. all values lie in [0, 1];
    3. ``snout_tip == 0`` (the yaml's midline axis origin);
    4. the yaml's ``ordered_ap_sequence.sequence`` is non-decreasing in s,
       skipping the intermediate gill slits 2-6 (which are deliberately not
       keypoints) and mapping ``gill_slit_1``/``gill_slit_7`` to their dorsal
       origins;
    5. the three members of ``unordered_posterior_trio`` are each bounded by
       ``pectoral_origin`` and ``anal_fin_origin`` -- and NO order among the
       three is checked, because the schema forbids enforcing one.

    Raises ``ValueError`` on the first violation. Returns ``None``.
    """
    for name, value in sorted(stations.items()):
        if name not in schema.keypoints:
            raise ValueError("station %r is not a keypoint in %s" % (name, schema.name))
        if not (0.0 <= float(value) <= 1.0):
            raise ValueError("station %r = %r outside [0, 1]" % (name, value))
    if "snout_tip" in stations and abs(float(stations["snout_tip"])) > 1e-12:
        raise ValueError("snout_tip must be s = 0 (yaml midline axis origin)")

    alias = {
        "gill_slit_1": "gill_slit_1_dorsal_origin",
        "gill_slit_7": "gill_slit_7_dorsal_origin",
    }
    trio = set(schema.posterior_trio)
    ordered = []
    for name in schema.ap_sequence:
        key = alias.get(name, name)
        if key in trio or key not in stations:
            continue  # slits 2-6 are not keypoints; trio order is forbidden
        ordered.append((key, float(stations[key])))
    for (a, sa), (b, sb) in zip(ordered[:-1], ordered[1:]):
        if sb < sa:
            raise ValueError(
                "ordered_ap_sequence violated: %s (s=%.4f) must not be posterior "
                "to %s (s=%.4f)" % (a, sa, b, sb)
            )
    lo_name, hi_name = "pectoral_origin", "anal_fin_origin"
    if lo_name in stations and hi_name in stations:
        lo, hi = float(stations[lo_name]), float(stations[hi_name])
        for name in schema.posterior_trio:
            if name not in stations:
                continue
            v = float(stations[name])
            if not (lo <= v <= hi):
                raise ValueError(
                    "%s (s=%.4f) must lie between pectoral_origin (%.4f) and "
                    "anal_fin_origin (%.4f); the schema asserts that bracket and "
                    "nothing finer" % (name, v, lo, hi)
                )
    return None


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------


class Region(object):
    """An axis-aligned box in chart space: an s interval x a phi sector.

    ``phi`` containment is wrap-aware: a point is inside when
    ``|wrap_phi(phi - phi_center)| <= phi_halfwidth``. A halfwidth of pi
    therefore covers the full circumference.

    ``kind`` is ``"exclusion"`` (no identity pattern may exist there and no
    identity signal may be counted there) or ``"soft"`` (informational only;
    :func:`mask_from_regions` ignores soft regions unless asked for them).
    """

    __slots__ = ("name", "s_lo", "s_hi", "phi_center", "phi_halfwidth", "kind", "note")

    def __init__(self, name, s_lo, s_hi, phi_center, phi_halfwidth,
                 kind="exclusion", note=""):
        if s_hi < s_lo:
            raise ValueError("region %r has s_hi < s_lo" % name)
        if not (0.0 <= float(phi_halfwidth) <= math.pi + 1e-12):
            raise ValueError("region %r phi_halfwidth outside [0, pi]" % name)
        self.name = name
        self.s_lo = float(np.clip(s_lo, 0.0, 1.0))
        self.s_hi = float(np.clip(s_hi, 0.0, 1.0))
        self.phi_center = float(wrap_phi(phi_center))
        self.phi_halfwidth = float(phi_halfwidth)
        self.kind = kind
        self.note = note

    def contains(self, s, phi):
        """Boolean array: which (s, phi) points fall inside this region."""
        s = np.asarray(s, dtype=np.float64)
        d = np.abs(wrap_phi(np.asarray(phi, dtype=np.float64) - self.phi_center))
        return (s >= self.s_lo) & (s <= self.s_hi) & (d <= self.phi_halfwidth)

    def area(self):
        """Chart area in (s x radian) units."""
        return (self.s_hi - self.s_lo) * 2.0 * self.phi_halfwidth

    def __repr__(self):
        return "Region(%s, s=[%.3f, %.3f], phi=%.2f+/-%.2f, %s)" % (
            self.name, self.s_lo, self.s_hi, self.phi_center,
            self.phi_halfwidth, self.kind,
        )


# Angular extents, all [DERIVED] from sevengill gross anatomy and all
# overridable. The s pads are in s-units (fractions of the whole axis).
REGION_GEOMETRY = {
    # Eyes sit dorsolaterally on a broad, dorsoventrally flattened head, so the
    # eye centre is ABOVE the lateral meridian (|phi| < pi/2).
    "eye_phi_center": 1.20,
    "eye_phi_halfwidth": 0.35,
    "eye_s_pad": 0.020,
    # Nares open ventrolaterally on the snout, well below the lateral meridian.
    "naris_phi_center": 2.10,
    "naris_phi_halfwidth": 0.30,
    "naris_s_pad": 0.015,
    # A broad terminal-to-subterminal mouth: the whole ventral head from the
    # snout back to the rictus, plus the jaw line.
    "mouth_phi_halfwidth": 1.20,
    "mouth_s_pad": 0.010,
    # Seven slits. The yaml (id 6) records that sevengill slits run much
    # further ventrally than a lamnid's -- "the first pair very nearly meets on
    # the throat" -- so the branchial band runs from just below the
    # dorsolateral shoulder all the way around the throat.
    "gill_phi_onset": 0.75,
    "gill_s_pad": 0.010,
    # Optional fin-insertion pads. The yaml's chart block excludes fin
    # SURFACES from the identity surface and retains insertion CURVES as
    # landmarks; these pads exclude a collar around each insertion curve.
    "pectoral_phi_center": 2.20,
    "pectoral_phi_halfwidth": 0.55,
    "pelvic_phi_center": 2.55,
    "pelvic_phi_halfwidth": 0.45,
    "dorsal_phi_center": 0.00,
    "dorsal_phi_halfwidth": 0.30,
    "anal_phi_center": math.pi,
    "anal_phi_halfwidth": 0.35,
    "fin_s_pad": 0.010,
}


def exclusion_regions(schema, stations=None, include_fin_insertions=False,
                      ventral_hard_exclude_phi=None, geometry=None):
    """Analytic exclusion regions in chart space.

    ``ventral_hard_exclude_phi``: if given, everything with
    ``|phi| >= ventral_hard_exclude_phi`` is hard-excluded over the whole body
    (a countershading cut expressed as an exclusion). The DEFAULT is ``None``:
    countershading is normally a soft density prior
    (:func:`countershading_weight`), not an exclusion, because a pale ventrum
    still carries faint speckling.

    Regions are analytic and resolution-independent, so the same list drives
    both rejection at sampling time and rasterisation at any resolution.
    """
    st = dict(default_stations(schema)) if stations is None else dict(stations)
    validate_stations(st, schema)
    g = dict(REGION_GEOMETRY)
    if geometry:
        g.update(geometry)
    out = []

    eye = st["eye_center"]
    for side, sign in (("left", +1.0), ("right", -1.0)):
        out.append(Region(
            "eye_%s" % side,
            eye - g["eye_s_pad"], eye + g["eye_s_pad"],
            sign * g["eye_phi_center"], g["eye_phi_halfwidth"],
            note="eye_center (yaml id %d), dorsolateral on a broad flat head"
                 % schema.keypoints["eye_center"],
        ))
    naris = st["naris_anterior_margin"]
    for side, sign in (("left", +1.0), ("right", -1.0)):
        out.append(Region(
            "naris_%s" % side,
            naris - g["naris_s_pad"], naris + g["naris_s_pad"],
            sign * g["naris_phi_center"], g["naris_phi_halfwidth"],
            note="naris_anterior_margin (yaml id %d); anterior bound of the "
                 "La Jolla freckle patch, so the patch itself is NOT excluded"
                 % schema.keypoints["naris_anterior_margin"],
        ))
    out.append(Region(
        "mouth_jaw",
        0.0, st["mouth_rictus"] + g["mouth_s_pad"],
        math.pi, g["mouth_phi_halfwidth"],
        note="snout to mouth_rictus (yaml id %d), ventral band: lips, jaw "
             "line and labial folds carry no identity speckling"
             % schema.keypoints["mouth_rictus"],
    ))
    out.append(Region(
        "gill_slits",
        st["gill_slit_1_dorsal_origin"] - g["gill_s_pad"],
        st["gill_slit_7_dorsal_origin"] + g["gill_s_pad"],
        math.pi, math.pi - g["gill_phi_onset"],
        note="seven slits, gill_slit_1 to gill_slit_7 (yaml ids %d..%d); band "
             "runs to the throat per the yaml's id-6 note"
             % (schema.keypoints["gill_slit_1_dorsal_origin"],
                schema.keypoints["gill_slit_7_dorsal_origin"]),
    ))
    if include_fin_insertions:
        pad = g["fin_s_pad"]
        for key, lo_kp, hi_kp in (
            ("pectoral", "pectoral_origin", "pectoral_insertion"),
            ("pelvic", "pelvic_origin", "pelvic_origin"),
            ("dorsal", "dorsal_fin_origin", "dorsal_fin_insertion"),
            ("anal", "anal_fin_origin", "anal_fin_insertion"),
        ):
            lo, hi = st[lo_kp], st[hi_kp]
            for side, sign in ((("left", +1.0), ("right", -1.0))
                               if key in ("pectoral", "pelvic")
                               else (("", 1.0),)):
                name = "%s_insertion" % key if not side else "%s_insertion_%s" % (key, side)
                out.append(Region(
                    name, lo - pad, hi + pad,
                    sign * g["%s_phi_center" % key], g["%s_phi_halfwidth" % key],
                    note="collar around the %s insertion curve (yaml chart "
                         "block: fins excluded from the identity surface)" % key,
                ))
    if ventral_hard_exclude_phi is not None:
        onset = float(ventral_hard_exclude_phi)
        if not (0.0 < onset <= math.pi):
            raise ValueError("ventral_hard_exclude_phi must be in (0, pi]")
        out.append(Region(
            "ventral_countershading", 0.0, 1.0, math.pi, math.pi - onset,
            note="hard countershading cut at |phi| >= %.2f rad (opt-in)" % onset,
        ))
    return out


def regions_contain(regions, s, phi, kinds=("exclusion",)):
    """Boolean array: which (s, phi) points fall in ANY region of ``kinds``."""
    s = np.asarray(s, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    out = np.zeros(np.broadcast(s, phi).shape, dtype=bool)
    for region in regions:
        if kinds is not None and region.kind not in kinds:
            continue
        out |= region.contains(s, phi)
    return out


def mask_from_regions(regions, resolution, kinds=("exclusion",)):
    """Rasterise regions to a boolean chart mask of shape ``(H_phi, W_s)``."""
    S, PHI = chart_meshgrid(resolution)
    return regions_contain(regions, S, PHI, kinds=kinds)


def build_exclusion_mask(schema_yaml_path, resolution=(128, 256), stations=None,
                         include_fin_insertions=False,
                         ventral_hard_exclude_phi=None, geometry=None):
    """Boolean chart mask, True where NO identity pattern may exist.

    True pixels are excluded from BOTH pattern generation and identity
    scoring: eyes, nares, the mouth/jaw band, the seven gill slits, and
    optionally the fin insertion collars and a hard ventral countershading
    band. Shape is ``resolution = (H_phi, W_s)``; see the module docstring for
    the pixel-centre convention.

    ``schema_yaml_path`` is ``keypoints_sevengill_v1.yaml``; landmark
    ORDERING and names come from it, station VALUES from ``stations`` (default
    :func:`default_stations`, all [UNVERIFIED]).
    """
    schema = load_schema(schema_yaml_path)
    regions = exclusion_regions(
        schema, stations=stations,
        include_fin_insertions=include_fin_insertions,
        ventral_hard_exclude_phi=ventral_hard_exclude_phi,
        geometry=geometry,
    )
    return mask_from_regions(regions, resolution)


# ---------------------------------------------------------------------------
# Countershading
# ---------------------------------------------------------------------------

# Evidence: Notorynchus cepedianus is countershaded -- dark speckling on a
# grey-brown dorsum, lighter ventrally. The onset/full angles below are
# [DERIVED] from that qualitative description; no measured phi profile exists.
COUNTERSHADING_DEFAULTS = dict(phi_onset=1.05, phi_full=2.60, floor=0.05)


def countershading_weight_at(phi, phi_onset=None, phi_full=None, floor=None):
    """Multiplicative speckle weight in [floor, 1] as a function of phi.

    1 from the dorsal midline out to ``phi_onset``, then a smoothstep down to
    ``floor`` at ``phi_full`` and beyond (the pale ventrum). ``floor = 0``
    makes the ventrum bare; the default 0.05 keeps it faint but non-zero,
    which is what "lighter ventrally" means -- not "unmarked".
    """
    p = COUNTERSHADING_DEFAULTS
    phi_onset = p["phi_onset"] if phi_onset is None else float(phi_onset)
    phi_full = p["phi_full"] if phi_full is None else float(phi_full)
    floor = p["floor"] if floor is None else float(floor)
    if not (0.0 <= phi_onset < phi_full <= math.pi):
        raise ValueError("need 0 <= phi_onset < phi_full <= pi")
    a = np.abs(wrap_phi(phi))
    u = np.clip((a - phi_onset) / (phi_full - phi_onset), 0.0, 1.0)
    smooth = u * u * (3.0 - 2.0 * u)
    return floor + (1.0 - floor) * (1.0 - smooth)


def countershading_weight(resolution, phi_onset=None, phi_full=None, floor=None):
    """``countershading_weight_at`` rasterised to a ``(H_phi, W_s)`` chart."""
    _, PHI = chart_meshgrid(resolution)
    return countershading_weight_at(PHI, phi_onset, phi_full, floor)
