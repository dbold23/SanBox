"""Synthetic OSEA-style sightings of the real scanned sevengill.

WHAT THIS IS.  A generator of photographs that look like the ones the OSEA
tagger actually ingests -- a small sevengill held in hand or lying in a tub ON
THE BOAT, IN AIR, in daylight; a lateral-oblique close-up of the HEAD AND
FOREBODY from slightly above; hands and tub edges cutting in; wet specular
skin; dark round blotches on grey-brown skin -- built from the REAL scanned
body (prototype 04's ``results/real_v11`` bind pose, via :mod:`real_body`) and
a prototype 05 ``Individual`` for the spot constellation.

WHY IT IS BUILT THIS WAY.  Prototype 06's premise is that the matcher never
sees RGB: real photographs and synthetic renders both reach it as OSEA
DETECTIONS (a body polygon plus spot boxes).  So the pixels here exist only to
be run through the same YOLO pair that runs on the real photos.  That sets the
priorities: the SPOTS must be the right size, contrast and density at the right
place on the body, the body must have the right silhouette and pose, and the
scene must have the right junk in it (hands, tub, blur, JPEG).  It does NOT
have to be a physically correct render, and it is not one -- see LIMITATIONS.

THE PIPELINE, one frame:

    Individual (05)  --render_chart-->  darkness chart (512, 2048)
    assets/chart_skin_x4.png ---------> base skin chart, same grid
              albedo chart = skin * tint * (1 - amp * darkness) * eye

    RealBody (06)   --pose(amp,wave,phase,yaw)-->  posed vertices
    render.render(Instance(color=1, NO texture), Camera, DirectionalLight)
              -> chart_s, chart_phi, normal, ndotl, shadow, masks

    DEFERRED TEXTURE: for every subject pixel,
              rgb = (ambient + I*ndotl*(not shadow))
                    * albedo(chart_s, chart_phi) * tone(phi_eff)
                    + Blinn-Phong specular(normal, L, V)
    background + hand occluders + gaussian blur + JPEG

The COUNTERSHADING tone is the one factor applied per PIXEL rather than baked
into the chart, because a fin blade's tone follows its own upper/lower face and
not its ``phi`` around the girth -- ``phi_eff`` is the chart ``phi`` on the body
and ``arccos(n_z)`` on a fin (see :func:`shade_subject`).

DEFERRED, not baked into a UV texture, for two reasons.  The scan's atlas is a
1 M-vertex photogrammetry unwrap whose texel density is wildly uneven, so
baking a 2048-wide chart through it would lose spot detail exactly where the
atlas is sparse; and the chart maps ``render`` already produces are EXACT
per-pixel ground truth, so sampling the albedo through them costs one bilinear
lookup and introduces no resampling of its own.

CONVENTIONS.  Charts are ``(H_phi, W_s)`` -- prototype 05's ``pattern`` layout;
``s in [0,1]`` snout to caudal terminus, ``phi in [-pi,pi)`` with 0 dorsal,
``+pi/2`` the animal's LEFT, ``+-pi`` ventral.  ``assets/chart_skin_x4.png`` is
a verified 4x nearest upscale of prototype 04's
``results/real/identity/chart_skin.png``, which is ``chart_to_image`` of a
``normalize="extent"`` chart -- the SAME normalisation :mod:`real_body` puts on
the vertices, so skin and geometry line up without a fitting step.

LIMITATIONS (each is a deliberate simplification, not an oversight):
  * No global illumination, no subsurface scattering, no refraction through the
    water film.  Specular is Blinn-Phong with a random strength/shininess.
  * The background is procedural (a palette colour, a low-frequency gradient
    and noise), not a photograph of a deck.
  * Hand occluders are ellipsoids, not hands.  They occlude and shadow like
    hands; they do not look like hands under a zoom.
  * Fins ride the chart under a bend, so a blade at radius ``r`` stretches by
    ``1 +- kappa*r`` (``real_body.fin_stretch``).  Poses here are gentle and the
    framing is anterior, where the only fin is the pectoral leading edge.
  * ``exclusion=None`` is passed to ``render``: no spot is ever PLACED in an
    excluded region (``pattern.render_chart`` masks them at stamp time), so the
    per-pixel exclusion pull-through would only re-derive information the chart
    already enforces, at the cost of a schema load per frame.

Run::

    python synth_render.py --out results/synth_smoke --identities 3 \\
        --sightings 2 --seed 0 --contact
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from typing import NamedTuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROTOTYPES = os.path.dirname(_HERE)
for _d in (_HERE,
           os.path.join(_PROTOTYPES, "02-centerline-chart"),
           os.path.join(_PROTOTYPES, "04-sevengill-rig"),
           os.path.join(_PROTOTYPES, "05-synthetic-identities")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import drift  # noqa: E402
import exclusions  # noqa: E402
import make_dataset  # noqa: E402
import nuisance  # noqa: E402
import pattern  # noqa: E402
import render  # noqa: E402

import real_body  # noqa: E402

__all__ = [
    "DEFAULT_CONFIG",
    "SCHEMA_PATH",
    "load_config",
    "pattern_context",
    "skin_chart",
    "albedo_chart",
    "sample_chart",
    "eye_chart_mask",
    "SceneDraw",
    "draw_scene",
    "frame_camera",
    "background_image",
    "hand_occluders",
    "render_sighting",
    "plan_same_side_sightings",
    "identity_timeline",
    "generate",
    "contact_sheet",
    "zoom_contact",
]

#: The keypoint schema prototype 05 needs.  ``pattern.DEFAULT_SCHEMA_PATH`` is
#: an absolute path from the machine 05 was written on; this is the copy in
#: this worktree.  It is assigned onto ``pattern`` at import time (a module
#: attribute, not a file edit) so that ``make_dataset.build_pattern_context``,
#: which reads that constant, finds it.
SCHEMA_PATH = os.path.normpath(os.path.join(
    _PROTOTYPES, "..", "phase1b", "p0-sevengill-schema",
    "keypoints_sevengill_v1.yaml"))
if os.path.exists(SCHEMA_PATH):
    pattern.DEFAULT_SCHEMA_PATH = SCHEMA_PATH

DEFAULT_SKIN_PNG = os.path.join(_HERE, "assets", "chart_skin_x4.png")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Everything calibratable, in one place, so a run is reproducible from
#: ``results/<corpus>/config.json`` alone.  Ranges are ``[lo, hi]`` and are
#: drawn uniformly unless the comment says otherwise.
#:
#: The numbers under ``pattern`` and ``skin`` are CALIBRATED AGAINST THE OSEA
#: PHOTOGRAPHS listed in ``REFERENCE_PHOTOS``.  Measured with a background-
#: subtracted blob detector (``bg = gaussian(45); bg - image > 18/255``, area
#: 150-9000 px) on a flank crop of each real 4032-wide frame: median blob
#: radius 12.0-16.3 px and blob spacing 128-183 px, i.e. 6-8 px and 64-92 px at
#: this prototype's 2016-wide output.  The same detector at half the filter
#: scale on the smoke corpus gives median radius 5.6-10.2 px and spacing
#: 123-199 px -- size matched, density still ~1.5-2x sparse.  Skin tone and
#: spot contrast were matched by eye on
#: ``results/synth_smoke/zoomed_contact.png``, which shows both at one scale.
#: [EVIDENCE GRADE: spot size/spacing MEASURED on 3 real frames; tone, tint and
#: specular calibrated by eye against the same 3 frames.]
DEFAULT_CONFIG = {
    "body": {
        "cell_mm": 1.5,               # real_body decimation (see its README table)
    },
    "chart_resolution": [512, 2048],  # (H_phi, W_s) for pattern + albedo
    "pattern": {
        "n_spots": 880,               # -> mean NN spacing 0.0134 s-units
        "radius_median": 0.0030,      # s-units, equivalent-circle radius
        "radius_log_sigma": 0.38,
        "min_sep": 0.0095,            # s-units in the scaled chart metric
        "darkness_mean": 0.86,
        "darkness_sigma": 0.12,
        # A real sevengill carries spots over the WHOLE flank, down to the
        # ventral transition (look at any of REFERENCE_PHOTOS).  The first
        # values here (onset 1.30, dorsal_exponent 0.80) faded them out from
        # |phi| ~ 1.3 rad and the placement prior thinned them further, which
        # left the camera-facing flank nearly bare.
        "cs_phi_onset": 2.30,         # full-strength spots out to |phi| = 2.30
        "cs_phi_full": 3.05,          # fading only at the ventral midline
        "cs_floor": 0.05,             # residual spot amplitude on the ventrum
        "dorsal_exponent": 0.30,      # placement prior ((1+cos phi)/2)**this
        "ecc_sigma": 0.28,            # eccentricity = 1 + |N(0, sigma)|
        "ecc_max": 2.40,
        "darkness_min": 0.15,         # truncation of the darkness normal
        "darkness_max": 1.00,
        "edge_softness": 0.25,        # -> pattern._EDGE_SOFTNESS (see pattern_context)
        "n_common": 110,              # shared, identity-free texture
        "common_darkness": 0.22,
        "amplitude": 0.92,            # albedo = skin * (1 - amplitude*darkness)
        # 05's ``gill_slits`` exclusion (s 0.13-0.23, |phi| > 0.75) is a
        # SCORING mask -- "never read identity off a gill slit" -- not a claim
        # that a sevengill has no spots there.  The OSEA frames plainly do have
        # spots across the gill region, so leaving the region in PLACEMENT
        # stamps a conspicuous bare band across every synthetic forebody, which
        # is a domain gap the detector would see.  Dropped here for rendering;
        # a downstream scorer still applies the exclusion through the chart
        # ground truth, exactly as it must for a real photograph.
        "drop_regions": ["gill_slits"],
    },
    "skin": {
        "png": None,                  # None -> assets/chart_skin_x4.png
        # tone_ventral was 1.18: a multiplier above 1 on a lit surface clips to
        # white, and on the rounded LEADING EDGE of a pectoral -- where
        # ``fin_tone_from_normal`` sweeps arccos(n_z) through the whole ramp in
        # a few pixels -- it drew a hard white band along the blade and the fin
        # root on every frame.  0.95 keeps the ventrum pale without blowing out.
        "tone_dorsal": 0.40,          # multiplies the de-lit skin chart
        "tone_ventral": 0.95,
        "tint": [1.10, 0.96, 1.00],   # the brown-purple cast of the real skin
        "phi_onset": 1.75,            # countershading ramp (radians from dorsal)
        "phi_full": 2.95,
        "fill_luminance": 0.86,       # chart cells above this are unobserved
        "smooth_px": 2.0,             # kills the x4 nearest-upscale blockiness
        "mottle": 0.045,              # fine multiplicative skin grain, sigma
        "mottle_px": 3.0,             # its correlation length in chart pixels
        "mottle_seed": 20260902,
        "eye_darkness": 0.26,         # albedo multiplier at the eye centre
        "eye_radius_m": 0.009,        # eye radius on the surface (see eye_chart_mask)
        "fin_tone_from_normal": True, # see tone_multiplier()
    },
    "pose": {
        "amp": [0.0, 0.35],
        "wave": [0.5, 0.75, 1.0],     # choice, not a range
        "yaw_deg": [-14.0, 14.0],
    },
    "camera": {
        "resolution": [1512, 2016],   # (H, W); a real frame is 3024x4032
        "fov_y_deg": 44.0,
        # Framing.  ``s_target`` is the station the camera CENTRES on and
        # ``s_frame_max`` the station whose projected span is ``width_frac`` of
        # the image, so the snout at s = 0 lands inside the frame only when
        # ``s_target < 0.5 * s_frame_max / width_frac``.  The first values
        # (s_target 0.25 against s_frame_max 0.26-0.38) failed that for every
        # draw, which pushed the snout off the edge and made the detected body
        # span the full width on every frame -- bbox_width_frac 1.00 against a
        # real median of 0.913.  Elevation is capped at 35 deg because above
        # that the view is near-dorsal, the pectorals splay, and the silhouette
        # becomes a fat cross (aspect ~1.05, area_norm ~0.31) that no real
        # hand-held photograph looks like.
        "elevation_deg": [0.0, 35.0], # above the lateral plane
        "azimuth_deg": [-25.0, 25.0], # about the dorsal axis
        "roll_deg": [-30.0, 30.0],
        "width_frac": [0.74, 0.92],   # framed span as a fraction of image width
        "s_frame_max": [0.30, 0.44],  # the tail runs out of frame past this
        "s_target": [0.14, 0.22],     # scalar accepted for old config.json
    },
    "light": {
        "elevation_deg": [30.0, 70.0],
        "azimuth_offset_deg": [-60.0, 60.0],   # from the camera azimuth
        "ambient": 0.35,
        "intensity": 0.85,
    },
    "specular": {
        # Wet skin under the sun: the real frames carry broad, strong highlights
        # along the dorsal ridge and over the gill region.
        "strength": [0.25, 0.60],
        "shininess": [18.0, 55.0],
    },
    "background": {
        # Sampled from the reference frames: a pale pink tub, a teal deck mat,
        # a blue tub, grey fibreglass, wet dark deck.
        "palette": [[0.86, 0.73, 0.69], [0.60, 0.75, 0.71], [0.55, 0.67, 0.78],
                    [0.72, 0.73, 0.72], [0.38, 0.42, 0.42]],
        "gradient": 0.28,             # peak-to-peak of the low-frequency ramp
        "noise": 0.035,               # sigma of the per-pixel noise
    },
    "occluders": {
        "count_probs": [0.35, 0.45, 0.20],   # P(0), P(1), P(2)
        "colors": [[0.62, 0.46, 0.36], [0.48, 0.34, 0.27], [0.72, 0.56, 0.46]],
        "scale_frac": [0.22, 0.42],   # long axis as a fraction of frame height
        "edge_offset": [0.55, 1.00],  # |offset| in frame half-widths
        "depth_frac": [0.45, 0.80],
        "cast_shadow_prob": 0.6,
    },
    "degrade": {
        "blur_sigma": [0.0, 1.5],
        "jpeg_quality": [75, 95],
    },
    "corpus": {
        "years": 3.0,
        "start_date": "2019-03-01",
        # None  -> 05's make_dataset.plan_sightings verbatim (field-catalogue
        #          statistics: singletons, a count drawn around the target, the
        #          flank flipped per sighting).
        # int k -> plan_same_side_sightings: exactly `sightings_per_individual`
        #          dates with at least k of them on one flank, which is what a
        #          matcher benchmark needs.  Set on the calibrated corpus.
        "min_same_side": None,
    },
}

#: The real frames the skin and spot parameters were matched against.
REFERENCE_PHOTOS = (
    "tagger/data/images_raw/IMG_20190828_104902.jpg",
    "tagger/data/images_raw/IMG_20190828_103029.jpg",
    "tagger/data/images_raw/PXL_20251017_205326759.jpg",
)


def _deep_update(base, patch):
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path=None, overrides=None):
    """The effective config: defaults, then ``path``'s JSON, then ``overrides``."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        with open(str(path)) as fh:
            _deep_update(cfg, json.load(fh))
    _deep_update(cfg, overrides or {})
    return cfg


def pattern_context(cfg):
    """``make_dataset.build_pattern_context`` with ``config["pattern"]`` applied.

    Every knob under ``pattern`` lands here, in one place, so ``generate`` and
    ``bench`` cannot drift apart.  Two of them are not ``PatternParams`` fields:

    * ``edge_softness`` sets ``pattern._EDGE_SOFTNESS``, the fraction of a
      spot's radius over which its coverage ramps to zero.  It is a module
      constant in prototype 05 and is assigned here as a module ATTRIBUTE (the
      same mechanism ``DEFAULT_SCHEMA_PATH`` already uses); 05's file is not
      edited.  It is global state, so it is set on every call rather than once
      at import.
    * ``drop_regions`` removes exclusion regions from PLACEMENT only.

    ``cs_floor`` is the countershading floor -- the residual spot amplitude on
    the ventrum -- and rides in the ``countershading`` dict with the two phis.
    """
    p = cfg["pattern"]
    pattern._EDGE_SOFTNESS = float(p.get("edge_softness", 0.25))
    ctx = make_dataset.build_pattern_context(
        n_spots=int(p["n_spots"]), n_common=int(p["n_common"]),
        chart_resolution=tuple(cfg["chart_resolution"]))
    ctx.params.radius_median = float(p["radius_median"])
    ctx.params.radius_log_sigma = float(p["radius_log_sigma"])
    ctx.params.min_sep = float(p["min_sep"])
    ctx.params.darkness_mean = float(p["darkness_mean"])
    ctx.params.darkness_sigma = float(p["darkness_sigma"])
    ctx.params.common_darkness = float(p["common_darkness"])
    ctx.params.dorsal_exponent = float(p["dorsal_exponent"])
    ctx.params.ecc_sigma = float(p["ecc_sigma"])
    ctx.params.ecc_max = float(p["ecc_max"])
    ctx.params.darkness_min = float(p["darkness_min"])
    ctx.params.darkness_max = float(p["darkness_max"])
    ctx.params.countershading = {"phi_onset": float(p["cs_phi_onset"]),
                                 "phi_full": float(p["cs_phi_full"]),
                                 "floor": float(p["cs_floor"])}
    drop = set(p.get("drop_regions") or ())
    if drop:
        ctx.regions = tuple(r for r in ctx.regions if r.name not in drop)
    return ctx


# ---------------------------------------------------------------------------
# Corpus plan
# ---------------------------------------------------------------------------

def plan_same_side_sightings(rng, n_sightings, years, start="2019-03-01",
                             min_same_side=2, primary_side_prob=0.72):
    """Dates and sides for one animal, with a guaranteed same-side recapture.

    05's :func:`make_dataset.plan_sightings` models a FIELD catalogue: it draws
    deliberate singletons (15%), varies the sighting count AROUND the target
    (2..target+2) and flips the flank per sighting at ``primary_side_prob``.
    That is the right generator for a protocol study and the wrong one for a
    matcher benchmark.  Measured on the smoke corpus it produced 3 identities x
    2 sightings in which BOTH same-individual pairs were L-vs-R -- pairs that
    share no spots by construction, so the resulting AUROC measured nothing.

    This is the benchmark plan instead: exactly ``n_sightings`` distinct dates,
    of which at least ``min_same_side`` are on one flank, so every identity
    contributes at least one scorable same-side pair.  Everything else is 05's:
    the first sighting lands in the first 35% of the window, gaps are drawn
    LOG-UNIFORMLY over the remainder (so the recapture-interval buckets are
    populated by construction, not by luck), and same-day collisions are pushed
    forward a day so no two sightings of one animal share an encounter.

    Returns a list of ``(date_str, side)`` in date order.
    """
    window = int(round(float(years) * 365.25))
    t0 = np.datetime64(str(start))
    n = max(1, int(n_sightings))
    first = int(rng.integers(0, max(1, int(0.35 * window))))
    if n == 1:
        offsets = [0]
    else:
        span = max(window - first, 30)
        raw = np.exp(rng.uniform(math.log(7.0), math.log(float(span)), size=n - 1))
        offsets = [0] + sorted(int(round(v)) for v in raw)
    dates = [t0 + np.timedelta64(first + o, "D") for o in offsets]
    uniq = []
    for d in dates:
        if not uniq or d > uniq[-1]:
            uniq.append(d)
        else:
            uniq.append(uniq[-1] + np.timedelta64(1, "D"))
    primary = "L" if rng.random() < 0.5 else "R"
    other = "R" if primary == "L" else "L"
    sides = [primary if rng.random() < primary_side_prob else other for _ in uniq]
    need = min(int(min_same_side), len(uniq))
    shortfall = need - sides.count(primary)
    if shortfall > 0:
        flippable = [i for i, s in enumerate(sides) if s != primary]
        rng.shuffle(flippable)
        for i in flippable[:shortfall]:
            sides[i] = primary
    return [(str(d), sd) for d, sd in zip(uniq, sides)]


def identity_timeline(ctx, seed, index, n_sightings, years, start_date,
                      min_same_side=2, length_bracket=None):
    """05's :func:`make_dataset.individual_timeline` on the benchmark plan.

    Identical seeding (``[seed, 1, index]`` for the plan and the length,
    ``[seed, 2, index]`` for the pattern), identical
    ``pattern.Individual.generate``, and identical ``drift.resight`` between
    consecutive dates -- so what a sighting shows is 05's DRIFTED pattern for
    that date, not a fresh draw.  Only :func:`plan_same_side_sightings`
    replaces ``plan_sightings``.
    """
    bracket = length_bracket or make_dataset.LENGTH_CM_BRACKET
    rng_i = np.random.default_rng([int(seed), 1, int(index)])
    identity = "syn%04d" % int(index)
    length_cm = float(rng_i.uniform(*bracket))
    plan = plan_same_side_sightings(rng_i, n_sightings, years, start=str(start_date),
                                    min_same_side=min_same_side)
    ind = pattern.Individual.generate(
        seed=int(np.random.default_rng([int(seed), 2, int(index)])
                 .integers(0, 2 ** 31 - 1)),
        params=ctx.params, identity=identity, date=plan[0][0],
        length_cm=length_cm, regions=ctx.regions)
    states = []
    prev_date = plan[0][0]
    for date, side in plan:
        if date != prev_date:
            ind = drift.resight(ind, prev_date, date, growth_model=ctx.growth)
            prev_date = date
        states.append((date, side, ind))
    return identity, length_cm, states


def _u(rng, span):
    """Uniform on ``[span[0], span[1]]``, REVERSED RANGES ALLOWED.

    ``low + (high - low) * rng.random()`` is byte-identical to
    ``rng.uniform(low, high)`` -- numpy computes exactly that from one
    ``next_double`` -- but does not raise on ``high < low``.  A reversed range
    negates each drawn value instead of mirroring the sample, which is what
    lets ``calibrate.BEFORE_OVERRIDE`` replay the pre-fix camera elevation
    frame for frame under the corrected sign.
    """
    lo, hi = float(span[0]), float(span[1])
    return float(lo + (hi - lo) * rng.random())


def _span(rng, value):
    """``_u`` for a knob that may still be written as a bare scalar.

    ``camera.s_target`` was a single number; it is a ``[lo, hi]`` range now
    (the station the camera centres on is what decides whether the snout falls
    inside the frame, and the real photographs vary it), and an old
    ``config.json`` must keep loading.
    """
    if isinstance(value, (list, tuple)):
        return _u(rng, value)
    return float(value)


# ---------------------------------------------------------------------------
# Charts: skin, tone, albedo, sampling
# ---------------------------------------------------------------------------

def _load_skin_png(path, resolution):
    """``assets/chart_skin_x4.png`` resampled to ``(H_phi, W_s, 3)`` in [0, 1].

    The PNG is already ``(H_phi, W_s)`` -- ``texture_identity.chart_to_image``
    transposes the bake layout on the way out -- so only the ``s`` axis needs
    resampling for the default 2048.  Verified: the file box-downsamples 4x to
    ``04-sevengill-rig/results/real/identity/chart_skin.png`` exactly.
    """
    from PIL import Image

    h, w = int(resolution[0]), int(resolution[1])
    img = Image.open(str(path)).convert("RGB")
    if img.size != (w, h):
        img = img.resize((w, h), Image.BILINEAR)
    return np.asarray(img, dtype=np.float64) / 255.0


DEFAULT_EYE_JSON = os.path.normpath(os.path.join(
    _PROTOTYPES, "04-sevengill-rig", "results", "real_v11", "eye", "eye_patch.json"))


def eye_chart_mask(resolution, body, radius_m=0.009, path=DEFAULT_EYE_JSON):
    """Soft ``(H_phi, W_s)`` elliptical mask over the two eyes, or ``None``.

    The centres are the ones prototype 04's eye-patch step MEASURED on this
    exact mesh (``results/real_v11/eye/eye_patch.json``, rest-pose 3-D points),
    charted here with the same ``tube_coords`` the body uses: they land at
    ``s ~ 0.024``, ``|phi| ~ 84 deg``.  Prototype 05's schema station puts
    ``eye_center`` at ``s = 0.06`` instead, and its ``eye_left``/``eye_right``
    exclusion RECTANGLES span ``s 0.04-0.08`` -- behind this scan's eye, and
    rectangular, so drawing them paints a dark oblong on the snout ridge.  A
    generic station table is the wrong instrument for "where is the eye on THIS
    animal"; a measurement on the mesh is the right one.

    ``radius_m`` is the eye's radius on the surface; it converts to half-widths
    ``radius / s_span_m`` in ``s`` and ``radius / r`` in ``phi``, so a thin
    snout (``r ~ 15 mm`` here) correctly gives an eye that wraps a lot of the
    girth.  The falloff is a smoothstep, ``1`` at the centre and ``0`` at the
    rim.  Why draw an eye at all: the OSEA spot detector was trained on real
    frames where every animal has one, so a synthetic frame without an eye is
    missing a feature the real corpus always presents to it.
    """
    if not os.path.exists(str(path)) or radius_m <= 0:
        return None
    with open(str(path)) as fh:
        doc = json.load(fh)
    pts = np.array([doc["source_eye_rest"], doc["target_eye_rest"]], dtype=float)
    import mesh3d

    coords = mesh3d.tube_coords(pts, body.centerline,
                                mesh3d.canonical_frames(len(body.centerline)))
    lo, hi = body.s_raw_range
    span = float(hi - lo)
    h, w = int(resolution[0]), int(resolution[1])
    s_axis, phi_axis = pattern.chart_axes((h, w))
    mask = np.zeros((h, w), dtype=float)
    for k in range(len(pts)):
        s0 = (float(coords.s[k]) - lo) / span
        phi0 = float(coords.phi[k])
        half_s = float(radius_m) / span
        half_phi = float(radius_m) / max(float(coords.r[k]), 1e-6)
        ds = (s_axis[None, :] - s0) / half_s
        dp = pattern.wrap_phi(phi_axis[:, None] - phi0) / half_phi
        d = np.hypot(ds, dp)
        u = np.clip(1.0 - d, 0.0, 1.0)
        mask = np.maximum(mask, u * u * (3.0 - 2.0 * u))
    return mask


def skin_chart(resolution, config):
    """Base skin albedo chart ``(H_phi, W_s, 3)``, WITHOUT the countershading.

    The de-lit scan skin (``assets/chart_skin_x4.png``) carries the real
    animal's fine tone variation and is nearly flat at 0.65 grey.  Four things
    happen to it here and the countershading tone is deliberately NOT one of
    them -- that is applied per PIXEL at shade time (:func:`tone_multiplier`),
    because a fin blade's tone follows its own upper/lower face, not its ``phi``
    around the body axis.

    1. Cells no texel reached come back near-white from the 04 read (the caudal
       overhang and the dropped fin texels are hole-filled to white); any cell
       brighter than ``skin.fill_luminance`` is replaced by the chart's median
       skin colour, so an unobserved cell renders as skin, not as a highlight.
    2. ``skin.smooth_px`` of Gaussian blur removes the blockiness the 4x
       NEAREST upscale in ``chart_skin_x4.png`` baked in (its native grid is
       240 x 128).
    3. A warm ``skin.tint``.
    4. ``skin.mottle`` of band-limited multiplicative grain, so the skin is not
       a perfectly clean background for a spot detector.  Seeded from
       ``skin.mottle_seed``, hence identical for every individual: it is skin
       texture, not identity.

    Returns ``(chart, stats)``.
    """
    h, w = int(resolution[0]), int(resolution[1])
    cfg = config["skin"]
    base = _load_skin_png(cfg.get("png") or DEFAULT_SKIN_PNG, (h, w))

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        # numpy 2.0.2 on Accelerate raises spurious FP warnings from matmul.
        lum = base @ np.array([0.2126, 0.7152, 0.0722])
    unobserved = lum > float(cfg["fill_luminance"])
    if unobserved.any() and (~unobserved).any():
        base = base.copy()
        base[unobserved] = np.median(base[~unobserved], axis=0)

    from scipy.ndimage import gaussian_filter
    smooth = float(cfg.get("smooth_px", 0.0))
    if smooth > 0:
        base = gaussian_filter(base, sigma=(smooth, smooth, 0.0), mode="wrap")

    mottle_sigma = float(cfg.get("mottle", 0.0))
    if mottle_sigma > 0:
        rng = np.random.default_rng(int(cfg.get("mottle_seed", 0)))
        grain = rng.standard_normal((h, w))
        px = float(cfg.get("mottle_px", 3.0))
        grain = gaussian_filter(grain, sigma=px, mode="wrap")
        grain = grain / max(float(grain.std()), 1e-9)
        base = base * (1.0 + mottle_sigma * grain)[..., None]

    tint = np.asarray(cfg["tint"], dtype=np.float64)
    out = np.clip(base * tint[None, None, :], 0.0, 1.0)
    return out, {"unobserved_frac": float(unobserved.mean()),
                 "median_rgb": [float(v) for v in np.median(out.reshape(-1, 3), axis=0)]}


def tone_multiplier(phi, config):
    """Countershading albedo multiplier: dark dorsum, pale ventrum.

    ``phi`` is an angle FROM THE DORSAL DIRECTION -- for a body pixel the chart
    ``phi``, for a fin pixel ``arccos(n_z)`` of its world normal (see
    :func:`shade_subject`).  The ramp is the same smoothstep
    ``exclusions.countershading_weight_at`` uses for speckle amplitude, so the
    tone and the spot attenuation agree on where the ventrum starts.
    """
    cfg = config["skin"]
    weight = exclusions.countershading_weight_at(
        phi, phi_onset=float(cfg["phi_onset"]), phi_full=float(cfg["phi_full"]))
    floor = exclusions.COUNTERSHADING_DEFAULTS["floor"]
    ventral = np.clip((1.0 - weight) / max(1.0 - floor, 1e-9), 0.0, 1.0)
    return (float(cfg["tone_dorsal"])
            + (float(cfg["tone_ventral"]) - float(cfg["tone_dorsal"])) * ventral)


def albedo_chart(skin, darkness, amplitude, eye_mask=None, eye_darkness=1.0):
    """``skin * (1 - amplitude * darkness)``, with the eyes darkened.

    ``darkness`` is ``pattern.render_chart``'s field: 0 unmarked, 1 fully dark.
    The eye layer is separate because it is ANATOMY, not identity: the real
    detector fires on a sevengill's eye, so a synthetic frame without one is
    missing a feature the real corpus always has.
    """
    d = np.asarray(darkness, dtype=np.float64)
    out = np.asarray(skin, dtype=np.float64) * (1.0 - float(amplitude) * d)[..., None]
    if eye_mask is not None and eye_darkness < 1.0:
        m = np.asarray(eye_mask, dtype=np.float64)[..., None]
        out = out * (1.0 - m * (1.0 - float(eye_darkness)))
    return np.clip(out, 0.0, 1.0)


def sample_chart(chart, s, phi):
    """Bilinear lookup of a ``(H_phi, W_s, C)`` chart at ``(s, phi)``.

    ``s`` clamps to the body ends; ``phi`` WRAPS (rows are periodic), so a
    pixel on the ventral seam blends the two ends of the array instead of
    clamping to one of them.  Cell centres follow ``pattern.chart_axes``:
    column ``i`` is ``s = (i+0.5)/W``, row ``j`` is
    ``phi = -pi + (j+0.5)*2pi/H`` -- the same indexing as
    ``bake.chart_indices``.  NaN inputs return NaN.
    """
    arr = np.asarray(chart, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[..., None]
    h, w = arr.shape[0], arr.shape[1]
    s = np.asarray(s, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    bad = ~(np.isfinite(s) & np.isfinite(phi))

    x = np.clip(np.nan_to_num(s, nan=0.0) * w - 0.5, 0.0, w - 1.0)
    y = (np.nan_to_num(phi, nan=0.0) + math.pi) / (2.0 * math.pi) * h - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    x1 = np.minimum(x0 + 1, w - 1)
    y0m = y0 % h
    y1m = (y0 + 1) % h
    out = (arr[y0m, x0] * (1 - fx) * (1 - fy) + arr[y0m, x1] * fx * (1 - fy)
           + arr[y1m, x0] * (1 - fx) * fy + arr[y1m, x1] * fx * fy)
    if bad.any():
        out = out.copy()
        out[bad] = np.nan
    return out[..., 0] if np.ndim(chart) == 2 else out


# ---------------------------------------------------------------------------
# Scene draw
# ---------------------------------------------------------------------------

class SceneDraw(NamedTuple):
    """Every seeded choice for one frame, kept as recordable data."""

    pose: dict
    side: str
    camera: dict
    light: dict
    specular: dict
    background: dict
    occluders: list
    degrade: dict


def draw_scene(rng, config, side="L"):
    """Draw one frame's nuisance parameters."""
    cam = config["camera"]
    lig = config["light"]
    spec = config["specular"]
    bg = config["background"]
    occ = config["occluders"]
    deg = config["degrade"]
    pos = config["pose"]

    n_occ = int(rng.choice(len(occ["count_probs"]),
                           p=np.asarray(occ["count_probs"], dtype=float)
                           / float(np.sum(occ["count_probs"]))))
    occluders = []
    for _ in range(n_occ):
        sign_x = 1.0 if rng.random() < 0.5 else -1.0
        sign_y = 1.0 if rng.random() < 0.5 else -1.0
        occluders.append({
            "color": [float(v) for v in
                      occ["colors"][int(rng.integers(len(occ["colors"])))]],
            "scale_frac": _u(rng, occ["scale_frac"]),
            "offset_x": sign_x * _u(rng, occ["edge_offset"]),
            "offset_y": sign_y * _u(rng, occ["edge_offset"]),
            "depth_frac": _u(rng, occ["depth_frac"]),
            "yaw_deg": float(rng.uniform(-90.0, 90.0)),
            "aspect": [1.0, float(rng.uniform(0.30, 0.55)),
                       float(rng.uniform(0.22, 0.40))],
            "casts_shadow": bool(rng.random() < float(occ["cast_shadow_prob"])),
        })

    return SceneDraw(
        pose={"amp": _u(rng, pos["amp"]),
              "wave": float(rng.choice(pos["wave"])),
              "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
              "yaw_deg": _u(rng, pos["yaw_deg"])},
        side=str(side),
        camera={"elevation_deg": _u(rng, cam["elevation_deg"]),
                "azimuth_deg": _u(rng, cam["azimuth_deg"]),
                "roll_deg": _u(rng, cam["roll_deg"]),
                "width_frac": _u(rng, cam["width_frac"]),
                "s_frame_max": _u(rng, cam["s_frame_max"]),
                "s_target": _span(rng, cam["s_target"]),
                "fov_y_deg": float(cam["fov_y_deg"]),
                "resolution": [int(cam["resolution"][0]), int(cam["resolution"][1])]},
        light={"elevation_deg": _u(rng, lig["elevation_deg"]),
               "azimuth_offset_deg": _u(rng, lig["azimuth_offset_deg"]),
               "ambient": float(lig["ambient"]),
               "intensity": float(lig["intensity"])},
        specular={"strength": _u(rng, spec["strength"]),
                  "shininess": _u(rng, spec["shininess"])},
        background={"color": [float(v) for v in
                              bg["palette"][int(rng.integers(len(bg["palette"])))]],
                    "gradient": float(bg["gradient"]),
                    "gradient_angle": float(rng.uniform(0.0, 2.0 * math.pi)),
                    "noise": float(bg["noise"]),
                    "seed": int(rng.integers(0, 2 ** 31 - 1))},
        occluders=occluders,
        degrade={"blur_sigma": _u(rng, deg["blur_sigma"]),
                 "jpeg_quality": int(rng.integers(int(deg["jpeg_quality"][0]),
                                                  int(deg["jpeg_quality"][1]) + 1))},
    )


# ---------------------------------------------------------------------------
# Camera framing
# ---------------------------------------------------------------------------

def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _rotate(vec, axis, angle_rad):
    """Rodrigues rotation of ``vec`` about unit ``axis``."""
    a = _unit(axis)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return vec * c + np.cross(a, vec) * s + a * float(np.dot(a, vec)) * (1.0 - c)


def frame_camera(vertices, s, draw, iterations=4):
    """A pinhole camera framing the head and forebody.

    The view direction starts LATERAL: perpendicular to the local body axis at
    ``s_target`` and horizontal (the animal's left for ``side="L"``, right for
    ``"R"``), then is tipped ``elevation_deg`` toward dorsal ``+Z`` and swung
    ``azimuth_deg`` about ``+Z``.  ``roll_deg`` rolls the image plane.

    The distance is solved so that the projected horizontal extent of the
    framed set (vertices with ``s <= s_frame_max``) is ``width_frac`` of the
    image width.  For a pinhole camera that extent scales as ``1/distance``, so
    a fixed-point iteration converges in two or three steps; ``iterations``
    runs a couple more for safety.  The tail is simply left out of frame.
    """
    cam = draw.camera
    v = np.asarray(vertices, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    h, w = int(cam["resolution"][0]), int(cam["resolution"][1])

    framed = s <= float(cam["s_frame_max"])
    if framed.sum() < 16:
        raise ValueError("s_frame_max=%r leaves %d vertices"
                         % (cam["s_frame_max"], int(framed.sum())))
    pts = v[framed]

    s_t = float(cam["s_target"])
    band = np.abs(s - s_t) < 0.015
    if band.sum() < 8:
        band = np.abs(s - s_t) < 0.05
    target = v[band].mean(axis=0)

    # Local body axis: the ring at s_target+d minus the ring at s_target-d.
    def _ring(centre):
        m = np.abs(s - centre) < 0.02
        return v[m].mean(axis=0) if m.any() else target

    axis = _unit(_ring(s_t + 0.05) - _ring(s_t - 0.05))
    dorsal = np.array([0.0, 0.0, 1.0])
    lateral = _unit(np.cross(axis, dorsal))      # animal's LEFT (B = T x N)
    if str(draw.side).upper().startswith("R"):
        lateral = -lateral

    # The sign is +: ``direction`` points from the target TOWARD the eye
    # (``eye = target + direction * dist``), and Rodrigues about
    # ``cross(lateral, dorsal)`` carries ``lateral`` toward ``+dorsal`` for a
    # POSITIVE angle.  The first version negated it, which put the eye BELOW
    # the animal and framed the countershaded, near-bare ventrum on every
    # frame -- the measured cause of "spots only in a dorsal band near the
    # silhouette".  See tests/test_calibrate.py::test_elevation_puts_the_eye_above.
    direction = _rotate(lateral, np.cross(lateral, dorsal),
                        math.radians(float(cam["elevation_deg"])))
    direction = _unit(_rotate(direction, dorsal,
                              math.radians(float(cam["azimuth_deg"]))))

    tan_y = math.tan(math.radians(0.5 * float(cam["fov_y_deg"])))
    tan_x = tan_y * (w / float(h))
    span = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    dist = max(span / (2.0 * tan_x * float(cam["width_frac"])), 1e-3)

    camera = None
    for _ in range(int(iterations)):
        camera = render.Camera(
            eye=target + direction * dist, target=target, up=dorsal,
            resolution=(h, w), kind="pinhole", fov_y_deg=float(cam["fov_y_deg"]),
            roll_deg=float(cam["roll_deg"]))
        px, _, pz = camera.project(pts)
        ok = np.isfinite(px) & (pz > camera.near)
        if not ok.any():
            dist *= 2.0
            continue
        extent = float(px[ok].max() - px[ok].min())
        if extent <= 1e-6:
            break
        dist *= extent / (float(cam["width_frac"]) * w)
    return camera, float(dist), target, direction


def _light_for(camera, draw):
    """A sun from the camera's side, ``elevation_deg`` above horizontal."""
    lig = draw.light
    look = camera.eye - camera.target
    az = math.atan2(float(look[1]), float(look[0])) + math.radians(
        float(lig["azimuth_offset_deg"]))
    el = math.radians(float(lig["elevation_deg"]))
    toward_sun = np.array([math.cos(el) * math.cos(az),
                           math.cos(el) * math.sin(az),
                           math.sin(el)])
    return render.DirectionalLight(direction=-toward_sun,
                                   intensity=float(lig["intensity"]),
                                   ambient=float(lig["ambient"]))


# ---------------------------------------------------------------------------
# Background and occluders
# ---------------------------------------------------------------------------

def background_image(resolution, draw):
    """A deck/tub-like ground: a palette colour, a low-frequency ramp, noise."""
    h, w = int(resolution[0]), int(resolution[1])
    bg = draw.background
    rng = np.random.default_rng(int(bg["seed"]))
    yy, xx = np.mgrid[0:h, 0:w]
    u = (xx / max(w - 1.0, 1.0) - 0.5)
    v = (yy / max(h - 1.0, 1.0) - 0.5)
    ang = float(bg["gradient_angle"])
    ramp = (u * math.cos(ang) + v * math.sin(ang))
    base = np.asarray(bg["color"], dtype=np.float64)[None, None, :]
    img = base * (1.0 + float(bg["gradient"]) * ramp)[..., None]
    # A second, coarser blob so the ground is not a pure linear ramp.
    coarse = rng.normal(size=(6, 8))
    from scipy.ndimage import zoom
    coarse = zoom(coarse, (h / 6.0, w / 8.0), order=3)[:h, :w]
    img = img * (1.0 + 0.06 * coarse[..., None])
    img = img + rng.normal(scale=float(bg["noise"]), size=(h, w, 1))
    return np.clip(img, 0.0, 1.0)


def hand_occluders(camera, draw):
    """0-2 hand-like ellipsoids placed near the frame edge, as occluders.

    Built as icospheres scaled to ``aspect``, then handed to
    ``nuisance.place_occluder``, which is what prototype 05 uses to put a
    second animal in the foreground: it puts the centroid at ``depth_frac`` of
    the eye->target distance and at ``(offset_x, offset_y)`` frame half-widths
    off centre, so an offset near 1.0 lands the blob on the frame edge.
    """
    import trimesh

    out = []
    for k, spec in enumerate(draw.occluders):
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
        scale = float(spec["scale_frac"])
        verts = np.asarray(sphere.vertices, dtype=np.float64) * np.asarray(
            spec["aspect"], dtype=np.float64)[None, :]
        inst = render.Instance(vertices=verts, faces=np.asarray(sphere.faces),
                               color=spec["color"], role="occluder",
                               casts_shadow=bool(spec["casts_shadow"]),
                               name="hand_%d" % k)
        # scale_frac is in frame heights at the occluder's depth; the ellipsoid
        # is built with a unit long axis, so that is exactly the scale.
        dist = float(np.linalg.norm(camera.target - camera.eye)) * float(spec["depth_frac"])
        half_h = nuisance._frame_half_height(camera, dist)
        placement = nuisance.OccluderPlacement(
            depth_frac=float(spec["depth_frac"]),
            offset_x=float(spec["offset_x"]), offset_y=float(spec["offset_y"]),
            yaw_deg=float(spec["yaw_deg"]), scale=scale * 2.0 * half_h,
            casts_shadow=bool(spec["casts_shadow"]))
        out.append(nuisance.place_occluder(inst, camera, placement,
                                           name="hand_%d" % k))
    return out


# ---------------------------------------------------------------------------
# Shading
# ---------------------------------------------------------------------------

def _view_directions(camera):
    """Unit vector from each pixel's surface point TOWARD the eye, ``(H, W, 3)``.

    Exact for a pinhole camera and needs no world positions: invert
    ``Camera.project``'s pixel mapping to get the NDC of each pixel centre,
    build the ray in camera axes and rotate it into world.
    """
    h, w = camera.resolution
    px = np.arange(w, dtype=np.float64)[None, :]
    py = np.arange(h, dtype=np.float64)[:, None]
    ndc_x = (px + 0.5) / w * 2.0 - 1.0
    ndc_y = 1.0 - (py + 0.5) / h * 2.0
    if camera.kind == "pinhole":
        tan_y = math.tan(math.radians(0.5 * camera.fov_y_deg))
        tan_x = tan_y * camera.aspect
        dx = ndc_x * tan_x
        dy = ndc_y * tan_y
        ray = (camera.right[None, None, :] * dx[..., None]
               + camera.up[None, None, :] * dy[..., None]
               + camera.forward[None, None, :])
        ray = ray / np.linalg.norm(ray, axis=-1, keepdims=True)
    else:
        ray = np.broadcast_to(camera.forward, (h, w, 3))
    return -ray


def shade_subject(out, camera, light, draw, albedo, config, fin_frac=None):
    """Deferred-textured subject RGB, ``(H, W, 3)``, valid on ``visible_skin``.

    ``shade = ambient + intensity * max(ndotl, 0) * (not shadow)`` -- recomputed
    here rather than divided out of ``out["rgb"]``, because ``render`` clips its
    product to [0, 1] and an unclipped albedo of 1 would saturate the lit side.
    The specular is Blinn-Phong on the interpolated normal, suppressed in
    shadow, which is what makes wet skin read as wet.

    THE FIN TONE.  Countershading is a body-frame property and the chart ``phi``
    expresses it correctly on the body -- but a pectoral blade sits at
    ``|phi| ~ 1.8-2.1``, deep in the pale ventral half of the ramp, so tone by
    ``phi`` alone paints the whole blade white on BOTH faces.  A shark's
    pectoral is dark on top and pale underneath, which is ``arccos(n_z)`` of its
    own normal, not its position around the girth.  So fin pixels take their
    tone angle from the normal and body pixels from the chart, blended over one
    triangle by ``fin_frac`` (the fraction of a face's vertices that are fin) so
    the fin root has no hard seam.  ``skin.fin_tone_from_normal = false`` turns
    this off and tones everything by ``phi``.
    """
    vis = out["visible_skin"]
    h, w = camera.resolution
    rgb = np.zeros((h, w, 3), dtype=np.float64)
    if not vis.any():
        return rgb

    ndotl = np.nan_to_num(out["ndotl"], nan=0.0)
    lit = np.clip(ndotl, 0.0, 1.0) * (~out["shadow"])
    shade = float(draw.light["ambient"]) + float(draw.light["intensity"]) * lit

    chart_s = out["chart_s"][vis]
    chart_phi = out["chart_phi"][vis]
    alb = sample_chart(albedo, chart_s, chart_phi)
    alb = np.where(np.isfinite(alb), alb, 0.5)

    normal = out["normal"][vis]
    tone = tone_multiplier(np.nan_to_num(chart_phi, nan=0.0), config)
    if fin_frac is not None and bool(config["skin"].get("fin_tone_from_normal", True)):
        blend = np.zeros(vis.shape, dtype=np.float64)
        sel = (out["instance"] == 0) & (out["face"] >= 0)
        blend[sel] = fin_frac[out["face"][sel]]
        blend = blend[vis]
        if blend.any():
            face_phi = np.arccos(np.clip(normal[:, 2], -1.0, 1.0))
            tone = tone * (1.0 - blend) + tone_multiplier(face_phi, config) * blend
    alb = alb * tone[:, None]

    view = _view_directions(camera)[vis]
    half = light.L[None, :] + view
    half = half / np.maximum(np.linalg.norm(half, axis=-1, keepdims=True), 1e-12)
    ndoth = np.clip(np.sum(normal * half, axis=-1), 0.0, 1.0)
    spec = (float(draw.specular["strength"])
            * ndoth ** float(draw.specular["shininess"])
            * (ndotl[vis] > 0.0) * (~out["shadow"][vis]))

    rgb[vis] = alb * shade[vis][:, None] + spec[:, None] * light.color[None, :]
    return np.clip(rgb, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Spot ground truth
# ---------------------------------------------------------------------------

def spot_ground_truth(out, spots, phi_scale):
    """Image-space ground truth for every rendered spot.

    For each spot the nearest VISIBLE subject pixel is found in the scaled
    chart metric ``d = hypot(s - s0, phi_scale * wrap(phi - phi0))``, the same
    metric ``pattern`` places and drifts spots in.  A spot counts as visible
    when that nearest pixel is within its own ``radius``; then

    * ``cx, cy`` are that pixel's centre;
    * ``radius_px`` is the AREA-EQUIVALENT radius ``sqrt(n / pi)`` of the
      visible pixels inside ``d <= radius`` -- honest under foreshortening,
      where a spot's footprint is an ellipse, and directly comparable to the
      ``w``/``h`` a spot detector would emit;
    * ``rendered_darkness`` is what ``pattern.render_chart`` actually stamped
      (0 for a spot not yet born at this date, or fully countershaded away).

    Spots with ``rendered_darkness <= 0`` are reported with ``visible: false``
    and no centre: nothing was drawn.
    """
    vis = out["visible_skin"]
    idx = np.flatnonzero(vis.ravel())
    h, w = vis.shape
    rows = []
    if idx.size == 0:
        for sp in spots:
            rows.append(_spot_row(sp, None, 0, phi_scale))
        return rows

    vs = out["chart_s"].ravel()[idx]
    vphi = out["chart_phi"].ravel()[idx]
    good = np.isfinite(vs) & np.isfinite(vphi)
    idx, vs, vphi = idx[good], vs[good], vphi[good]
    order = np.argsort(vs, kind="stable")
    idx, vs, vphi = idx[order], vs[order], vphi[order]

    for sp in spots:
        rad = float(sp["radius"])
        if float(sp["rendered_darkness"]) <= 0.0:
            rows.append(_spot_row(sp, None, 0, phi_scale))
            continue
        s0, phi0 = float(sp["s"]), float(sp["phi"])
        lo = int(np.searchsorted(vs, s0 - rad, side="left"))
        hi = int(np.searchsorted(vs, s0 + rad, side="right"))
        if hi <= lo:
            rows.append(_spot_row(sp, None, 0, phi_scale))
            continue
        ds = vs[lo:hi] - s0
        dphi = pattern.wrap_phi(vphi[lo:hi] - phi0) * float(phi_scale)
        d = np.hypot(ds, dphi)
        k = int(np.argmin(d))
        n_inside = int((d <= rad).sum())
        if float(d[k]) > rad:
            rows.append(_spot_row(sp, None, 0, phi_scale))
            continue
        flat = int(idx[lo + k])
        rows.append(_spot_row(sp, (flat % w, flat // w), n_inside, phi_scale))
    return rows


def _spot_row(sp, centre, n_inside, phi_scale):
    row = {
        "id": int(sp["id"]),
        "s": float(sp["s"]),
        "phi": float(sp["phi"]),
        "radius": float(sp["radius"]),
        "rendered_darkness": float(sp["rendered_darkness"]),
        "visible": centre is not None,
        "cx": None, "cy": None, "radius_px": None, "n_pixels": int(n_inside),
    }
    if centre is not None:
        row["cx"] = float(centre[0])
        row["cy"] = float(centre[1])
        row["radius_px"] = float(math.sqrt(max(n_inside, 1) / math.pi))
    return row


# ---------------------------------------------------------------------------
# One sighting
# ---------------------------------------------------------------------------

def render_sighting(body, individual, draw, skin, config, date=None,
                    eye_mask=None, fin_frac=None):
    """Render one frame.  Returns a dict; writes nothing.

    Keys: ``image`` (float RGB), ``out`` (the raw ``render`` bundle),
    ``spots`` (the ground-truth rows), ``camera``, ``light``, ``timings``.
    ``eye_mask`` and ``fin_frac`` are caches: both depend only on the body and
    the chart resolution, so :func:`generate` builds them once.
    """
    t = {}
    if fin_frac is None:
        fin_frac = body.is_fin[body.faces].mean(axis=1)
    t0 = time.time()
    verts = real_body.pose(body, **draw.pose)
    t["pose"] = time.time() - t0

    t0 = time.time()
    darkness, spot_table = pattern.render_chart(
        individual, resolution=tuple(config["chart_resolution"]), date=date)
    albedo = albedo_chart(skin, darkness, config["pattern"]["amplitude"],
                          eye_mask=eye_mask,
                          eye_darkness=float(config["skin"].get("eye_darkness", 1.0)))
    t["chart"] = time.time() - t0

    camera, dist, target, direction = frame_camera(verts, body.s, draw)
    light = _light_for(camera, draw)

    subject = render.Instance(vertices=verts, faces=body.faces, color=(1.0, 1.0, 1.0),
                              vertex_s=body.s, vertex_phi=body.phi, name="subject")
    instances = [subject] + hand_occluders(camera, draw)

    t0 = time.time()
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        # See skin_chart: numpy 2.0.2 + Accelerate warns spuriously on matmul.
        out = render.render(instances, camera, light=light, exclusion=None,
                            background=(0.0, 0.0, 0.0), shadows=True)
    t["raster"] = time.time() - t0

    t0 = time.time()
    image = background_image(camera.resolution, draw)
    occluded = out["coverage"] & ~out["visible_skin"]
    if occluded.any():
        image[occluded] = out["rgb"][occluded]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        subj = shade_subject(out, camera, light, draw, albedo, config,
                             fin_frac=fin_frac)
    image[out["visible_skin"]] = subj[out["visible_skin"]]
    t["shade"] = time.time() - t0

    t0 = time.time()
    sigma = float(draw.degrade["blur_sigma"])
    if sigma > 0.01:
        from scipy.ndimage import gaussian_filter
        image = gaussian_filter(image, sigma=(sigma, sigma, 0.0), mode="nearest")
    image = np.clip(image, 0.0, 1.0)
    t["degrade"] = time.time() - t0

    t0 = time.time()
    rows = spot_ground_truth(out, spot_table, individual.params.phi_scale)
    t["spot_gt"] = time.time() - t0

    cover = out["coverage"]
    if cover.any():
        cols = np.flatnonzero(cover.any(axis=0))
        rws = np.flatnonzero(cover.any(axis=1))
        bbox = [int(cols[0]), int(rws[0]),
                int(cols[-1] - cols[0] + 1), int(rws[-1] - rws[0] + 1)]
    else:
        bbox = None

    return {
        "image": image,
        "out": out,
        "spots": rows,
        "camera": camera,
        "light": light,
        "timings": t,
        "geometry": {
            "distance_m": float(dist),
            "target": [float(v) for v in target],
            "direction": [float(v) for v in direction],
            "coverage_bbox_xywh": bbox,
            "coverage_frac": float(cover.mean()),
            "visible_skin_frac": float(out["visible_skin"].mean()),
            "occlusion_frac": float(out["occlusion"].mean()),
            "shadow_frac": float(out["shadow"].mean()),
            "fin_stretch": real_body.fin_stretch(body, draw.pose["amp"]),
        },
    }


def write_sighting(out_dir, image_id, result, quality):
    """Write ``body/<id>.jpg``, ``gt/<id>.npz`` and ``gt/<id>_spots.json``."""
    from PIL import Image

    body_dir = os.path.join(out_dir, "body")
    gt_dir = os.path.join(out_dir, "gt")
    os.makedirs(body_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    jpg = os.path.join(body_dir, image_id + ".jpg")
    arr = (np.clip(result["image"], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(jpg, quality=int(quality), subsampling=0)

    o = result["out"]
    npz = os.path.join(gt_dir, image_id + ".npz")
    np.savez_compressed(
        npz,
        chart_s=o["chart_s"].astype(np.float32),
        chart_phi=o["chart_phi"].astype(np.float32),
        visible_skin=o["visible_skin"], shadow=o["shadow"],
        occlusion=o["occlusion"])

    spots_path = os.path.join(gt_dir, image_id + "_spots.json")
    with open(spots_path, "w") as fh:
        json.dump(result["spots"], fh, indent=1, sort_keys=True)
    return {"image": jpg, "gt": npz, "spots": spots_path}


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def generate(out_dir, n_individuals=3, sightings_per_individual=2, seed=0,
             config=None, report=True):
    """Render a corpus; returns the summary dict it also writes.

    Deterministic in ``(seed, config)``: identities come from
    ``make_dataset.individual_timeline`` (which is 05's single source of truth
    for what an animal looked like on a date) and every nuisance draw comes
    from a generator seeded by ``(seed, index, sighting)``.
    """
    cfg = config or load_config()
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    t_start = time.time()
    body = real_body.load_cached(cfg["body"]["cell_mm"])
    skin, skin_stats = skin_chart(cfg["chart_resolution"], cfg)

    ctx = pattern_context(cfg)
    min_same_side = cfg["corpus"].get("min_same_side")

    eye_mask = eye_chart_mask(cfg["chart_resolution"], body,
                              radius_m=float(cfg["skin"].get("eye_radius_m", 0.0)))
    fin_frac = body.is_fin[body.faces].mean(axis=1)

    truth_path = os.path.join(out_dir, "truth.jsonl")
    rows = []
    with open(truth_path, "w") as truth:
        for index in range(int(n_individuals)):
            if min_same_side is None:
                identity, length_cm, states = make_dataset.individual_timeline(
                    ctx, seed=seed, index=index,
                    sightings_per_individual=int(sightings_per_individual),
                    years=float(cfg["corpus"]["years"]),
                    start_date=str(cfg["corpus"]["start_date"]))
                # ``plan_sightings`` draws a count AROUND the target (2..target+2,
                # or 1 for a deliberate singleton); this prototype renders at most
                # the requested number so a corpus size is what it says it is.
                states = states[:int(sightings_per_individual)]
            else:
                identity, length_cm, states = identity_timeline(
                    ctx, seed=seed, index=index,
                    n_sightings=int(sightings_per_individual),
                    years=float(cfg["corpus"]["years"]),
                    start_date=str(cfg["corpus"]["start_date"]),
                    min_same_side=int(min_same_side))
            for k, (date, side, individual) in enumerate(states):
                rng = np.random.default_rng([int(seed), 7, int(index), int(k)])
                draw = draw_scene(rng, cfg, side=side)
                image_id = "%s_%02d" % (identity, k)
                res = render_sighting(body, individual, draw, skin, cfg, date=date,
                                      eye_mask=eye_mask, fin_frac=fin_frac)
                paths = write_sighting(out_dir, image_id, res,
                                       draw.degrade["jpeg_quality"])
                n_vis = int(sum(1 for r in res["spots"] if r["visible"]))
                row = {
                    "image_id": image_id,
                    "identity": identity,
                    "sighting": int(k),
                    "date": str(date),
                    "side": side,
                    "length_cm": float(length_cm),
                    "pose": draw.pose,
                    "camera": dict(draw.camera,
                                   distance_m=res["geometry"]["distance_m"],
                                   target=res["geometry"]["target"],
                                   direction=res["geometry"]["direction"]),
                    "light": dict(draw.light,
                                  direction=[float(v) for v in res["light"].direction]),
                    "specular": draw.specular,
                    "background": draw.background,
                    "occluders": draw.occluders,
                    "degrade": draw.degrade,
                    "n_spots": int(len(res["spots"])),
                    "n_visible_spots": n_vis,
                    "geometry": res["geometry"],
                    "timings": {kk: round(vv, 3) for kk, vv in res["timings"].items()},
                    "paths": {kk: os.path.relpath(vv, out_dir)
                              for kk, vv in paths.items()},
                }
                truth.write(json.dumps(row, sort_keys=True) + "\n")
                truth.flush()
                rows.append(row)
                if report:
                    print("%-14s %s %s  %3d/%3d spots  bbox %s  %.1fs"
                          % (image_id, date, side, n_vis, len(res["spots"]),
                             res["geometry"]["coverage_bbox_xywh"],
                             sum(res["timings"].values())))

    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2, sort_keys=True)

    vis = [r["n_visible_spots"] for r in rows]
    radii = [sp["radius_px"] for r in rows
             for sp in json.load(open(os.path.join(out_dir, r["paths"]["spots"])))
             if sp["radius_px"]]
    summary = {
        "out_dir": out_dir,
        "seed": int(seed),
        "n_individuals": int(n_individuals),
        "n_frames": len(rows),
        "plan": ("plan_same_side_sightings(min_same_side=%d)" % int(min_same_side)
                 if min_same_side is not None else "make_dataset.plan_sightings"),
        "body": body.meta,
        "skin": skin_stats,
        "schema": SCHEMA_PATH,
        "visible_spots": {"min": int(min(vis)) if vis else 0,
                          "median": float(np.median(vis)) if vis else 0.0,
                          "max": int(max(vis)) if vis else 0},
        "spot_radius_px": ({"p5": float(np.percentile(radii, 5)),
                            "median": float(np.median(radii)),
                            "p95": float(np.percentile(radii, 95))}
                           if radii else None),
        "coverage_frac_median": float(np.median(
            [r["geometry"]["coverage_frac"] for r in rows])) if rows else 0.0,
        "seconds": round(time.time() - t_start, 2),
        "seconds_per_frame": round((time.time() - t_start) / max(len(rows), 1), 2),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    if report:
        print("wrote %d frames to %s in %.1fs (%.2fs/frame)"
              % (len(rows), out_dir, summary["seconds"],
                 summary["seconds_per_frame"]))
    return summary


# ---------------------------------------------------------------------------
# Contact sheets
# ---------------------------------------------------------------------------

def contact_sheet(out_dir, path=None, cell_width=640, cols=3, pad=8,
                  label_height=22, background=(28, 30, 32)):
    """Every frame of a corpus in one grid, labelled with its image id."""
    from PIL import Image, ImageDraw

    path = path or os.path.join(out_dir, "smoke_contact.png")
    rows = [json.loads(line) for line in
            open(os.path.join(out_dir, "truth.jsonl")) if line.strip()]
    if not rows:
        raise ValueError("no frames in %s" % out_dir)
    tiles = []
    for r in rows:
        img = Image.open(os.path.join(out_dir, r["paths"]["image"])).convert("RGB")
        scale = cell_width / float(img.width)
        tiles.append((r, img.resize((cell_width, int(round(img.height * scale))),
                                    Image.LANCZOS)))
    cell_h = max(t.height for _, t in tiles) + label_height
    n_cols = min(int(cols), len(tiles))
    n_rows = int(math.ceil(len(tiles) / float(n_cols)))
    sheet = Image.new("RGB", (n_cols * cell_width + (n_cols + 1) * pad,
                              n_rows * cell_h + (n_rows + 1) * pad), background)
    draw = ImageDraw.Draw(sheet)
    for k, (r, tile) in enumerate(tiles):
        cx = pad + (k % n_cols) * (cell_width + pad)
        cy = pad + (k // n_cols) * (cell_h + pad)
        sheet.paste(tile, (cx, cy))
        draw.text((cx + 4, cy + tile.height + 4),
                  "%s  %s %s  %d/%d spots  q%d  blur %.2f"
                  % (r["image_id"], r["date"], r["side"], r["n_visible_spots"],
                     r["n_spots"], r["degrade"]["jpeg_quality"],
                     r["degrade"]["blur_sigma"]),
                  fill=(230, 230, 225))
    sheet.save(path)
    return path


def _flank_crop(out_dir, row, size=900, s_centre=None, phi_centre=1.15):
    """A ``size``-square crop centred on the flank, from the chart ground truth.

    The centre is the visible pixel nearest ``(s_centre, |phi| = phi_centre)``
    in the scaled chart metric -- i.e. mid-flank, halfway between the dorsal
    ridge and the ventrum, where the OSEA photographs show their spots.
    """
    from PIL import Image

    img = Image.open(os.path.join(out_dir, row["paths"]["image"])).convert("RGB")
    with np.load(os.path.join(out_dir, row["paths"]["gt"])) as z:
        cs, cp, vis = z["chart_s"], z["chart_phi"], z["visible_skin"]
    if s_centre is None:
        s_centre = 0.55 * float(row["camera"]["s_frame_max"])
    sign = 1.0 if str(row["side"]).upper().startswith("L") else -1.0
    idx = np.flatnonzero(vis.ravel())
    if idx.size:
        d = np.hypot(cs.ravel()[idx] - s_centre,
                     0.085 * pattern.wrap_phi(cp.ravel()[idx] - sign * phi_centre))
        flat = int(idx[int(np.argmin(d))])
        cx, cy = flat % vis.shape[1], flat // vis.shape[1]
    else:
        cy, cx = vis.shape[0] // 2, vis.shape[1] // 2
    half = size // 2
    cx = int(np.clip(cx, half, img.width - half))
    cy = int(np.clip(cy, half, img.height - half))
    return img.crop((cx - half, cy - half, cx + half, cy + half))


def zoom_contact(out_dir, real_paths, path=None, size=900, real_crop_frac=0.45,
                 n_frames=2, pad=10, label_height=24, background=(28, 30, 32)):
    """Synthetic flank zooms beside real OSEA photographs, at matched scale.

    Both sides are ``size`` px across.  The synthetic crop is ``size`` pixels of
    a 2016-wide render; the real crop is ``real_crop_frac`` of the full-
    resolution photograph's width (a 4032-wide frame, so 0.45 -> 1814 px)
    resampled to ``size``.  Because the two framings cover a similar arc of the
    animal, a spot that is the right size in the render is the same size on the
    page as a spot in the photograph -- which is the only thing this sheet is
    for.  A missing full-resolution file falls back to its 512-wide thumbnail.
    """
    from PIL import Image, ImageDraw, ImageOps

    path = path or os.path.join(out_dir, "zoomed_contact.png")
    rows = [json.loads(line) for line in
            open(os.path.join(out_dir, "truth.jsonl")) if line.strip()]
    rows = sorted(rows, key=lambda r: -r["n_visible_spots"])[:int(n_frames)]

    panels = []
    for r in rows:
        panels.append(("synthetic %s (%d visible spots)"
                       % (r["image_id"], r["n_visible_spots"]),
                       _flank_crop(out_dir, r, size=size)))
    for p in real_paths:
        img = ImageOps.exif_transpose(Image.open(str(p))).convert("RGB")
        side = int(round(real_crop_frac * img.width))
        side = min(side, img.width, img.height)
        cx, cy = img.width // 2, int(0.55 * img.height)
        cy = int(np.clip(cy, side // 2, img.height - side // 2))
        crop = img.crop((cx - side // 2, cy - side // 2,
                         cx + side // 2, cy + side // 2)).resize(
                             (size, size), Image.LANCZOS)
        panels.append(("real %s" % os.path.basename(str(p)), crop))

    cols = 2
    n_rows = int(math.ceil(len(panels) / float(cols)))
    cell_h = size + label_height
    sheet = Image.new("RGB", (cols * size + (cols + 1) * pad,
                              n_rows * cell_h + (n_rows + 1) * pad), background)
    draw = ImageDraw.Draw(sheet)
    for k, (label, tile) in enumerate(panels):
        cx = pad + (k % cols) * (size + pad)
        cy = pad + (k // cols) * (cell_h + pad)
        sheet.paste(tile, (cx, cy))
        draw.text((cx + 4, cy + size + 5), label, fill=(230, 230, 225))
    sheet.save(path)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_real_paths():
    main = os.path.normpath(os.path.join(_PROTOTYPES, "..", "..", "..", ".."))
    out = []
    for rel in REFERENCE_PHOTOS[:2]:
        p = os.path.join(main, rel)
        if os.path.exists(p):
            out.append(p)
    return out


def bench(cells=(1.5, 2.0, 2.5, 4.0, 6.0), n_frames=3, seed=0, config=None,
          report=True):
    """Rasterisation cost per frame at several decimation levels.

    Times the two things that scale with the face count -- ``render.render``
    with the shadow pass on, and the same call with it off, whose difference is
    the shadow map -- on real draws from :func:`draw_scene`, at the production
    resolution.  Everything else in a frame (chart, deferred texture,
    background, blur, JPEG, spot ground truth) is independent of the mesh and
    is timed once and reported alongside.
    """
    cfg = config or load_config()
    skin, _ = skin_chart(cfg["chart_resolution"], cfg)
    ctx = pattern_context(cfg)
    _, _, states = make_dataset.individual_timeline(ctx, seed=seed, index=0,
                                                    sightings_per_individual=2)
    individual = states[0][2]

    rows = []
    for cell in cells:
        body = real_body.load_cached(float(cell))
        fin_frac = body.is_fin[body.faces].mean(axis=1)
        with_shadow, without, whole = [], [], []
        for k in range(int(n_frames)):
            draw = draw_scene(np.random.default_rng([int(seed), 99, k]), cfg)
            verts = real_body.pose(body, **draw.pose)
            camera, _, _, _ = frame_camera(verts, body.s, draw)
            light = _light_for(camera, draw)
            inst = render.Instance(vertices=verts, faces=body.faces,
                                   color=(1.0, 1.0, 1.0), vertex_s=body.s,
                                   vertex_phi=body.phi, name="subject")
            insts = [inst] + hand_occluders(camera, draw)
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                t0 = time.time()
                render.render(insts, camera, light=light, exclusion=None,
                              background=(0.0, 0.0, 0.0), shadows=True)
                with_shadow.append(time.time() - t0)
                t0 = time.time()
                render.render(insts, camera, light=light, exclusion=None,
                              background=(0.0, 0.0, 0.0), shadows=False)
                without.append(time.time() - t0)
            t0 = time.time()
            res = render_sighting(body, individual, draw, skin, cfg,
                                  fin_frac=fin_frac)
            whole.append(time.time() - t0)
        row = {
            "cell_mm": float(cell),
            "n_vertices": int(len(body.vertices)),
            "n_faces": int(len(body.faces)),
            "raster_with_shadow_s": float(np.median(with_shadow)),
            "raster_without_shadow_s": float(np.median(without)),
            "shadow_map_s": float(np.median(with_shadow) - np.median(without)),
            "whole_frame_s": float(np.median(whole)),
            "last_frame_stage_s": {k: round(v, 3) for k, v in res["timings"].items()},
            "resolution": list(cfg["camera"]["resolution"]),
            "n_frames": int(n_frames),
        }
        rows.append(row)
        if report:
            print("cell %4.1f mm  V %7d  F %7d  raster %.2fs (shadow map %.2fs) "
                  " whole frame %.2fs"
                  % (cell, row["n_vertices"], row["n_faces"],
                     row["raster_with_shadow_s"], row["shadow_map_s"],
                     row["whole_frame_s"]))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(_HERE, "results", "synth_smoke"))
    ap.add_argument("--identities", type=int, default=3)
    ap.add_argument("--sightings", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--cell-mm", type=float, default=None)
    ap.add_argument("--min-same-side", type=int, default=None,
                    help="benchmark plan: exactly --sightings dates per identity, "
                         "at least this many on one flank (default: 05's "
                         "field-catalogue plan_sightings)")
    ap.add_argument("--contact", action="store_true",
                    help="also write smoke_contact.png / zoomed_contact.png")
    ap.add_argument("--real", nargs="*", default=None,
                    help="real photographs for the zoom sheet")
    ap.add_argument("--bench", nargs="?", const="1.5,2.0,2.5,4,6", default=None,
                    help="time the rasteriser at these cell sizes and exit")
    args = ap.parse_args(argv)

    if args.bench:
        cells = [float(c) for c in str(args.bench).split(",")]
        rows = bench(cells=cells, seed=args.seed,
                     config=load_config(args.config))
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        path = os.path.join(os.path.dirname(args.out) or ".", "bench.json")
        with open(path, "w") as fh:
            json.dump(rows, fh, indent=2, sort_keys=True)
        print("wrote", path)
        return 0

    overrides = {}
    if args.cell_mm is not None:
        overrides["body"] = {"cell_mm": float(args.cell_mm)}
    if args.min_same_side is not None:
        overrides["corpus"] = {"min_same_side": int(args.min_same_side)}
    cfg = load_config(args.config, overrides)
    generate(args.out, n_individuals=args.identities,
             sightings_per_individual=args.sightings, seed=args.seed, config=cfg)
    if args.contact:
        print("contact:", contact_sheet(args.out))
        real = args.real if args.real is not None else _default_real_paths()
        if real:
            print("zoom:   ", zoom_contact(args.out, real))
        else:
            print("zoom:    skipped (no real photographs found)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
