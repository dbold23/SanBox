"""COPY A REAL INDIVIDUAL: near-lateral photograph -> canonical chart space.

This is the owner's requirement (2), "copy real individuals' actual skin
patterns onto the 3D model", implemented as the cheap 2D path:

    photo + body mask  ->  centerline  ->  (s, r) strip  ->  (s, phi) chart

and then :func:`copy_from_photo` hands that chart to ``pattern.copy_from_chart``
(module P), which is what actually installs it as an individual.  Baking the
chart onto a mesh is :func:`bake.bake_chart_to_texture`.

SCOPE, STATED UP FRONT.  This module assumes a NEAR-LATERAL view of a locally
circular cross-section.  That assumption buys a closed-form ``r -> phi``
inverse and nothing else.  It is WRONG for oblique views, for rolled animals,
and near the flattened head, and it is SUPERSEDED -- not merely complemented --
by the full-3D fit path: fit the prototype 04 rig to the photograph, render,
and solve for the chart by analysis-by-synthesis.  That path handles arbitrary
pose, resolves the near/far surface ambiguity properly, and gives a per-pixel
visibility term instead of this module's cosine proxy.  Use this module for
archival near-lateral catalogue shots and for bootstrapping; use the 3D fit for
anything oblique.

THE r -> phi CONVERSION
-----------------------
For an orthographic lateral view of a circular cross-section of radius
``R_local``, a surface point at circumferential angle ``phi`` projects to
signed image offset ``r = R_local * cos(phi)`` -- because the binding chart
convention measures ``phi`` FROM THE DORSAL MIDLINE (see ``bake``), and the
dorsal direction is the image-vertical one, not the view direction.  Hence

    phi = side_sign * arccos(dorsal_sign * r / R_local)

with ``side_sign = +1`` for a left-side photograph (visible half
``phi in (0, pi)``, since ``+pi/2`` IS the animal's left) and ``-1`` for a
right-side one.

The brief writes this as ``phi = asin(r / R_local)``.  That is the same
geometry with ``phi`` measured from the FLANK instead of from the dorsum; the
two differ by the quarter turn that the binding convention fixes:

    arccos(x) = pi/2 - asin(x)   =>   phi_dorsal_origin = +-(pi/2 - asin(r/R))

The dorsal-origin form is used here because it is what
``mesh3d.TubeCoords.phi`` and ``bake`` already mean, and having two ``phi``
zeros in one pipeline is exactly the sort of thing that silently rotates every
pattern by 90 degrees.

CONFIDENCE
----------
``d(phi)/dr = -1 / (R_local * sin(phi))`` blows up as ``|r| -> R_local``: at the
silhouette the view grazes the surface, one pixel of image covers an unbounded
sweep of girth, and the recovered ``phi`` is worthless.  The confidence
returned is the foreshortening factor

    c = sin(|phi|) = sqrt(1 - (r / R_local)^2)

which is both the reciprocal of that Jacobian (up to ``R_local``) and the
cosine of the angle between the surface normal and the view direction -- the
same number a renderer would use.  It is multiplied by an exclusion mask
(eye, mouth) and by sample validity, and it is what downstream code must
weight by; a chart cell with ``confidence == 0`` carries no measurement.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import NamedTuple

import numpy as np
from scipy import ndimage

from bake import (
    DELIGHT_SIGMA_PHI,
    DELIGHT_SIGMA_S,
    masked_gaussian,
    luminance,
    splat_to_chart,
    wrap_to_pi,
)

__all__ = [
    "SCHEMA_PATH",
    "UnbakeResult",
    "load_schema_landmarks",
    "eye_mouth_exclusion",
    "resolve_exclusion_mask",
    "local_half_width",
    "dorsal_sign_from_point",
    "photo_to_chart",
    "copy_from_photo",
    "EYE_EXCLUSION_RADIUS",
    "MOUTH_VENTRAL_PHI",
]

_P02 = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "02-centerline-chart")
)

SCHEMA_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..",
        "phase1b", "p0-sevengill-schema", "keypoints_sevengill_v1.yaml",
    )
)

# Exclusion geometry.  DELIBERATELY NOT arc-length fractions: Schema S1 does
# not publish an ``axis_fraction`` for any head landmark (only the seven
# midline semilandmarks carry one, at k/8), and its own open_questions call the
# fin stations "provisional ... [UNVERIFIED]".  So the exclusion regions are
# built around MEASURED landmark positions handed in by the caller, and only
# their SIZE is a constant here.
#
# EYE_EXCLUSION_RADIUS: geodesic radius of the eye exclusion disc, in body
#   lengths.  [BRACKET 0.010-0.035] [EVIDENCE GRADE: UNVERIFIED -- no published
#   Notorynchus cepedianus eye-diameter/TL ratio was retrieved; the default is
#   a round number inside the bracket, sized to also swallow the eyelid margins
#   and the specular highlight that sits on them.  Schema S1 calls the eye
#   "relatively small ... no nictitating membrane" (id 2), which argues for the
#   low end.  Re-fit from annotated frames.]
# MOUTH_VENTRAL_PHI: the mouth exclusion is the head band anterior to the
#   rictus, restricted to |phi| above this, i.e. the ventral sector.  The
#   sevengill mouth is a broad ventral crescent reaching back to the rictus;
#   2.0 rad = 115 deg from the dorsum leaves the dorsolateral head -- where the
#   La Jolla freckle patch actually lives (Schema S1 id 1, head_patch_bounds =
#   naris -> gill slit 1) -- fully available.
#   [BRACKET 1.7-2.4 rad] [EVIDENCE GRADE: UNVERIFIED, geometric.]
EYE_EXCLUSION_RADIUS = 0.020
MOUTH_VENTRAL_PHI = 2.0

# Nominal girth radius used to make ``phi`` distances commensurable with ``s``
# distances when measuring a disc on the skin.  [BRACKET 0.06-0.12 body
# lengths] [EVIDENCE GRADE: derived from the fixture profile in
# ``fixtures.shark_radius_profile`` (r_max ~ 0.12 L); a real value comes from
# the fitted rig's radius profile, which is why callers can override it.]
NOMINAL_RADIUS = 0.09


# Prototype 02 is loaded BY FILE PATH, not by name.
#
# A plain ``from chart import rectify`` after putting ../02-centerline-chart on
# sys.path is a trap in this prototype: prototype 05's own package directory is
# already first on sys.path, so the moment a sibling module here is named
# ``chart.py`` (an entirely natural name for chart-space code) the import
# silently binds to the wrong module and ``rectify`` disappears -- or worse,
# resolves to a different function of the same name.  Loading from an explicit
# file location under a ``_p02_`` prefix makes the dependency unambiguous.
# Prototype 02's own modules import each other by bare name, so those two names
# are aliased for the duration of the exec and then restored.
_P02_CACHE = {}
_P02_ORDER = ("centerline", "frames", "chart")


def _p02(name):
    """Import ``prototypes/02-centerline-chart/<name>.py`` by path."""
    import importlib.util

    if name in _P02_CACHE:
        return _P02_CACHE[name]
    for dep in _P02_ORDER:
        if dep == name:
            break
        if dep not in _P02_CACHE:
            _p02(dep)

    path = os.path.join(_P02, name + ".py")
    if not os.path.exists(path):
        raise RuntimeError(
            "prototype 02 module %r not found at %s -- unbake needs "
            "extract_centerline and rectify from 02-centerline-chart" % (name, path)
        )
    spec = importlib.util.spec_from_file_location("_p02_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    saved = {n: sys.modules.get(n) for n in _P02_ORDER}
    for n in _P02_ORDER:
        if n in _P02_CACHE:
            sys.modules[n] = _P02_CACHE[n]
    try:
        spec.loader.exec_module(mod)
    finally:
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m
    _P02_CACHE[name] = mod
    return mod


class UnbakeResult(NamedTuple):
    """Output of :func:`photo_to_chart`.

    chart:       ``(n_s, n_phi)`` albedo-MULTIPLIER chart (1 = unmarked skin),
                 the same semantics ``bake.bake_chart_to_texture`` consumes.
                 ``NaN`` where nothing was observed.
    confidence:  ``(n_s, n_phi)`` in ``[0, 1]``; 0 = no usable measurement.
    strip:       ``(n_s_strip, n_r)`` rectified ``(s, r)`` strip, for debugging.
    centerline:  ``(n_stations, 2)`` extracted centerline in image pixels.
    half_width:  the scalar strip half-width in pixels used by ``chart.rectify``.
    radius_px:   ``(n_s_strip,)`` local body half-width per strip row, pixels.
    dorsal_sign: ``+1`` if chart ``r > 0`` points dorsally, ``-1`` otherwise.
    side_sign:   ``+1`` for a left-side photograph, ``-1`` for a right-side one.
    notes:       list of human-readable caveats raised while unbaking.
    """

    chart: np.ndarray
    confidence: np.ndarray
    strip: np.ndarray
    centerline: np.ndarray
    half_width: float
    radius_px: np.ndarray
    dorsal_sign: int
    side_sign: int
    notes: list


# ---------------------------------------------------------------------------
# exclusions, derived from Schema S1 landmarks (never from hard-coded fractions)
# ---------------------------------------------------------------------------

def load_schema_landmarks(path=SCHEMA_PATH):
    """Read the Schema S1 keypoint names/types/tiers from the YAML.

    Returns ``{name: {"id", "type", "tier", "description"}}``.

    The schema is READ, not paraphrased, because the exclusion regions must be
    anchored to real landmark names -- ``eye_center`` (id 2, Type I) and
    ``mouth_rictus`` (id 3, Type I) -- and because a typo'd landmark name would
    otherwise silently produce an empty exclusion.  The YAML publishes NO
    arc-length fraction for either point, so their ``(s, phi)`` must be
    supplied per-image by the caller (annotation, pose model, or the fitted
    rig).  That is the whole reason ``eye_mouth_exclusion`` takes landmarks
    rather than constants.
    """
    import yaml  # local: only the exclusion path needs it

    with open(path, "r") as fh:
        doc = yaml.safe_load(fh)
    out = {}
    for kp in doc.get("keypoints", []):
        out[kp["name"]] = {
            "id": int(kp["id"]),
            "type": kp.get("type"),
            "tier": kp.get("tier"),
            "description": kp.get("description", ""),
        }
    return out


def _chart_meshgrid(chart_shape):
    n_s, n_phi = int(chart_shape[0]), int(chart_shape[1])
    s_ax = (np.arange(n_s) + 0.5) / n_s
    phi_ax = -np.pi + (np.arange(n_phi) + 0.5) * (2 * np.pi / n_phi)
    return np.meshgrid(s_ax, phi_ax, indexing="ij")


def eye_mouth_exclusion(
    chart_shape,
    landmarks,
    eye_radius=EYE_EXCLUSION_RADIUS,
    mouth_ventral_phi=MOUTH_VENTRAL_PHI,
    nominal_radius=NOMINAL_RADIUS,
    snout_s=0.0,
    validate=True,
):
    """Chart mask of regions that carry no identity pattern: eye and mouth.

    Args:
        chart_shape: ``(n_s, n_phi)``.
        landmarks: ``{name: (s, phi)}`` in chart coordinates.  Recognised:
            ``eye_center`` (Schema S1 id 2) and ``mouth_rictus`` (id 3).
            Missing ones simply contribute nothing, and are reported.
        eye_radius: geodesic radius of the eye disc, in body lengths.
        mouth_ventral_phi: the mouth band covers ``|phi| >= this``.
        snout_s: anterior end of the mouth band (0 = the snout).
        validate: check the landmark names against the YAML schema.

    Returns:
        ``(mask, notes)``: ``mask`` is a bool ``(n_s, n_phi)`` array, ``True``
        where the surface is EXCLUDED; ``notes`` is a list of strings.

    Construction, and why it is shaped this way.  The eye is a compact feature,
    so it is a disc on the skin (``phi`` distances converted to arc length via
    ``nominal_radius``, so the disc is round on the animal rather than round in
    index space).  The mouth is NOT compact: on a sevengill it is a broad
    ventral crescent running back to the rictus, so it is the head band
    ``s in [snout_s, s_rictus]`` intersected with the ventral sector.  That
    leaves the dorsolateral head -- the naris-to-gill-slit-1 "freckle patch"
    that Schema S1 names as the head-patch crop (id 1, ``chart.head_patch_bounds``)
    -- entirely inside the identity surface, which is the point.

    NOT INCLUDED: the fins, because Schema S1's ``chart`` block already declares
    ``identity_surface: trunk flank only; fins excluded`` and a tube chart holds
    no fin material to exclude.

    ALSO NOT INCLUDED, and this one is a KNOWN GAP rather than a decision: the
    seven gill slits.  :func:`resolve_exclusion_mask` settles that question in
    favour of excluding them and uses module P's mask, which does.  This
    function stays eye-and-mouth-only because it is the degraded fallback for
    when module P is absent: it has no station table, and deriving a
    seven-slit band from two head landmarks would be inventing geometry.  A
    chart built through this fallback therefore still contains gill pixels --
    which is why the fallback says so in its notes.
    """
    n_s, n_phi = int(chart_shape[0]), int(chart_shape[1])
    S, P = _chart_meshgrid((n_s, n_phi))
    mask = np.zeros((n_s, n_phi), dtype=bool)
    notes = []

    if validate:
        try:
            known = load_schema_landmarks()
        except Exception as exc:                      # pragma: no cover - env
            known = None
            notes.append("schema not read (%s); landmark names unvalidated" % exc)
        if known is not None:
            unknown = [k for k in landmarks if k not in known]
            if unknown:
                raise ValueError(
                    "landmark names not in Schema S1 (%s): %r"
                    % (os.path.basename(SCHEMA_PATH), sorted(unknown))
                )

    if "eye_center" in landmarks:
        s0, p0 = (float(v) for v in landmarks["eye_center"])
        d = np.hypot(S - s0, wrap_to_pi(P - p0) * float(nominal_radius))
        mask |= d <= float(eye_radius)
    else:
        notes.append("no eye_center landmark: eye NOT excluded")

    if "mouth_rictus" in landmarks:
        s_r = float(landmarks["mouth_rictus"][0])
        band = (S >= float(snout_s)) & (S <= s_r)
        mask |= band & (np.abs(P) >= float(mouth_ventral_phi))
    else:
        notes.append("no mouth_rictus landmark: mouth NOT excluded")

    return mask, notes


# Names probed on module P, in order.  ``pattern`` deliberately ships its hook
# as ``chart_exclusion_mask`` rather than ``exclusion_mask`` so that landing
# module P could not silently change this module's behaviour, and left the
# activation to "whoever owns the reconciliation".  That is this module, and
# this tuple is the activation -- written here rather than as an alias in
# ``pattern.py``, so the decision lives with the consumer that has to defend it.
_PATTERN_EXCLUSION_HOOKS = ("exclusion_mask", "chart_exclusion_mask")


def resolve_exclusion_mask(chart_shape, landmarks=None, exclusion_mask=None):
    """Get an exclusion mask, preferring module P's canonical one.

    Resolution order:
      1. an explicit ``exclusion_mask`` argument (always wins);
      2. ``pattern.exclusion_mask`` or ``pattern.chart_exclusion_mask``, called
         as ``(chart_shape, landmarks, axis_order="s_major")`` -- module P owns
         the canonical exclusion geometry and the station table it needs;
      3. this module's :func:`eye_mouth_exclusion`, if landmarks were given;
      4. no exclusion at all, with a note saying so.

    The lazy import is deliberate: ``bake``/``unbake`` must be testable with
    ``pattern`` absent, and must not import it at module load.

    THE GILL-SLIT ARBITRATION, settled here.  Module P's mask excludes the
    seven gill slits (``EXCLUSION_MASK_INCLUDE_GILL_SLITS = True``); this
    module's local :func:`eye_mouth_exclusion` does not, and ``pattern.py``
    recorded the disagreement rather than resolving it.  Resolved in favour of
    EXCLUDING them, for three reasons:

      * The objection this module originally raised -- that gill slit 1 is the
        chart's own arc-length origin (Schema S1 ``chart.arc_length_origin``)
        -- does not survive contact: re-anchoring runs on LANDMARKS, not on the
        identity image, so removing gill pixels from the identity chart costs
        the anchor nothing.  Module P's rebuttal is correct.
      * A gill slit is a dark linear aperture that EVERY individual has, whose
        appearance changes with respiration and viewing angle.  In a synthetic
        re-ID corpus that is the textbook shortcut feature: present, salient,
        pose-correlated and identity-free.
      * It costs nothing that matters.  The La Jolla freckle patch that carries
        the head signal lies between the nares and gill slit 1 (Schema S1
        ``chart.head_patch_bounds``), anterior to the excluded band.

    ``eye_mouth_exclusion`` is left gill-free on purpose: it is the degraded
    fallback for when module P is absent, it has no station table, and a
    fallback that guesses at a seven-slit band from two head landmarks would be
    inventing geometry.  When ``pattern`` is importable its mask is used and
    this question does not arise.

    Returns ``(mask, notes)``.
    """
    n_s, n_phi = int(chart_shape[0]), int(chart_shape[1])
    if exclusion_mask is not None:
        m = np.asarray(exclusion_mask, dtype=bool)
        if m.shape != (n_s, n_phi):
            raise ValueError(
                "exclusion_mask shape %r != chart_shape %r" % (m.shape, (n_s, n_phi))
            )
        return m, ["exclusion mask supplied by caller"]

    try:
        import pattern  # noqa: F401  (module P; optional here by design)
    except Exception:
        pattern = None
    if pattern is not None:
        for hook_name in _PATTERN_EXCLUSION_HOOKS:
            hook = getattr(pattern, hook_name, None)
            if hook is None:
                continue
            try:
                m = np.asarray(
                    hook((n_s, n_phi), landmarks, axis_order="s_major"), dtype=bool
                )
            except TypeError:
                try:
                    m = np.asarray(hook((n_s, n_phi), landmarks), dtype=bool)
                except Exception as exc:
                    return (
                        np.zeros((n_s, n_phi), dtype=bool),
                        ["pattern.%s failed (%s); no exclusion applied"
                         % (hook_name, exc)],
                    )
            except Exception as exc:
                return (
                    np.zeros((n_s, n_phi), dtype=bool),
                    ["pattern.%s failed (%s); no exclusion applied"
                     % (hook_name, exc)],
                )
            if m.shape != (n_s, n_phi):
                return (
                    np.zeros((n_s, n_phi), dtype=bool),
                    ["pattern.%s returned shape %r, expected %r (s-major); "
                     "no exclusion applied" % (hook_name, m.shape, (n_s, n_phi))],
                )
            return m, ["exclusion mask from pattern.%s" % hook_name]

    if landmarks:
        m, notes = eye_mouth_exclusion((n_s, n_phi), landmarks)
        return m, ["exclusion mask from unbake.eye_mouth_exclusion "
                   "(module P absent; gill slits NOT excluded)"] + notes
    return (
        np.zeros((n_s, n_phi), dtype=bool),
        ["NO exclusion applied: no pattern module hook and no landmarks given "
         "-- eye and mouth pixels WILL enter the chart"],
    )


# ---------------------------------------------------------------------------
# photo -> chart
# ---------------------------------------------------------------------------

def local_half_width(mask, centerline):
    """Local body half-width in pixels at each centerline station.

    The Euclidean distance transform of the (filled) mask, sampled along the
    centerline: for a locally tubular silhouette that is exactly the distance
    to the nearest silhouette edge, i.e. the projected radius ``R_local``.

    LIMIT: at a fin insertion or a bite notch the nearest edge is not the
    dorso-ventral one and ``R_local`` is underestimated, which pushes
    ``|r| / R_local`` above 1 and voids those samples.  That is the desired
    failure -- silent, local, and visible in the confidence map -- rather than
    a wrong ``phi``.
    """
    m = ndimage.binary_fill_holes(np.asarray(mask).astype(bool))
    edt = ndimage.distance_transform_edt(m)
    cl = np.asarray(centerline, dtype=float)
    return ndimage.map_coordinates(edt, [cl[:, 1], cl[:, 0]], order=1, mode="nearest")


def dorsal_sign_from_point(dorsal_point, centerline, half_width, n_s_strip, n_r):
    """Exact dorsal sign from one image point known to lie dorsally.

    Projects the point onto the same ``(s, r)`` chart the strip uses and reads
    the sign of its ``r``.  Any dorsal landmark works; Schema S1's
    ``dorsal_fin_origin`` (id 14, Type I, tier 1, lateral_visibility high) is
    the natural one on a sevengill because the single dorsal sits far posterior
    over the pelvics and is unambiguous, but the dorsal terminus of a gill slit
    (``gill_slit_1_dorsal_origin``, id 5, the chart's own arc-length origin)
    serves just as well and is annotated more often.

    This replaces a photometric guess with a geometric fact, and it is the
    recommended route.  It fails only if the supplied point is not actually
    dorsal, or lands within a pixel of the centerline.
    """
    image_to_chart = _p02("chart").image_to_chart

    pt = np.asarray(dorsal_point, dtype=float).reshape(1, 2)
    s_idx, r_idx = image_to_chart(centerline, half_width, n_s_strip, n_r, pt)[0]
    centre = 0.5 * (n_r - 1)
    offset = r_idx - centre
    if abs(offset) < 0.5:
        return 1, (
            "dorsal_point is within half a chart cell of the centerline "
            "(r_idx %.2f vs centre %.2f); falling back to dorsal_sign +1"
            % (r_idx, centre)
        )
    sign = 1 if offset > 0 else -1
    return sign, (
        "dorsal_sign %+d from dorsal_point at chart (s_idx %.1f, r_idx %.1f)"
        % (sign, s_idx, r_idx)
    )


# Relative-contrast band used by the countershading heuristic.  The rim
# (|r|/R > 0.85) is dropped because it is grazing and dark for geometric
# reasons in every image; the axis strip (|r|/R < 0.2) is dropped because
# dorsal and ventral meet there and carry no discriminating tone.
_COUNTERSHADE_BAND = (0.20, 0.85)

# Below this relative difference between the two half-means the countershading
# cue is not separable from noise and the caller is told so.
# [BRACKET 0.02-0.08] [EVIDENCE GRADE: UNVERIFIED, chosen so the fixture's
# Lambert-vs-countershading near-cancellation trips it rather than guessing.]
_COUNTERSHADE_MIN_MARGIN = 0.03


def _infer_dorsal_sign(strip, valid, ratio_pos):
    """Which side of the strip is the dorsum, from countershading.

    Sevengill countershading is "dark speckling on grey-brown dorsum, lighter
    ventrally" [species anatomy, prototype brief], so the darker half of the
    rectified strip is the dorsal half.

    Args:
        strip: ``(n_s, n_r)`` rectified luminance.
        valid: finite-sample mask.
        ratio_pos: ``r / R_local`` per strip cell, WITHOUT any dorsal sign
            applied, so this function can pick the sign.

    Returns ``(sign, note)`` where ``sign`` is ``+1`` if chart ``r > 0`` points
    dorsally.

    HEURISTIC, AND STRUCTURALLY WEAK.  Countershading is an ADAPTATION FOR
    CANCELLING exactly the gradient this heuristic reads: a dark dorsum under
    an overhead sun renders at nearly the same luminance as a pale ventrum in
    shadow -- that is what countershading is FOR.  So in the commonest
    underwater lighting the cue self-destructs, and it is not a subtle effect:
    this prototype's own fixture renderer, with a plain overhead key light and
    a plain cosine countershading term, lands the two halves within ~1% of each
    other.  It also fails on backlit or silhouetted animals, a cast shadow on
    the ventrum, a surface caustic on the dorsum, and washed-out juvenile or
    turbid-water shots.

    When the two half-means are within ``_COUNTERSHADE_MIN_MARGIN`` the
    returned sign is a coin flip and the note says so.  PREFER ``dorsal_point``
    (one image-space point known to be dorsal, e.g. Schema S1's
    ``dorsal_fin_origin``, id 14, Type I, tier 1, lateral_visibility high) or an
    explicit ``dorsal_sign``.  Getting this sign wrong does not degrade the
    chart, it MIRRORS it about the dorso-ventral axis, so it is silent and
    total.
    """
    lo_r, hi_r = _COUNTERSHADE_BAND
    band = valid & (np.abs(ratio_pos) >= lo_r) & (np.abs(ratio_pos) <= hi_r)
    pos = strip[band & (ratio_pos > 0)]
    neg = strip[band & (ratio_pos < 0)]
    if pos.size < 32 or neg.size < 32:
        return 1, "dorsal_sign fallback +1 (too few valid samples to infer)"
    m_pos, m_neg = float(np.mean(pos)), float(np.mean(neg))
    denom = max(abs(m_pos) + abs(m_neg), 1e-9)
    margin = abs(m_pos - m_neg) / (0.5 * denom)
    sign = 1 if m_pos < m_neg else -1
    if margin < _COUNTERSHADE_MIN_MARGIN:
        return sign, (
            "dorsal_sign %+d is UNRELIABLE: the two flank halves differ by only "
            "%.1f%% in mean luminance, below the %.0f%% countershading margin. "
            "Pass dorsal_sign explicitly." % (sign, 100 * margin,
                                              100 * _COUNTERSHADE_MIN_MARGIN)
        )
    return sign, (
        "dorsal_sign %+d inferred from countershading (%s half darker by %.1f%%)"
        % (sign, "r>0" if sign > 0 else "r<0", 100 * margin)
    )


def photo_to_chart(
    photo,
    mask,
    side="L",
    chart_shape=(128, 256),
    n_s_strip=384,
    n_r=129,
    n_stations=256,
    dorsal_sign=None,
    dorsal_point=None,
    landmarks=None,
    exclusion_mask=None,
    normalize=True,
    sigma_s=DELIGHT_SIGMA_S,
    sigma_phi=DELIGHT_SIGMA_PHI,
    min_confidence=0.0,
    half_width_scale=1.15,
    seed=0,
):
    """Unbake a near-lateral photograph into a canonical ``(s, phi)`` chart.

    Args:
        photo: ``(H, W)`` or ``(H, W, 3)`` image, float ``[0, 1]`` or uint8.
        mask: ``(H, W)`` body mask (bool-ish).
        side: ``"L"`` or ``"R"`` -- the catalogue's side column, i.e. WHICH
            FLANK the camera sees.  Sets the sign of ``phi``.
        chart_shape: ``(n_s, n_phi)`` of the output chart.
        n_s_strip, n_r: resolution of the intermediate ``(s, r)`` strip.
        n_stations: centerline stations for ``extract_centerline``.
        dorsal_sign: ``+1``/``-1`` for which side of the strip's ``r`` axis is
            dorsal, or ``None`` to derive it.
        dorsal_point: ``(x, y)`` image pixel known to lie on the DORSAL side --
            e.g. Schema S1's ``dorsal_fin_origin`` or
            ``gill_slit_1_dorsal_origin`` annotation.  Used when
            ``dorsal_sign`` is ``None``; this is the reliable route.  With
            neither, the countershading heuristic guesses and says how much it
            trusts itself in ``notes`` -- and it is right to distrust itself,
            see :func:`_infer_dorsal_sign`.
        landmarks: ``{name: (s, phi)}`` for the exclusion regions.
        exclusion_mask: explicit ``(n_s, n_phi)`` bool override.
        normalize: divide the luminance chart by its own low-frequency content,
            turning it into an albedo MULTIPLIER chart (1 = unmarked).  This is
            the photo-side twin of ``bake``'s de-lighting and removes BOTH the
            capture's shading and the species' countershading -- see the note
            below.
        min_confidence: cells below this confidence are set to ``NaN``.
        half_width_scale: the strip half-width is this times the largest local
            half-width, so the strip always contains the whole silhouette.
        seed: accepted for API stability; every step here is deterministic.

    Returns:
        :class:`UnbakeResult`.

    WHAT ``normalize`` DOES AND DOES NOT SEPARATE.  Dividing by a large-scale
    blur removes any smooth multiplicative field: lighting, water column
    attenuation, vignetting -- and countershading, which is genuine species
    albedo and is smooth.  That is the same limitation ``bake``'s de-lighting
    has and the same answer applies: countershading belongs to the species
    layer, not the identity layer, and is re-applied after the fact.  What
    survives the divide is what varies faster than ``sigma_s`` / ``sigma_phi``
    -- the speckling, i.e. the identity.

    KNOWN BIASES.
      * ``s`` is arc length along the MEDIAL AXIS OF THE SILHOUETTE, which
        retracts from the true snout and caudal tips; both ends of ``s`` are
        therefore compressed relative to a snout-to-caudal parameterisation.
        Schema S1's own rectifier notes the same head-ward bias.  Re-anchor on
        the gill-slit contours (``chart.arc_length_origin`` in the schema) when
        landmarks are available.
      * A bent animal is fine (the centerline follows the bend) but a
        FORESHORTENED one is not: an animal angled toward the camera has a
        shortened silhouette and every ``s`` is wrong.  Nothing here detects
        that; the 3D fit path does.
      * Only the camera-facing half of the girth is observed.  The returned
        chart covers roughly ``phi in (0, pi)`` (side ``L``) with confidence
        tapering to 0 at both ends; the far half is ``NaN``.  Two opposite-side
        photographs of the same individual are needed for a full chart, and
        Schema S1 warns that cross-flank matching is near-chance
        ("cross-flank Rank-1 fell to 0.70% zero-shot"), so do NOT mirror one
        flank onto the other to fill it.
    """
    extract_centerline = _p02("centerline").extract_centerline
    rectify = _p02("chart").rectify

    if side not in ("L", "R"):
        raise ValueError("side must be 'L' or 'R', got %r" % (side,))
    side_sign = 1 if side == "L" else -1
    notes = []

    img = np.asarray(photo)
    if img.dtype == np.uint8:
        img = img.astype(float) / 255.0
    img = np.asarray(img, dtype=float)
    gray = luminance(img) if img.ndim == 3 else img
    body = np.asarray(mask).astype(bool)

    cl = extract_centerline(body, n_stations=n_stations, seed=seed)
    radius_stations = local_half_width(body, cl)
    half_width = float(half_width_scale * np.max(radius_stations))
    if not np.isfinite(half_width) or half_width <= 0:
        raise ValueError("degenerate body mask: local half-width is zero everywhere")

    strip = rectify(gray, cl, half_width, n_s_strip, n_r, mask=body, fill=np.nan)
    valid = np.isfinite(strip)
    if valid.sum() < 64:
        raise ValueError("rectified strip has almost no valid samples (%d)" % valid.sum())

    # Local radius per strip row: resample the per-station radius onto the strip's
    # own uniform-in-arc-length station grid.
    r_of_row = np.interp(
        np.linspace(0.0, 1.0, n_s_strip),
        np.linspace(0.0, 1.0, len(radius_stations)),
        radius_stations,
    )

    r_px = np.linspace(-half_width, half_width, n_r)
    S_strip = np.repeat(np.linspace(0.0, 1.0, n_s_strip)[:, None], n_r, axis=1)
    R_local = np.repeat(np.maximum(r_of_row, 1e-6)[:, None], n_r, axis=1)
    ratio_pos = r_px[None, :] / R_local

    if dorsal_sign is None and dorsal_point is not None:
        dorsal_sign, note = dorsal_sign_from_point(
            dorsal_point, cl, half_width, n_s_strip, n_r
        )
        notes.append(note)
    if dorsal_sign is None:
        dorsal_sign, note = _infer_dorsal_sign(strip, valid, ratio_pos)
        notes.append(note)
    dorsal_sign = int(np.sign(dorsal_sign)) or 1
    ratio = dorsal_sign * ratio_pos

    on_body = valid & (np.abs(ratio) <= 1.0)
    phi = side_sign * np.arccos(np.clip(ratio, -1.0, 1.0))
    conf_geom = np.sqrt(np.maximum(1.0 - np.clip(ratio, -1.0, 1.0) ** 2, 0.0))

    n_s, n_phi = int(chart_shape[0]), int(chart_shape[1])
    excl, excl_notes = resolve_exclusion_mask((n_s, n_phi), landmarks, exclusion_mask)
    notes.extend(excl_notes)

    w = np.where(on_body, conf_geom, 0.0)
    sel = w > 1e-6
    acc, wsum = splat_to_chart(
        S_strip[sel], phi[sel], strip[sel], n_s, n_phi, weights=w[sel]
    )
    have = wsum > 0
    chart = np.full((n_s, n_phi), np.nan)
    chart[have] = acc[have] / wsum[have]

    # Confidence: mean geometric confidence per cell, i.e. the weighted mean of
    # sin|phi| over the samples that landed there.  A cell reached only by
    # grazing samples keeps a low score even though it has many of them.
    acc_c, _ = splat_to_chart(
        S_strip[sel], phi[sel], w[sel], n_s, n_phi, weights=w[sel]
    )
    confidence = np.zeros((n_s, n_phi))
    confidence[have] = np.clip(acc_c[have] / wsum[have], 0.0, 1.0)
    confidence[excl] = 0.0
    chart[excl] = np.nan

    if normalize:
        ok = have & np.isfinite(chart) & (confidence > 1e-6)
        if ok.sum() < 16:
            notes.append("normalize skipped: too few confident cells")
        else:
            sig = (max(sigma_s * n_s, 0.5), max(sigma_phi / (2 * np.pi) * n_phi, 0.5))
            low, _ = masked_gaussian(np.nan_to_num(chart), ok, sig, wrap_axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                chart = np.where(ok, chart / np.maximum(low, 1e-6), np.nan)

    if min_confidence > 0:
        chart = np.where(confidence >= float(min_confidence), chart, np.nan)

    frac = float(np.mean(confidence > 0.35))
    if frac < 0.15:
        notes.append(
            "only %.1f%% of chart cells exceed confidence 0.35 -- this view "
            "recovered very little surface; check the mask and the side label"
            % (100 * frac)
        )
        warnings.warn(notes[-1], RuntimeWarning, stacklevel=2)

    return UnbakeResult(
        chart=chart,
        confidence=confidence,
        strip=strip,
        centerline=cl,
        half_width=half_width,
        radius_px=r_of_row,
        dorsal_sign=dorsal_sign,
        side_sign=side_sign,
        notes=notes,
    )


def copy_from_photo(
    photo,
    mask,
    side="L",
    identity="copied",
    date=None,
    pattern_kwargs=None,
    **kwargs
):
    """Full copy-a-real-individual path: photograph -> ``pattern.Individual``.

    Runs :func:`photo_to_chart`, then hands the chart and its confidence to
    module P's ``pattern.copy_from_chart``.  ``pattern`` is imported LAZILY and
    only here, so every other entry point in this module works without it.

    THE LAYOUT / SEMANTICS HANDSHAKE IS EXPLICIT.  ``pattern`` lays charts out
    ``(H_phi, W_s)`` and stores DARKNESS; this module produces ``(n_s, n_phi)``
    ALBEDO MULTIPLIERS.  ``pattern.copy_from_chart`` can auto-detect both, but
    auto-detection of the axis order is undecidable on a square chart and the
    semantics rule is a mean test -- so this call passes
    ``axis_order="s_major"`` and ``chart_semantics="albedo"`` outright.  Do not
    remove them "because auto works": auto works until someone asks for a
    256x256 chart.

    Args:
        identity: individual id, forwarded to ``pattern.copy_from_chart``.
        date: sighting date for the fitted spots; forwarded when given.
        pattern_kwargs: extra keyword arguments for ``pattern.copy_from_chart``
            (``threshold``, ``min_confidence``, ``length_cm``, ...).
        **kwargs: forwarded to :func:`photo_to_chart`.

    Returns:
        ``(individual, unbake_result)`` -- whatever ``pattern.copy_from_chart``
        returns, paired with the :class:`UnbakeResult` it was built from, so
        the caller keeps the confidence map and the notes.

    Raises:
        RuntimeError: if ``pattern`` (module P) is not importable or lacks
        ``copy_from_chart``, naming the fallback -- call
        :func:`photo_to_chart` and keep the chart.
    """
    result = photo_to_chart(photo, mask, side=side, **kwargs)
    try:
        import pattern
    except Exception as exc:
        raise RuntimeError(
            "pattern (module P) is not importable (%s). photo_to_chart() "
            "already produced the chart and confidence; call it directly and "
            "pass the chart to pattern.copy_from_chart once module P lands. "
            "The contract is copy_from_chart(chart, confidence=..., "
            "identity=..., axis_order='s_major', chart_semantics='albedo')."
            % exc
        )
    if not hasattr(pattern, "copy_from_chart"):
        raise RuntimeError(
            "pattern is importable but has no copy_from_chart(); the unbake -> "
            "pattern contract is copy_from_chart(chart, confidence=..., "
            "identity=..., axis_order='s_major', chart_semantics='albedo') "
            "with chart an albedo-multiplier (n_s, n_phi) array"
        )
    call = dict(
        confidence=np.nan_to_num(result.confidence, nan=0.0),
        identity=identity,
        axis_order="s_major",
        chart_semantics="albedo",
    )
    if date is not None:
        call["date"] = date
    call.update(dict(pattern_kwargs or {}))
    chart = np.nan_to_num(result.chart, nan=1.0)     # unobserved reads as skin
    return pattern.copy_from_chart(chart, **call), result
