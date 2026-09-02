"""Synthetic sevengill re-ID dataset generator -- the whole engine, end to end.

WHAT THIS SCRIPT IS
-------------------
It wires the engine modules of prototype 05 into one command that writes a
corpus ``prototypes/01-melops-ablation/melops_data.load_melops`` can read
UNCHANGED, so ``run_ablation.py`` and ``diagnose.py`` run on synthetic
sevengills with zero edits:

    pattern.py   RANDOMIZE  an individual's speckle field in CHART SPACE
    unbake.py    COPY       a real pattern out of a photograph into that chart
    drift.py     RESIGHT    the same animal months/years later (growth + drift)
    bake.py      BAKE       the chart onto the model's UV texture
    render.py    RENDER     a numpy z-buffer frame + pixel-aligned chart GT
    nuisance.py  DEGRADE    kelp/occluders, turbidity, caustics, blur, jitter
    exclusions.py           the anatomical exclusion mask (eye, mouth, ...)

``unbake`` is not called from this file -- the corpus generator randomizes.
It is the COPY entry point (``unbake.copy_from_photo``) that produces an
``Individual`` this generator can then drift and render exactly like a
randomized one; see README.md.

Per individual: one pattern.  Per sighting: a date, growth-driven drift, a
pose, a side (L or R -- rendered by MOVING THE CAMERA, never by mirroring),
camera + light + nuisance draws, a render, a crop by the body mask, and three
LTWH boxes cut in ARC LENGTH through the chart ground truth.

Usage
-----
    python make_dataset.py --out DIR --n-individuals 40 \
        --sightings-per-individual 6 --years 4 \
        [--head-signal 1.0 --flank-signal 1.0] \
        [--occlusion 0.3 --shadow 0.5 --turbidity 0.4] [--seed 0] \
        [--length-noise 0.07]

Outputs in ``DIR``
------------------
    metadata.csv            the melops_data contract (see that module's
                            ``_REQUIRED_PLAIN_COLUMNS`` / ``_check_catalogue``)
    Melops_metadata.txt     filename_year,length -- what
                            ``readout_length_controlled.py`` reads.  The length
                            is the RECORDED one (true length + measurement
                            error); the true one is in truth.jsonl.
    body/<image_id>.png     the crop referenced by ``path``
    masks/<image_id>_identity.png   the render-time identity mask, same crop
    gt/<image_id>.npz       chart_s, chart_phi and every mask, same crop
    truth.jsonl             one JSON record per image: pose, camera, light,
                            nuisance draws, growth ratio, elapsed days, spot
                            count, TRUE and RECORDED length, GT paths
    dataset.json            the run's arguments, constants and summary counts

WHY CHART SPACE (the binding design decision)
---------------------------------------------
Patterns are generated in canonical chart coordinates ``(s, phi)`` -- arc-length
fraction snout(0)->caudal(1) and circumferential angle (0 dorsal, +pi/2 the
animal's LEFT, +-pi ventral) -- NOT in the mesh's UV atlas.  A chart-space
pattern is mesh-agnostic, maps to any pose through the rig, and gives every
rendered pixel exact ground-truth chart coordinates (``out["chart_s"]`` /
``out["chart_phi"]``).  Baking to a UV texture is a separate, later step that
needs per-vertex ``(s, phi)``; here it comes from ``fixtures.make_uv_tube``,
and for a real mesh it comes from prototype 04 (see SWAP-IN POINTS below).

SWAP-IN POINTS FOR PROTOTYPE 04 (never imported here; only assumed)
-------------------------------------------------------------------
1. GEOMETRY + CHART COORDS.  Replace :func:`build_model` with::

       tc = mesh3d.tube_coords(mesh, centerline)
       vertex_s = tc.s / tc.total_length
       vertex_phi = tc.phi              # convention already matches

   Everything downstream (bake, render, the chart GT, the boxes) is written
   against ``(mesh, vertex_s, vertex_phi)`` and needs no other change.
2. POSE.  Replace :func:`pose_vertices` with the rig's own posing.  The tube
   bend here is a stand-in: a planar centerline of curvature
   ``kappa(s) = amp * cos(2*pi*wave*s + phase)`` re-swept with the SAME
   ``(s, phi)`` per vertex, so it is exactly arc-length preserving and the
   chart GT is unchanged by the pose.  It has no vertebral limits, no fin
   articulation and no volume preservation.  ``PoseParams`` is the record the
   rig should fill in instead.

EVIDENCE AND GRADES for the constants introduced HERE are in
:data:`EVIDENCE`.  Constants owned by the other modules keep their own grades
there (drift's jitter rate, nuisance's visibility bracket, exclusions'
station table, ...); this file never re-states them as if they were measured.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import bake            # noqa: E402
import drift           # noqa: E402
import exclusions      # noqa: E402
import fixtures        # noqa: E402
import nuisance        # noqa: E402
import pattern         # noqa: E402
import render          # noqa: E402

try:                                        # Pillow is the only image writer
    from PIL import Image
except ImportError:                         # pragma: no cover - env has it
    Image = None


# ---------------------------------------------------------------------------
# Constants introduced by THIS module, each with a bracket and a grade.
# ---------------------------------------------------------------------------

#: Total length of the model in world units, with 1 world unit = 1 metre, so
#: nuisance.py's metre-based attenuation is meaningful without a conversion.
BODY_LENGTH_M = 2.0
#: Camera-to-subject range, inside the typical La Jolla visibility bracket.
SUBJECT_RANGE_M = 2.5
#: Base skin albedo: a mid grey-brown dorsum.
SKIN_BASE_RGB = (0.46, 0.43, 0.37)
#: Ventral albedo gain at full countershading (1 + this, at |phi| >= phi_full).
VENTRAL_LIGHTEN = 0.55
#: Darkness -> albedo-multiplier amplitude when a chart is baked.
PATTERN_AMPLITUDE = 0.85
#: Total length of a sampled animal, centimetres, at its FIRST sighting.
LENGTH_CM_BRACKET = (140.0, 285.0)
#: Relative sd of the RECORDED length of a sighting: field length estimates are
#: not the animal.  Placeholder bracket, see EVIDENCE.
LENGTH_MEASUREMENT_RSD = 0.07
#: Peak lateral curvature of the stand-in pose, radians of heading per unit s.
POSE_AMP_BRACKET = (0.0, 0.9)
#: Pixels of padding around the body silhouette when cropping.
CROP_PAD_PX = 6
#: Lateral component of the key light, as a fraction of its vertical one, when
#: the light is on the CAMERA's side of the animal (a lit near flank).
LIGHT_FRONTAL_LATERAL_BRACKET = (0.35, 0.95)
#: Probability the key light is instead on the FAR side (a rim-lit animal).
BACKLIT_PROB = 0.22
#: Height of the shadow-casting canopy above the animal, in body lengths.
CANOPY_HEIGHT_FRAC = 0.55
#: Blades in that canopy.
CANOPY_N_BLADES = 5

EVIDENCE = {
    "BODY_LENGTH_M": (
        "2.0 m [SECONDARY: species description -- Notorynchus cepedianus "
        "commonly 1.5-3 m TL]. Bracket [1.5, 3.0]. Only sets the world scale "
        "that nuisance.py's per-metre attenuation is applied in."),
    "SUBJECT_RANGE_M": (
        "2.5 m [DERIVED from nuisance.VISIBILITY_M_BRACKET (3-6 m, SECONDARY: "
        "dive-operator copy in docs/sevengill-canonical-reid/"
        "01-evidence-and-answers.md)]: a diver-held frame-filling shot must be "
        "inside the visibility, or the animal is fog. Bracket [1.5, 4.0]."),
    "SKIN_BASE_RGB": (
        "(0.46, 0.43, 0.37) [DERIVED from the species description 'dark "
        "speckling on a grey-brown dorsum, lighter ventrally']. No "
        "spectrophotometry of sevengill skin was retrieved; this is a "
        "plausible mid grey-brown, NOT a measurement."),
    "VENTRAL_LIGHTEN": (
        "0.55 [DERIVED]: the ventral albedo is 1.55x the dorsal one, applied "
        "through exclusions.countershading_weight_at so the pattern prior and "
        "the tone prior are the SAME curve. Bracket [0.3, 1.0]; countershading "
        "itself is [SECONDARY: species description]."),
    "PATTERN_AMPLITUDE": (
        "0.85 [DERIVED]: a fully dark chart texel (darkness 1) becomes an "
        "albedo multiplier of 0.15, i.e. a near-black speckle on grey-brown "
        "skin. Bracket [0.6, 1.0]. No measured sevengill speckle contrast "
        "exists; this is the knob to fit when photographs arrive."),
    "LENGTH_CM_BRACKET": (
        "[140, 285] cm [UNVERIFIED]: consistent with drift.VonBertalanffyGrowth's "
        "l_inf_cm=290 default (itself [UNVERIFIED]) and with the species "
        "description's adult range. Lengths only drive growth-scaled spot "
        "SPACING (spots are a fixed cell population that spreads as the animal "
        "grows -- derived, see drift.py) and the size-assortativity readout. "
        "Widened from the earlier [150, 275] so the sampled population overlaps "
        "the species range rather than a narrow slice of it; see "
        "LENGTH_MEASUREMENT_RSD for why a wide bracket is NOT an identity code."),
    "LENGTH_MEASUREMENT_RSD": (
        "0.07 relative sd [UNVERIFIED -- placeholder bracket [0.03, 0.15] for "
        "photogrammetric / laser-scaled total-length error on a free-swimming "
        "shark]. No sevengill length-error study was retrieved; the bracket is "
        "a stand-in for one and MUST be replaced before any length-stratified "
        "claim is made. It exists because without it the recorded length is a "
        "near-unique IDENTITY CODE: each animal draws one initial length and "
        "then only grows, so 1-NN identity from length alone ran far above "
        "chance and readout_length_controlled.py's size-assortativity index "
        "was measuring a label, not a body. The TRUE length stays in "
        "truth.jsonl (``length_cm`` / ``length_mm``); only the sidecar "
        "Melops_metadata.txt carries the noisy RECORDED value, which is what a "
        "field catalogue actually has."),
    "POSE_AMP_BRACKET": (
        "[0, 0.9] rad [none -- placeholder]: no sevengill vertebral-flexion "
        "measurement was retrieved. This bracket exists to make the corpus "
        "non-rigid, not to model a real swimming envelope. Prototype 04's rig "
        "replaces it (see the module docstring)."),
    "CROP_PAD_PX": (
        "6 px [DERIVED]: a hand-drawn or detector box is never tight; a small "
        "pad keeps the silhouette off the crop edge. Bracket [0, 16]."),
    "LIGHT_FRONTAL_LATERAL_BRACKET": (
        "[0.35, 0.95] x the vertical component [none -- SCENE-SETUP "
        "CONVENTION, not a physical claim]. A purely overhead sun on a "
        "laterally-viewed cylinder puts the terminator ON the camera-facing "
        "flank: N.L ~ 0 over the whole near side, so roughly half of every "
        "body would be attached shadow and would leave the identity mask. A "
        "diver does not shoot that way -- they put the sun, or a strobe, "
        "behind their own shoulder. So the key light's lateral component is "
        "drawn on the CAMERA's side of the animal. Measured on the 10x6 demo "
        "corpus: identity pixels / body pixels 0.393 -> 0.472 overall, which "
        "splits as 0.578 on the front-lit frames and 0.180 on the backlit "
        "ones -- the remaining loss is ATTACHED shadow on the ventral quarter "
        "and is real, not an artefact."),
    "BACKLIT_PROB": (
        "0.22 [none -- scene-setup convention]. Backlit frames are kept on "
        "purpose: a rim-lit animal is a real and hard encounter, and a corpus "
        "in which the near flank is ALWAYS lit would train a model that has "
        "never seen one. Bracket [0.1, 0.4]."),
    "CANOPY_HEIGHT_FRAC": (
        "0.55 body lengths [DERIVED, geometric]. The canopy must be far "
        "enough up-light to sit outside the camera's fitted ortho frame "
        "(half-height ~0.20 body lengths at the default framing) so it casts "
        "WITHOUT occluding -- which is what keeps the cast-shadow mask and "
        "the occlusion mask independent. Bracket [0.35, 1.0]."),
    "CANOPY_N_BLADES": (
        "5 [none -- visual placeholder]. Giant kelp canopy density was not "
        "retrieved; this is enough blades to dapple rather than to stripe."),
}


# ---------------------------------------------------------------------------
# 1. The model: geometry + per-vertex chart coordinates
# ---------------------------------------------------------------------------

class Model(object):
    """The 3D model plus everything the bake and the render need.

    Attributes:
        tube: the ``fixtures.UVTube`` (rest pose).
        mesh: ``trimesh.Trimesh`` with ``visual.uv`` -- what ``bake`` rasterises.
        vertex_s, vertex_phi: ``(V,)`` canonical chart coordinates per vertex.
            THE PROTOTYPE-04 SWAP-IN POINT: these two arrays are the entire
            contract (``tc = mesh3d.tube_coords(mesh, centerline)``;
            ``vertex_s = tc.s / tc.total_length``; ``vertex_phi = tc.phi``).
        uv_raster: cached UV rasterisation, reused for every bake (it depends
            only on the atlas and the texture size, never on the pattern).
        station_s: arc-length fraction per station station, from the schema.
    """

    def __init__(self, tube, tex_size, stations):
        self.tube = tube
        self.mesh = tube.mesh
        self.vertex_s = tube.vertex_s
        self.vertex_phi = tube.vertex_phi
        self.uv = np.asarray(tube.mesh.visual.uv)
        self.faces = np.asarray(tube.mesh.faces)
        self.tex_size = (int(tex_size), int(tex_size))
        self.uv_raster = bake.rasterize_uv(self.mesh, self.tex_size)
        self.stations = dict(stations)
        n_st = tube.grid_shape[0]
        self._station_s = np.linspace(0.0, 1.0, n_st)
        self._radius = np.asarray(tube.radius, dtype=np.float64)
        self.vertex_radius = np.interp(self.vertex_s, self._station_s, self._radius)


def build_model(n_stations=56, n_around=36, tex_size=128, length=BODY_LENGTH_M,
                r_max=0.16, schema_path=None, stations=None):
    """Build the stand-in model.

    PROTOTYPE-04 SWAP-IN POINT.  Everything downstream consumes
    ``(mesh, vertex_s, vertex_phi)``; replace this function with the rig's
    mesh plus ``mesh3d.tube_coords`` and nothing else changes.
    """
    tube = fixtures.make_uv_tube(n_stations=n_stations, n_around=n_around,
                                 length=length, r_max=r_max)
    if stations is None:
        schema = exclusions.load_schema(schema_path or pattern.DEFAULT_SCHEMA_PATH)
        stations = exclusions.default_stations(schema)
    return Model(tube, tex_size, stations)


# ---------------------------------------------------------------------------
# 2. Pose (the stand-in for prototype 04's rig)
# ---------------------------------------------------------------------------

class PoseParams(object):
    """A planar lateral bend of the centreline -- a C or an S.

    ``kappa(s) = amp * cos(2*pi*wave*s + phase)`` in radians of heading per
    unit arc-length fraction.  ``wave`` near 0.5 gives a C, near 1.0 an S.
    The sweep is arc-length preserving by construction, so per-vertex ``s``
    and ``phi`` -- and therefore every chart ground-truth map -- are unchanged
    by the pose.  That invariance is the point: it is what lets the same
    identity be measured across poses.

    NOT ANATOMY.  See ``EVIDENCE["POSE_AMP_BRACKET"]``.
    """

    def __init__(self, amp=0.0, wave=0.5, phase=0.0, yaw_deg=0.0):
        self.amp = float(amp)
        self.wave = float(wave)
        self.phase = float(phase)
        self.yaw_deg = float(yaw_deg)

    @classmethod
    def sample(cls, rng, amp_bracket=POSE_AMP_BRACKET):
        return cls(amp=float(rng.uniform(*amp_bracket)),
                   wave=float(rng.choice([0.5, 0.75, 1.0])),
                   phase=float(rng.uniform(0.0, 2.0 * math.pi)),
                   yaw_deg=float(rng.uniform(-18.0, 18.0)))

    def as_dict(self):
        return {"amp": self.amp, "wave": self.wave, "phase": self.phase,
                "yaw_deg": self.yaw_deg, "kind": "tube_bend_stand_in"}

    def __repr__(self):
        return "PoseParams(amp=%.3f, wave=%.2f, phase=%.2f)" % (
            self.amp, self.wave, self.phase)


def pose_vertices(model, pose, n_samples=512):
    """Re-sweep the tube along a bent centreline; returns ``(V, 3)``.

    The frame is rotation-minimising for a planar curve: dorsal stays ``+Z``
    and ``left = cross(dorsal, tangent)``, exactly as ``fixtures.make_uv_tube``
    builds it, so ``pose_vertices(model, PoseParams())`` reproduces the rest
    pose to machine precision.
    """
    u = np.linspace(0.0, 1.0, int(n_samples))
    kappa = pose.amp * np.cos(2.0 * math.pi * pose.wave * u + pose.phase)
    theta = np.concatenate([[0.0], np.cumsum(0.5 * (kappa[1:] + kappa[:-1]) * np.diff(u))])
    theta = theta - theta.mean()
    theta = theta + math.radians(pose.yaw_deg)
    tang = np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
    step = np.diff(u)[:, None] * model_length(model)
    seg = 0.5 * (tang[1:] + tang[:-1]) * step
    centre = np.concatenate([np.zeros((1, 3)), np.cumsum(seg, axis=0)], axis=0)
    centre -= centre.mean(axis=0)

    s = model.vertex_s
    cx = np.interp(s, u, centre[:, 0])
    cy = np.interp(s, u, centre[:, 1])
    cz = np.interp(s, u, centre[:, 2])
    th = np.interp(s, u, theta)
    dorsal = np.zeros((len(s), 3))
    dorsal[:, 2] = 1.0
    left = np.column_stack([-np.sin(th), np.cos(th), np.zeros_like(th)])
    r = model.vertex_radius[:, None]
    off = (np.cos(model.vertex_phi)[:, None] * dorsal
           + np.sin(model.vertex_phi)[:, None] * left)
    return np.column_stack([cx, cy, cz]) + r * off


def model_length(model):
    """Total centreline length of the model in world units."""
    seg = np.linalg.norm(np.diff(model.tube.centerline, axis=0), axis=1)
    return float(seg.sum())


# ---------------------------------------------------------------------------
# 3. Sighting histories
# ---------------------------------------------------------------------------

def plan_sightings(rng, n_target, years, start="2019-03-01", singleton_prob=0.15,
                   primary_side_prob=0.72):
    """Dates and sides for one animal.

    Gaps are drawn LOG-UNIFORMLY over the study window so the recapture-gap
    buckets ``diagnose.py`` reports (0-30, 31-180, 181-365, 366-730, 731+ days)
    are all populated by construction rather than by luck.  Singletons are
    drawn deliberately: ``protocol.one_shot_open_set_split`` enrols
    pre-cutoff singletons with zero known queries and treats post-cutoff ones
    as novel queries, and both arms must survive (Melops is ~2.5 images per
    individual).

    Sides are drawn per sighting with a primary-side bias: one encounter
    usually yields one flank, but an animal is not always photographed from
    the same side -- and the cross-orientation arm needs both.
    """
    window = int(round(float(years) * 365.25))
    t0 = np.datetime64(str(start))
    if rng.random() < singleton_prob:
        n = 1
    else:
        lo = max(2, int(n_target) - 2)
        n = int(rng.integers(lo, int(n_target) + 3))
    first = int(rng.integers(0, max(1, int(0.35 * window))))
    if n == 1:
        offsets = [0]
    else:
        span = max(window - first, 30)
        raw = np.exp(rng.uniform(math.log(7.0), math.log(float(span)), size=n - 1))
        offsets = [0] + sorted(int(round(v)) for v in raw)
    dates = [t0 + np.timedelta64(first + o, "D") for o in offsets]
    # de-duplicate: two photographs of one animal on one day are the
    # same-session near-duplicates protocol.py deliberately excludes.
    uniq = []
    for d in dates:
        if not uniq or d > uniq[-1]:
            uniq.append(d)
        else:
            uniq.append(uniq[-1] + np.timedelta64(1, "D"))
    primary = "L" if rng.random() < 0.5 else "R"
    other = "R" if primary == "L" else "L"
    sides = [primary if rng.random() < primary_side_prob else other for _ in uniq]
    return [(str(d), sd) for d, sd in zip(uniq, sides)]


# ---------------------------------------------------------------------------
# 4. Chart -> texture
# ---------------------------------------------------------------------------

def chart_to_texture(model, individual, date=None, chart_resolution=(96, 192),
                     amplitude=PATTERN_AMPLITUDE, ventral_lighten=VENTRAL_LIGHTEN,
                     base_rgb=SKIN_BASE_RGB):
    """Render the individual's chart and bake it onto the model's UV atlas.

    Returns ``(texture_rgb, chart_darkness, spot_table)``.

    Two separable layers go into the albedo, and keeping them separable is the
    whole point of the chart:

    * IDENTITY -- ``pattern.render_chart``'s darkness field, converted to an
      albedo multiplier by ``bake.from_pattern_chart`` (1 = unmarked skin).
    * SPECIES TONE -- countershading, from
      ``exclusions.countershading_weight_at``, the SAME curve that already
      attenuates ventral speckle amplitude in ``pattern.render_chart``. The
      ventral albedo gain is ``1 + ventral_lighten * (1 - w)``.

    Capture lighting is the third layer and is NOT baked here at all: this
    albedo has no light in it, which is what ``bake``'s de-lighting exists to
    achieve for a photogrammetry texture (``delight=True``). With a synthetic
    albedo there is nothing to remove, so ``delight`` is off -- switching it on
    here would flatten the countershading, which is albedo, not light.
    """
    chart, spots = pattern.render_chart(individual, resolution=chart_resolution,
                                        date=date)
    mult = bake.from_pattern_chart(chart, semantics="darkness", amplitude=amplitude)
    n_s, n_phi = mult.shape
    _, phi_axis = bake.chart_axes(n_s, n_phi)
    w = exclusions.countershading_weight_at(phi_axis)
    tone = 1.0 + float(ventral_lighten) * (1.0 - w)
    chart_rgb = mult[:, :, None] * tone[None, :, None]
    tex = bake.bake_chart_to_texture(
        model.mesh, model.vertex_s, model.vertex_phi, chart_rgb,
        model.tex_size, base_albedo=base_rgb, delight=False,
        chart_semantics="multiplier", raster=model.uv_raster,
    )
    return np.asarray(tex[..., :3], dtype=np.float64), chart, spots


# ---------------------------------------------------------------------------
# 5. Scene draws
# ---------------------------------------------------------------------------

class SceneDraw(object):
    """Every seeded nuisance choice for one frame, kept as recordable data."""

    def __init__(self, pose, side, light_dir, ambient, has_kelp, kelp_casts,
                 has_shark_occluder, water, blur_px, jitter, caustic_time,
                 has_canopy=False, backlit=False):
        self.pose = pose
        self.side = side
        self.light_dir = tuple(float(v) for v in light_dir)
        self.ambient = float(ambient)
        self.has_kelp = bool(has_kelp)
        self.kelp_casts = bool(kelp_casts)
        self.has_shark_occluder = bool(has_shark_occluder)
        self.water = water
        self.blur_px = float(blur_px)
        self.jitter = jitter
        self.caustic_time = float(caustic_time)
        self.has_canopy = bool(has_canopy)
        self.backlit = bool(backlit)

    def as_dict(self):
        return {
            "pose": self.pose.as_dict(),
            "side": self.side,
            "light_direction": list(self.light_dir),
            "ambient": self.ambient,
            "backlit": self.backlit,
            "kelp": self.has_kelp,
            "kelp_casts_shadow": self.kelp_casts,
            "canopy_caster": self.has_canopy,
            "shark_occluder": self.has_shark_occluder,
            "visibility_m": float(self.water.turbidity.visibility_m),
            "attenuation_per_m": float(self.water.turbidity.broadband_c),
            "caustic_contrast": float(self.water.caustics.contrast),
            "caustic_time": self.caustic_time,
            "motion_blur_px": self.blur_px,
            "camera_jitter": {"translate": self.jitter.translate,
                              "aim": self.jitter.aim,
                              "roll_deg": self.jitter.roll_deg},
        }


def draw_scene(rng, side, occlusion=0.3, shadow=0.5, turbidity=0.4,
               pose_amp_bracket=POSE_AMP_BRACKET):
    """Sample one frame's nuisance.

    ``turbidity`` in [0, 1] interpolates between the best-day and the typical
    visibility brackets of ``nuisance.py`` (9-12 m -> 3-6 m, both [SECONDARY]),
    and scales caustic contrast and the motion-blur probability with it.
    ``occlusion`` is the probability the frame has an occluder at all;
    ``shadow`` the probability of an oblique key light plus a shadow-casting
    caster, which is what puts a hard CAST shadow on the skin.

    THE KEY LIGHT IS PLACED RELATIVE TO THE CAMERA, and this is the one scene
    choice here that is a convention rather than a physical model.  A purely
    overhead sun on a laterally-viewed cylinder puts the terminator on the
    camera-facing flank -- ``N.L ~ 0`` right where the pattern is -- so half of
    every animal would be attached shadow and would drop out of the identity
    mask for a reason that has nothing to do with the animal.  A diver instead
    keeps the sun or the strobe behind their own shoulder, so the lateral
    component of the light is drawn on the CAMERA's side (``light_dir[1]``
    takes the sign of the view direction's Y) for ``1 - BACKLIT_PROB`` of
    frames.  The remaining frames are deliberately backlit: a rim-lit animal
    is a real encounter and the corpus should contain it.

    ``shadow`` also gates a CANOPY CASTER -- blades placed up-light and above
    the animal, outside the camera frame (see :func:`canopy_casters`).  Kelp
    drifting in FRONT of the camera is usually not between the animal and the
    sun, so the in-frame kelp is what occludes and the canopy is what shadows;
    keeping them separate is what keeps the occlusion mask and the cast-shadow
    mask independent instead of entangled.
    """
    t = float(np.clip(turbidity, 0.0, 1.0))
    lo = (nuisance.VISIBILITY_M_BEST_BRACKET[0] * (1 - t)
          + nuisance.VISIBILITY_M_BRACKET[0] * t)
    hi = (nuisance.VISIBILITY_M_BEST_BRACKET[1] * (1 - t)
          + nuisance.VISIBILITY_M_BRACKET[1] * t)
    water = nuisance.WaterParams(
        turbidity=nuisance.TurbidityParams(visibility_m=float(rng.uniform(lo, hi))),
        caustics=nuisance.CausticParams(contrast=float(0.04 + 0.14 * t * rng.uniform(0.5, 1.5))),
        seed=int(rng.integers(0, 2 ** 31 - 1)),
    )
    blur_px = 0.0
    if rng.random() < 0.25 * t:
        blur_px = float(rng.uniform(2.0, 7.0))
        water = water.replace(blur=nuisance.MotionBlurParams(
            length_px=blur_px, angle_deg=float(rng.uniform(-30.0, 30.0))))

    oblique = rng.random() < float(shadow)
    tilt = 0.55 if oblique else 0.18
    # The camera's view direction for this side; its Y sign is the "toward the
    # camera" lateral direction for the light (see the docstring).
    look_y = float(_SIDE_VIEW_DIRECTION[side][1])
    backlit = bool(rng.random() < BACKLIT_PROB)
    lateral = float(rng.uniform(*LIGHT_FRONTAL_LATERAL_BRACKET))
    light_dir = (float(rng.uniform(-tilt, tilt)),
                 lateral * (-look_y if backlit else look_y),
                 -1.0)
    ambient = float(rng.uniform(0.18, 0.30)) if oblique else float(rng.uniform(0.26, 0.36))

    occl = rng.random() < float(occlusion)
    has_kelp = bool(occl and rng.random() < 0.75)
    has_shark = bool(occl and not has_kelp)
    # In-frame kelp casts a shadow only in a hard-light scene; that keeps the
    # dappled-through-the-occluder case a deliberate draw, not a side effect.
    kelp_casts = bool(has_kelp and oblique)
    # The canopy is the shadow knob's own instrument: it casts and never
    # occludes, so `--shadow` moves cast-shadow pixels without moving
    # occlusion pixels.
    has_canopy = bool(rng.random() < float(shadow))

    return SceneDraw(
        pose=PoseParams.sample(rng, pose_amp_bracket),
        side=side,
        light_dir=light_dir,
        ambient=ambient,
        has_kelp=has_kelp,
        kelp_casts=kelp_casts,
        has_shark_occluder=has_shark,
        has_canopy=has_canopy,
        backlit=backlit,
        water=water,
        blur_px=blur_px,
        jitter=nuisance.CameraJitterParams(
            translate=float(0.02 + 0.04 * rng.random()),
            aim=float(0.01 + 0.02 * rng.random()),
            roll_deg=float(rng.uniform(2.0, 9.0))),
        caustic_time=float(rng.uniform(0.0, 10.0)),
    )


#: The camera looks along -Y to see the animal's LEFT flank (phi = +pi/2 is
#: +Y).  Rendering the R side MOVES THE CAMERA to -Y; the image is never
#: mirrored, because a mirrored flank is a different flank and a re-ID model
#: that learns on mirrored data learns a lie (Schema S1: cross-flank Rank-1
#: fell to 0.70% zero-shot).
_SIDE_VIEW_DIRECTION = {"L": (0.0, -1.0, 0.0), "R": (0.0, 1.0, 0.0)}


#: The one-cell 8-connected chart dilation that closes the nearest-neighbour
#: leak in the SCORING mask.  It lives in ``render.py`` (next to
#: ``sample_chart_mask``, the lookup whose half-cell error it pays for) and is
#: re-exported here because this module builds the render mask explicitly.
dilate_chart_mask = render.dilate_chart_mask


def canopy_casters(centre, light, body_length, seed, n_blades=CANOPY_N_BLADES,
                   height_frac=CANOPY_HEIGHT_FRAC):
    """Kelp blades placed UP-LIGHT of the animal and outside the camera frame.

    They exist to dapple the skin with a hard CAST shadow while contributing
    nothing to the occlusion mask.  The bases are laid out on the plane
    ``centre - d * dist``, where ``d`` is the light's travel direction and
    ``dist`` puts them ``height_frac`` body lengths above the animal, well
    clear of the fitted ortho frame (half-height ~0.20 body lengths).  Blades
    run along the body axis so their shadows fall as bands ACROSS the flank,
    which is the orientation that actually cuts the pattern.

    Every blade is ``role="occluder"`` (that is what ``nuisance.kelp_ribbon``
    builds), so even if a framing change ever brought one into view it could
    not contribute identity pixels -- it would be honestly labelled occlusion.

    NOT ANATOMY OR OCEANOGRAPHY.  See ``EVIDENCE["CANOPY_HEIGHT_FRAC"]``.
    """
    rng = np.random.default_rng([int(seed), 0x0CA0])
    d = np.asarray(light.direction, dtype=np.float64)
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    centre = np.asarray(centre, dtype=np.float64).reshape(3)
    # Travel far enough along -d that the blades clear the frame vertically
    # even when the light is oblique.
    dist = float(height_frac) * float(body_length) / max(abs(float(d[2])), 0.35)
    origin = centre - d * dist
    axis = np.array([1.0, 0.0, 0.0])
    side = np.array([0.0, 1.0, 0.0])
    blades = []
    for k in range(int(n_blades)):
        kp = nuisance.KelpParams(
            n_blades=1,
            length=float(body_length) * float(rng.uniform(0.30, 0.60)),
            width=float(body_length) * float(rng.uniform(0.045, 0.085)),
            n_segments=12,
            twist_turns=float(rng.uniform(0.9, 2.1)),
            sway=float(body_length) * 0.06,
            casts_shadow=True,
        )
        base = origin + np.array([
            float(rng.uniform(-0.60, 0.30)) * body_length,
            float(rng.uniform(-0.30, 0.30)) * body_length,
            float(rng.uniform(-0.06, 0.06)) * body_length,
        ])
        blades.append(nuisance.kelp_ribbon(
            base, axis, side, kp, phase=float(rng.uniform(0.0, 2.0 * math.pi)),
            name="canopy%02d" % k))
    return blades


def render_sighting(model, texture, scene, resolution=(192, 384), seed=0,
                    exclusion=None, shadow_map_size=512):
    """Pose, frame, light, occlude, render and degrade one sighting.

    Returns the ``render.render`` output dict with ``rgb`` already degraded by
    ``nuisance.apply_water`` (every mask is untouched: they are geometric
    ground truth for one instant).
    """
    verts = pose_vertices(model, scene.pose)
    subject = render.Instance(
        vertices=verts, faces=model.faces, uv=model.uv, texture=texture,
        vertex_s=model.vertex_s, vertex_phi=model.vertex_phi,
        role="subject", name="subject")

    direction = _SIDE_VIEW_DIRECTION[scene.side]
    cam = render.Camera.fit_ortho(verts, direction=direction, resolution=resolution,
                                  margin=1.22, distance=SUBJECT_RANGE_M)
    cam = nuisance.jitter_camera(cam, scene.jitter, seed=seed)

    light = render.DirectionalLight(direction=scene.light_dir, ambient=scene.ambient)

    instances = [subject]
    if scene.has_kelp:
        blades = nuisance.kelp_curtain(
            cam, params=nuisance.KelpParams(n_blades=int(4 + (seed % 4)),
                                            casts_shadow=scene.kelp_casts),
            seed=seed + 1)
        instances.extend(blades)
    if scene.has_shark_occluder:
        placement = nuisance.OccluderPlacement.sample(seed + 2)
        instances.append(nuisance.place_occluder(subject, cam, placement,
                                                 name="shark_occluder"))
    if scene.has_canopy:
        instances.extend(canopy_casters(
            verts.mean(axis=0), light, model_length(model), seed + 3))

    out = render.render(instances, cam, light=light, exclusion=exclusion,
                        shadows=True, shadow_map_size=shadow_map_size)
    out = nuisance.apply_water(out, params=scene.water, time=scene.caustic_time)
    out["meta"]["camera"] = {
        "kind": cam.kind, "eye": [float(v) for v in cam.eye],
        "target": [float(v) for v in cam.target],
        "ortho_height": float(cam.ortho_height),
        "resolution": [int(cam.resolution[0]), int(cam.resolution[1])],
        "range_m": SUBJECT_RANGE_M,
    }
    return out


# ---------------------------------------------------------------------------
# 6. Crop + the three melops boxes
# ---------------------------------------------------------------------------

def crop_and_boxes(out, s_head_max, pad=CROP_PAD_PX):
    """Crop by the body mask and cut the three LTWH boxes in ARC LENGTH.

    The body mask is ``visible_skin | occlusion`` -- every pixel where a
    SUBJECT surface exists, whether or not something is in front of it. That
    is what a human annotator boxes.

    The head/headless split is cut through the chart ground truth at
    ``s_head_max`` (the schema's last gill slit), never guessed from the
    silhouette, so it means the same anatomical thing at every pose and every
    view angle. Boxes are ``[left, top, width, height]`` floats in CROP
    pixels, and head/headless are expressed INSIDE the body crop, which is
    exactly what ``melops_data.load_crop`` applies them to.

    Returns ``None`` if the body, the head or the trunk is not visible at all;
    the caller drops that sighting rather than inventing a box.
    """
    body = out["visible_skin"] | out["occlusion"]
    if not body.any():
        return None
    h, w = body.shape
    box = render.mask_bbox_ltwh(body, pad=pad)
    l0 = int(max(0, math.floor(box[0])))
    t0 = int(max(0, math.floor(box[1])))
    l1 = int(min(w, math.ceil(box[0] + box[2])))
    t1 = int(min(h, math.ceil(box[1] + box[3])))
    if l1 - l0 < 8 or t1 - t0 < 8:
        return None
    sl = (slice(t0, t1), slice(l0, l1))

    chart_s = out["chart_s"][sl]
    body_c = body[sl]
    head_mask = render.chart_span_mask(chart_s, 0.0, s_head_max, within=body_c)
    tail_mask = render.chart_span_mask(chart_s, s_head_max, 1.0 + 1e-6, within=body_c)
    bbox_head = render.mask_bbox_ltwh(head_mask)
    bbox_headless = render.mask_bbox_ltwh(tail_mask)
    if bbox_head is None or bbox_headless is None:
        return None
    crop_h, crop_w = t1 - t0, l1 - l0
    if min(bbox_head[2], bbox_head[3], bbox_headless[2], bbox_headless[3]) < 2.0:
        return None

    return {
        "slice": sl,
        "origin": (float(l0), float(t0)),
        "bbox_body": [0.0, 0.0, float(crop_w), float(crop_h)],
        "bbox_head": [float(v) for v in bbox_head],
        "bbox_headless": [float(v) for v in bbox_headless],
        "head_px": int(head_mask.sum()),
        "headless_px": int(tail_mask.sum()),
    }


def _format_bbox(bbox):
    """The melops_data on-disk bbox format, digit for digit."""
    return ",".join("%.2f" % float(v) for v in bbox)


# ---------------------------------------------------------------------------
# 7. The deterministic identity timeline (shared with chart_readout.py)
# ---------------------------------------------------------------------------

class PatternContext(object):
    """Everything the PATTERN side of a corpus is generated against.

    Split out of :func:`generate` so a reader of a finished corpus can rebuild
    the exact ``Individual`` states that produced it -- ``chart_readout.py``
    needs them to compute the TRUE chart similarity a rendered pair should
    have had.  If the two ever diverge the ground truth stops being ground
    truth, so there is one constructor and both callers use it.
    """

    __slots__ = ("schema", "stations", "regions", "excl_chart", "params",
                 "s_head_max", "growth")

    def __init__(self, schema, stations, regions, excl_chart, params,
                 s_head_max, growth):
        self.schema = schema
        self.stations = stations
        self.regions = regions
        self.excl_chart = excl_chart
        self.params = params
        self.s_head_max = s_head_max
        self.growth = growth


def build_pattern_context(head_signal=1.0, flank_signal=1.0, n_spots=220,
                          n_common=40, chart_resolution=(96, 192)):
    """Schema, stations, exclusion regions, pattern params and growth model."""
    schema = exclusions.load_schema(pattern.DEFAULT_SCHEMA_PATH)
    stations = exclusions.default_stations(schema)
    exclusions.validate_stations(stations, schema)
    regions = exclusions.exclusion_regions(schema, stations=stations)
    s_head_max = float(stations["gill_slit_7_dorsal_origin"])
    # Sampling exclusion (pattern.py): the exact cells, so no spot is PLACED
    # in an excluded region.  Render exclusion: the same mask grown by one
    # cell, so no pixel can LEAK through nearest-neighbour lookup.  Built
    # explicitly here (rather than left to render's ``exclusion="auto"``,
    # which applies the same dilation) because this corpus pins its own chart
    # resolution and station values.  See render.dilate_chart_mask.
    excl_chart = dilate_chart_mask(
        exclusions.mask_from_regions(regions, chart_resolution), n_cells=1)
    params = pattern.PatternParams(
        n_spots_target=int(n_spots),
        head_signal=float(head_signal),
        flank_signal=float(flank_signal),
        head_s_max=s_head_max,
        flank_s_max=float(stations["precaudal_pit"]),
        n_common=int(n_common),
    )
    return PatternContext(schema, stations, regions, excl_chart, params,
                          s_head_max, drift.VonBertalanffyGrowth())


def individual_timeline(context, seed, index, sightings_per_individual=6,
                        years=4, start_date="2019-03-01",
                        length_bracket=LENGTH_CM_BRACKET):
    """The deterministic history of individual ``index`` of run ``seed``.

    Returns ``(identity, initial_length_cm, states)`` where ``states`` is a
    list of ``(date, side, Individual)`` in plan order.  The SAME
    ``Individual`` object is returned for two sides photographed on one date
    -- one animal, one pattern, two flanks.

    This is the single source of truth for "what the animal actually looked
    like": :func:`generate` renders these states, and ``chart_readout.py``
    re-derives them to score a readout against the drift it was given.
    """
    rng_i = np.random.default_rng([int(seed), 1, int(index)])
    identity = "syn%04d" % int(index)
    length_cm = float(rng_i.uniform(*length_bracket))
    plan = plan_sightings(rng_i, sightings_per_individual, years,
                          start=start_date)
    ind = pattern.Individual.generate(
        seed=int(np.random.default_rng([int(seed), 2, int(index)])
                 .integers(0, 2 ** 31 - 1)),
        params=context.params, identity=identity, date=plan[0][0],
        length_cm=length_cm, regions=context.regions)
    states = []
    prev_date = plan[0][0]
    for date, side in plan:
        if date != prev_date:
            ind = drift.resight(ind, prev_date, date,
                                growth_model=context.growth)
            prev_date = date
        states.append((date, side, ind))
    return identity, length_cm, states


def measured_length_mm(true_length_cm, seed, index, sighting,
                       rsd=LENGTH_MEASUREMENT_RSD):
    """The RECORDED length of one sighting, in millimetres.

    A field catalogue does not hold the animal's length, it holds somebody's
    estimate of it.  Without this the recorded length is a near-unique
    identity code (see ``EVIDENCE["LENGTH_MEASUREMENT_RSD"]``).  Drawn from a
    generator seeded from ``(seed, 4, index, sighting)`` so it is reproducible
    and disturbs no other stream.
    """
    rng = np.random.default_rng([int(seed), 4, int(index), int(sighting)])
    factor = 1.0 + float(rsd) * float(rng.standard_normal())
    return float(true_length_cm) * 10.0 * max(factor, 0.2)


# ---------------------------------------------------------------------------
# 8. The generator
# ---------------------------------------------------------------------------

def generate(out_dir, n_individuals=40, sightings_per_individual=6, years=4,
             head_signal=1.0, flank_signal=1.0, occlusion=0.3, shadow=0.5,
             turbidity=0.4, seed=0, resolution=(192, 384), tex_size=128,
             chart_resolution=(96, 192), n_spots=220, n_common=40,
             n_stations=56, n_around=36, shadow_map_size=512,
             start_date="2019-03-01", length_noise=LENGTH_MEASUREMENT_RSD,
             save_gt=True, progress=False):
    """Write a full synthetic corpus. Returns the summary dict.

    Deterministic in ``seed``: every rng is derived from
    ``(seed, individual index, sighting index)``.
    """
    if Image is None:
        raise RuntimeError("Pillow is required to write the corpus")
    os.makedirs(out_dir, exist_ok=True)
    for sub in ("body", "masks", "gt"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    context = build_pattern_context(head_signal=head_signal,
                                    flank_signal=flank_signal, n_spots=n_spots,
                                    n_common=n_common,
                                    chart_resolution=chart_resolution)
    stations, regions = context.stations, context.regions
    s_head_max, excl_chart = context.s_head_max, context.excl_chart

    model = build_model(n_stations=n_stations, n_around=n_around,
                        tex_size=tex_size, stations=stations)

    rows = []
    truth = []
    lengths = {}
    n_dropped = 0
    drop_reasons = []
    for i in range(int(n_individuals)):
        identity, _initial_length_cm, states = individual_timeline(
            context, seed, i, sightings_per_individual=sightings_per_individual,
            years=years, start_date=start_date)
        first_date = states[0][0]
        for j, (date, side, ind) in enumerate(states):
            rng_j = np.random.default_rng([int(seed), 3, i, j])
            scene = draw_scene(rng_j, side, occlusion=occlusion, shadow=shadow,
                               turbidity=turbidity)
            texture, chart, spot_table = chart_to_texture(
                model, ind, date=date, chart_resolution=chart_resolution)
            frame_seed = int(rng_j.integers(0, 2 ** 31 - 1))
            out = render_sighting(model, texture, scene, resolution=resolution,
                                  seed=frame_seed,
                                  exclusion=(excl_chart, "phi_major"),
                                  shadow_map_size=shadow_map_size)
            crop = crop_and_boxes(out, s_head_max)
            image_id = "%s_s%02d" % (identity, j)
            if crop is None:
                n_dropped += 1
                drop_reasons.append({"image_id": image_id,
                                     "reason": "body/head/trunk not visible in frame"})
                continue
            sl = crop["slice"]
            rgb = np.clip(out["rgb"][sl] * 255.0 + 0.5, 0, 255).astype(np.uint8)
            rel_path = os.path.join("body", image_id + ".png")
            Image.fromarray(rgb).save(os.path.join(out_dir, rel_path))
            mask_rel = os.path.join("masks", image_id + "_identity.png")
            Image.fromarray((out["identity"][sl] * 255).astype(np.uint8)).save(
                os.path.join(out_dir, mask_rel))
            gt_rel = ""
            if save_gt:
                gt_rel = os.path.join("gt", image_id + ".npz")
                np.savez_compressed(
                    os.path.join(out_dir, gt_rel),
                    chart_s=out["chart_s"][sl].astype(np.float32),
                    chart_phi=out["chart_phi"][sl].astype(np.float32),
                    identity=out["identity"][sl],
                    visible_skin=out["visible_skin"][sl],
                    occlusion=out["occlusion"][sl],
                    shadow=out["shadow"][sl],
                    cast_shadow=out["cast_shadow"][sl],
                    exclusion=out["exclusion"][sl],
                )
            rows.append({
                "image_id": image_id,
                "identity": identity,
                "path": rel_path,
                "date": str(date),
                "side": side,
                "bbox_body": _format_bbox(crop["bbox_body"]),
                "bbox_head": _format_bbox(crop["bbox_head"]),
                "bbox_headless": _format_bbox(crop["bbox_headless"]),
            })
            # The RECORDED length carries measurement error; the TRUE one
            # stays in truth.jsonl. See EVIDENCE["LENGTH_MEASUREMENT_RSD"].
            recorded_mm = measured_length_mm(ind.length_cm, seed, i, j,
                                             rsd=length_noise)
            lengths[image_id] = recorded_mm
            elapsed = float(pattern.days_between(first_date, date))
            truth.append({
                "image_id": image_id,
                "identity": identity,
                "date": str(date),
                "side": side,
                "path": rel_path,
                "identity_mask_path": mask_rel,
                "chart_gt_path": gt_rel,
                "length_cm": float(ind.length_cm),
                "length_mm": float(ind.length_cm) * 10.0,
                "measured_length_mm": float(recorded_mm),
                "length_measurement_rsd": float(length_noise),
                "elapsed_days_since_first": elapsed,
                "n_spots": int(len(ind.spots)),
                "n_visible_spots": int((spot_table["rendered_darkness"] > 0.05).sum()),
                "n_scars": int(len(ind.scars)),
                "spot_spacing_chart": float(ind.spot_spacing("chart")),
                "spot_spacing_cm": float(ind.spot_spacing("cm")),
                "pattern_provenance": {k: v for k, v in ind.provenance.items()
                                       if k in ("origin", "seed", "elapsed_days",
                                                "growth_ratio", "jitter_sigma",
                                                "new_scars", "realised_spots")},
                "scene": scene.as_dict(),
                "camera": out["meta"]["camera"],
                "crop_origin_px": list(crop["origin"]),
                "bbox_body": crop["bbox_body"],
                "bbox_head": crop["bbox_head"],
                "bbox_headless": crop["bbox_headless"],
                "px": {
                    "body": int((out["visible_skin"] | out["occlusion"]).sum()),
                    "visible_skin": int(out["visible_skin"].sum()),
                    "identity": int(out["identity"].sum()),
                    "occlusion": int(out["occlusion"].sum()),
                    "shadow": int(out["shadow"].sum()),
                    "cast_shadow": int(out["cast_shadow"].sum()),
                    "exclusion": int(out["exclusion"].sum()),
                    "head": crop["head_px"],
                    "headless": crop["headless_px"],
                },
                "exclusion_source": out["meta"].get("exclusion_source"),
            })
        if progress:
            print("individual %d/%d  images=%d" % (i + 1, n_individuals, len(rows)),
                  file=sys.stderr)

    if not rows:
        raise RuntimeError("no images survived framing; loosen the nuisance knobs")

    meta_path = os.path.join(out_dir, "metadata.csv")
    columns = ("image_id", "identity", "path", "date", "side",
               "bbox_body", "bbox_head", "bbox_headless")
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # readout_length_controlled.py reads this: filename_year is its image key.
    with open(os.path.join(out_dir, "Melops_metadata.txt"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename_year", "length"])
        for row in rows:
            writer.writerow([row["image_id"], "%.1f" % lengths[row["image_id"]]])

    with open(os.path.join(out_dir, "truth.jsonl"), "w") as f:
        for rec in truth:
            f.write(json.dumps(rec, sort_keys=True) + "\n")

    ident_px = np.array([t["px"]["identity"] for t in truth], dtype=np.float64)
    body_px = np.array([t["px"]["body"] for t in truth], dtype=np.float64)
    summary = {
        "n_individuals": int(n_individuals),
        "n_images": len(rows),
        "n_dropped": int(n_dropped),
        "drop_reasons": drop_reasons[:20],
        "n_singletons": int(sum(1 for k, v in _counts(rows).items() if v == 1)),
        "sides": {s: int(sum(1 for r in rows if r["side"] == s)) for s in ("L", "R")},
        "date_range": [min(r["date"] for r in rows), max(r["date"] for r in rows)],
        "identity_pixel_fraction_mean": float((ident_px / np.maximum(body_px, 1)).mean()),
        "identity_pixel_fraction_frontlit": _frac_mean(
            truth, "identity", lambda t: not t["scene"]["backlit"]),
        "identity_pixel_fraction_backlit": _frac_mean(
            truth, "identity", lambda t: t["scene"]["backlit"]),
        "occlusion_pixel_fraction_mean": _frac_mean(truth, "occlusion"),
        "cast_shadow_pixel_fraction_mean": _frac_mean(truth, "cast_shadow"),
        "exclusion_pixel_fraction_mean": _frac_mean(truth, "exclusion"),
        "occluded_frames": int(sum(1 for t in truth if t["px"]["occlusion"] > 0)),
        "cast_shadow_frames": int(sum(1 for t in truth if t["px"]["cast_shadow"] > 0)),
        "canopy_frames": int(sum(1 for t in truth if t["scene"]["canopy_caster"])),
        "backlit_frames": int(sum(1 for t in truth if t["scene"]["backlit"])),
        "images_per_identity": {
            "min": int(min(_counts(rows).values())),
            "max": int(max(_counts(rows).values())),
            "mean": float(np.mean(list(_counts(rows).values()))),
        },
        "mean_visibility_m": float(np.mean([t["scene"]["visibility_m"] for t in truth])),
        "length_mm": {
            "true_mean": float(np.mean([t["length_mm"] for t in truth])),
            "true_sd": float(np.std([t["length_mm"] for t in truth])),
            "recorded_mean": float(np.mean([t["measured_length_mm"] for t in truth])),
            "recorded_sd": float(np.std([t["measured_length_mm"] for t in truth])),
            "measurement_rsd": float(length_noise),
        },
        "args": {
            "seed": int(seed), "years": float(years),
            "sightings_per_individual": int(sightings_per_individual),
            "head_signal": float(head_signal), "flank_signal": float(flank_signal),
            "occlusion": float(occlusion), "shadow": float(shadow),
            "turbidity": float(turbidity),
            "length_noise": float(length_noise),
            "resolution": [int(resolution[0]), int(resolution[1])],
            "tex_size": int(tex_size),
            "chart_resolution": [int(chart_resolution[0]), int(chart_resolution[1])],
            "n_spots": int(n_spots), "n_common": int(n_common),
            "start_date": str(start_date),
        },
        "constants": {k: EVIDENCE[k] for k in sorted(EVIDENCE)},
        "chart_convention": exclusions.CHART_CONVENTION,
        "head_cut_station": {"name": "gill_slit_7_dorsal_origin", "s": s_head_max,
                             "grade": exclusions.station_grades()["gill_slit_7_dorsal_origin"]},
        "exclusion_regions": [r.name for r in regions],
    }
    with open(os.path.join(out_dir, "dataset.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def _frac_mean(truth, key, where=None):
    """Mean of ``px[key] / px['body']`` over the truth records passing ``where``."""
    vals = [t["px"][key] / max(t["px"]["body"], 1)
            for t in truth if where is None or where(t)]
    return float(np.mean(vals)) if vals else None


def _counts(rows):
    out = {}
    for r in rows:
        out[r["identity"]] = out.get(r["identity"], 0) + 1
    return out


def _resolution(text):
    parts = str(text).lower().replace(",", "x").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("resolution must be HxW, got %r" % (text,))
    return (int(parts[0]), int(parts[1]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output corpus directory")
    p.add_argument("--n-individuals", type=int, default=40)
    p.add_argument("--sightings-per-individual", type=int, default=6,
                   help="target count; the realised count varies and includes "
                        "singletons on purpose")
    p.add_argument("--years", type=float, default=4.0, help="study-window length")
    p.add_argument("--head-signal", type=float, default=1.0,
                   help="identity amplitude anterior to the last gill slit "
                        "(0 = textured but uninformative)")
    p.add_argument("--flank-signal", type=float, default=1.0,
                   help="identity amplitude on the trunk")
    p.add_argument("--occlusion", type=float, default=0.3,
                   help="probability a frame has kelp or a second shark in front")
    p.add_argument("--shadow", type=float, default=0.5,
                   help="probability of an oblique key light plus a shadow caster")
    p.add_argument("--turbidity", type=float, default=0.4,
                   help="0 = best-day visibility, 1 = typical La Jolla murk")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resolution", type=_resolution, default=(192, 384),
                   help="render resolution HxW (default 192x384)")
    p.add_argument("--tex-size", type=int, default=128)
    p.add_argument("--chart-resolution", type=_resolution, default=(96, 192),
                   help="pattern chart resolution H_phi x W_s")
    p.add_argument("--n-spots", type=int, default=220)
    p.add_argument("--n-common", type=int, default=40,
                   help="shared non-identity speckle layer, so a zero-signal "
                        "region is textured but uninformative")
    p.add_argument("--start-date", default="2019-03-01")
    p.add_argument("--length-noise", type=float, default=LENGTH_MEASUREMENT_RSD,
                   help="relative sd of the RECORDED length per sighting "
                        "(default %.2f, a placeholder bracket -- see EVIDENCE). "
                        "0 makes the recorded length exact, which turns it back "
                        "into a near-unique identity code" % LENGTH_MEASUREMENT_RSD)
    p.add_argument("--no-gt", action="store_true", help="skip the chart-GT npz files")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    summary = generate(
        args.out, n_individuals=args.n_individuals,
        sightings_per_individual=args.sightings_per_individual,
        years=args.years, head_signal=args.head_signal,
        flank_signal=args.flank_signal, occlusion=args.occlusion,
        shadow=args.shadow, turbidity=args.turbidity, seed=args.seed,
        resolution=args.resolution, tex_size=args.tex_size,
        chart_resolution=args.chart_resolution, n_spots=args.n_spots,
        n_common=args.n_common, start_date=args.start_date,
        length_noise=args.length_noise,
        save_gt=not args.no_gt, progress=not args.quiet,
    )
    print("wrote %d images of %d individuals to %s (dropped %d)"
          % (summary["n_images"], summary["n_individuals"], args.out,
             summary["n_dropped"]))
    print("sides L/R = %d/%d | dates %s .. %s | mean visibility %.1f m"
          % (summary["sides"]["L"], summary["sides"]["R"],
             summary["date_range"][0], summary["date_range"][1],
             summary["mean_visibility_m"]))
    print("identity pixels / body pixels = %.3f | occluded frames %d | "
          "cast-shadow frames %d"
          % (summary["identity_pixel_fraction_mean"], summary["occluded_frames"],
             summary["cast_shadow_frames"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
