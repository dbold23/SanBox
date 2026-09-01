"""Module R (part 2) -- occluders and water for synthetic sevengill frames.

Everything in here is NUISANCE: it must change the photograph without
changing the identity.  That is the whole point of the synthetic corpus --
the re-ID experiment downstream is only meaningful if the pattern is held
fixed while the imaging conditions vary, so a drop in Rank-1 can be blamed on
kelp or turbidity rather than on the animal.

Four families, each a SEEDED PARAMETER OBJECT (``*Params``) plus a pure
function that applies it:

1. OCCLUDERS -- geometry placed between the camera and the animal.
   :func:`kelp_curtain` builds procedural ribbons (thin twisted quads);
   :func:`place_occluder` puts a second shark (any :class:`render.Instance`)
   in the foreground.  Both come back as ``role="occluder"`` instances, which
   is what makes ``render.render`` record their pixels in the OCCLUSION MASK
   and drop them from the identity mask.
2. TURBIDITY -- Beer-Lambert attenuation toward a blue-green veiling light,
   parameterised by HORIZONTAL VISIBILITY in metres because that is the only
   number the dive literature actually reports.
3. CAUSTICS -- a low-frequency multiplicative flicker.
4. CAMERA -- seeded jitter/roll, and an optional motion blur.

EVIDENCE for the water constants (this is the only part of module R that
makes a physical claim; the renderer itself is pure geometry):

* Visibility at the La Jolla sevengill aggregation is **typically 10-20 ft
  (3-6 m)**, best days 30-40 ft (9-12 m), in kelp, shot obliquely from a
  diver-held camera, with the aggregation concentrated March-April.
  Source: ``docs/sevengill-canonical-reid/01-evidence-and-answers.md``
  (encounter/imaging-conditions row).  That row itself says: **"These figures
  are dive-shop copy, not measurements -- if the number matters, get it from
  SCCOOS."**  [EVIDENCE GRADE: SECONDARY, dive-operator sources.  Bracket
  3-6 m typical / 9-12 m best.]  :data:`VISIBILITY_M_TYPICAL` sits at the
  midpoint of the typical bracket.
* Visibility -> attenuation uses the standard 2% apparent-contrast threshold:
  ``c = -ln(0.02) / visibility``.  [EVIDENCE GRADE: DERIVED -- a textbook
  visual-range convention, not a sevengill measurement.]
* The blue-green tint is the qualitative Jerlov coastal-water ordering (red
  attenuates fastest; the transmission minimum sits in the green; blue is
  pulled down by dissolved organics, so it is intermediate, not lowest).
  ``docs/.../03-candidate-approaches.md`` describes the target water as
  "turbid green kelp water".  [EVIDENCE GRADE: DERIVED from a qualitative
  description.  No measured spectral attenuation for this site exists in this
  repository -- do not quote :data:`CHANNEL_ATTENUATION_RATIOS` as measured.]
* Caustic amplitude has NO evidence behind it at all; it is a visual
  placeholder, and at 3-6 m visibility scattering has largely washed caustics
  out.  [EVIDENCE GRADE: none -- placeholder.]
"""

from __future__ import annotations

import math

import numpy as np

from render import Instance, transform_instance

__all__ = [
    "KelpParams",
    "kelp_ribbon",
    "kelp_curtain",
    "OccluderPlacement",
    "place_occluder",
    "TurbidityParams",
    "attenuation_per_metre",
    "channel_attenuation",
    "apply_turbidity",
    "CausticParams",
    "caustic_field",
    "apply_caustics",
    "CameraJitterParams",
    "jitter_camera",
    "MotionBlurParams",
    "motion_blur",
    "average_frames",
    "WaterParams",
    "apply_water",
    "VISIBILITY_M_TYPICAL",
    "VISIBILITY_M_BRACKET",
    "VISIBILITY_M_BEST_BRACKET",
    "CONTRAST_THRESHOLD",
    "CHANNEL_ATTENUATION_RATIOS",
    "VEILING_RGB",
    "KELP_RGB",
]

# --- water constants: see the module docstring for grades and brackets -----
VISIBILITY_M_BRACKET = (3.0, 6.0)        # [SECONDARY] typical, La Jolla
VISIBILITY_M_BEST_BRACKET = (9.0, 12.0)  # [SECONDARY] best days
VISIBILITY_M_TYPICAL = 4.5               # [DERIVED] midpoint of the bracket
CONTRAST_THRESHOLD = 0.02                # [DERIVED] standard visual-range convention
# (red, green, blue) attenuation relative to the broadband coefficient; the
# mean is 1 so changing the tint does not change the overall visibility.
CHANNEL_ATTENUATION_RATIOS = (1.60, 0.60, 0.80)   # [DERIVED] qualitative Jerlov coastal
VEILING_RGB = (0.10, 0.27, 0.25)                  # [DERIVED] "turbid green kelp water"
KELP_RGB = (0.16, 0.13, 0.07)                     # [none] visual placeholder, dark olive


def _rng(seed):
    return np.random.default_rng(np.random.SeedSequence(int(seed)))


# ---------------------------------------------------------------------------
# 1a. Kelp ribbons
# ---------------------------------------------------------------------------

class KelpParams(object):
    """Seeded parameters for a curtain of kelp blades.

    A blade is a THIN TWISTED QUAD STRIP: ``n_segments`` quads along its
    length, each rotated a little further about the blade axis.  The twist is
    not decoration -- it is what makes a blade's occluding width vary from
    full to nearly zero along its length, which is how real kelp occludes:
    intermittently, not as a solid bar.

    Args:
        n_blades: how many blades in the curtain.
        length: blade length in world units (the fixture shark is 1-2 units).
        width: blade width in world units.
        n_segments: quads per blade; more = smoother twist.
        twist_turns: full turns of twist over the blade's length.
        sway: lateral amplitude of the blade's own bend, in world units.
        depth_frac: where the curtain sits between the camera eye (0) and the
            look-at point (1).  Must be < 1 or the "occluder" is behind the
            animal and occludes nothing.
        spread: lateral/vertical scatter of blade bases, as a fraction of the
            camera's ortho height (or of the frame height at the curtain's
            depth for a pinhole camera).
        lean: blade tilt from vertical, radians, sampled uniformly in
            ``+-lean``.
        color: albedo.
        casts_shadow: default ``False``.  A blade drifting between the camera
            and the animal is usually NOT between the animal and the sun, and
            defaulting it to ``True`` would silently entangle the occlusion
            mask with the shadow mask (and break the "occlusion mask covers
            exactly the identity-mask reduction" property).  Set it ``True``
            deliberately when you want dappled kelp shadow.
    """

    def __init__(self, n_blades=6, length=1.4, width=0.09, n_segments=10,
                 twist_turns=1.3, sway=0.12, depth_frac=0.45, spread=0.55,
                 lean=0.30, color=KELP_RGB, casts_shadow=False):
        if not 0.0 < depth_frac < 1.0:
            raise ValueError("depth_frac must be in (0, 1); >= 1 is behind the subject")
        self.n_blades = int(n_blades)
        self.length = float(length)
        self.width = float(width)
        self.n_segments = max(int(n_segments), 1)
        self.twist_turns = float(twist_turns)
        self.sway = float(sway)
        self.depth_frac = float(depth_frac)
        self.spread = float(spread)
        self.lean = float(lean)
        self.color = tuple(float(c) for c in color)
        self.casts_shadow = bool(casts_shadow)

    def replace(self, **kw):
        base = dict(n_blades=self.n_blades, length=self.length, width=self.width,
                    n_segments=self.n_segments, twist_turns=self.twist_turns,
                    sway=self.sway, depth_frac=self.depth_frac,
                    spread=self.spread, lean=self.lean, color=self.color,
                    casts_shadow=self.casts_shadow)
        base.update(kw)
        return KelpParams(**base)

    def __repr__(self):
        return ("KelpParams(n_blades=%d, length=%.3g, width=%.3g, twist_turns=%.3g, "
                "depth_frac=%.3g)" % (self.n_blades, self.length, self.width,
                                      self.twist_turns, self.depth_frac))


def kelp_ribbon(base, axis, side, params, phase=0.0, sway_dir=None, name="kelp"):
    """One twisted ribbon as an :class:`render.Instance`.

    Args:
        base: ``(3,)`` world point at the bottom of the blade.
        axis: ``(3,)`` unit direction the blade runs along.
        side: ``(3,)`` unit direction of the blade's width at zero twist;
            it is re-orthogonalised against ``axis``.
        params: :class:`KelpParams` (length, width, twist, sway, colour).
        phase: twist phase offset, radians -- de-synchronises blades.
        sway_dir: direction of the blade's own bend (default: ``axis x side``).

    The strip is open (no cap), one quad thick and ``double_sided``: an open
    surface shaded from one side only would go black over half its length.
    """
    base = np.asarray(base, dtype=np.float64).reshape(3)
    axis = _unitv(axis)
    side = _unitv(side - axis * float(np.dot(side, axis)))
    if sway_dir is None:
        sway_dir = np.cross(axis, side)
    sway_dir = _unitv(sway_dir)

    n = params.n_segments
    t = np.linspace(0.0, 1.0, n + 1)
    centres = (base[None, :] + np.outer(t * params.length, axis)
               + np.outer(params.sway * np.sin(math.pi * t) * t, sway_dir))
    theta = phase + 2.0 * math.pi * params.twist_turns * t
    # Blade width vector rotates about the blade axis: at theta = +-pi/2 the
    # blade is edge-on to `side` and its silhouette narrows to nothing.
    perp = np.cross(axis, side)
    wvec = (np.cos(theta)[:, None] * side[None, :]
            + np.sin(theta)[:, None] * perp[None, :])
    # Taper: kelp blades are narrower at both ends.  The 0.25 floor keeps the
    # end quads from collapsing to zero area -- a degenerate triangle is
    # skipped by the rasteriser, so a blade with a true zero taper silently
    # loses its tip.
    taper = 0.25 + 0.75 * np.sin(np.pi * np.clip(t, 0.0, 1.0)) ** 0.35
    half = 0.5 * params.width * taper
    verts = np.concatenate([centres - half[:, None] * wvec,
                            centres + half[:, None] * wvec], axis=0)
    lo = np.arange(n)
    faces = np.concatenate([
        np.column_stack([lo, lo + 1, lo + 1 + (n + 1)]),
        np.column_stack([lo, lo + 1 + (n + 1), lo + (n + 1)]),
    ], axis=0)
    return Instance(vertices=verts, faces=faces, color=params.color,
                    role="occluder", double_sided=True,
                    casts_shadow=params.casts_shadow, name=name)


def _unitv(v):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("zero-length direction vector")
    return v / n


def _frame_half_height(camera, distance):
    """Half-height of the camera's view frustum at ``distance`` in front."""
    if camera.kind == "ortho":
        return 0.5 * camera.ortho_height
    return float(distance) * math.tan(math.radians(0.5 * camera.fov_y_deg))


def kelp_curtain(camera, params=None, seed=0, up=(0.0, 0.0, 1.0), target=None):
    """A seeded curtain of kelp blades BETWEEN the camera and its target.

    Blades stand roughly along ``up`` (kelp grows toward the surface), leaning
    and twisting by seeded amounts, scattered across the frame at
    ``params.depth_frac`` of the way from the eye to ``target`` (default the
    camera's look-at point).  Because the placement is expressed in the
    camera's own frame, the curtain is in front of the subject for any camera
    pose, which is what the occlusion test needs.

    Returns a list of ``role="occluder"`` instances -- append them to the
    subject list and ``render.render`` does the rest.
    """
    params = KelpParams() if params is None else params
    rng = _rng(seed)
    target = camera.target if target is None else np.asarray(target, dtype=np.float64)
    dist = float(np.linalg.norm(target - camera.eye)) * params.depth_frac
    centre = camera.eye + camera.forward * dist
    half_h = _frame_half_height(camera, dist)
    half_w = half_h * camera.aspect
    upv = _unitv(up)

    out = []
    for k in range(params.n_blades):
        dx = rng.uniform(-1.0, 1.0) * params.spread * half_w
        dy = rng.uniform(-1.0, 1.0) * params.spread * half_h
        dz = rng.uniform(-0.15, 0.15) * dist
        base = (centre + camera.right * dx + camera.up * dy
                + camera.forward * dz - upv * 0.5 * params.length)
        lean_a = rng.uniform(-params.lean, params.lean)
        lean_b = rng.uniform(-params.lean, params.lean)
        axis = _unitv(upv + lean_a * camera.right + lean_b * camera.forward)
        side = camera.right - axis * float(np.dot(camera.right, axis))
        if float(np.linalg.norm(side)) < 1e-6:
            side = camera.up - axis * float(np.dot(camera.up, axis))
        blade = params.replace(
            length=params.length * float(rng.uniform(0.75, 1.3)),
            width=params.width * float(rng.uniform(0.6, 1.4)),
            twist_turns=params.twist_turns * float(rng.uniform(0.6, 1.5)),
        )
        out.append(kelp_ribbon(base, axis, side, blade,
                               phase=float(rng.uniform(0.0, 2.0 * math.pi)),
                               name="kelp_%02d" % k))
    return out


# ---------------------------------------------------------------------------
# 1b. A second shark as a foreground occluder
# ---------------------------------------------------------------------------

class OccluderPlacement(object):
    """Where to put a second animal relative to the camera.

    ``depth_frac`` is the fraction of the eye->target distance at which the
    occluder's centroid sits (< 1 = in front of the subject); ``offset_x`` and
    ``offset_y`` are in units of the frame half-width/half-height at that
    depth, so ``offset_y = -0.4`` means "40% of the way down the frame"
    regardless of camera type; ``yaw_deg`` rotates it about world ``up``;
    ``scale`` shrinks or grows it (a smaller shark nearer the camera is the
    common real case).
    """

    def __init__(self, depth_frac=0.55, offset_x=0.0, offset_y=-0.35,
                 yaw_deg=25.0, scale=0.8, casts_shadow=False):
        if not 0.0 < depth_frac < 1.0:
            raise ValueError("depth_frac must be in (0, 1)")
        self.depth_frac = float(depth_frac)
        self.offset_x = float(offset_x)
        self.offset_y = float(offset_y)
        self.yaw_deg = float(yaw_deg)
        self.scale = float(scale)
        self.casts_shadow = bool(casts_shadow)

    @classmethod
    def sample(cls, seed, depth_frac=(0.35, 0.7), offset=0.45, yaw_deg=60.0,
               scale=(0.6, 1.0)):
        """A seeded random placement."""
        rng = _rng(seed)
        return cls(depth_frac=float(rng.uniform(*depth_frac)),
                   offset_x=float(rng.uniform(-offset, offset)),
                   offset_y=float(rng.uniform(-offset, offset)),
                   yaw_deg=float(rng.uniform(-yaw_deg, yaw_deg)),
                   scale=float(rng.uniform(*scale)))

    def __repr__(self):
        return ("OccluderPlacement(depth_frac=%.3g, offset=(%.3g, %.3g), "
                "yaw_deg=%.3g, scale=%.3g)" % (self.depth_frac, self.offset_x,
                                               self.offset_y, self.yaw_deg, self.scale))


def place_occluder(instance, camera, placement=None, up=(0.0, 0.0, 1.0),
                   target=None, name=None):
    """Copy ``instance`` into the foreground as an OCCLUDER.

    The copy keeps its texture, UV and chart coordinates (they are intrinsic
    to the surface) but its ``role`` becomes ``"occluder"``, so its pixels are
    NOT identity pixels even though it is a shark, and the subject pixels it
    hides land in the occlusion mask.  That is the honest way to model the
    aggregation: several sevengills in one frame, one of them the subject.
    """
    placement = OccluderPlacement() if placement is None else placement
    target = camera.target if target is None else np.asarray(target, dtype=np.float64)
    dist = float(np.linalg.norm(target - camera.eye)) * placement.depth_frac
    half_h = _frame_half_height(camera, dist)
    half_w = half_h * camera.aspect
    goal = (camera.eye + camera.forward * dist
            + camera.right * (placement.offset_x * half_w)
            + camera.up * (placement.offset_y * half_h))

    a = math.radians(placement.yaw_deg)
    u = _unitv(up)
    ca, sa = math.cos(a), math.sin(a)
    K = np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])
    rot = np.eye(3) * ca + sa * K + (1.0 - ca) * np.outer(u, u)   # Rodrigues

    centre = instance.vertices.mean(axis=0)
    return transform_instance(
        instance, rotation=rot, translation=goal - centre,
        scale=placement.scale, role="occluder",
        casts_shadow=placement.casts_shadow,
        name=name or ((instance.name or "instance") + "_occluder"))


# ---------------------------------------------------------------------------
# 2. Turbidity: Beer-Lambert attenuation toward a blue-green veiling light
# ---------------------------------------------------------------------------

def attenuation_per_metre(visibility_m, contrast_threshold=CONTRAST_THRESHOLD):
    """Broadband attenuation coefficient ``c`` [1/m] from a visibility range.

    An object is "just visible" when its apparent contrast has fallen to
    ``contrast_threshold`` (conventionally 2%), and contrast decays as
    ``exp(-c * range)``, so ``c = -ln(threshold) / visibility``.
    ``visibility_m = 4.5`` gives ``c ~= 0.87 /m``.
    [EVIDENCE GRADE: DERIVED -- standard visual-range convention.]
    """
    v = float(visibility_m)
    if v <= 0.0:
        raise ValueError("visibility_m must be > 0")
    t = float(contrast_threshold)
    if not 0.0 < t < 1.0:
        raise ValueError("contrast_threshold must be in (0, 1)")
    return -math.log(t) / v


class TurbidityParams(object):
    """Water between the camera and everything it sees.

    Args:
        visibility_m: horizontal visibility in metres.  Default
            :data:`VISIBILITY_M_TYPICAL` = 4.5 m, the midpoint of the 3-6 m
            La Jolla typical bracket [SECONDARY].
        contrast_threshold: the apparent contrast that defines "visible".
        channel_ratios: per-channel multipliers on ``c`` (red, green, blue).
            Their mean is 1, so retinting does not change overall visibility.
            [DERIVED -- qualitative Jerlov coastal ordering, NOT measured.]
        veiling: RGB of the light scattered into the path; this is what a
            pixel becomes at infinite range, i.e. the background colour.
        metres_per_world_unit: scene scale.  The fixture tube is 1-2 world
            units long and a sevengill is 1.5-3 m, so a scene built at
            "1 unit = 1 m" wants 1.0.  GET THIS RIGHT: it is the only thing
            tying the rendered fog to the metres in the evidence.
        gain: multiplies ``c`` for quick sweeps without touching visibility.

    Not modelled: forward scatter (a blur that grows with range), backscatter
    from the strobe (the "snowstorm" of a diver-held flash), or wavelength
    dependence of the veiling light itself.  Those are real and this is a
    single-scatter approximation.  [EVIDENCE GRADE: DERIVED, approximate.]
    """

    def __init__(self, visibility_m=VISIBILITY_M_TYPICAL,
                 contrast_threshold=CONTRAST_THRESHOLD,
                 channel_ratios=CHANNEL_ATTENUATION_RATIOS,
                 veiling=VEILING_RGB, metres_per_world_unit=1.0, gain=1.0):
        self.visibility_m = float(visibility_m)
        self.contrast_threshold = float(contrast_threshold)
        self.channel_ratios = np.asarray(channel_ratios, dtype=np.float64).reshape(3)
        self.veiling = np.asarray(veiling, dtype=np.float64).reshape(3)
        self.metres_per_world_unit = float(metres_per_world_unit)
        self.gain = float(gain)
        if self.metres_per_world_unit <= 0.0:
            raise ValueError("metres_per_world_unit must be > 0")

    @classmethod
    def sample(cls, seed, visibility_bracket=VISIBILITY_M_BRACKET, **kw):
        """A seeded draw of visibility from a bracket (default 3-6 m)."""
        rng = _rng(seed)
        return cls(visibility_m=float(rng.uniform(*visibility_bracket)), **kw)

    def replace(self, **kw):
        base = dict(visibility_m=self.visibility_m,
                    contrast_threshold=self.contrast_threshold,
                    channel_ratios=self.channel_ratios, veiling=self.veiling,
                    metres_per_world_unit=self.metres_per_world_unit,
                    gain=self.gain)
        base.update(kw)
        return TurbidityParams(**base)

    def __repr__(self):
        return ("TurbidityParams(visibility_m=%.3g, c=%.4g /m, ratios=%s)"
                % (self.visibility_m, self.broadband_c,
                   np.round(self.channel_ratios, 3).tolist()))

    @property
    def broadband_c(self):
        return self.gain * attenuation_per_metre(self.visibility_m,
                                                 self.contrast_threshold)


def channel_attenuation(params):
    """``(3,)`` per-channel attenuation in 1/WORLD UNIT (not per metre)."""
    return (params.broadband_c * params.channel_ratios
            * params.metres_per_world_unit)


def apply_turbidity(rgb, distance, params=None):
    """Attenuate ``rgb`` over ``distance`` and mix in the veiling light.

    ``out = rgb * T + veiling * (1 - T)`` with ``T = exp(-c_channel * d)``.
    ``distance`` is in WORLD UNITS (pass ``render.render``'s ``"depth"``);
    ``+inf`` -- the background -- gives ``T = 0``, i.e. pure veiling light, so
    an untextured far background becomes the water colour for free.

    Contrast between any two surfaces at the same range is multiplied by
    exactly ``T``, which is strictly decreasing in range: that is the sense in
    which turbidity "reduces contrast with distance", and it is a test.
    """
    params = TurbidityParams() if params is None else params
    rgb = np.asarray(rgb, dtype=np.float64)
    d = np.asarray(distance, dtype=np.float64)
    if d.shape != rgb.shape[:-1]:
        raise ValueError("distance shape %r does not match rgb %r"
                         % (d.shape, rgb.shape))
    c = channel_attenuation(params)
    dd = np.where(np.isfinite(d), np.maximum(d, 0.0), np.inf)
    with np.errstate(over="ignore"):
        T = np.exp(-dd[..., None] * c)          # (..., 3), broadcast over channels
    T = np.where(np.isfinite(dd)[..., None], T, 0.0)
    return np.clip(rgb * T + params.veiling * (1.0 - T), 0.0, 1.0)


# ---------------------------------------------------------------------------
# 3. Caustics
# ---------------------------------------------------------------------------

class CausticParams(object):
    """A low-frequency multiplicative flicker standing in for surface caustics.

    ``n_waves`` plane waves with seeded directions, frequencies (in cycles
    across the frame) and phase speeds are summed and normalised to [-1, 1];
    the image is multiplied by ``1 + contrast * field``.  ``time`` advances
    the phases, so a resighting sequence flickers instead of freezing.

    This is a SCREEN-SPACE approximation: real caustics are projected onto the
    body by the water surface and follow its geometry.  At 3-6 m in kelp they
    are also largely washed out by scattering, which is why the default
    contrast is small.  [EVIDENCE GRADE: none -- visual placeholder.]
    """

    def __init__(self, contrast=0.12, n_waves=4, freq=(0.8, 3.0), speed=0.6,
                 lit_only=True):
        self.contrast = float(contrast)
        self.n_waves = int(n_waves)
        self.freq = (float(freq[0]), float(freq[1]))
        self.speed = float(speed)
        self.lit_only = bool(lit_only)
        if self.n_waves < 1:
            raise ValueError("n_waves must be >= 1")

    def replace(self, **kw):
        base = dict(contrast=self.contrast, n_waves=self.n_waves,
                    freq=self.freq, speed=self.speed, lit_only=self.lit_only)
        base.update(kw)
        return CausticParams(**base)

    def __repr__(self):
        return ("CausticParams(contrast=%.3g, n_waves=%d, freq=%s, speed=%.3g)"
                % (self.contrast, self.n_waves, self.freq, self.speed))


def caustic_field(shape, params=None, seed=0, time=0.0):
    """``(H, W)`` field in [-1, 1]; deterministic in ``(seed, time)``."""
    params = CausticParams() if params is None else params
    h, w = int(shape[0]), int(shape[1])
    rng = _rng(seed)
    y = (np.arange(h) + 0.5) / h
    x = (np.arange(w) + 0.5) / w
    X, Y = np.meshgrid(x, y)
    field = np.zeros((h, w), dtype=np.float64)
    total = 0.0
    for _ in range(params.n_waves):
        ang = rng.uniform(0.0, 2.0 * math.pi)
        f = rng.uniform(*params.freq)
        ph = rng.uniform(0.0, 2.0 * math.pi)
        sp = rng.uniform(0.5, 1.5) * params.speed
        amp = rng.uniform(0.6, 1.0)
        field += amp * np.cos(2.0 * math.pi * f * (math.cos(ang) * X + math.sin(ang) * Y)
                              + ph + sp * float(time))
        total += amp
    return field / max(total, 1e-12)


def apply_caustics(rgb, params=None, seed=0, time=0.0, mask=None):
    """Multiply ``rgb`` by ``1 + contrast * caustic_field``.

    ``mask`` (e.g. ``out["visible_skin"] & ~out["shadow"]``) restricts the
    flicker to lit surfaces; ``CausticParams.lit_only`` only documents the
    intent, the caller supplies the mask.
    """
    params = CausticParams() if params is None else params
    rgb = np.asarray(rgb, dtype=np.float64)
    field = caustic_field(rgb.shape[:2], params, seed=seed, time=time)
    gain = 1.0 + params.contrast * field
    if mask is not None:
        gain = np.where(np.asarray(mask, dtype=bool), gain, 1.0)
    return np.clip(rgb * gain[..., None], 0.0, 1.0)


# ---------------------------------------------------------------------------
# 4. Camera jitter and motion blur
# ---------------------------------------------------------------------------

class CameraJitterParams(object):
    """Hand-held camera shake: position, aim and roll.

    ``translate`` and ``aim`` are fractions of the frame half-height at the
    subject distance, so the same numbers mean the same visual displacement
    for an orthographic and a pinhole camera.  ``roll_deg`` is the 1-sigma
    roll: a diver-held camera is never level, and a re-ID model that has only
    ever seen level frames learns the horizon.
    """

    def __init__(self, translate=0.04, aim=0.02, roll_deg=6.0):
        self.translate = float(translate)
        self.aim = float(aim)
        self.roll_deg = float(roll_deg)

    def replace(self, **kw):
        base = dict(translate=self.translate, aim=self.aim, roll_deg=self.roll_deg)
        base.update(kw)
        return CameraJitterParams(**base)

    def __repr__(self):
        return ("CameraJitterParams(translate=%.3g, aim=%.3g, roll_deg=%.3g)"
                % (self.translate, self.aim, self.roll_deg))


def jitter_camera(camera, params=None, seed=0):
    """A seeded jittered copy of ``camera`` (same resolution and kind)."""
    params = CameraJitterParams() if params is None else params
    rng = _rng(seed)
    dist = float(np.linalg.norm(camera.target - camera.eye))
    half_h = _frame_half_height(camera, dist)
    dx, dy = rng.normal(0.0, params.translate * half_h, size=2)
    ax, ay = rng.normal(0.0, params.aim * half_h, size=2)
    roll = float(rng.normal(0.0, params.roll_deg))
    eye = camera.eye + camera.right * dx + camera.up * dy
    target = camera.target + camera.right * ax + camera.up * ay
    return camera.replace(eye=eye, target=target,
                          roll_deg=camera.roll_deg + roll)


class MotionBlurParams(object):
    """Linear motion blur: ``length_px`` long at ``angle_deg`` in the image."""

    def __init__(self, length_px=7.0, angle_deg=0.0):
        self.length_px = float(length_px)
        self.angle_deg = float(angle_deg)

    def replace(self, **kw):
        base = dict(length_px=self.length_px, angle_deg=self.angle_deg)
        base.update(kw)
        return MotionBlurParams(**base)

    def __repr__(self):
        return ("MotionBlurParams(length_px=%.3g, angle_deg=%.3g)"
                % (self.length_px, self.angle_deg))


def motion_blur(image, params=None):
    """Convolve an image with a line kernel.  APPROXIMATE, and only for RGB.

    A true motion blur integrates the SCENE over the exposure, which moves
    occlusion boundaries; a fixed line kernel smears across them instead.  Use
    :func:`average_frames` over several renders when the difference matters
    (fast subject, long exposure).  Masks are never blurred: they are
    geometric ground truth for one instant, and a blurred boolean is a lie.
    """
    from scipy import ndimage

    params = MotionBlurParams() if params is None else params
    img = np.asarray(image, dtype=np.float64)
    n = max(int(round(params.length_px)), 1)
    if n == 1:
        return img.copy()
    k = np.zeros((n, n), dtype=np.float64)
    a = math.radians(params.angle_deg)
    t = np.linspace(-0.5, 0.5, 4 * n)
    xs = np.clip(np.rint((n - 1) / 2.0 + t * (n - 1) * math.cos(a)), 0, n - 1).astype(int)
    ys = np.clip(np.rint((n - 1) / 2.0 - t * (n - 1) * math.sin(a)), 0, n - 1).astype(int)
    np.add.at(k, (ys, xs), 1.0)
    k /= k.sum()
    if img.ndim == 2:
        return ndimage.convolve(img, k, mode="nearest")
    return np.stack([ndimage.convolve(img[..., c], k, mode="nearest")
                     for c in range(img.shape[2])], axis=-1)


def average_frames(frames):
    """Mean of a stack of RGB renders -- the honest motion blur."""
    arr = np.asarray(frames, dtype=np.float64)
    if arr.ndim != 4:
        raise ValueError("frames must be (n, H, W, 3), got %r" % (arr.shape,))
    return arr.mean(axis=0)


# ---------------------------------------------------------------------------
# The whole water column in one seeded object
# ---------------------------------------------------------------------------

class WaterParams(object):
    """Turbidity + caustics (+ optional motion blur) as one seeded bundle."""

    def __init__(self, turbidity=None, caustics=None, blur=None, seed=0):
        self.turbidity = TurbidityParams() if turbidity is None else turbidity
        self.caustics = CausticParams() if caustics is None else caustics
        self.blur = blur
        self.seed = int(seed)

    @classmethod
    def sample(cls, seed, visibility_bracket=VISIBILITY_M_BRACKET):
        rng = _rng(seed)
        return cls(turbidity=TurbidityParams(
                       visibility_m=float(rng.uniform(*visibility_bracket))),
                   caustics=CausticParams(
                       contrast=float(rng.uniform(0.05, 0.18))),
                   seed=int(seed))

    def replace(self, **kw):
        base = dict(turbidity=self.turbidity, caustics=self.caustics,
                    blur=self.blur, seed=self.seed)
        base.update(kw)
        return WaterParams(**base)

    def __repr__(self):
        return ("WaterParams(seed=%d, %r, %r, blur=%r)"
                % (self.seed, self.turbidity, self.caustics, self.blur))


def apply_water(out, params=None, seed=None, time=0.0, copy=True):
    """Apply turbidity, then caustics, then optional blur to a render dict.

    ORDER MATTERS and it is physical: caustics illuminate the SUBJECT, so they
    multiply the radiance leaving the body, which then travels through the
    water; but the veiling light is added along the whole path and is not
    itself caustic-modulated.  Applying caustics to the already-fogged image
    would put flicker on the distant background, which is wrong.  So: shade
    (in ``render.render``) -> caustics on lit skin -> turbidity.

    MASKS ARE NOT TOUCHED.  They are geometric ground truth; only ``rgb``
    changes, and ``meta["nuisance"]`` records what was applied.
    """
    params = WaterParams() if params is None else params
    seed = params.seed if seed is None else int(seed)
    res = dict(out) if copy else out
    rgb = np.asarray(out["rgb"], dtype=np.float64)

    lit = None
    if params.caustics is not None and params.caustics.lit_only:
        if "visible_skin" in out and "shadow" in out:
            lit = np.asarray(out["visible_skin"]) & ~np.asarray(out["shadow"])
    if params.caustics is not None:
        rgb = apply_caustics(rgb, params.caustics, seed=seed + 1, time=time,
                             mask=lit)
    if params.turbidity is not None:
        rgb = apply_turbidity(rgb, out["depth"], params.turbidity)
    if params.blur is not None:
        rgb = motion_blur(rgb, params.blur)

    res["rgb"] = rgb
    meta = dict(out.get("meta", {}))
    meta["nuisance"] = {
        "turbidity": repr(params.turbidity),
        "caustics": repr(params.caustics),
        "blur": repr(params.blur),
        "seed": seed,
        "time": float(time),
        "order": "caustics(lit skin) -> turbidity -> blur",
    }
    res["meta"] = meta
    return res
