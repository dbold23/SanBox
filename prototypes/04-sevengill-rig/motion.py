"""Swimming kinematics for a sevengill rig: body wave -> curvature -> joint rotations -> clips.

Module C of prototype 04. Takes a *rest-pose* skeleton description and emits animation
clips. It never touches meshes, GLB files, weights or ``rig.py``; the only things it
needs from a rig are joint names, the parent array, the arc-length station of each
spine joint and which joints belong to which fin.

AXIS CONVENTION
---------------
Inherited from ``blender/operators/create_shark_armature.py``:
**snout +X, tail -X, lateral +/-Y, dorsal +Z.**  Arc length ``s`` runs 0 at the snout
to 1 at the caudal upper-lobe tip, so *increasing s moves in -X*.  Lateral body
undulation is therefore displacement along **Y**, and axial (yaw) bending is rotation
about the **dorsoventral axis Z** -- exactly the "Z if X is the body axis and Y
lateral" convention the brief names.  Roll/bank is about **X**, fin pitch about **Y**.

SIGN OF THE YAW.  With the chain marching along -X, a bone rotated by ``theta`` about
+Z points along ``(-cos theta, -sin theta, 0)``.  Matching that to the midline tangent
``(-1, y'(u), 0)/sqrt(1+y'^2)`` gives ``theta = -arctan(y')``, so the *cumulative*
yaw at bone ``j`` is minus the tangent angle and the *local* yaw at joint ``j`` is

    yaw_j = - integral of kappa(u) du over joint j's arc-length cell.

``BODY_AXIS_YAW_SIGN`` below carries that minus sign as a named constant; flip it if a
rig ever puts the snout at -X.

INPUT CONTRACT (module B -> module C)
-------------------------------------
Anything with the shape of :class:`MotionSkeleton`:

* ``names``      -- list of J joint names, parents before children.
* ``parents``    -- (J,) int, -1 at the root.
* ``spine_names``, ``spine_fractions`` -- the serial spine in head->tail order and its
  arc-length fractions ``s_j`` in [0, 1].  These are the joints of
  ``phase1b/p0-sevengill-schema/skeleton_sevengill.py``; nothing here invents a spine.
* ``fins``       -- ``{fin_name: (root_index, tip_index)}``.
* ``fps``, ``body_length``.

``MotionSkeleton.from_skeleton`` duck-types ``rig.Skeleton`` (``.names/.parents/.kinds/
.fractions/.fins/.joints``) without importing ``rig``, so the two modules stay
independent and testable apart.

OUTPUT CONTRACT (module C -> module B)
--------------------------------------
:class:`Clip` -- ``{"name": str, "times": (T,), "quats": (T, J, 4) xyzw}``.  ``Clip``
also answers to ``clip["rotations"]`` and ``clip.to_animation()`` so it drops straight
into ``gltf_export.write_skinned_glb(..., animations=[clip.to_animation()])``, which
wants that key.  Every joint gets a channel; undriven joints carry identity.

BIOLOGY AND ITS SOURCES
-----------------------
Every constant below is a named module-level parameter carrying its literature
bracket.  Citations are to the programme scan,
``docs/sevengill-canonical-reid/01-evidence-and-answers.md`` (Q4b, fish
biomechanics).  The load-bearing caveat from that scan is repeated here because it
governs how these numbers may be quoted:

    **No swimming kinematics have ever been published for *Notorynchus cepedianus*,
    or for any hexanchid** -- no amplitude envelope, no propulsive wavelength, no
    maximum curvature (scan, Q4b: "NO SOURCE EXISTS", 22 searches).  The sevengill is
    bracketed, not measured: subcarangiform sharks below (Webb & Keyes 1982; Donley &
    Shadwick 2003; Berio et al. 2025 on *Scyliorhinus canicula*), anguilliform
    elongate vertebrates above (Gillis 1997; the eel figures in Di Santo et al. 2021;
    released lamprey midlines).  The defaults here sit inside that bracket by
    construction and are **plausible, not measured**.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "BODY_AXIS_YAW_SIGN",
    "MODES",
    "MODE_CONFIG",
    "DEFAULT_FIN_DRIVES",
    "ESCAPE_MAX_TOTAL_TURN_DEG",
    "ESCAPE_PEAK_CURVATURE_UNCAPPED_PER_BL",
    "escape_peak_curvature_cap",
    "escape_total_turn_deg",
    "escape_closure_bl",
    "mode_body_amplitude_bl",
    "fin_amplitude_scale",
    "WaveParams",
    "EscapeParams",
    "FinChannel",
    "FinDrive",
    "MotionSkeleton",
    "Clip",
    "amplitude_envelope",
    "lateral_wave",
    "wave_heading",
    "curvature",
    "joint_cell_edges",
    "joint_yaw_angles",
    "integrate_spine",
    "make_clip",
    "spine_yaw_angles",
    "fundamental_phase",
    "phase_report",
    "dct_energy_fraction",
    "tail_tip_amplitude",
    "implied_skin_strain",
    "peak_curvature",
    "params_for_mode",
    "kinematics_report",
    "default_spine_fractions",
    "escape_curvature",
    "resolve_fin_drive",
    "quat_from_axis_angle",
    "quat_mul",
    "euler_zxy_quat",
    "quat_yaw_z",
]


# ---------------------------------------------------------------------------
# Schema import (the DCT bending basis lives there; do not re-derive it here)
# ---------------------------------------------------------------------------
def _import_schema_skeleton():
    """Import ``skeleton_sevengill`` from the phase1b schema package.

    Path resolution mirrors ``rig.py``: ``$SEVENGILL_SCHEMA_DIR`` if set, else
    ``<repo>/phase1b/p0-sevengill-schema`` relative to this file.
    """
    override = os.environ.get("SEVENGILL_SCHEMA_DIR")
    if override:
        candidate = override
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.abspath(
            os.path.join(here, os.pardir, os.pardir, "phase1b", "p0-sevengill-schema")
        )
    if not os.path.isfile(os.path.join(candidate, "skeleton_sevengill.py")):
        raise ImportError(
            "skeleton_sevengill.py not found in %r; set SEVENGILL_SCHEMA_DIR" % candidate
        )
    if candidate not in sys.path:
        sys.path.insert(0, candidate)
    import skeleton_sevengill  # noqa: E402  (path is only valid after the insert)

    return skeleton_sevengill


SCHEMA = _import_schema_skeleton()

SPINE_JOINTS: List[str] = list(SCHEMA.SPINE_JOINTS)
NUM_SPINE_JOINTS: int = len(SPINE_JOINTS)          # 13
NUM_SPINE_SEGMENTS: int = NUM_SPINE_JOINTS - 1     # 12

#: Sign relating a *cumulative* +Z yaw to the midline tangent angle.  -1 because the
#: body axis runs snout +X -> tail -X (see the module docstring).
BODY_AXIS_YAW_SIGN: float = -1.0


# ---------------------------------------------------------------------------
# Literature brackets.  Each default is the midpoint of its bracket unless noted.
# ---------------------------------------------------------------------------

#: Tail-beat frequency for steady cruise, Hz.  Brief's bracket 0.5-1.5 Hz.  Sharks of
#: this size cruise slowly; no hexanchid measurement exists (scan Q4b).
CRUISE_TAILBEAT_HZ: float = 0.9
CRUISE_TAILBEAT_HZ_BRACKET: Tuple[float, float] = (0.5, 1.5)

#: Propulsive wavelength in body lengths.  Di Santo et al. 2021 (PNAS 118) at ~1 BL/s:
#: **eel 0.58 BL, trout 1.00, mackerel 0.96, tuna 1.17** -- an eel-like swimmer fits
#: nearly two wave cycles on its body where a carangiform fits one.  The bracket for
#: anguilliform swimmers is 0.6-1.0 BL.
#:
#: WHY THE DEFAULT SITS AT THE LONG END OF THAT BRACKET, not the eel end.  Curvature
#: scales as ``A * (2*pi/lambda)^2``, and curvature is what sets tissue strain
#: (Donley & Shadwick 2003).  Pairing the eel wavelength 0.6 BL with the 0.10-0.20 BL
#: tail amplitude bracket implies a mid-body longitudinal skin strain of roughly
#: 15-25% -- two to four times the only strain figure any shark measurement supports.
#: The two brackets are therefore not independently satisfiable at their extremes, and
#: strain is the one of the two that has a shark measurement behind it.  0.90 BL is
#: the longest wavelength that stays inside the stated bracket while
#: :func:`implied_skin_strain` lands inside :data:`SKIN_STRAIN_BRACKET`.  Override it
#: (and re-read the strain diagnostic) to explore the eel end deliberately.
CRUISE_WAVELENGTH_BL: float = 0.90
CRUISE_WAVELENGTH_BL_BRACKET: Tuple[float, float] = (0.6, 1.0)

#: Tail-tip lateral HALF-amplitude in body lengths (peak, not peak-to-peak).  Brief's
#: bracket 0.10-0.20 BL for steady cruise; ~0.2 BL peak-to-peak is the classic fish
#: figure.  Webb & Keyes 1982 report specific amplitude A/L for six shark species but
#: the per-species values were not retrievable (scan Q4b).
CRUISE_TAIL_AMPLITUDE_BL: float = 0.11
CRUISE_TAIL_AMPLITUDE_BL_BRACKET: Tuple[float, float] = (0.10, 0.20)

#: A(0) / A(1): how much the head yaws relative to the tail.  Kajiura et al. 2022
#: (*Carcharhinus perezi*): "tail amplitude exceeded head yaw amplitude by roughly
#: 80%", i.e. head ~ 0.2 of tail.  Webb & Keyes 1982's rate-of-change-of-A/L result is
#: the same statement: the anterior trunk barely bends, the posterior does the work.
HEAD_AMPLITUDE_RATIO: float = 0.15
HEAD_AMPLITUDE_RATIO_BRACKET: Tuple[float, float] = (0.10, 0.35)

#: Amplitude envelope shape.  Di Santo et al. 2021 model the lateral amplitude
#: envelope along the body with a **second-degree polynomial shared by most species**;
#: "quadratic" is that.  "exponential" is the harder anguilliform caricature, kept as
#: an option for the upper end of the bracket.
DEFAULT_ENVELOPE: str = "quadratic"

#: Half-width of the trunk in body lengths, used only by :func:`implied_skin_strain`.
#: An elongate shark is roughly 0.10-0.12 BL wide, so the skin sits ~0.05 BL off the
#: vertebral neutral axis. [derived from body proportions, not a published figure]
TRUNK_HALF_WIDTH_BL: float = 0.05

#: Longitudinal strain bracket the scan supports.  Donley & Shadwick 2003 (JEB 206)
#: measured leopard-shark RED MUSCLE strain at ~1.0 BL/s as +/-3.9% anterior, +/-6.6%
#: mid-body, +/-4.8% posterior, in phase with local midline curvature and reproduced
#: by a bending-beam model.  Skin is farther from the neutral axis than superficial
#: red muscle, so **skin strain >= muscle strain**; the scan's own derived figure for
#: a ~0.10 m half-width at R ~ 0.8-1.0 m is roughly 10-12% per flank.  Report this as
#: a bound DERIVED from published muscle strain, never as a measured skin strain.
SKIN_STRAIN_BRACKET: Tuple[float, float] = (0.039, 0.13)

#: Sustained curvature offset for a turn, 1/BL.  A steady turn of radius ~2 BL is a
#: gentle cruise turn; 0.5/BL is radius 2 BL. [plausible, not measured]
TURN_CURVATURE_OFFSET_PER_BL: float = 0.5

#: Bank angle applied at the root during a turn, degrees.  Sharks roll into turns;
#: magnitude unmeasured for this species. [plausible, not measured]
TURN_BANK_DEG: float = 8.0

#: C-start stage durations, seconds.  Stage 1 = the rapid whole-body bend to one side
#: (~0.1 s per the brief), stage 2 = the propulsive contralateral stroke, then a coast.
ESCAPE_STAGE1_S: float = 0.10
ESCAPE_STAGE2_S: float = 0.12
ESCAPE_COAST_TAU_S: float = 0.08
#: Head-to-tail travel delay of the C-start wave, seconds (wave speed ~ 30 BL/s).
ESCAPE_TRAVEL_DELAY_S: float = 0.03
#: Arc-length fraction over which the C-start curvature ramps in from the rigid head.
ESCAPE_HEAD_RAMP_S: float = 0.30

#: THE UNCAPPED C-start caricature's peak curvature, 1/BL (radius ~0.17 BL at the
#: tightest point of the C).  No hexanchid escape response has been filmed; this
#: number was originally chosen so that the NET turning of the midline at the extreme
#: of stage 1 -- the quantity that is actually observable in published fish C-starts --
#: reached ~285 deg.
#:
#: WHY IT IS NO LONGER THE DEFAULT.  285 deg of total turning closes an elongate body
#: into an O: at that setting the last spine joint comes back to within 0.175 BL of the
#: snout and the caudal lobe, rigidly carried past it, overlaps the head outright.
#: README section 7 states plainly that this rig has **no self-contact handling** --
#: "a C-start bends the body far enough that the tail can intersect the head.  Nothing
#: detects or resolves that; the clip is kinematic, not simulated."  A default that is
#: *guaranteed* to violate the one caveat the prototype names is not a caricature, it
#: is a bug, so the default is capped instead of shipped self-intersecting.
ESCAPE_PEAK_CURVATURE_UNCAPPED_PER_BL: float = 6.0

#: The cap: the largest TOTAL turning of the midline (the sum of the spine joint
#: yaws, i.e. the integral of curvature over the whole body) that the default C-start
#: is allowed to reach, in degrees.  180 deg is the body closed into a U -- the
#: tightest published-looking C-start that still leaves the tail pointing away from
#: the head rather than back through it, so the "no self-contact" caveat above stays a
#: caveat about extreme *overrides* instead of a description of the default.
#: [plausible, not measured -- like every other escape number here]
ESCAPE_MAX_TOTAL_TURN_DEG: float = 180.0


def _escape_shape_integral(head_ramp_s=ESCAPE_HEAD_RAMP_S):
    """``integral of E(s) ds`` over [0, 1] for the C-start's head ramp, exactly.

    ``E(s) = smoothstep(s / head_ramp)``, and ``integral of smoothstep on [0,1] = 1/2``,
    so the integral is ``head_ramp/2 + (1 - head_ramp)``.  At the extreme of stage 1 the
    time profile ``g`` is 1 over essentially the whole body, so the total turning of
    the midline there is ``peak_curvature * this integral`` radians -- which is what
    makes the cap below a closed form rather than a search.
    """
    h = float(head_ramp_s)
    return 0.5 * h + (1.0 - h)


def escape_peak_curvature_cap(max_total_turn_deg=ESCAPE_MAX_TOTAL_TURN_DEG,
                              head_ramp_s=ESCAPE_HEAD_RAMP_S):
    """Peak curvature, 1/BL, whose stage-1 extreme turns the midline by exactly
    ``max_total_turn_deg``.  See :data:`ESCAPE_MAX_TOTAL_TURN_DEG`."""
    return math.radians(float(max_total_turn_deg)) / _escape_shape_integral(head_ramp_s)


#: C-start peak curvature, 1/BL -- :data:`ESCAPE_PEAK_CURVATURE_UNCAPPED_PER_BL` capped
#: by :data:`ESCAPE_MAX_TOTAL_TURN_DEG`.  3.696 /BL, radius ~0.27 BL at the tightest
#: point of the C, total turning ~177 deg with the head-to-tail travel delay included.
#:
#: What distinguishes an escape from a cruise is NOT whole-body peak curvature.  The
#: most curved point on a cruising anguilliform swimmer is its own tail tip, and that
#: stays comparable.  The signature is that during a C-start the WHOLE body -- head
#: and anterior trunk included -- bends the SAME WAY AT ONCE, so net turning is an
#: order of magnitude larger than a cruise's, whose travelling S-wave cancels itself.
#: (Measured on the defaults: escape 176.7 deg of net turning against cruise's 29.9,
#: while over the anterior half the escape's peak curvature is still 2.5x the cruise's.)
ESCAPE_PEAK_CURVATURE_PER_BL: float = min(
    ESCAPE_PEAK_CURVATURE_UNCAPPED_PER_BL, escape_peak_curvature_cap()
)

#: Default frame rate for clips.
DEFAULT_FPS: float = 30.0


# ---------------------------------------------------------------------------
# Small quaternion helpers (xyzw, Hamilton product), numpy-only.
# ---------------------------------------------------------------------------
_AXES: Dict[str, np.ndarray] = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


def _axis_vector(axis):
    """Accept ``"x"``/``"y"``/``"z"`` or a 3-vector; return a unit (3,) array."""
    if isinstance(axis, str):
        try:
            return _AXES[axis.lower()]
        except KeyError:
            raise ValueError("axis must be 'x', 'y', 'z' or a 3-vector; got %r" % (axis,))
    v = np.asarray(axis, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("axis vector is zero")
    return v / n


def quat_from_axis_angle(axis, angle):
    """Quaternion(s) in (x, y, z, w) for a rotation of ``angle`` about ``axis``.

    ``angle`` may be an array; the result is ``angle.shape + (4,)``.
    """
    a = _axis_vector(axis)
    ang = np.asarray(angle, dtype=float)
    half = 0.5 * ang
    s = np.sin(half)
    out = np.empty(ang.shape + (4,), dtype=float)
    out[..., 0] = a[0] * s
    out[..., 1] = a[1] * s
    out[..., 2] = a[2] * s
    out[..., 3] = np.cos(half)
    return out


def quat_mul(q1, q2):
    """Hamilton product ``q1 * q2`` in (x, y, z, w); broadcasts over leading axes.

    Corresponds to matrix composition ``R(q1) @ R(q2)``.
    """
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    )


def euler_zxy_quat(yaw=0.0, roll=0.0, pitch=0.0):
    """Quaternion for ``R = Rz(yaw) @ Rx(roll) @ Ry(pitch)`` in (x, y, z, w).

    The order is fixed and documented rather than inferred: yaw (dorsoventral axis Z)
    is the axial bending DOF and is applied outermost, roll (about the body axis X) is
    bank, pitch (about the lateral axis Y) is fin angle of attack.  All three are
    small for every mode implemented here, so the ordering is a convention, not a
    source of visible difference.
    """
    q = quat_from_axis_angle("z", yaw)
    q = quat_mul(q, quat_from_axis_angle("x", roll))
    q = quat_mul(q, quat_from_axis_angle("y", pitch))
    return q


def quat_yaw_z(q):
    """Extract the +Z rotation angle from quaternion(s), exact for pure-Z rotations.

    Used only to read yaw back out of a clip when the per-frame yaw profile was not
    cached; :class:`Clip` caches it in ``meta["spine_yaw"]``, which is exact for every
    joint including a banked root.
    """
    q = np.asarray(q, dtype=float)
    return 2.0 * np.arctan2(q[..., 2], q[..., 3])


# ---------------------------------------------------------------------------
# Wave parameters
# ---------------------------------------------------------------------------
class WaveParams(object):
    """Parameters of the travelling body wave.

    Lateral displacement of the midline, with ``s`` the arc-length fraction along the
    centerline (0 snout, 1 caudal upper-lobe tip) and ``t`` seconds::

        y(s, t) = envelope_gain(t) * A(s) * sin(2*pi*(s/lambda - f*t + phase))

    Lengths are in BODY LENGTHS.  Curvature is therefore in 1/BL and joint angles --
    which are integrals of curvature over arc length -- are dimensionless and
    independent of the animal's actual size.  ``body_length`` only enters when a
    displacement is wanted in world units (:func:`integrate_spine`).

    Args:
        frequency_hz: tail-beat frequency ``f``.  Bracket ``CRUISE_TAILBEAT_HZ_BRACKET``.
        wavelength_bl: propulsive wavelength ``lambda`` in BL.
        tail_amplitude_bl: ``A(1)``, the tail-tip lateral half-amplitude in BL.
        head_amplitude_ratio: ``A(0) / A(1)``.
        envelope: ``"quadratic"`` (Di Santo et al. 2021's shared second-degree
            polynomial) or ``"exponential"``.
        phase: constant phase offset in cycles.
        curvature_offset_per_bl: sustained asymmetric curvature added to the wave
            (a turn), 1/BL, applied through the same spatial envelope shape.
        bank_deg: constant roll about +X applied at the root joint.
        gain: ``"steady"`` (gain 1) or ``"burst_coast"`` (a smooth periodic
            kick-then-coast envelope, used by ``glide``) or ``"decay"`` (monotone
            exponential decay with ``decay_tau_s``, used for a one-shot glide).
        decay_tau_s: time constant of the ``"decay"`` gain, seconds.
        coast_floor: residual amplitude fraction at the bottom of a burst-and-coast
            cycle.
        coast_power: sharpness of the burst-and-coast kick.
    """

    __slots__ = (
        "frequency_hz",
        "wavelength_bl",
        "tail_amplitude_bl",
        "head_amplitude_ratio",
        "envelope",
        "phase",
        "curvature_offset_per_bl",
        "bank_deg",
        "gain",
        "decay_tau_s",
        "coast_floor",
        "coast_power",
    )

    def __init__(
        self,
        frequency_hz=CRUISE_TAILBEAT_HZ,
        wavelength_bl=CRUISE_WAVELENGTH_BL,
        tail_amplitude_bl=CRUISE_TAIL_AMPLITUDE_BL,
        head_amplitude_ratio=HEAD_AMPLITUDE_RATIO,
        envelope=DEFAULT_ENVELOPE,
        phase=0.0,
        curvature_offset_per_bl=0.0,
        bank_deg=0.0,
        gain="steady",
        decay_tau_s=1.0,
        coast_floor=0.25,
        coast_power=2.0,
    ):
        if frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be > 0; got %r" % (frequency_hz,))
        if wavelength_bl <= 0.0:
            raise ValueError("wavelength_bl must be > 0; got %r" % (wavelength_bl,))
        if tail_amplitude_bl < 0.0:
            raise ValueError("tail_amplitude_bl must be >= 0; got %r" % (tail_amplitude_bl,))
        if not 0.0 < head_amplitude_ratio <= 1.0:
            raise ValueError(
                "head_amplitude_ratio must be in (0, 1]; got %r" % (head_amplitude_ratio,)
            )
        if envelope not in ("quadratic", "exponential"):
            raise ValueError("envelope must be 'quadratic' or 'exponential'; got %r" % (envelope,))
        if gain not in ("steady", "burst_coast", "decay"):
            raise ValueError("gain must be 'steady', 'burst_coast' or 'decay'; got %r" % (gain,))
        if decay_tau_s <= 0.0:
            raise ValueError("decay_tau_s must be > 0")
        if not 0.0 <= coast_floor <= 1.0:
            raise ValueError("coast_floor must be in [0, 1]")
        self.frequency_hz = float(frequency_hz)
        self.wavelength_bl = float(wavelength_bl)
        self.tail_amplitude_bl = float(tail_amplitude_bl)
        self.head_amplitude_ratio = float(head_amplitude_ratio)
        self.envelope = envelope
        self.phase = float(phase)
        self.curvature_offset_per_bl = float(curvature_offset_per_bl)
        self.bank_deg = float(bank_deg)
        self.gain = gain
        self.decay_tau_s = float(decay_tau_s)
        self.coast_floor = float(coast_floor)
        self.coast_power = float(coast_power)

    @property
    def period_s(self):
        """Tail-beat period, seconds."""
        return 1.0 / self.frequency_hz

    def replace(self, **kwargs):
        """Return a copy with the named fields overridden."""
        fields = {name: getattr(self, name) for name in self.__slots__}
        unknown = set(kwargs) - set(fields)
        if unknown:
            raise TypeError("unknown WaveParams field(s): %s" % ", ".join(sorted(unknown)))
        fields.update(kwargs)
        return WaveParams(**fields)

    def __repr__(self):
        return "WaveParams(f=%.3g Hz, lambda=%.3g BL, A_tail=%.3g BL, envelope=%r, gain=%r)" % (
            self.frequency_hz,
            self.wavelength_bl,
            self.tail_amplitude_bl,
            self.envelope,
            self.gain,
        )


class EscapeParams(object):
    """C-start parameters (mode ``"escape"``).

    Stage 1 is the rapid whole-body bend to one side; stage 2 is the contralateral
    propulsive stroke; then the body coasts back toward straight.  The curvature field
    is prescribed directly (not through ``y``), because a C-start is not a travelling
    sinusoid::

        kappa(s, t) = peak * E(s) * g(t - travel_delay * s)

    with ``E`` a smooth ramp from the rigid head into the trunk and ``g`` the piecewise
    stage profile ``0 -> +1 -> -1 -> 0``.

    Every number is a caricature: no hexanchid escape response has been published.
    """

    __slots__ = (
        "peak_curvature_per_bl",
        "stage1_s",
        "stage2_s",
        "coast_tau_s",
        "travel_delay_s",
        "head_ramp_s",
        "direction",
    )

    def __init__(
        self,
        peak_curvature_per_bl=ESCAPE_PEAK_CURVATURE_PER_BL,
        stage1_s=ESCAPE_STAGE1_S,
        stage2_s=ESCAPE_STAGE2_S,
        coast_tau_s=ESCAPE_COAST_TAU_S,
        travel_delay_s=ESCAPE_TRAVEL_DELAY_S,
        head_ramp_s=ESCAPE_HEAD_RAMP_S,
        direction=1.0,
    ):
        if stage1_s <= 0 or stage2_s <= 0 or coast_tau_s <= 0:
            raise ValueError("escape stage durations must be > 0")
        if not 0.0 < head_ramp_s < 1.0:
            raise ValueError("head_ramp_s must be in (0, 1)")
        self.peak_curvature_per_bl = float(peak_curvature_per_bl)
        self.stage1_s = float(stage1_s)
        self.stage2_s = float(stage2_s)
        self.coast_tau_s = float(coast_tau_s)
        self.travel_delay_s = float(travel_delay_s)
        self.head_ramp_s = float(head_ramp_s)
        self.direction = float(direction)

    @property
    def duration_s(self):
        """A sensible full clip length: both stages plus three coast time constants."""
        return self.stage1_s + self.stage2_s + 3.0 * self.coast_tau_s + self.travel_delay_s

    def __repr__(self):
        return "EscapeParams(peak=%.3g /BL, stage1=%.3g s, stage2=%.3g s)" % (
            self.peak_curvature_per_bl,
            self.stage1_s,
            self.stage2_s,
        )


# ---------------------------------------------------------------------------
# The body wave
# ---------------------------------------------------------------------------
def amplitude_envelope(s, params, derivatives=False):
    """Lateral amplitude envelope ``A(s)`` in body lengths.

    ``"quadratic"``: ``A(s) = A1 * (h + (1 - h) * s^2)`` -- the second-degree
    polynomial envelope Di Santo et al. 2021 found shared across most of 44 fish
    species, pinned by ``A(0) = h*A1`` (head yaw, Kajiura et al. 2022) and
    ``A(1) = A1`` (tail tip).  Its slope is zero at the snout, which is the
    "anterior trunk barely bends" result of Webb & Keyes 1982.

    ``"exponential"``: ``A(s) = A1 * h^(1-s)`` -- the harder anguilliform caricature.

    Args:
        s: scalar or array of arc-length fractions.
        params: :class:`WaveParams`.
        derivatives: if True return ``(A, dA/ds, d2A/ds2)`` instead of ``A``.
    """
    s = np.asarray(s, dtype=float)
    a1 = params.tail_amplitude_bl
    h = params.head_amplitude_ratio
    if params.envelope == "quadratic":
        a = a1 * (h + (1.0 - h) * s ** 2)
        if not derivatives:
            return a
        da = a1 * (1.0 - h) * 2.0 * s
        dda = np.full_like(s, a1 * (1.0 - h) * 2.0)
        return a, da, dda
    k = math.log(1.0 / h)
    a = a1 * np.exp(k * (s - 1.0))
    if not derivatives:
        return a
    return a, k * a, (k ** 2) * a


def _gain(t, params):
    """Time-varying amplitude gain ``g(t)``, dimensionless.

    ``"steady"``   -> 1.
    ``"burst_coast"`` -> a smooth, strictly PERIODIC kick-then-coast envelope with
        period ``1/f``: full amplitude at the start of each tail-beat, decaying to
        ``coast_floor`` mid-cycle and returning.  This is what ``glide`` uses, and it
        is why a glide clip is still a seamless loop: every quantity in the clip is a
        function of ``t`` through terms of period ``1/f``.  Burst-and-coast (a.k.a.
        kick-and-glide) is a real fish gait, so the caricature is not arbitrary --
        but its shape here is chosen, not fitted.
    ``"decay"``    -> monotone ``exp(-t / tau)``, for a one-shot glide transition.
        NOT periodic; a clip using it cannot be a seamless loop and ``make_clip``
        refuses to mark it as one.
    """
    t = np.asarray(t, dtype=float)
    if params.gain == "steady":
        return np.ones_like(t)
    if params.gain == "decay":
        return np.exp(-t / params.decay_tau_s)
    kick = 0.5 * (1.0 + np.cos(2.0 * math.pi * params.frequency_hz * t))
    return params.coast_floor + (1.0 - params.coast_floor) * kick ** params.coast_power


def lateral_wave(s, t, params, derivatives=False):
    """Lateral midline displacement ``y(s, t)`` in body lengths, and its s-derivatives.

    ``y(s, t) = g(t) * A(s) * sin(theta)``, ``theta = 2*pi*(s/lambda - f*t + phase)``.

    Derivatives are analytic (never finite-differenced), which is what makes the
    curvature exact and the finite-difference test in ``test_motion.py`` meaningful::

        y'  = g * (A' sin + A k cos)
        y'' = g * (A'' sin + 2 A' k cos - A k^2 sin),   k = 2*pi/lambda

    ``s`` and ``t`` broadcast against each other.

    Args:
        s: arc-length fraction(s).
        t: time(s) in seconds.
        params: :class:`WaveParams`.
        derivatives: if True return ``(y, dy/ds, d2y/ds2)``.
    """
    s = np.asarray(s, dtype=float)
    t = np.asarray(t, dtype=float)
    k = 2.0 * math.pi / params.wavelength_bl
    theta = 2.0 * math.pi * (s / params.wavelength_bl - params.frequency_hz * t + params.phase)
    g = _gain(t, params)
    if not derivatives:
        return g * amplitude_envelope(s, params) * np.sin(theta)
    a, da, dda = amplitude_envelope(s, params, derivatives=True)
    sin, cos = np.sin(theta), np.cos(theta)
    y = g * a * sin
    dy = g * (da * sin + a * k * cos)
    ddy = g * (dda * sin + 2.0 * da * k * cos - a * k ** 2 * sin)
    return y, dy, ddy


def _turn_offset_shape(s):
    """Spatial shape of the sustained turn curvature offset, peaking at the tail.

    Same normalisation as the quadratic amplitude envelope (0.2 at the snout, 1 at the
    tail): a turning shark bends its posterior trunk more than its rigid head.
    """
    s = np.asarray(s, dtype=float)
    h = HEAD_AMPLITUDE_RATIO
    return h + (1.0 - h) * s ** 2


def curvature(s, t, params):
    """Signed midline curvature ``kappa(s, t)`` in 1/BL.

    ``kappa = y'' / (1 + y'^2)^(3/2)`` -- the exact planar-curve curvature of the
    midline, not the small-amplitude approximation ``y''`` -- plus, for a turn, the
    sustained asymmetric offset ``curvature_offset_per_bl * shape(s)``.

    NOTE on the parameterisation, which is where the sign and factor traps live.
    ``y`` is written against the axial coordinate ``u = s`` (in BL), while the rig
    treats the same ``s`` as ARC LENGTH (bone ``j`` has rest length
    ``(s_{j+1} - s_j) * L``).  Two consequences, both deliberate:

    * The tangent angle ``psi = arctan(y')`` satisfies ``dpsi/du = y'/(1+y'^2)``, so
      ``kappa = (dpsi/du) / sqrt(1 + y'^2)`` -- curvature is turning per unit ARC
      LENGTH, not per unit axial coordinate.  The extra ``sqrt`` is not optional;
      ``test_motion.py`` asserts the identity by finite differences.
    * Integrating ``kappa`` over arc length therefore builds a curve whose LENGTH is
      exactly ``L`` and whose curvature profile is the prescribed one, which is what a
      skeleton with fixed bone lengths can actually represent.  The prescribed
      ``y(u)`` is the approximation, not the target; the two agree to
      ``O((dy/du)^2)`` and :func:`tail_tip_amplitude` reports the measured gap.
    """
    _, dy, ddy = lateral_wave(s, t, params, derivatives=True)
    kappa = ddy / (1.0 + dy ** 2) ** 1.5
    if params.curvature_offset_per_bl != 0.0:
        kappa = kappa + params.curvature_offset_per_bl * _turn_offset_shape(s)
    return kappa


def wave_heading(t, params):
    """Tangent angle of the prescribed midline AT THE SNOUT, ``arctan(y'(0, t))``, rad.

    This is the animal's head yaw: the global heading of the whole chain, which no
    integral of curvature can recover because curvature is heading-INDEPENDENT.  Feed
    it to :func:`joint_yaw_angles` as ``heading=`` and the reconstructed midline
    reproduces the prescribed wave in world space; leave it out and the reconstruction
    is the same shape rotated so that the snout always points along -X.
    """
    _, dy, _ = lateral_wave(0.0, t, params, derivatives=True)
    return np.arctan(dy)


def _smoothstep(x):
    """C1 smoothstep on [0, 1], clamped outside."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def escape_curvature(s, t, escape):
    """C-start curvature field ``kappa(s, t)`` in 1/BL.  See :class:`EscapeParams`."""
    s = np.asarray(s, dtype=float)
    t = np.asarray(t, dtype=float)
    shape = _smoothstep(s / escape.head_ramp_s)
    tau = t - escape.travel_delay_s * s
    t1, t2 = escape.stage1_s, escape.stage2_s
    g = np.zeros(np.broadcast(tau, shape).shape, dtype=float)
    tau_b = np.broadcast_to(tau, g.shape)
    stage1 = (tau_b >= 0.0) & (tau_b < t1)
    stage2 = (tau_b >= t1) & (tau_b < t1 + t2)
    coast = tau_b >= t1 + t2
    g = np.where(stage1, _smoothstep(tau_b / t1), g)
    g = np.where(stage2, 1.0 - 2.0 * _smoothstep((tau_b - t1) / t2), g)
    g = np.where(coast, -np.exp(-(tau_b - t1 - t2) / escape.coast_tau_s), g)
    return escape.direction * escape.peak_curvature_per_bl * shape * g


# ---------------------------------------------------------------------------
# Curvature -> joint rotations
# ---------------------------------------------------------------------------
def joint_cell_edges(s_j):
    """Arc-length cell boundaries for a spine, one cell per joint.

    Joint ``j``'s rotation orients the bone from joint ``j`` to joint ``j+1`` (head at
    the joint, tail toward the first child -- ``create_shark_armature.py``).  Its cell
    therefore runs from the midpoint of the previous bone to the midpoint of its own,
    with the first cell opening at ``s = 0`` (the snout) and the last closing at
    ``s = 1`` (the caudal tip), so the cells tile [0, 1] exactly and

        sum_j (integral of kappa over cell j) == total turning of the midline.

    Args:
        s_j: (J,) strictly increasing arc-length fractions in [0, 1].

    Returns:
        (J + 1,) edges.
    """
    s = np.asarray(s_j, dtype=float).reshape(-1)
    if len(s) < 2:
        raise ValueError("need at least 2 spine stations; got %d" % len(s))
    if not np.all(np.diff(s) > 0):
        raise ValueError("spine arc-length fractions must be strictly increasing: %r" % (s,))
    if s[0] < 0.0 or s[-1] > 1.0:
        raise ValueError("spine arc-length fractions must lie in [0, 1]: %r" % (s,))
    mids = 0.5 * (s[:-1] + s[1:])
    return np.concatenate([[0.0], mids, [1.0]])


def joint_yaw_angles(s_j, t, kappa_fn, n_quad=9, heading=None):
    """Local +Z yaw angle of every spine joint at time(s) ``t``, radians.

    ``yaw_j = BODY_AXIS_YAW_SIGN * integral of kappa(s, t) ds over joint j's cell``,
    the cells being :func:`joint_cell_edges`.  The integral is Simpson's rule on
    ``n_quad`` samples per cell (``n_quad`` odd), which is exact enough that the
    reconstructed midline matches the analytic one to the parameterisation error and
    not to quadrature error.

    Args:
        s_j: (J,) spine arc-length fractions.
        t: scalar or (T,) times in seconds.
        kappa_fn: callable ``(s, t) -> kappa`` in 1/BL, broadcasting over both.
        n_quad: odd number of Simpson samples per cell.
        heading: optional scalar or (T,) absolute tangent angle of the midline at
            ``s = 0`` (see :func:`wave_heading`).  Added to the ROOT joint's yaw with
            the same ``BODY_AXIS_YAW_SIGN``, because curvature carries shape but not
            heading.  Default None = the snout keeps pointing along -X.

    Returns:
        (J,) if ``t`` is scalar, else (T, J).
    """
    if n_quad < 3 or n_quad % 2 == 0:
        raise ValueError("n_quad must be an odd integer >= 3; got %r" % (n_quad,))
    edges = joint_cell_edges(s_j)
    lo, hi = edges[:-1], edges[1:]
    n_cells = len(lo)
    # (n_cells, n_quad) sample grid
    u = np.linspace(0.0, 1.0, n_quad)
    grid = lo[:, None] + (hi - lo)[:, None] * u[None, :]
    w = np.ones(n_quad)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    w = w * (1.0 / (3.0 * (n_quad - 1)))          # Simpson weights on a unit interval

    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    # kappa on (T, n_cells, n_quad)
    kappa = kappa_fn(grid[None, :, :], t_arr[:, None, None])
    kappa = np.broadcast_to(kappa, (len(t_arr), n_cells, n_quad))
    integral = np.einsum("tcq,q->tc", kappa, w) * (hi - lo)[None, :]
    out = BODY_AXIS_YAW_SIGN * integral
    if heading is not None:
        out = out.copy()
        out[:, 0] += BODY_AXIS_YAW_SIGN * np.broadcast_to(
            np.asarray(heading, dtype=float).reshape(-1), (len(t_arr),)
        )
    if np.isscalar(t) or np.asarray(t).ndim == 0:
        return out[0]
    return out


def integrate_spine(s_j, yaw, body_length=1.0, origin=(0.0, 0.0, 0.0)):
    """Forward-kinematic midline from local yaw angles: an exact arc-length integrator.

    Mirrors the spine part of ``rig.forward_kinematics`` without importing it: the
    cumulative world yaw at bone ``j`` is ``sum_{k<=j} yaw_k`` and each bone keeps its
    REST length ``(s_{j+1} - s_j) * body_length``, so the reconstructed midline is
    inextensible by construction.  Bones point along -X at rest (snout +X, tail -X),
    hence the ``(-cos, -sin, 0)`` direction.

    Args:
        s_j: (J,) arc-length fractions.
        yaw: (J,) or (T, J) local +Z yaw angles in radians.
        body_length: total centerline length in world units.
        origin: world position of the first spine joint.

    Returns:
        (J, 3) or (T, J, 3) joint positions.
    """
    s = np.asarray(s_j, dtype=float).reshape(-1)
    yaw = np.asarray(yaw, dtype=float)
    single = yaw.ndim == 1
    y2 = yaw.reshape(1, -1) if single else yaw
    if y2.shape[-1] != len(s):
        raise ValueError("yaw last axis (%d) must match len(s_j) (%d)" % (y2.shape[-1], len(s)))
    seg = np.diff(s) * float(body_length)                      # (J-1,)
    cum = np.cumsum(y2, axis=-1)[:, :-1]                       # (T, J-1) bone headings
    step = np.stack(
        [-np.cos(cum) * seg[None, :], -np.sin(cum) * seg[None, :], np.zeros_like(cum)], axis=-1
    )
    pts = np.concatenate(
        [np.zeros((len(y2), 1, 3)), np.cumsum(step, axis=1)], axis=1
    ) + np.asarray(origin, dtype=float).reshape(1, 1, 3)
    return pts[0] if single else pts


# ---------------------------------------------------------------------------
# Fins
# ---------------------------------------------------------------------------
class FinChannel(object):
    """One driven rotation channel of a fin joint.

    Args:
        axis: ``"x"`` (roll / dihedral / lateral lean), ``"y"`` (pitch, i.e. angle of
            attack for a laterally-extended pectoral) or ``"z"`` (yaw / sweep).
        amplitude_deg: peak angle in degrees, for ``source="wave"``.
        phase_lag_deg: how far the fin trails the body, in degrees of the tail-beat
            cycle.  Defined OPERATIONALLY and identically for both sources: the fin is
            driven by what the body was doing ``lag/(360*f)`` seconds ago.  Positive
            lag therefore means the fin peaks LATER -- passive compliance.
        source: ``"wave"`` -> ``amplitude * sin(theta(s, t - lag_s))``, where ``theta``
            is the phase of :func:`lateral_wave` at the fin's station.  Because that
            phase is ``2*pi*(s/lambda - f*t)``, a delay in ``t`` is ``+lag`` inside the
            sine: ``sin(theta(s,t) + lag_rad)``.  Getting this sign backwards makes
            every fin LEAD the body, which is why :func:`phase_report` measures it and
            ``test_motion.py`` asserts on the measurement.
            ``"curvature"`` -> ``gain_deg_per_curvature * kappa(s, t - lag_s)``, a
            passive fin leaning with the local body bend it felt a moment ago.
        gain_deg_per_curvature: degrees per (1/BL) of local curvature, for
            ``source="curvature"``.
    """

    __slots__ = ("axis", "amplitude_deg", "phase_lag_deg", "source", "gain_deg_per_curvature")

    def __init__(
        self,
        axis="z",
        amplitude_deg=0.0,
        phase_lag_deg=0.0,
        source="wave",
        gain_deg_per_curvature=0.0,
    ):
        if source not in ("wave", "curvature"):
            raise ValueError("source must be 'wave' or 'curvature'; got %r" % (source,))
        _axis_vector(axis)  # validate early
        self.axis = axis
        self.amplitude_deg = float(amplitude_deg)
        self.phase_lag_deg = float(phase_lag_deg)
        self.source = source
        self.gain_deg_per_curvature = float(gain_deg_per_curvature)

    def __repr__(self):
        return "FinChannel(axis=%r, amp=%.3g deg, lag=%.3g deg, source=%r)" % (
            self.axis,
            self.amplitude_deg,
            self.phase_lag_deg,
            self.source,
        )


class FinDrive(object):
    """How one fin moves: channels on its root joint, scaled/lagged copies on its tip.

    The tip repeats the root's channels with amplitude ``tip_gain`` and an extra
    ``tip_extra_lag_deg`` of phase, which is the cheapest caricature of passive
    compliance: a membranous fin's distal region follows its base.

    Args:
        channels: sequence of :class:`FinChannel`.
        tip_gain: tip amplitude as a fraction of the root's.
        tip_extra_lag_deg: extra phase lag at the tip, degrees.
    """

    __slots__ = ("channels", "tip_gain", "tip_extra_lag_deg")

    def __init__(self, channels=(), tip_gain=0.6, tip_extra_lag_deg=25.0):
        self.channels = tuple(channels)
        self.tip_gain = float(tip_gain)
        self.tip_extra_lag_deg = float(tip_extra_lag_deg)

    def __repr__(self):
        return "FinDrive(%d channels, tip_gain=%.3g, tip_extra_lag=%.3g deg)" % (
            len(self.channels),
            self.tip_gain,
            self.tip_extra_lag_deg,
        )


# Defaults keyed by the FAMILY of the fin, matched against a fin name by prefix, so
# "pectoral", "pectoral_left" and "pectoral_l" all resolve to the pectoral drive.
#
# ALL of these amplitudes and lags are PLAUSIBLE, NOT MEASURED.  There is no published
# fin kinematics for any hexanchid (scan Q4b), and the general shark literature the
# scan retrieved reports body-wave quantities, not fin angles.  They exist so the rig
# has fin "plasticity" -- fins that live with the body wave rather than ride it
# rigidly -- and every one of them is a parameter the caller can override.
DEFAULT_FIN_DRIVES: Dict[str, FinDrive] = {
    # Pectorals: pitch (angle of attack, about the lateral axis Y) leads the dihedral
    # flap (about the body axis X).  A quarter-cycle lag behind the local body wave is
    # the classic pitch/heave relationship of an oscillating foil.
    "pectoral": FinDrive(
        channels=(
            FinChannel(axis="y", amplitude_deg=7.0, phase_lag_deg=90.0),
            FinChannel(axis="x", amplitude_deg=5.0, phase_lag_deg=150.0),
        ),
        tip_gain=0.7,
        tip_extra_lag_deg=30.0,
    ),
    # Single dorsal, set far posterior over the pelvics.  Passive: it leans laterally
    # with the local body curvature rather than being actuated.
    "dorsal": FinDrive(
        channels=(
            FinChannel(
                axis="x",
                source="curvature",
                gain_deg_per_curvature=2.5,
                phase_lag_deg=20.0,
            ),
        ),
        tip_gain=0.8,
        tip_extra_lag_deg=15.0,
    ),
    "pelvic": FinDrive(
        channels=(FinChannel(axis="z", amplitude_deg=3.0, phase_lag_deg=45.0),),
        tip_gain=0.6,
        tip_extra_lag_deg=25.0,
    ),
    "anal": FinDrive(
        channels=(FinChannel(axis="z", amplitude_deg=3.5, phase_lag_deg=55.0),),
        tip_gain=0.6,
        tip_extra_lag_deg=25.0,
    ),
    # The long upper lobe of the strongly heterocercal caudal trails the peduncle:
    # 75 deg sits mid-bracket of the brief's 60-90 deg of passive compliance.
    "caudal_upper": FinDrive(
        channels=(FinChannel(axis="z", amplitude_deg=14.0, phase_lag_deg=75.0),),
        tip_gain=0.8,
        tip_extra_lag_deg=25.0,
    ),
    "caudal_lower": FinDrive(
        channels=(FinChannel(axis="z", amplitude_deg=8.0, phase_lag_deg=60.0),),
        tip_gain=0.7,
        tip_extra_lag_deg=20.0,
    ),
    # Bare "caudal" for rigs that do not split the lobes.
    "caudal": FinDrive(
        channels=(FinChannel(axis="z", amplitude_deg=12.0, phase_lag_deg=75.0),),
        tip_gain=0.8,
        tip_extra_lag_deg=25.0,
    ),
}


def resolve_fin_drive(fin_name, drives=None):
    """Look up the :class:`FinDrive` for ``fin_name`` by longest matching family prefix.

    Returns ``None`` when no family matches, which leaves that fin at identity.
    """
    table = DEFAULT_FIN_DRIVES if drives is None else drives
    if fin_name in table:
        return table[fin_name]
    best = None
    for family in table:
        if fin_name.startswith(family) and (best is None or len(family) > len(best)):
            best = family
    return table[best] if best is not None else None


# ---------------------------------------------------------------------------
# Fin amplitude scaling: fins move as hard as the BODY is moving, not harder
# ---------------------------------------------------------------------------
#: Cruise's default tail-tip half-amplitude, the denominator every mode's fin
#: amplitude is scaled against.  Alias of :data:`CRUISE_TAIL_AMPLITUDE_BL`, named
#: separately because it is a REFERENCE here, not a parameter of the current clip:
#: overriding a clip's ``tail_amplitude_bl`` must move the fins, so the denominator
#: has to stay pinned to the cruise default.
CRUISE_DEFAULT_TAIL_AMPLITUDE_BL: float = CRUISE_TAIL_AMPLITUDE_BL

#: Clamp on the fin amplitude scale.  A mode ten times cruise's amplitude does not get
#: ten times the fin deflection: fins are membranes on a joint with a finite range,
#: and past roughly 2x the caricature stops meaning anything.
FIN_AMPLITUDE_SCALE_CLAMP: Tuple[float, float] = (0.0, 2.0)


def mode_body_amplitude_bl(mode, params, escape=None, s_j=None, n_t=97):
    """The mode's tail-tip lateral half-amplitude in BL -- what its fins scale against.

    For every wave mode this is just ``params.tail_amplitude_bl``: the amplitude the
    wave was asked for.  ``escape`` has no wave amplitude at all -- its shape is
    prescribed as a curvature field -- so its equivalent amplitude is MEASURED off the
    posed midline: the largest lateral offset of the last spine joint from the snout,
    forward-kinematically, over the clip.  That is the same physical quantity
    ``tail_amplitude_bl`` names for a cruise, which is what makes the ratio in
    :func:`fin_amplitude_scale` comparable across modes.

    Args:
        mode: mode name.
        params: :class:`WaveParams` of the clip.
        escape: :class:`EscapeParams`, required for ``mode="escape"``.
        s_j: spine arc-length fractions; defaults to :func:`default_spine_fractions`.
        n_t: samples used for the escape measurement.
    """
    if mode != "escape":
        return float(params.tail_amplitude_bl)
    escape = escape or EscapeParams()
    s = default_spine_fractions() if s_j is None else np.asarray(s_j, dtype=float)
    t = np.linspace(0.0, escape.duration_s, int(n_t))
    yaw = joint_yaw_angles(s, t, lambda ss, tt: escape_curvature(ss, tt, escape))
    pts = integrate_spine(s, yaw, body_length=1.0)
    return float(np.max(np.abs(pts[:, -1, 1] - pts[:, 0, 1])))


def fin_amplitude_scale(mode, params, escape=None, s_j=None):
    """Scale applied to every ``source="wave"`` fin amplitude of ``mode``.

    ``clamp(mode_body_amplitude_bl / CRUISE_DEFAULT_TAIL_AMPLITUDE_BL)``, clamped to
    :data:`FIN_AMPLITUDE_SCALE_CLAMP`.

    WHY THIS EXISTS.  :data:`DEFAULT_FIN_DRIVES` amplitudes are ABSOLUTE degrees, and
    they were written for a cruise.  Applied unscaled they made a ``rest`` clip -- a
    body wave 1/11th of cruise's, "near-zero articulation" by its own MODE_CONFIG
    description -- flap its pectorals, pelvics and caudal lobes exactly as hard as a
    cruising animal, which reads as a shark holding station while beating its fins.
    Fins are driven BY the body wave here, so their amplitude has to be proportional
    to it.

    ``source="curvature"`` channels (the dorsal) are deliberately NOT scaled: they are
    already proportional, being ``gain_deg_per_curvature * kappa(s, t)`` with the
    mode's own curvature in the multiplication.  Scaling them again would square the
    dependence.
    """
    ref = float(CRUISE_DEFAULT_TAIL_AMPLITUDE_BL)
    if ref <= 0.0:
        return 1.0
    ratio = mode_body_amplitude_bl(mode, params, escape=escape, s_j=s_j) / ref
    lo, hi = FIN_AMPLITUDE_SCALE_CLAMP
    return float(min(max(ratio, lo), hi))


# ---------------------------------------------------------------------------
# Skeleton view
# ---------------------------------------------------------------------------
class MotionSkeleton(object):
    """The minimum a rig must expose for this module to animate it.

    Attributes:
        names: list of J joint names, parents before children.
        parents: (J,) int, -1 at the root.
        spine_names: the serial spine head->tail (schema joint names).
        spine_fractions: (J_spine,) arc-length fractions ``s_j`` in [0, 1].
        fins: ``{fin_name: (root_index, tip_index)}``.
        fps: frames per second for clips built from this skeleton.
        body_length: total centerline length in world units (only affects
            :func:`integrate_spine` diagnostics, never joint angles).
    """

    def __init__(self, names, parents, spine_names, spine_fractions, fins, fps=DEFAULT_FPS,
                 body_length=1.0):
        self.names = list(names)
        self.parents = np.asarray(parents, dtype=int)
        self.spine_names = list(spine_names)
        self.spine_fractions = np.asarray(spine_fractions, dtype=float)
        self.fins = {k: (int(v[0]), int(v[1])) for k, v in dict(fins).items()}
        self.fps = float(fps)
        self.body_length = float(body_length)
        self._index = {n: i for i, n in enumerate(self.names)}

        n = len(self.names)
        if self.parents.shape != (n,):
            raise ValueError("parents must be (%d,); got %r" % (n, (self.parents.shape,)))
        for j, p in enumerate(self.parents):
            if p >= j and p != -1:
                raise ValueError(
                    "joint %r (index %d) has parent %d: parents must precede children"
                    % (self.names[j], j, p)
                )
        if len(self.spine_names) != len(self.spine_fractions):
            raise ValueError("spine_names and spine_fractions disagree in length")
        missing = [nm for nm in self.spine_names if nm not in self._index]
        if missing:
            raise ValueError("spine joints missing from names: %r" % (missing,))
        joint_cell_edges(self.spine_fractions)  # validates monotone, in [0, 1]
        for fin, (r, tip) in self.fins.items():
            if not (0 <= r < n and 0 <= tip < n):
                raise ValueError("fin %r has out-of-range joint indices" % (fin,))
        if self.fps <= 0:
            raise ValueError("fps must be > 0")

    @property
    def num_joints(self):
        return len(self.names)

    def index(self, name):
        return self._index[name]

    @property
    def spine_indices(self):
        """(J_spine,) indices of the spine joints in ``names`` order."""
        return np.asarray([self._index[n] for n in self.spine_names], dtype=int)

    def station_of(self, joint_index):
        """Arc-length fraction of the nearest spine ANCESTOR of ``joint_index``.

        A fin's phase is the body wave's phase where the fin attaches, so a fin root
        (parented to a spine joint) and its tip (parented to the root) both report the
        station of that spine joint.
        """
        spine_pos = {int(i): float(s) for i, s in zip(self.spine_indices, self.spine_fractions)}
        j = int(joint_index)
        guard = 0
        while j >= 0:
            if j in spine_pos:
                return spine_pos[j]
            j = int(self.parents[j])
            guard += 1
            if guard > len(self.names):
                raise ValueError("parent chain does not terminate")
        raise ValueError("joint %d has no spine ancestor" % (joint_index,))

    @classmethod
    def from_skeleton(cls, skeleton, fps=DEFAULT_FPS, body_length=None):
        """Adapt a ``rig.Skeleton`` (duck-typed -- ``rig`` is never imported).

        Requires ``.names``, ``.parents``, ``.kinds``, ``.fractions`` and ``.fins``.
        ``body_length`` defaults to the rest chord of the spine divided by the span of
        its arc-length fractions, which recovers the full centerline length from a
        straight rest pose.
        """
        names = list(skeleton.names)
        kinds = list(skeleton.kinds)
        fractions = np.asarray(skeleton.fractions, dtype=float)
        spine_names = [n for n, k in zip(names, kinds) if k == "spine"]
        spine_idx = [names.index(n) for n in spine_names]
        spine_fracs = fractions[spine_idx]
        if body_length is None:
            body_length = 1.0
            joints = getattr(skeleton, "joints", None)
            if joints is not None and len(spine_idx) >= 2:
                pts = np.asarray(joints, dtype=float)[spine_idx]
                chord = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
                span = float(spine_fracs[-1] - spine_fracs[0])
                if chord > 0 and span > 0:
                    body_length = chord / span
        return cls(
            names=names,
            parents=skeleton.parents,
            spine_names=spine_names,
            spine_fractions=spine_fracs,
            fins=skeleton.fins,
            fps=fps,
            body_length=body_length,
        )

    def __repr__(self):
        return "MotionSkeleton(%d joints, %d spine, %d fins, %.3g fps)" % (
            self.num_joints,
            len(self.spine_names),
            len(self.fins),
            self.fps,
        )


# ---------------------------------------------------------------------------
# Clips
# ---------------------------------------------------------------------------
class Clip(object):
    """An animation clip: ``{"name", "times", "quats"}``.

    Attributes:
        name: clip name.
        times: (T,) seconds, strictly increasing, starting at 0.
        quats: (T, J, 4) local joint rotations in glTF (x, y, z, w) order, unit norm.
        joint_names: list of J names, aligned with axis 1 of ``quats``.
        fps: nominal frame rate.
        loop: True when ``quats[-1] == quats[0]`` and ``times[-1]`` is the loop point.
        meta: diagnostics -- ``"mode"``, ``"params"``, ``"spine_yaw"`` (T, J_spine)
            exact yaw angles, ``"period_s"``.

    ``clip["rotations"]`` is an alias for ``quats`` so the object also satisfies the
    key ``gltf_export.write_skinned_glb`` reads; :meth:`to_animation` returns that as
    a plain dict.
    """

    def __init__(self, name, times, quats, joint_names, fps=DEFAULT_FPS, loop=False, meta=None):
        self.name = str(name)
        self.times = np.asarray(times, dtype=float).reshape(-1)
        self.quats = np.asarray(quats, dtype=float)
        self.joint_names = list(joint_names)
        self.fps = float(fps)
        self.loop = bool(loop)
        self.meta = dict(meta or {})
        t, j = len(self.times), len(self.joint_names)
        if self.quats.shape != (t, j, 4):
            raise ValueError(
                "quats must be (%d, %d, 4); got %r" % (t, j, (self.quats.shape,))
            )
        if t < 2 or not np.all(np.diff(self.times) > 0):
            raise ValueError("times must be strictly increasing with >= 2 samples")

    @property
    def rotations(self):
        """Alias for ``quats``; the key ``gltf_export.write_skinned_glb`` reads."""
        return self.quats

    @property
    def num_frames(self):
        return len(self.times)

    @property
    def duration_s(self):
        return float(self.times[-1] - self.times[0])

    def to_animation(self):
        """Plain dict for ``gltf_export.write_skinned_glb(animations=[...])``."""
        return {"name": self.name, "times": self.times, "rotations": self.quats}

    # -- dict-style access, so the raw {name, times, quats} contract also works ----
    _KEYS = ("name", "times", "quats", "rotations", "joint_names", "fps", "loop", "meta")

    def __getitem__(self, key):
        if key in self._KEYS:
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        return key in self._KEYS

    def keys(self):
        return list(self._KEYS)

    def __repr__(self):
        return "Clip(%r, %d frames, %d joints, %.3g s, loop=%s)" % (
            self.name,
            self.num_frames,
            len(self.joint_names),
            self.duration_s,
            self.loop,
        )


# ---------------------------------------------------------------------------
# Modes.  Names are pose_sampler.MODE_CONFIG's, unchanged, plus "glide".
# ---------------------------------------------------------------------------
MODE_CONFIG: Dict[str, Dict[str, object]] = {
    "cruise": {
        "description": "Steady travelling body wave, amplitude growing toward the tail",
        "implemented": True,
        "loop": True,
        "params": {},
    },
    "turn": {
        "description": "Steady wave plus a sustained asymmetric curvature offset and a bank",
        "implemented": True,
        "loop": True,
        "params": {
            "curvature_offset_per_bl": TURN_CURVATURE_OFFSET_PER_BL,
            "bank_deg": TURN_BANK_DEG,
            "tail_amplitude_bl": 0.9 * CRUISE_TAIL_AMPLITUDE_BL,
        },
    },
    "escape": {
        "description": "C-start: rapid whole-body bend to one side, then a propulsive stroke",
        "implemented": True,
        "loop": False,
        "params": {},
    },
    "rest": {
        "description": "Near-zero articulation; a slow, tiny residual wave",
        "implemented": True,
        "loop": True,
        "params": {"tail_amplitude_bl": 0.010, "frequency_hz": 0.30},
    },
    "glide": {
        "description": "Burst-and-coast: the wave amplitude decays after each kick",
        "implemented": True,
        "loop": True,
        "params": {
            "tail_amplitude_bl": 0.8 * CRUISE_TAIL_AMPLITUDE_BL,
            "frequency_hz": 0.6,
            "gain": "burst_coast",
            "coast_floor": 0.20,
            "coast_power": 3.0,
        },
    },
    "breach": {
        "description": "Extreme dorsoventral flex (pose_sampler name; not implemented)",
        "implemented": False,
        "loop": False,
        "params": {},
    },
    "strike": {
        "description": "Rapid head extension at prey (pose_sampler name; not implemented)",
        "implemented": False,
        "loop": False,
        "params": {},
    },
}

MODES: List[str] = list(MODE_CONFIG)

_NOT_IMPLEMENTED_NOTE = (
    "mode %r is a pose_sampler.MODE_CONFIG name kept for interface parity but not "
    "implemented here. %s It needs a dorsoventral (about Y) bending channel and a "
    "non-periodic time profile, neither of which this module's lateral-wave "
    "formulation provides; and there is no measured sevengill kinematics to shape it "
    "with (see docs/sevengill-canonical-reid/01-evidence-and-answers.md Q4b)."
)


def params_for_mode(mode, overrides=None):
    """:class:`WaveParams` for ``mode``, with ``overrides`` applied last.

    Raises NotImplementedError for ``breach`` and ``strike``.
    """
    if mode not in MODE_CONFIG:
        raise ValueError("unknown mode %r; known modes: %s" % (mode, ", ".join(MODES)))
    cfg = MODE_CONFIG[mode]
    if not cfg["implemented"]:
        raise NotImplementedError(_NOT_IMPLEMENTED_NOTE % (mode, cfg["description"] + "."))
    kwargs = dict(cfg["params"])
    if overrides:
        kwargs.update(overrides)
    return WaveParams(**kwargs)


def _clip_times(duration, fps):
    """``max(2, round(duration*fps)) + 1`` samples spanning [0, duration] inclusive.

    The endpoint is always included.  For a looping clip that endpoint sits exactly ON
    the loop point, so ``times[-1]`` is the period and frame ``T-1`` reproduces frame
    0 -- the glTF convention for a clip played back cyclically, and the reason
    ``duration`` must be a whole number of periods for a loop.
    """
    n = max(2, int(round(duration * fps)))
    return np.linspace(0.0, duration, n + 1)


def make_clip(
    skeleton,
    mode="cruise",
    params=None,
    escape=None,
    fps=None,
    n_periods=1,
    duration=None,
    loop=None,
    fin_drives=None,
    include_head_yaw=True,
    jitter_deg=0.0,
    seed=0,
    name=None,
):
    """Build a :class:`Clip` for one locomotion mode.

    Args:
        skeleton: :class:`MotionSkeleton` (or anything ``MotionSkeleton.from_skeleton``
            accepts -- a ``rig.Skeleton`` is adapted automatically).
        mode: one of :data:`MODES`.  ``breach``/``strike`` raise NotImplementedError.
        params: :class:`WaveParams`; defaults to :func:`params_for_mode`.
        escape: :class:`EscapeParams`, used only by ``mode="escape"``.
        fps: overrides ``skeleton.fps``.
        n_periods: clip length in tail-beat periods (ignored when ``duration`` is set,
            and for ``escape``, whose length comes from its stage durations).
        duration: explicit clip length in seconds.
        loop: force the clip to be a seamless loop (last frame == first).  Defaults to
            the mode's ``MODE_CONFIG["loop"]``.  A loop requires an integer number of
            periods and a periodic gain; ``make_clip`` raises rather than silently
            emitting a clip that pops.
        fin_drives: ``{family_or_fin_name: FinDrive}``; defaults to
            :data:`DEFAULT_FIN_DRIVES`.  Pass ``{}`` to leave all fins at identity.
        include_head_yaw: carry the midline's own heading at ``s = 0`` on the root
            joint (see :func:`wave_heading`).  True -- the default -- makes the posed
            midline reproduce the prescribed wave in world space, and gives the head
            the few degrees of yaw a real swimming shark has.  False pins the snout
            along -X, which is convenient when an external controller owns heading.
        jitter_deg: optional per-joint, per-frame Gaussian yaw jitter in degrees, for
            dataset variation.  0 (the default) makes the clip fully deterministic;
            any non-zero value is drawn from ``np.random.default_rng(seed)`` and is
            applied so that a loop stays seamless (the last frame reuses the first
            frame's draw).
        seed: explicit RNG seed for ``jitter_deg``.
        name: clip name; defaults to ``mode``.

    Returns:
        :class:`Clip` with ``quats`` of shape (T, J, 4), xyzw, unit norm.
    """
    if not isinstance(skeleton, MotionSkeleton):
        skeleton = MotionSkeleton.from_skeleton(skeleton, fps=fps or DEFAULT_FPS)
    if mode not in MODE_CONFIG:
        raise ValueError("unknown mode %r; known modes: %s" % (mode, ", ".join(MODES)))
    cfg = MODE_CONFIG[mode]
    if not cfg["implemented"]:
        raise NotImplementedError(_NOT_IMPLEMENTED_NOTE % (mode, cfg["description"] + "."))

    params = params_for_mode(mode, None) if params is None else params
    escape = escape or EscapeParams()
    fps = float(skeleton.fps if fps is None else fps)
    loop = bool(cfg["loop"]) if loop is None else bool(loop)

    if mode == "escape":
        if loop:
            raise ValueError("mode 'escape' is a one-shot transient and cannot loop")
        span = escape.duration_s if duration is None else float(duration)
    else:
        if duration is not None:
            span = float(duration)
            if loop:
                cycles = span * params.frequency_hz
                if abs(cycles - round(cycles)) > 1e-9:
                    raise ValueError(
                        "a seamless loop needs a whole number of tail-beat periods; "
                        "duration %.6g s at %.6g Hz is %.6g cycles" % (span, params.frequency_hz, cycles)
                    )
        else:
            span = float(n_periods) * params.period_s
    if loop and params.gain == "decay":
        raise ValueError(
            "gain='decay' is monotone in t and cannot produce a seamless loop; use "
            "gain='burst_coast' for a looping glide or loop=False for a transition"
        )
    if span <= 0:
        raise ValueError("clip duration must be > 0")

    times = _clip_times(span, fps)
    s_j = skeleton.spine_fractions

    if mode == "escape":
        kappa_fn = lambda s, t: escape_curvature(s, t, escape)  # noqa: E731
        heading = None          # a C-start starts from a straight, forward-pointing head
    else:
        kappa_fn = lambda s, t: curvature(s, t, params)         # noqa: E731
        heading = wave_heading(times, params) if include_head_yaw else None

    spine_yaw = joint_yaw_angles(s_j, times, kappa_fn, heading=heading)   # (T, J_spine)

    if jitter_deg:
        rng = np.random.default_rng(seed)
        noise = np.deg2rad(jitter_deg) * rng.standard_normal(spine_yaw.shape)
        if loop:
            noise[-1] = noise[0]
        spine_yaw = spine_yaw + noise

    n_joints = skeleton.num_joints
    quats = np.zeros((len(times), n_joints, 4), dtype=float)
    quats[..., 3] = 1.0

    spine_idx = skeleton.spine_indices
    root_idx = int(spine_idx[0])
    bank = math.radians(params.bank_deg) if mode != "escape" else 0.0
    for k, j in enumerate(spine_idx):
        roll = bank if int(j) == root_idx else 0.0
        quats[:, int(j), :] = euler_zxy_quat(yaw=spine_yaw[:, k], roll=roll)

    _apply_fin_drives(skeleton, quats, times, params, escape, mode, fin_drives)

    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
    if loop:
        quats[-1] = quats[0]

    meta = {
        "mode": mode,
        "params": params,
        "escape": escape if mode == "escape" else None,
        "spine_yaw": spine_yaw,
        "spine_fractions": np.asarray(s_j, dtype=float),
        "period_s": params.period_s,
        "body_length": skeleton.body_length,
        "fin_amplitude_scale": fin_amplitude_scale(
            mode, params, escape=escape, s_j=s_j
        ),
    }
    return Clip(
        name=name or mode,
        times=times,
        quats=quats,
        joint_names=skeleton.names,
        fps=fps,
        loop=loop,
        meta=meta,
    )


def _apply_fin_drives(skeleton, quats, times, params, escape, mode, fin_drives):
    """Write fin joint rotations into ``quats`` in place.

    Fin phase is taken from the body wave at the fin's spine station, so every lag in
    :class:`FinChannel` is a lag against a signal that exists and can be measured
    (:func:`phase_report`).  During an escape there is no periodic wave, so
    ``source="wave"`` channels fall back to being driven by the local C-start
    curvature scaled to the same peak amplitude -- the fins still live with the body.

    Every ``source="wave"`` amplitude is multiplied by
    :func:`fin_amplitude_scale`, so a mode whose body barely moves gets fins that
    barely move.  ``source="curvature"`` channels are left alone -- they already carry
    the mode's own curvature (see :func:`fin_amplitude_scale`).
    """
    if fin_drives is not None and len(fin_drives) == 0:
        return
    f = params.frequency_hz
    amp_scale = fin_amplitude_scale(
        mode, params, escape=escape, s_j=skeleton.spine_fractions
    )
    for fin_name, (root_i, tip_i) in skeleton.fins.items():
        drive = resolve_fin_drive(fin_name, fin_drives)
        if drive is None:
            continue
        s_fin = skeleton.station_of(root_i)
        for target, gain, extra_lag in (
            (root_i, 1.0, 0.0),
            (tip_i, drive.tip_gain, drive.tip_extra_lag_deg),
        ):
            angles = {"x": 0.0, "y": 0.0, "z": 0.0}
            for ch in drive.channels:
                lag_deg = ch.phase_lag_deg + extra_lag
                if mode == "escape":
                    lag_s = lag_deg / 360.0 * max(escape.stage1_s + escape.stage2_s, 1e-6)
                    kappa = escape_curvature(s_fin, times - lag_s, escape)
                    if ch.source == "curvature":
                        val = np.deg2rad(ch.gain_deg_per_curvature) * kappa
                    else:
                        peak = max(abs(escape.peak_curvature_per_bl), 1e-9)
                        val = amp_scale * np.deg2rad(ch.amplitude_deg) * kappa / peak
                else:
                    lag_s = lag_deg / 360.0 / f
                    delayed = times - lag_s
                    if ch.source == "curvature":
                        val = np.deg2rad(ch.gain_deg_per_curvature) * curvature(
                            s_fin, delayed, params
                        )
                    else:
                        theta = 2.0 * math.pi * (
                            s_fin / params.wavelength_bl - f * delayed + params.phase
                        )
                        val = (
                            amp_scale
                            * np.deg2rad(ch.amplitude_deg)
                            * _gain(delayed, params)
                            * np.sin(theta)
                        )
                angles[ch.axis] = angles[ch.axis] + gain * np.asarray(val, dtype=float)
            quats[:, int(target), :] = euler_zxy_quat(
                yaw=angles["z"], roll=angles["x"], pitch=angles["y"]
            )


# ---------------------------------------------------------------------------
# Analysis / reporting
# ---------------------------------------------------------------------------
def spine_yaw_angles(clip, skeleton=None):
    """(T, J_spine) local +Z yaw angles of the spine for a clip.

    Prefers the exact profile cached in ``clip.meta["spine_yaw"]``; falls back to
    reading the Z component out of the quaternions, which is exact for a pure-Z
    rotation (every spine joint except a banked root).
    """
    cached = clip.meta.get("spine_yaw") if hasattr(clip, "meta") else None
    if cached is not None:
        return np.asarray(cached, dtype=float)
    if skeleton is None:
        raise ValueError("clip has no cached spine_yaw; pass the skeleton")
    return quat_yaw_z(clip.quats[:, skeleton.spine_indices, :])


def fundamental_phase(signal, times, frequency_hz):
    """Amplitude and phase of the ``frequency_hz`` component of a real signal.

    Fits ``signal ~ mean + amp * sin(2*pi*f*t + phase)`` by projection onto sin/cos --
    a one-bin DFT, robust to the harmonics that the exact curvature formula's
    ``(1 + y'^2)^(3/2)`` denominator injects into an otherwise sinusoidal drive.

    Args:
        signal: (T,) real samples.
        times: (T,) seconds.
        frequency_hz: the frequency to fit.

    Returns:
        ``(amplitude, phase_rad)``; ``phase_rad`` in (-pi, pi].
    """
    x = np.asarray(signal, dtype=float).reshape(-1)
    t = np.asarray(times, dtype=float).reshape(-1)
    if len(x) != len(t):
        raise ValueError("signal and times must have the same length")
    # Drop a duplicated loop-point sample so the projection sees whole cycles only.
    if len(t) > 2 and abs((t[-1] - t[0]) * frequency_hz - round((t[-1] - t[0]) * frequency_hz)) < 1e-9:
        x, t = x[:-1], t[:-1]
    w = 2.0 * math.pi * frequency_hz * t
    x = x - x.mean()
    a = 2.0 * np.mean(x * np.sin(w))
    b = 2.0 * np.mean(x * np.cos(w))
    return float(math.hypot(a, b)), float(math.atan2(b, a))


def phase_report(clip, skeleton, joint_names=None, axis=None):
    """Measured amplitude and phase lag of driven joints against the local body wave.

    For each requested joint this fits the fundamental of one rotation channel and
    reports its LAG behind the local lateral displacement ``sin(theta(s_j, t))`` -- so
    for a ``source="wave"`` :class:`FinChannel` the number that comes back IS the
    ``phase_lag_deg`` that was asked for, which is what makes the configured lag
    testable rather than merely documented.

    The algebra, since the sign traps here are real.  A wave channel emits
    ``A sin(c - w)`` with ``c = 2*pi*(s/lambda + phase) + lag`` and ``w = 2*pi*f*t``;
    :func:`fundamental_phase` fits ``amp*sin(w + phi)``, and ``sin(c - w) =
    sin(w + (pi - c))``, so ``phi = pi - c`` and
    ``lag = pi - phi - 2*pi*(s/lambda + phase)``.

    A ``source="curvature"`` channel is NOT expected to report its configured lag:
    curvature is close to ``-k^2 * y``, so a passive curvature-driven fin reads back
    roughly ``lag + 180 deg``.  That offset is a property of the drive, not an error.

    Args:
        clip: a :class:`Clip` from a periodic mode.
        skeleton: the :class:`MotionSkeleton` it was built from.
        joint_names: which joints to report; default every joint.
        axis: ``"x"``/``"y"``/``"z"`` to force a channel; default is the component with
            the largest excursion.

    Returns:
        ``{joint_name: {"amplitude_deg", "lag_deg", "station", "axis"}}``.
    """
    params = clip.meta["params"]
    f = params.frequency_hz
    names = list(skeleton.names) if joint_names is None else list(joint_names)
    out = {}
    for nm in names:
        j = skeleton.index(nm)
        q = clip.quats[:, j, :]
        if axis is None:
            comp = int(np.argmax(np.max(np.abs(q[:, :3]), axis=0)))
        else:
            comp = "xyz".index(axis.lower())
        angle = 2.0 * np.arcsin(np.clip(q[:, comp], -1.0, 1.0))
        amp, phi = fundamental_phase(angle, clip.times, f)
        s_fin = skeleton.station_of(j)
        ref = 2.0 * math.pi * (s_fin / params.wavelength_bl + params.phase)
        lag = (math.pi - phi - ref) % (2.0 * math.pi)
        out[nm] = {
            "amplitude_deg": math.degrees(amp),
            "lag_deg": math.degrees(lag),
            "station": s_fin,
            "axis": "xyz"[comp],
        }
    return out


def dct_energy_fraction(clip, skeleton=None, n_modes=(4, SCHEMA.NUM_BENDING_MODES)):
    """Fraction of spine bending energy captured by the schema's DCT bending basis.

    The per-frame tangent-angle profile over the 12 spine SEGMENTS is the cumulative
    sum of the local yaw angles (bone ``j``'s heading is ``sum_{k<=j} yaw_k``).  That
    profile is projected with ``skeleton_sevengill.project_to_bending_modes``, which
    removes the profile mean (a global heading, not a bend) before projecting onto the
    orthonormal DCT-II basis -- so this function does not re-derive the basis, it uses
    the schema's.

    Energy fraction ``= sum_frames ||coeffs||^2 / sum_frames ||centred profile||^2``.

    Args:
        clip: a :class:`Clip`.
        skeleton: only needed if the clip has no cached ``spine_yaw``.
        n_modes: iterable of mode counts to report.

    Returns:
        ``{n: fraction}`` plus key ``"total_energy"``.
    """
    import torch  # local: the schema module is torch-based, this module is not

    yaw = spine_yaw_angles(clip, skeleton)                       # (T, J_spine)
    if yaw.shape[1] != NUM_SPINE_JOINTS:
        raise ValueError(
            "the DCT basis is defined for the schema's %d-joint spine; got %d"
            % (NUM_SPINE_JOINTS, yaw.shape[1])
        )
    profile = np.cumsum(yaw, axis=1)[:, :-1]                     # (T, 12) bone headings
    ang = torch.as_tensor(profile, dtype=torch.float32)
    centred = ang - ang.mean(dim=-1, keepdim=True)
    total = float((centred ** 2).sum())
    out = {"total_energy": total}
    for n in n_modes:
        basis = SCHEMA.build_bending_basis(n_modes=int(n))
        coeffs = SCHEMA.project_to_bending_modes(ang, basis=basis)
        out[int(n)] = 1.0 if total <= 0.0 else float((coeffs ** 2).sum()) / total
    return out


def tail_tip_amplitude(params, s_j=None, body_length=1.0, n_t=128):
    """Tail-tip lateral half-amplitude, analytic and forward-kinematic, in BL.

    Two numbers, because they answer different questions:

    * ``analytic_bl`` -- ``max_t |y(1, t)|``, i.e. ``A(1)`` for a steady gain.  This is
      the prescribed quantity.
    * ``fk_bl`` -- the peak |Y| of the actual POSED tail tip: the midline reconstructed
      by :func:`integrate_spine` from the joint angles (arc length exactly preserved),
      carried past the last spine station to ``s = 1`` along the last bone, which is
      what linear blend skinning does to a caudal tip bound to that bone.  This is the
      excursion a camera would measure, and the one the 0.10-0.20 BL literature
      bracket is stated in.

    ``fk_over_analytic`` is the ``O((dy/ds)^2)`` cost of writing ``y`` against the
    axial coordinate while the rig integrates against arc length (see
    :func:`curvature`); it is < 1 because an inextensible midline of the same
    curvature does not reach as far sideways as the prescribed graph.

    Args:
        params: :class:`WaveParams`.
        s_j: (J,) spine arc-length fractions; the schema default is used if None.
        body_length: world units; amplitudes are divided back out, so this only
            affects numerical conditioning.
        n_t: samples over one tail-beat period.
    """
    if s_j is None:
        s_j = default_spine_fractions()
    s_j = np.asarray(s_j, dtype=float)
    times = np.linspace(0.0, params.period_s, n_t, endpoint=False)
    analytic = float(np.max(np.abs(lateral_wave(1.0, times, params))))
    heading = wave_heading(times, params)
    yaw = joint_yaw_angles(
        s_j, times, lambda s, t: curvature(s, t, params), heading=heading
    )
    y_head = lateral_wave(s_j[0], times, params) * body_length
    pts = np.stack(
        [integrate_spine(s_j, yaw[i], body_length, origin=(0.0, y_head[i], 0.0))
         for i in range(len(times))]
    )
    # Carry the midline from the last spine station out to s = 1 along the last bone.
    tail_dir = pts[:, -1, :] - pts[:, -2, :]
    tail_dir = tail_dir / np.linalg.norm(tail_dir, axis=-1, keepdims=True)
    tip = pts[:, -1, :] + tail_dir * (1.0 - s_j[-1]) * body_length
    fk = float(np.max(np.abs(tip[:, 1])) / body_length)
    return {
        "analytic_bl": analytic,
        "fk_bl": fk,
        "fk_over_analytic": fk / analytic if analytic > 0 else float("nan"),
        "last_station": float(s_j[-1]),
        "fk_last_joint_bl": float(np.max(np.abs(pts[:, -1, 1])) / body_length),
    }


def escape_total_turn_deg(escape=None, s_j=None, n_t=241):
    """Largest TOTAL turning of the midline during a C-start, degrees.

    ``max over t of |sum_j yaw_j(t)|`` -- the net heading change from snout to caudal
    tip, which is the quantity :data:`ESCAPE_MAX_TOTAL_TURN_DEG` caps and the one
    published fish C-starts actually report.  180 deg is a U; 360 deg is a closed
    loop with the tail through the head.
    """
    escape = escape or EscapeParams()
    s = default_spine_fractions() if s_j is None else np.asarray(s_j, dtype=float)
    t = np.linspace(0.0, escape.duration_s, int(n_t))
    yaw = joint_yaw_angles(s, t, lambda ss, tt: escape_curvature(ss, tt, escape))
    return float(np.degrees(np.abs(yaw.sum(axis=1)).max()))


def escape_closure_bl(escape=None, s_j=None, n_t=241):
    """Smallest distance from the LAST spine joint back to the first, in BL.

    The self-contact diagnostic behind :data:`ESCAPE_MAX_TOTAL_TURN_DEG`: it is how
    close the C-start brings the caudal axis to the snout.  It ignores body thickness
    and the caudal lobe carried rigidly past the last joint, so it is an OPTIMISTIC
    bound -- the real surfaces touch well before this reaches zero.
    """
    escape = escape or EscapeParams()
    s = default_spine_fractions() if s_j is None else np.asarray(s_j, dtype=float)
    t = np.linspace(0.0, escape.duration_s, int(n_t))
    yaw = joint_yaw_angles(s, t, lambda ss, tt: escape_curvature(ss, tt, escape))
    pts = integrate_spine(s, yaw, body_length=1.0)
    return float(np.linalg.norm(pts[:, -1, :] - pts[:, 0, :], axis=1).min())


def peak_curvature(kappa_fn, times, s_range=(0.0, 1.0), n_s=201):
    """Peak ``|kappa|`` in 1/BL over an arc-length window and a set of times.

    ``s_range`` matters: the very tail (``s -> 1``) of any travelling-wave envelope is
    the most curved point on the animal, so a whole-body peak is dominated by the
    caudal fin membrane rather than by the vertebral column.  Comparisons between
    modes are more informative over the trunk, e.g. ``s_range=(0.1, 0.9)``.
    """
    s = np.linspace(s_range[0], s_range[1], n_s)
    t = np.asarray(times, dtype=float).reshape(-1)
    return float(np.max(np.abs(kappa_fn(s[None, :], t[:, None]))))


def implied_skin_strain(params, half_width_bl=TRUNK_HALF_WIDTH_BL,
                        stations=(0.25, 0.50, 0.75), n_t=128):
    """Peak longitudinal skin strain implied by the body wave, dimensionless.

    Beam bending about the vertebral neutral axis: ``strain = r * kappa``, with ``r``
    the distance from the axis to the skin.  Donley & Shadwick 2003 (JEB 206) showed
    for a leopard shark that this simple model reproduces measured red-muscle strain
    at all three longitudinal stations, and that strain is in phase with local midline
    curvature -- which is exactly why a curvature-parameterised pose model is also a
    strain model.

    The returned numbers are a **bound derived from published muscle strain**, not a
    published skin-strain measurement: red muscle is superficial, skin is farther from
    the neutral axis, so skin strain >= muscle strain.  Compare against
    :data:`SKIN_STRAIN_BRACKET` and against Donley & Shadwick's three probes at
    ~1.0 BL/s (anterior +/-3.9%, mid-body +/-6.6%, posterior +/-4.8%).

    Note the expected DISAGREEMENT at the posterior station.  A leopard shark peaks
    mid-body -- a subcarangiform signature -- while a monotonically growing
    anguilliform envelope keeps climbing to the tail.  A posterior value above the
    leopard shark's is the model saying "more anguilliform than a *Triakis*", which is
    the hedged morphological expectation the scan permits; it is not a fitted result.

    Returns:
        ``{"anterior", "mid_body", "posterior", "stations", "half_width_bl"}`` with
        strains as fractions.  Keys map to ``stations`` in order.
    """
    t = np.linspace(0.0, params.period_s, n_t, endpoint=False)
    vals = [float(np.max(np.abs(curvature(s, t, params))) * half_width_bl) for s in stations]
    out = {"stations": tuple(float(s) for s in stations), "half_width_bl": float(half_width_bl)}
    for key, v in zip(("anterior", "mid_body", "posterior"), vals):
        out[key] = v
    out["by_station"] = dict(zip(out["stations"], vals))
    return out


def default_spine_fractions(precaudal_fraction=0.78):
    """The 13 schema spine stations as arc-length fractions of the full centerline.

    Mirrors ``rig.spine_arclength_fractions`` (same numbers, same reasoning) so this
    module can be exercised and tested with no rig present.  The seven trunk stations
    come from ``skeleton_sevengill.MIDLINE_AXIS_FRACTIONS``; the three head stations,
    the precaudal pit and the two caudal-axis stations are the prototype's declared
    values [UNVERIFIED -- no measured *Notorynchus cepedianus* proportions exist].
    """
    pf = float(precaudal_fraction)
    head = {"spine_00_cranium": 0.030, "spine_01_branchial_1": 0.065,
            "spine_02_branchial_7": 0.100}
    caudal = {"spine_11_caudal_axis_1": 0.40, "spine_12_caudal_axis_2": 0.80}
    midline = tuple(SCHEMA.MIDLINE_AXIS_FRACTIONS)
    out = []
    for name in SPINE_JOINTS:
        if name in head:
            out.append(head[name] * pf)
        elif name in caudal:
            out.append(pf + caudal[name] * (1.0 - pf))
        elif name == "spine_10_precaudal":
            out.append(pf)
        else:
            out.append(midline[SPINE_JOINTS.index(name) - 3] * pf)
    return np.asarray(out, dtype=float)


def kinematics_report(params=None, s_j=None):
    """Human-readable summary of what the shipped defaults imply.  Used by ``__main__``.

    Returns a list of ``(label, value_string)`` pairs; nothing is printed here.
    """
    params = params_for_mode("cruise") if params is None else params
    s_j = default_spine_fractions() if s_j is None else np.asarray(s_j, dtype=float)
    amp = tail_tip_amplitude(params, s_j, body_length=1.0)
    strain = implied_skin_strain(params)
    t = np.linspace(0.0, params.period_s, 128, endpoint=False)
    esc = EscapeParams()
    t_e = np.linspace(0.0, esc.duration_s, 64)
    return [
        ("tail-beat frequency", "%.2f Hz  (bracket %.1f-%.1f)"
         % ((params.frequency_hz,) + CRUISE_TAILBEAT_HZ_BRACKET)),
        ("propulsive wavelength", "%.2f BL  (bracket %.1f-%.1f)"
         % ((params.wavelength_bl,) + CRUISE_WAVELENGTH_BL_BRACKET)),
        ("tail-tip amplitude, prescribed", "%.3f BL  (bracket %.2f-%.2f)"
         % ((amp["analytic_bl"],) + CRUISE_TAIL_AMPLITUDE_BL_BRACKET)),
        ("tail-tip amplitude, posed (FK)", "%.3f BL  (%.0f%% of prescribed)"
         % (amp["fk_bl"], 100.0 * amp["fk_over_analytic"])),
        ("implied skin strain ant/mid/post", "%.1f%% / %.1f%% / %.1f%%  (bracket %.1f-%.1f%%)"
         % (100 * strain["anterior"], 100 * strain["mid_body"], 100 * strain["posterior"],
            100 * SKIN_STRAIN_BRACKET[0], 100 * SKIN_STRAIN_BRACKET[1])),
        ("peak curvature, cruise trunk", "%.2f /BL"
         % peak_curvature(lambda s, tt: curvature(s, tt, params), t, (0.1, 0.9))),
        ("peak curvature, escape trunk", "%.2f /BL"
         % peak_curvature(lambda s, tt: escape_curvature(s, tt, esc), t_e, (0.1, 0.9))),
    ]


if __name__ == "__main__":  # pragma: no cover - a convenience report, not an API
    print("sevengill swimming kinematics -- shipped cruise defaults")
    print("(plausible, not measured: no hexanchid kinematics has ever been published)")
    for label, value in kinematics_report():
        print("  %-34s %s" % (label, value))
