"""Behavioural tests for ``motion.py``.

Fast (no meshes, no GLB, no torch autograd) and behavioural: every test asserts on a
quantity the brief names -- wave/curvature consistency, loop seamlessness, the tail-tip
amplitude bracket, escape vs cruise, configured fin phase lags, DCT energy, quaternion
shape and norm -- rather than on internal structure.

``rig.py`` is deliberately NOT imported; the fixtures below are the whole input
contract, which is the point of stating one.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import motion as M  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures: the smallest thing that satisfies the input contract.
# ---------------------------------------------------------------------------
FIN_LAYOUT = (
    ("pectoral_left", "spine_02_branchial_7"),
    ("pectoral_right", "spine_02_branchial_7"),
    ("dorsal", "spine_08_trunk_06"),
    ("pelvic", "spine_08_trunk_06"),
    ("anal", "spine_09_trunk_07"),
    ("caudal_upper", "spine_11_caudal_axis_1"),
    ("caudal_lower", "spine_10_precaudal"),
)


def build_skeleton(fps=60.0, body_length=2.4):
    """A 13-joint schema spine plus two joints per fin -- the anatomy the brief lists.

    Seven gill slits are a surface feature, not joints; the single dorsal sits over the
    pelvics (``spine_08_trunk_06``, s ~ 0.585) and there is no second dorsal; the
    strongly heterocercal caudal is split into an upper lobe carried by the vertebral
    axis and a weak lower lobe hanging off the precaudal joint.
    """
    names = list(M.SPINE_JOINTS)
    parents = [-1] + list(range(len(M.SPINE_JOINTS) - 1))
    fins = {}
    for fin, parent in FIN_LAYOUT:
        root = len(names)
        names.append("%s_fin_root" % fin)
        parents.append(names.index(parent))
        names.append("%s_fin_tip" % fin)
        parents.append(root)
        fins[fin] = (root, root + 1)
    return M.MotionSkeleton(
        names=names,
        parents=parents,
        spine_names=M.SPINE_JOINTS,
        spine_fractions=M.default_spine_fractions(),
        fins=fins,
        fps=fps,
        body_length=body_length,
    )


class FakeRigSkeleton(object):
    """Duck-type of ``rig.Skeleton``, to test ``MotionSkeleton.from_skeleton``."""

    def __init__(self):
        sk = build_skeleton()
        self.names = sk.names
        self.parents = sk.parents
        self.kinds = ["spine"] * len(M.SPINE_JOINTS) + ["fin_root", "fin_tip"] * len(sk.fins)
        frac = np.full(len(self.names), np.nan)
        frac[: len(M.SPINE_JOINTS)] = sk.spine_fractions
        self.fractions = frac
        self.fins = sk.fins
        # Straight rest pose: snout +X, tail -X, spanning 3.0 world units of centerline.
        self.joints = np.zeros((len(self.names), 3))
        self.joints[: len(M.SPINE_JOINTS), 0] = 1.5 - 3.0 * sk.spine_fractions


@pytest.fixture(scope="module")
def sk():
    return build_skeleton()


UNIFORM_S = np.linspace(0.0, 1.0, 101)


def central_diff(f, h):
    """Second-order central difference at interior points.

    Used instead of ``np.gradient`` because numpy's default ``edge_order=1`` makes the
    two boundary samples first-order accurate, which swamps the interior error and
    would force a tolerance loose enough to hide a real mistake.
    """
    return (f[2:] - f[:-2]) / (2.0 * h)


# ---------------------------------------------------------------------------
# The wave and its curvature
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("envelope", ["quadratic", "exponential"])
def test_amplitude_envelope_endpoints_and_monotonicity(envelope):
    p = M.WaveParams(envelope=envelope)
    a = M.amplitude_envelope(UNIFORM_S, p)
    assert a[0] == pytest.approx(p.tail_amplitude_bl * p.head_amplitude_ratio, rel=1e-12)
    assert a[-1] == pytest.approx(p.tail_amplitude_bl, rel=1e-12)
    assert np.all(np.diff(a) >= -1e-15), "amplitude must grow toward the tail"


@pytest.mark.parametrize("envelope", ["quadratic", "exponential"])
def test_envelope_derivatives_match_finite_differences(envelope):
    p = M.WaveParams(envelope=envelope)
    s = np.linspace(0.05, 0.95, 400)
    a, da, dda = M.amplitude_envelope(s, p, derivatives=True)
    h = s[1] - s[0]
    assert np.allclose(central_diff(a, h), da[1:-1], rtol=1e-5, atol=1e-9)
    assert np.allclose(central_diff(da, h), dda[1:-1], rtol=1e-5, atol=1e-9)


def test_wave_s_derivatives_match_finite_differences():
    """The analytic y' and y'' are what the curvature formula consumes; check them."""
    p = M.WaveParams()
    s = np.linspace(0.02, 0.98, 2001)
    h = s[1] - s[0]
    for t in (0.0, 0.137, 0.61):
        y, dy, ddy = M.lateral_wave(s, t, p, derivatives=True)
        assert np.allclose(y, M.lateral_wave(s, t, p), rtol=0, atol=1e-15)
        assert np.max(np.abs(central_diff(y, h) - dy[1:-1])) < 5e-6
        assert np.max(np.abs(central_diff(dy, h) - ddy[1:-1])) < 1e-4


def test_curvature_is_turning_per_unit_arc_length_not_per_axial_coordinate():
    """The identity behind curvature-to-joints, including the factor easy to drop.

    ``psi = arctan(y')`` gives ``dpsi/ds = y''/(1+y'^2)``, but curvature is turning per
    unit ARC length, so ``kappa = (dpsi/ds) / sqrt(1+y'^2) = y''/(1+y'^2)^(3/2)``.
    Dropping the extra sqrt is a silent double-digit curvature error at cruise amplitude.
    """
    p = M.WaveParams()
    s = np.linspace(0.02, 0.98, 4001)
    h = s[1] - s[0]
    _, dy, _ = M.lateral_wave(s, 0.21, p, derivatives=True)
    psi = np.arctan(dy)
    kappa = M.curvature(s, 0.21, p)
    expected = central_diff(psi, h) / np.sqrt(1.0 + dy[1:-1] ** 2)
    assert np.max(np.abs(expected - kappa[1:-1])) < 1e-4
    # And the naive version really is different, i.e. this test has teeth.
    naive = central_diff(psi, h)
    assert np.max(np.abs(naive - kappa[1:-1])) > 0.05 * np.max(np.abs(kappa))


def test_joint_cells_tile_the_body_and_yaw_sums_to_total_turning():
    """Sum of per-joint yaw == -(total tangent-angle change), by construction."""
    p = M.WaveParams()
    s_j = M.default_spine_fractions()
    edges = M.joint_cell_edges(s_j)
    assert edges[0] == 0.0 and edges[-1] == 1.0
    assert np.all(np.diff(edges) > 0)
    assert len(edges) == len(s_j) + 1

    t = 0.33
    yaw = M.joint_yaw_angles(s_j, t, lambda ss, tt: M.curvature(ss, tt, p))
    assert yaw.shape == (len(s_j),)
    # Summed per-cell Simpson integrals == an independent fine trapezoid over [0, 1].
    fine = np.linspace(0.0, 1.0, 20001)
    total = np.trapezoid(M.curvature(fine, t, p), fine)
    assert yaw.sum() == pytest.approx(M.BODY_AXIS_YAW_SIGN * total, abs=1e-5)

    # The heading term lands on the root joint and nowhere else.
    with_head = M.joint_yaw_angles(
        s_j, t, lambda ss, tt: M.curvature(ss, tt, p), heading=M.wave_heading(t, p)
    )
    assert np.allclose(with_head[1:], yaw[1:], atol=0)
    assert with_head[0] - yaw[0] == pytest.approx(
        M.BODY_AXIS_YAW_SIGN * float(M.wave_heading(t, p))
    )


def test_joint_cell_edges_rejects_a_non_monotone_spine():
    with pytest.raises(ValueError):
        M.joint_cell_edges([0.1, 0.3, 0.2, 0.9])
    with pytest.raises(ValueError):
        M.joint_cell_edges([0.1, 1.4])


def test_forward_kinematics_reproduces_the_prescribed_wave():
    """The load-bearing consistency test: y(s,t) -> kappa -> joint yaw -> FK -> y(s,t).

    On a finely discretised straight spine and at small amplitude (where writing y
    against the axial coordinate is a good approximation to writing it against arc
    length), the reconstructed midline must reproduce the analytic wave to a small
    fraction of its own amplitude.
    """
    p = M.WaveParams(tail_amplitude_bl=0.02)
    s = UNIFORM_S
    for t in (0.0, 0.19, 0.44):
        yaw = M.joint_yaw_angles(
            s, t, lambda ss, tt: M.curvature(ss, tt, p), heading=M.wave_heading(t, p)
        )
        y0 = float(M.lateral_wave(s[0], t, p))
        pts = M.integrate_spine(s, yaw, body_length=1.0, origin=(0.0, y0, 0.0))
        assert pts.shape == (len(s), 3)
        expected = M.lateral_wave(s, t, p)
        err = np.max(np.abs(pts[:, 1] - expected))
        assert err < 0.03 * p.tail_amplitude_bl, "FK lateral error %.4g" % err
        # Without the heading term the whole reconstruction is rotated by psi(0) --
        # a ~100%-of-amplitude error here. Guard against anyone dropping it.
        no_head = M.integrate_spine(
            s,
            M.joint_yaw_angles(s, t, lambda ss, tt: M.curvature(ss, tt, p)),
            1.0,
            origin=(0.0, y0, 0.0),
        )
        if abs(float(M.wave_heading(t, p))) > 1e-3:
            assert np.max(np.abs(no_head[:, 1] - expected)) > 10.0 * err
        # x must march monotonically toward -X: snout +X, tail -X.
        assert np.all(np.diff(pts[:, 0]) < 0)
        # Arc length is preserved exactly by the integrator.
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        assert np.allclose(seg, np.diff(s), rtol=1e-12)


def test_forward_kinematics_at_cruise_amplitude_is_still_close():
    """At the shipped cruise amplitude the parameterisation error is bounded and known."""
    p = M.params_for_mode("cruise")
    s = UNIFORM_S
    errs = []
    for t in np.linspace(0.0, p.period_s, 9):
        yaw = M.joint_yaw_angles(
            s, t, lambda ss, tt: M.curvature(ss, tt, p), heading=M.wave_heading(t, p)
        )
        y0 = float(M.lateral_wave(s[0], t, p))
        pts = M.integrate_spine(s, yaw, 1.0, origin=(0.0, y0, 0.0))
        errs.append(np.max(np.abs(pts[:, 1] - M.lateral_wave(s, t, p))))
    # The measured O((dy/ds)^2) gap at the shipped amplitude is ~18% of A(1).
    assert max(errs) < 0.25 * p.tail_amplitude_bl


def test_zero_amplitude_gives_a_straight_spine():
    p = M.WaveParams(tail_amplitude_bl=0.0)
    s = M.default_spine_fractions()
    yaw = M.joint_yaw_angles(s, 0.4, lambda ss, tt: M.curvature(ss, tt, p))
    assert np.allclose(yaw, 0.0, atol=1e-15)
    pts = M.integrate_spine(s, yaw, body_length=2.0)
    assert np.allclose(pts[:, 1:], 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Literature brackets
# ---------------------------------------------------------------------------
def test_shipped_defaults_sit_inside_their_literature_brackets():
    lo, hi = M.CRUISE_TAILBEAT_HZ_BRACKET
    assert lo <= M.CRUISE_TAILBEAT_HZ <= hi
    lo, hi = M.CRUISE_WAVELENGTH_BL_BRACKET
    assert lo <= M.CRUISE_WAVELENGTH_BL <= hi
    lo, hi = M.CRUISE_TAIL_AMPLITUDE_BL_BRACKET
    assert lo <= M.CRUISE_TAIL_AMPLITUDE_BL <= hi
    lo, hi = M.HEAD_AMPLITUDE_RATIO_BRACKET
    assert lo <= M.HEAD_AMPLITUDE_RATIO <= hi


def test_cruise_tail_tip_amplitude_is_in_the_published_bracket():
    """0.10-0.20 BL half-amplitude at the tail tip for steady cruise."""
    p = M.params_for_mode("cruise")
    lo, hi = M.CRUISE_TAIL_AMPLITUDE_BL_BRACKET
    amp = M.tail_tip_amplitude(p, M.default_spine_fractions(), body_length=2.4)
    assert lo <= amp["analytic_bl"] <= hi
    assert lo <= amp["fk_bl"] <= hi
    # The two parameterisations must agree to within the O((dy/ds)^2) gap, not diverge.
    # An inextensible midline of the same curvature does not reach as far sideways
    # as the prescribed graph, so the ratio is < 1 -- but only by that gap.
    assert 0.85 <= amp["fk_over_analytic"] < 1.0
    assert amp["fk_last_joint_bl"] < amp["fk_bl"]


def test_implied_skin_strain_lands_in_the_derived_bracket():
    """Donley & Shadwick 2003 + the beam argument: skin strain >= muscle strain.

    Mid-body is the station their probe and the scan's derived 10-12% figure refer to.
    The posterior station is EXPECTED to exceed the leopard shark's 4.8%: a
    monotonically growing anguilliform envelope peaks at the tail, a subcarangiform
    one peaks mid-body.
    """
    p = M.params_for_mode("cruise")
    strain = M.implied_skin_strain(p)
    lo, hi = M.SKIN_STRAIN_BRACKET
    assert lo <= strain["anterior"] <= hi
    assert lo <= strain["mid_body"] <= hi
    assert strain["anterior"] < strain["mid_body"] < strain["posterior"]


# ---------------------------------------------------------------------------
# Clips: shape, norm, loops
# ---------------------------------------------------------------------------
IMPLEMENTED = [m for m in M.MODES if M.MODE_CONFIG[m]["implemented"]]
LOOPING = [m for m in IMPLEMENTED if M.MODE_CONFIG[m]["loop"]]


@pytest.mark.parametrize("mode", IMPLEMENTED)
def test_clip_shapes_and_unit_quaternions(mode, sk):
    clip = M.make_clip(sk, mode)
    t = clip.num_frames
    assert clip.quats.shape == (t, sk.num_joints, 4)
    assert clip.times.shape == (t,)
    norms = np.linalg.norm(clip.quats, axis=-1)
    assert np.max(np.abs(norms - 1.0)) < 1e-12
    assert clip.times[0] == 0.0
    assert np.all(np.diff(clip.times) > 0)
    assert clip.name == mode


@pytest.mark.parametrize("mode", LOOPING)
def test_looping_clips_are_seamless(mode, sk):
    """Last frame == first frame, and the clip spans a whole number of periods."""
    clip = M.make_clip(sk, mode, n_periods=2)
    assert clip.loop
    assert np.array_equal(clip.quats[-1], clip.quats[0])
    assert clip.times[0] == 0.0
    period = clip.meta["params"].period_s
    assert clip.times[-1] == pytest.approx(2.0 * period, rel=1e-12)


def test_a_loop_requires_a_whole_number_of_periods(sk):
    with pytest.raises(ValueError, match="whole number of tail-beat periods"):
        M.make_clip(sk, "cruise", duration=0.7, loop=True)


def test_escape_refuses_to_loop(sk):
    with pytest.raises(ValueError, match="cannot loop"):
        M.make_clip(sk, "escape", loop=True)


def test_monotone_decay_gain_refuses_to_loop(sk):
    p = M.params_for_mode("glide", {"gain": "decay", "decay_tau_s": 0.8})
    with pytest.raises(ValueError, match="cannot produce a seamless loop"):
        M.make_clip(sk, "glide", params=p, loop=True)
    clip = M.make_clip(sk, "glide", params=p, loop=False, duration=2.0)
    assert not clip.loop
    assert not np.array_equal(clip.quats[-1], clip.quats[0])


@pytest.mark.parametrize("mode", ["breach", "strike"])
def test_pose_sampler_names_kept_but_unimplemented(mode, sk):
    assert mode in M.MODE_CONFIG and not M.MODE_CONFIG[mode]["implemented"]
    with pytest.raises(NotImplementedError):
        M.make_clip(sk, mode)
    with pytest.raises(NotImplementedError):
        M.params_for_mode(mode)


def test_mode_names_match_pose_sampler(sk):
    """The six pose_sampler.MODE_CONFIG names survive verbatim, plus 'glide'."""
    for name in ("cruise", "turn", "breach", "strike", "rest", "escape"):
        assert name in M.MODE_CONFIG
    assert "glide" in M.MODE_CONFIG


def test_unknown_mode_raises(sk):
    with pytest.raises(ValueError, match="unknown mode"):
        M.make_clip(sk, "backflip")


# ---------------------------------------------------------------------------
# Mode behaviour
# ---------------------------------------------------------------------------
def _net_turning(clip):
    """Head-to-tail heading change per frame, radians: the C-vs-S discriminator."""
    return np.abs(clip.meta["spine_yaw"].sum(axis=1))


def test_escape_curvature_far_exceeds_cruise(sk):
    """A C-start bends the WHOLE body one way at once; a cruise wave cancels itself.

    Whole-body PEAK curvature is deliberately not the assertion: the most curved point
    on a cruising anguilliform swimmer is its own tail tip, and it stays comparable.
    The two quantities that do separate the modes are net turning of the midline and
    curvature in the anterior-to-mid trunk, which barely bends during a cruise.
    """
    cruise = M.make_clip(sk, "cruise", n_periods=2)
    escape = M.make_clip(sk, "escape")

    # ~5x on the defaults (176.4 deg against 35.5).  It was ~8x before
    # ESCAPE_MAX_TOTAL_TURN_DEG capped the C-start; the cap roughly halved the
    # escape's turning and the discriminator still holds by a wide margin.
    assert _net_turning(escape).max() > 4.0 * _net_turning(cruise).max()
    # At least a hard C, and no more than the self-contact cap (see
    # ESCAPE_MAX_TOTAL_TURN_DEG and test_escape_default_respects_the_self_contact_cap).
    turn_deg = math.degrees(_net_turning(escape).max())
    assert 0.9 * M.ESCAPE_MAX_TOTAL_TURN_DEG < turn_deg <= M.ESCAPE_MAX_TOTAL_TURN_DEG

    p = cruise.meta["params"]
    e = escape.meta["escape"]
    # The head and ANTERIOR trunk: the stretch of body a cruise wave barely bends and
    # a C-start bends as hard as everywhere else.  Extending this window past ~0.5
    # walks into the growing tail of the cruise envelope and the ratio collapses --
    # which is the docstring's point, not a weakness of the test.
    anterior = (0.0, 0.35)
    k_cruise = M.peak_curvature(
        lambda s, t: M.curvature(s, t, p), cruise.times, anterior
    )
    k_escape = M.peak_curvature(
        lambda s, t: M.escape_curvature(s, t, e), escape.times, anterior
    )
    assert k_escape > 2.0 * k_cruise

    yaw_c, yaw_e = cruise.meta["spine_yaw"], escape.meta["spine_yaw"]
    fore = sk.spine_fractions <= 0.35          # head + anterior trunk, same window
    assert np.abs(yaw_e[:, fore]).max() > 2.0 * np.abs(yaw_c[:, fore]).max()


def test_escape_stages_bend_one_way_then_the_other_then_relax(sk):
    e = M.EscapeParams()
    clip = M.make_clip(sk, "escape", escape=e)
    net = clip.meta["spine_yaw"].sum(axis=1)
    t = clip.times
    stage1 = net[(t > 0.5 * e.stage1_s) & (t <= e.stage1_s)]
    stage2_end = e.stage1_s + e.stage2_s
    stage2 = net[(t > stage2_end - 0.3 * e.stage2_s) & (t <= stage2_end)]
    coast = net[t > stage2_end + 3.0 * e.coast_tau_s]
    assert np.all(np.sign(stage1) == np.sign(stage1[0]))
    assert np.sign(stage2).sum() * np.sign(stage1).sum() < 0, "stroke must reverse"
    if len(coast):
        assert np.abs(coast).max() < 0.25 * np.abs(net).max()


def test_escape_direction_flips_with_the_parameter(sk):
    a = M.make_clip(sk, "escape", escape=M.EscapeParams(direction=1.0))
    b = M.make_clip(sk, "escape", escape=M.EscapeParams(direction=-1.0))
    assert np.allclose(a.meta["spine_yaw"], -b.meta["spine_yaw"], atol=1e-12)


def test_turn_holds_a_sustained_asymmetric_bend_and_a_bank(sk):
    """Cycle-mean curvature is zero for a cruise and non-zero, one-signed, for a turn."""
    cruise = M.make_clip(sk, "cruise", n_periods=1)
    turn = M.make_clip(sk, "turn", n_periods=1)
    mean_c = cruise.meta["spine_yaw"][:-1].mean(axis=0)
    mean_t = turn.meta["spine_yaw"][:-1].mean(axis=0)
    assert np.abs(mean_c).max() < 1e-3
    assert np.abs(mean_t).max() > 10.0 * max(np.abs(mean_c).max(), 1e-9)
    assert np.all(np.sign(mean_t[1:]) == np.sign(mean_t[1]))

    root = sk.index(sk.spine_names[0])
    bank = math.radians(M.TURN_BANK_DEG)
    # The root carries the bank; no other spine joint does.  q = Rz(yaw)*Rx(roll),
    # so the x component is cos(yaw/2)*sin(roll/2), not sin(roll/2).
    yaw0 = turn.meta["spine_yaw"][0, 0]
    assert abs(turn.quats[0, root, 0]) == pytest.approx(
        abs(math.cos(yaw0 / 2.0) * math.sin(bank / 2.0)), rel=1e-9
    )
    for name in sk.spine_names[1:]:
        assert abs(turn.quats[0, sk.index(name), 0]) < 1e-12


def test_rest_is_near_zero_articulation(sk):
    rest = M.make_clip(sk, "rest")
    cruise = M.make_clip(sk, "cruise")
    assert np.abs(rest.meta["spine_yaw"]).max() < math.radians(3.0)
    assert np.abs(rest.meta["spine_yaw"]).max() < 0.15 * np.abs(cruise.meta["spine_yaw"]).max()


def test_glide_amplitude_decays_after_each_kick(sk):
    """Burst-and-coast: full amplitude at the loop point, a coast floor mid-cycle."""
    p = M.params_for_mode("glide")
    period = p.period_s
    g0 = float(M._gain(np.array([0.0]), p)[0])
    g_mid = float(M._gain(np.array([0.5 * period]), p)[0])
    assert g0 == pytest.approx(1.0, rel=1e-12)
    assert g_mid == pytest.approx(p.coast_floor, rel=1e-9)
    # Monotone decay through the first half-cycle, and periodic overall.
    ts = np.linspace(0.0, 0.5 * period, 40)
    g = M._gain(ts, p)
    assert np.all(np.diff(g) <= 1e-12)
    assert float(M._gain(np.array([period]), p)[0]) == pytest.approx(1.0, rel=1e-9)

    clip = M.make_clip(sk, "glide")
    yaw = np.abs(clip.meta["spine_yaw"]).max(axis=1)
    assert yaw[0] > yaw[len(yaw) // 2]


# ---------------------------------------------------------------------------
# Fins
# ---------------------------------------------------------------------------
def test_fin_phase_lags_are_what_was_configured(sk):
    """Every wave-driven fin channel reads back its own ``phase_lag_deg``."""
    clip = M.make_clip(sk, "cruise", n_periods=2)
    for fin, (_, _) in sk.fins.items():
        drive = M.resolve_fin_drive(fin)
        for ch in drive.channels:
            if ch.source != "wave":
                continue
            root = M.phase_report(clip, sk, ["%s_fin_root" % fin], axis=ch.axis)[
                "%s_fin_root" % fin
            ]
            tip = M.phase_report(clip, sk, ["%s_fin_tip" % fin], axis=ch.axis)[
                "%s_fin_tip" % fin
            ]
            assert root["lag_deg"] == pytest.approx(ch.phase_lag_deg % 360.0, abs=0.5)
            assert tip["lag_deg"] == pytest.approx(
                (ch.phase_lag_deg + drive.tip_extra_lag_deg) % 360.0, abs=0.5
            )
            assert root["amplitude_deg"] == pytest.approx(ch.amplitude_deg, rel=0.02)
            assert tip["amplitude_deg"] == pytest.approx(
                ch.amplitude_deg * drive.tip_gain, rel=0.02
            )


def test_caudal_upper_lobe_lag_is_in_the_60_to_90_degree_bracket():
    drive = M.resolve_fin_drive("caudal_upper")
    lag = drive.channels[0].phase_lag_deg
    assert 60.0 <= lag <= 90.0


def test_a_positive_lag_makes_the_fin_peak_later_than_the_body(sk):
    """Guards the sign trap: with theta = 2*pi*(s/lambda - f*t), lag is +lag INSIDE sin."""
    fins = {"caudal_upper": sk.fins["caudal_upper"]}
    small = M.MotionSkeleton(
        sk.names, sk.parents, sk.spine_names, sk.spine_fractions, fins, fps=400.0
    )
    p = M.params_for_mode("cruise")
    drives = {"caudal_upper": M.FinDrive(
        channels=(M.FinChannel(axis="z", amplitude_deg=10.0, phase_lag_deg=90.0),),
        tip_gain=1.0, tip_extra_lag_deg=0.0)}
    clip = M.make_clip(small, "cruise", params=p, fin_drives=drives)
    s_fin = small.station_of(sk.fins["caudal_upper"][0])
    body = M.lateral_wave(s_fin, clip.times, p)
    fin = clip.quats[:, sk.fins["caudal_upper"][0], 2]
    lead = int(np.argmax(body))
    follow = int(np.argmax(fin))
    delay = (clip.times[follow] - clip.times[lead]) % p.period_s
    assert delay == pytest.approx(0.25 * p.period_s, abs=2.0 / small.fps)


def test_dorsal_is_driven_by_local_curvature_not_by_a_wave_amplitude():
    drive = M.resolve_fin_drive("dorsal")
    assert [c.source for c in drive.channels] == ["curvature"]
    assert drive.channels[0].axis == "x", "a passive dorsal leans laterally, about +X"


def test_empty_fin_drives_leave_fins_at_identity(sk):
    clip = M.make_clip(sk, "cruise", fin_drives={})
    for fin, (root, tip) in sk.fins.items():
        assert np.allclose(clip.quats[:, root, :], [0.0, 0.0, 0.0, 1.0])
        assert np.allclose(clip.quats[:, tip, :], [0.0, 0.0, 0.0, 1.0])


def test_fin_drive_families_resolve_by_prefix():
    assert M.resolve_fin_drive("pectoral_left") is M.DEFAULT_FIN_DRIVES["pectoral"]
    assert M.resolve_fin_drive("caudal_upper") is M.DEFAULT_FIN_DRIVES["caudal_upper"]
    assert M.resolve_fin_drive("caudal") is M.DEFAULT_FIN_DRIVES["caudal"]
    assert M.resolve_fin_drive("second_dorsal") is None  # sevengills have one dorsal


def test_fins_still_move_during_an_escape(sk):
    clip = M.make_clip(sk, "escape")
    root = sk.fins["caudal_upper"][0]
    assert np.abs(clip.quats[:, root, 2]).max() > math.sin(math.radians(2.0) / 2.0)


# ---------------------------------------------------------------------------
# Fin amplitude scaling (fix M1)
# ---------------------------------------------------------------------------
def _fin_root_peak_deg(clip, skeleton):
    """{fin: peak rotation angle of its ROOT joint over the clip, degrees}."""
    out = {}
    for name, (root, _tip) in skeleton.fins.items():
        q = clip.quats[:, int(root), :]
        half = np.arccos(np.clip(np.abs(q[:, 3]), -1.0, 1.0))
        out[name] = float(np.degrees(2.0 * half).max())
    return out


def test_rest_fins_barely_move_compared_with_cruise(sk):
    """REGRESSION (fix M1): 'rest' used to flap its fins exactly as hard as a cruise.

    DEFAULT_FIN_DRIVES amplitudes are ABSOLUTE degrees written for a cruise, and
    nothing scaled them, so the mode whose own MODE_CONFIG description is "near-zero
    articulation" beat its pectorals, pelvics and caudal lobes at full cruise
    amplitude.  Rest's body wave is 0.010 / 0.110 = 9.1% of cruise's, so its fins
    must be too -- comfortably inside 15% for EVERY fin, root and tip.
    """
    cruise = M.make_clip(sk, "cruise", n_periods=2)
    rest = M.make_clip(sk, "rest", n_periods=2)
    peak_c = _fin_root_peak_deg(cruise, sk)
    peak_r = _fin_root_peak_deg(rest, sk)
    for fin in sk.fins:
        assert peak_c[fin] > 0.5, "%s does not move at cruise; the test is vacuous" % fin
        assert peak_r[fin] <= 0.15 * peak_c[fin], (
            "%s: rest %.3f deg vs cruise %.3f deg (%.1f%%)"
            % (fin, peak_r[fin], peak_c[fin], 100.0 * peak_r[fin] / peak_c[fin])
        )


def test_the_fin_amplitude_scale_is_the_modes_body_amplitude_over_cruises(sk):
    for mode, expected in (
        ("cruise", 1.0),
        ("turn", 0.9 * M.CRUISE_TAIL_AMPLITUDE_BL / M.CRUISE_DEFAULT_TAIL_AMPLITUDE_BL),
        ("glide", 0.8 * M.CRUISE_TAIL_AMPLITUDE_BL / M.CRUISE_DEFAULT_TAIL_AMPLITUDE_BL),
        ("rest", 0.010 / M.CRUISE_DEFAULT_TAIL_AMPLITUDE_BL),
    ):
        params = M.params_for_mode(mode)
        assert M.fin_amplitude_scale(mode, params) == pytest.approx(expected, rel=1e-12)
        assert M.make_clip(sk, mode).meta["fin_amplitude_scale"] == pytest.approx(
            expected, rel=1e-12
        )


def test_cruise_fin_amplitudes_are_untouched_by_the_scale(sk):
    """Cruise IS the reference, so its fins must come out bit-identical to before."""
    assert M.fin_amplitude_scale("cruise", M.params_for_mode("cruise")) == 1.0
    clip = M.make_clip(sk, "cruise", n_periods=2)
    for fin in sk.fins:
        drive = M.resolve_fin_drive(fin)
        for ch in drive.channels:
            if ch.source != "wave":
                continue
            measured = M.phase_report(
                clip, sk, ["%s_fin_root" % fin], axis=ch.axis
            )["%s_fin_root" % fin]["amplitude_deg"]
            assert measured == pytest.approx(ch.amplitude_deg, rel=0.02)


def test_escape_fins_scale_up_but_are_clamped(sk):
    """An escape throws the body much further than a cruise, so its fins scale UP --
    and the clamp stops the caricature running away with it."""
    params = M.params_for_mode("escape")
    raw = M.mode_body_amplitude_bl("escape", params) / M.CRUISE_DEFAULT_TAIL_AMPLITUDE_BL
    assert raw > M.FIN_AMPLITUDE_SCALE_CLAMP[1], "raw %.3f" % raw
    assert M.fin_amplitude_scale("escape", params) == M.FIN_AMPLITUDE_SCALE_CLAMP[1]

    escape = M.make_clip(sk, "escape")
    peak_e = _fin_root_peak_deg(escape, sk)

    # Isolated proof that the clamped scale is what is being applied: halving every
    # wave amplitude in the drive table must halve every fin angle exactly.
    halved = {
        family: M.FinDrive(
            channels=tuple(
                M.FinChannel(
                    axis=ch.axis,
                    amplitude_deg=ch.amplitude_deg / M.FIN_AMPLITUDE_SCALE_CLAMP[1],
                    phase_lag_deg=ch.phase_lag_deg,
                    source=ch.source,
                    gain_deg_per_curvature=ch.gain_deg_per_curvature,
                )
                for ch in drive.channels
            ),
            tip_gain=drive.tip_gain,
            tip_extra_lag_deg=drive.tip_extra_lag_deg,
        )
        for family, drive in M.DEFAULT_FIN_DRIVES.items()
    }
    peak_h = _fin_root_peak_deg(M.make_clip(sk, "escape", fin_drives=halved), sk)
    for fin in sk.fins:
        if M.resolve_fin_drive(fin).channels[0].source != "wave":
            continue        # the dorsal is curvature-driven and deliberately unscaled
        # rel, not exact: a two-channel fin composes its axes into one quaternion,
        # so the ANGLE of the product is only linear in the amplitudes to O(theta^2).
        assert peak_e[fin] == pytest.approx(
            M.FIN_AMPLITUDE_SCALE_CLAMP[1] * peak_h[fin], rel=1e-3
        ), fin

    # And in absolute terms the posterior fins beat harder than at cruise.  The
    # PECTORALS do not, and must not: they sit at s = 0.08, inside the C-start's rigid
    # head ramp, where the escape's own curvature field is still near zero.
    peak_c = _fin_root_peak_deg(M.make_clip(sk, "cruise", n_periods=2), sk)
    for fin in ("pelvic", "anal", "caudal_upper", "caudal_lower"):
        assert peak_e[fin] > 1.5 * peak_c[fin], (
            "%s: escape %.3f vs cruise %.3f" % (fin, peak_e[fin], peak_c[fin])
        )
        assert peak_e[fin] <= 1.05 * M.FIN_AMPLITUDE_SCALE_CLAMP[1] * peak_c[fin]
    assert peak_e["pectoral_left"] < peak_c["pectoral_left"]


def test_curvature_driven_fins_are_not_scaled_twice(sk):
    """The dorsal's drive is ``gain * kappa``, already proportional to the mode.

    Scaling it by the amplitude ratio as well would square the dependence, so it is
    excluded -- and the proof is that the dorsal's rest/cruise ratio tracks the
    CURVATURE ratio at its own station, not the square of it.
    """
    assert [c.source for c in M.resolve_fin_drive("dorsal").channels] == ["curvature"]
    root = sk.fins["dorsal"][0]
    s_fin = sk.station_of(root)
    ratios = {}
    for mode in ("cruise", "rest"):
        p = M.params_for_mode(mode)
        clip = M.make_clip(sk, mode, n_periods=2)
        q = clip.quats[:, int(root), :]
        ratios[mode] = (
            float(np.abs(q[:, 0]).max()),
            float(np.abs(M.curvature(s_fin, clip.times, p)).max()),
        )
    angle_ratio = ratios["rest"][0] / ratios["cruise"][0]
    kappa_ratio = ratios["rest"][1] / ratios["cruise"][1]
    assert angle_ratio == pytest.approx(kappa_ratio, rel=0.02)


def test_fin_amplitude_scale_follows_an_explicit_amplitude_override(sk):
    """The scale is against CRUISE'S DEFAULT, so overriding a clip's amplitude moves
    its fins -- it is not silently pinned to whatever the clip asked for."""
    p = M.params_for_mode("cruise", {"tail_amplitude_bl": 0.5 * M.CRUISE_TAIL_AMPLITUDE_BL})
    assert M.fin_amplitude_scale("cruise", p) == pytest.approx(0.5)
    half = M.make_clip(sk, "cruise", params=p, n_periods=2)
    full = M.make_clip(sk, "cruise", n_periods=2)
    peak_h, peak_f = _fin_root_peak_deg(half, sk), _fin_root_peak_deg(full, sk)
    assert peak_h["caudal_upper"] == pytest.approx(0.5 * peak_f["caudal_upper"], rel=0.02)


# ---------------------------------------------------------------------------
# The C-start self-contact cap (fix M3)
# ---------------------------------------------------------------------------
def test_escape_default_respects_the_self_contact_cap(sk):
    """REGRESSION (fix M3): the shipped C-start no longer closes the body into an O.

    At the old 6 /BL the midline turned 276.6 deg end to end, bringing the last spine
    joint to 0.175 BL of the snout and the posed caudal lobe to 0.038 BL of the posed
    head -- straight through the "no self-contact handling" caveat in README section
    7.  The default is capped at ESCAPE_MAX_TOTAL_TURN_DEG of total turning instead.
    """
    assert M.ESCAPE_PEAK_CURVATURE_PER_BL < M.ESCAPE_PEAK_CURVATURE_UNCAPPED_PER_BL
    assert M.ESCAPE_PEAK_CURVATURE_PER_BL == pytest.approx(
        M.escape_peak_curvature_cap(), rel=1e-12
    )
    assert M.escape_total_turn_deg() <= M.ESCAPE_MAX_TOTAL_TURN_DEG
    assert M.escape_total_turn_deg(s_j=sk.spine_fractions) <= M.ESCAPE_MAX_TOTAL_TURN_DEG

    clip = M.make_clip(sk, "escape")
    turn = np.degrees(np.abs(clip.meta["spine_yaw"].sum(axis=1)).max())
    assert turn <= M.ESCAPE_MAX_TOTAL_TURN_DEG
    # still a hard C, not a shrug
    assert turn > 0.9 * M.ESCAPE_MAX_TOTAL_TURN_DEG
    # and it opens the body back up: the tail no longer reaches the head
    assert M.escape_closure_bl() > 3.0 * M.escape_closure_bl(
        M.EscapeParams(peak_curvature_per_bl=M.ESCAPE_PEAK_CURVATURE_UNCAPPED_PER_BL)
    )


def test_the_cap_is_a_parameter_and_the_uncapped_caricature_is_still_reachable(sk):
    """Nothing is hidden: the cap is a named constant and an explicit EscapeParams
    still gets whatever the caller asks for."""
    loose = M.escape_peak_curvature_cap(max_total_turn_deg=360.0)
    assert loose == pytest.approx(2.0 * M.escape_peak_curvature_cap(180.0), rel=1e-12)
    wild = M.EscapeParams(peak_curvature_per_bl=M.ESCAPE_PEAK_CURVATURE_UNCAPPED_PER_BL)
    assert M.escape_total_turn_deg(wild) > M.ESCAPE_MAX_TOTAL_TURN_DEG
    clip = M.make_clip(sk, "escape", escape=wild)
    assert clip.meta["escape"] is wild


def test_escape_shape_integral_matches_a_numeric_quadrature():
    s = np.linspace(0.0, 1.0, 200001)
    for ramp in (0.15, 0.30, 0.60):
        numeric = np.trapezoid(M._smoothstep(s / ramp), s)
        assert M._escape_shape_integral(ramp) == pytest.approx(numeric, rel=1e-6)


# ---------------------------------------------------------------------------
# DCT bending basis
# ---------------------------------------------------------------------------
def test_cruise_spine_profile_is_low_dimensional_in_the_schema_basis(sk):
    """>= 0.9 of the bending energy in 6 DCT modes for cruise; report 4 as well."""
    clip = M.make_clip(sk, "cruise", n_periods=2)
    frac = M.dct_energy_fraction(clip)
    assert frac[M.SCHEMA.NUM_BENDING_MODES] >= 0.90
    assert frac[4] >= 0.80
    assert frac[4] <= frac[M.SCHEMA.NUM_BENDING_MODES] <= 1.0 + 1e-9


@pytest.mark.parametrize("mode", IMPLEMENTED)
def test_every_mode_is_low_dimensional_in_six_modes(mode, sk):
    clip = M.make_clip(sk, mode)
    assert M.dct_energy_fraction(clip)[M.SCHEMA.NUM_BENDING_MODES] >= 0.90


def test_dct_uses_the_schema_basis_and_rejects_a_foreign_spine():
    """The basis is defined on the schema's 12 segments; a 5-joint spine must not pass."""
    names = ["j%d" % i for i in range(5)]
    tiny = M.MotionSkeleton(
        names, [-1, 0, 1, 2, 3], names, np.linspace(0.05, 0.95, 5), {}, fps=30.0
    )
    clip = M.make_clip(tiny, "cruise")
    with pytest.raises(ValueError, match="13-joint spine"):
        M.dct_energy_fraction(clip)


# ---------------------------------------------------------------------------
# Contract plumbing
# ---------------------------------------------------------------------------
def test_clip_answers_to_both_contracts(sk):
    clip = M.make_clip(sk, "cruise")
    assert clip["name"] == "cruise"
    assert clip["times"] is clip.times
    assert clip["quats"] is clip.quats
    assert clip["rotations"] is clip.quats          # the key gltf_export reads
    anim = clip.to_animation()
    assert set(anim) == {"name", "times", "rotations"}
    assert anim["rotations"].shape == (clip.num_frames, sk.num_joints, 4)
    assert np.all(np.diff(anim["times"]) > 0)       # gltf_export's precondition
    with pytest.raises(KeyError):
        clip["nope"]


def test_motion_skeleton_adapts_a_rig_skeleton_without_importing_rig(sk):
    adapted = M.MotionSkeleton.from_skeleton(FakeRigSkeleton(), fps=24.0)
    assert adapted.spine_names == list(M.SPINE_JOINTS)
    assert np.allclose(adapted.spine_fractions, sk.spine_fractions)
    assert adapted.fps == 24.0
    # Recovered from the straight rest pose: 3.0 world units of full centerline.
    assert adapted.body_length == pytest.approx(3.0, rel=1e-9)
    clip = M.make_clip(adapted, "cruise")
    assert clip.quats.shape[1] == adapted.num_joints


def test_station_of_walks_up_to_the_nearest_spine_ancestor(sk):
    root, tip = sk.fins["dorsal"]
    s = sk.spine_fractions[sk.spine_names.index("spine_08_trunk_06")]
    assert sk.station_of(root) == pytest.approx(s)
    assert sk.station_of(tip) == pytest.approx(s)


def test_skeleton_rejects_children_before_parents():
    with pytest.raises(ValueError, match="parents must precede children"):
        M.MotionSkeleton(["a", "b"], [1, -1], ["a", "b"], [0.1, 0.9], {})


def test_jitter_is_seeded_deterministic_and_loop_preserving(sk):
    a = M.make_clip(sk, "cruise", jitter_deg=0.5, seed=7)
    b = M.make_clip(sk, "cruise", jitter_deg=0.5, seed=7)
    c = M.make_clip(sk, "cruise", jitter_deg=0.5, seed=8)
    assert np.array_equal(a.quats, b.quats)
    assert not np.array_equal(a.quats, c.quats)
    assert np.array_equal(a.quats[-1], a.quats[0])
    assert not np.array_equal(a.quats, M.make_clip(sk, "cruise").quats)


def test_quaternion_helpers_agree_with_rotation_matrices():
    """q must mean R = Rz(yaw) @ Rx(roll) @ Ry(pitch); checked against explicit matrices."""
    def rot(axis, a):
        c, s = math.cos(a), math.sin(a)
        if axis == "x":
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
        if axis == "y":
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)

    yaw, roll, pitch = 0.31, -0.17, 0.09
    q = M.euler_zxy_quat(yaw=yaw, roll=roll, pitch=pitch)
    x, y, z, w = q
    r_q = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    assert np.allclose(r_q, rot("z", yaw) @ rot("x", roll) @ rot("y", pitch), atol=1e-12)
    assert M.quat_yaw_z(M.quat_from_axis_angle("z", yaw)) == pytest.approx(yaw)


def test_fundamental_phase_recovers_a_known_sinusoid():
    f = 1.3
    t = np.linspace(0.0, 2.0 / f, 200, endpoint=False)
    amp, phase = M.fundamental_phase(0.4 + 2.5 * np.sin(2 * math.pi * f * t + 0.9), t, f)
    assert amp == pytest.approx(2.5, rel=1e-6)
    assert phase == pytest.approx(0.9, abs=1e-6)


def test_wave_params_validation_and_replace():
    p = M.WaveParams()
    assert p.replace(frequency_hz=1.4).frequency_hz == 1.4
    assert p.replace(frequency_hz=1.4).wavelength_bl == p.wavelength_bl
    with pytest.raises(TypeError):
        p.replace(not_a_field=1)
    for bad in ({"frequency_hz": 0.0}, {"wavelength_bl": -1.0},
                {"head_amplitude_ratio": 0.0}, {"envelope": "cubic"}, {"gain": "warp"}):
        with pytest.raises(ValueError):
            M.WaveParams(**bad)
