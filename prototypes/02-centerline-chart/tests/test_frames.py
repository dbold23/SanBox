from __future__ import annotations

import numpy as np

from frames import rotation_minimizing_frames, tangents_normals_2d


def test_2d_frames_orthonormal():
    t = np.linspace(0, 1, 200)
    poly = np.column_stack([300 * t, 40 * np.sin(2 * np.pi * t)])
    tan, nor = tangents_normals_2d(poly)
    assert np.allclose(np.linalg.norm(tan, axis=1), 1.0, atol=1e-9)
    assert np.allclose(np.linalg.norm(nor, axis=1), 1.0, atol=1e-9)
    assert np.allclose(np.sum(tan * nor, axis=1), 0.0, atol=1e-9)
    # Left-normal convention: n is t rotated +90 deg.
    assert np.allclose(nor[:, 0], -tan[:, 1])
    assert np.allclose(nor[:, 1], tan[:, 0])


def _twist_rate(pts, tan, nor):
    """Component of dN/di along the binormal = discrete twist about the tangent."""
    b = np.cross(tan, nor)
    dn = np.gradient(nor, axis=0)
    return np.sum(dn * b, axis=1)


def test_rmf_orthonormal_right_handed():
    t = np.linspace(0, 4 * np.pi, 400)
    pts = np.column_stack([np.cos(t), np.sin(t), 0.3 * t])
    tan, nor, bin_ = rotation_minimizing_frames(pts)
    for u, v in ((tan, nor), (tan, bin_), (nor, bin_)):
        assert np.abs(np.sum(u * v, axis=1)).max() < 1e-8
    for u in (tan, nor, bin_):
        assert np.allclose(np.linalg.norm(u, axis=1), 1.0, atol=1e-8)
    assert np.allclose(np.cross(tan, nor), bin_, atol=1e-8)


def test_rmf_zero_twist_on_planar_curve():
    # Planar S-curve in the xy-plane: an RMF seeded in-plane must stay
    # in-plane (zero twist); one seeded out-of-plane must stay out-of-plane.
    u = np.linspace(0, 1, 500)
    pts = np.column_stack([u * 10, np.sin(2 * np.pi * u), np.zeros_like(u)])
    tan, nor, _ = rotation_minimizing_frames(pts, initial_normal=[0.0, 0.0, 1.0])
    assert np.abs(nor[:, :2]).max() < 1e-6          # normal stays +z
    assert np.abs(_twist_rate(pts, tan, nor)).max() < 1e-6


def test_rmf_matches_analytic_normal_on_circle():
    t = np.linspace(0, 1.5 * np.pi, 600)
    pts = np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)])
    radial_in = -pts  # inward analytic normal (unit, since R=1)
    _, nor, _ = rotation_minimizing_frames(pts, initial_normal=radial_in[0])
    # On a planar circle the RMF normal is exactly the (transported) radial.
    err = np.linalg.norm(nor - radial_in, axis=1)
    assert err.max() < 5e-3


def test_rmf_minimal_twist_on_helix():
    # Helix x = (a cos t, a sin t, b t): torsion tau = b / (a^2 + b^2).
    # The Frenet frame twists about the tangent at rate tau; the RMF must
    # untwist it, i.e. RMF normal = rotate(Frenet normal, -tau * s).
    a, b = 1.0, 0.4
    c2 = a * a + b * b
    tau = b / c2
    t = np.linspace(0, 6 * np.pi, 3000)
    s = np.sqrt(c2) * t
    pts = np.column_stack([a * np.cos(t), a * np.sin(t), b * t])

    n_frenet = np.column_stack([-np.cos(t), -np.sin(t), np.zeros_like(t)])
    b_frenet = np.column_stack(
        [b * np.sin(t), -b * np.cos(t), np.full_like(t, a)]
    ) / np.sqrt(c2)
    ang = -tau * s
    analytic = np.cos(ang)[:, None] * n_frenet + np.sin(ang)[:, None] * b_frenet

    tan, nor, _ = rotation_minimizing_frames(pts, initial_normal=n_frenet[0])
    err = np.linalg.norm(nor - analytic, axis=1)
    assert err.max() < 5e-3
    # And directly: near-zero twist about the tangent everywhere.
    assert np.abs(_twist_rate(pts, tan, nor)).max() < 1e-4
