"""Procedural sevengill (*Notorynchus cepedianus*) mesh, with ground truth.

Why this file exists: the real input to this prototype is a textured Meshy-AI GLB
whose rest pose is a strong lateral C-curve.  That asset is not present in this
session, so every stage of the pipeline is developed and tested against a
procedural stand-in whose centerline, per-vertex fin labels and tube coordinates
are known exactly.  The same ``mesh3d`` code runs on either.

Contracts
---------
``make_sevengill(...) -> trimesh.Trimesh``
    A single connected surface: a body of revolution (blunt head, tapered
    peduncle, mild dorsoventral ellipticity) with eight **solid** fins welded
    into it.  Each fin is a closed, two-sided loft of a NACA-00xx section --
    round nose, thin trailing edge, maximum thickness ``FIN_THICKNESS_RATIO`` of
    the local chord at the root -- whose root loop is a *slit* opened in the
    body grid itself: the root column is split into two lips pushed apart by the
    local section half-thickness, its neighbours are slid aside to make room,
    and a single triangle closes the notch at each end of the slit.  Nothing is
    duplicated and nothing is orphaned; the mesh graph is connected exactly as a
    scanned/photogrammetric mesh would be, and the fin shells add no boundary
    and no non-manifold edge.  Carries UVs and a procedural texture.
    ``mesh.metadata`` holds the ground truth: ``centerline`` (N,3),
    ``vertex_labels`` (V,) of str, ``fins`` (per-fin construction parameters),
    ``total_length``, ``tube_length``, ``gill_u``.

``bend(mesh, curve=None, ...) -> (bent_mesh, info)``
    Forward tube-coordinate transport of a *straight* mesh onto ``curve``.  Uses
    exactly the machinery ``mesh3d.debend`` inverts -- ``mesh3d.map_mesh`` --
    so a round trip through the ground-truth centerline is exact to float
    precision and any residual measures centerline-extraction error alone.

Canonical pose (shared with ``blender/operators/create_shark_armature.py``):
snout at **+X**, tail at **-X**, dorsal **+Z**, animal's left **+Y**.
Arc length ``s`` runs head -> tail, so the body tangent is ``-X``.

Deterministic: the only stochastic content is the texture, driven by ``seed``.
"""

from __future__ import annotations

import numpy as np
import trimesh

__all__ = [
    "make_sevengill",
    "bend",
    "c_curve",
    "s_curve",
    "export_glb",
    "preview_png",
    "fin_section_report",
    "FIN_SPECS",
    "LABELS",
]

# Fin label vocabulary; ``body`` is index 0 so ``labels != "body"`` is the fin set.
LABELS = (
    "body",
    "pectoral_L",
    "pectoral_R",
    "dorsal",
    "pelvic_L",
    "pelvic_R",
    "anal",
    "caudal_upper",
    "caudal_lower",
)

# Body radius profile: knots in tube fraction u (0 = snout, 1 = end of the body
# tube at the precaudal peduncle) against fraction of the maximum radius.
# Shaped from the hexanchiform habitus: blunt broad head, girth maximum just
# behind the branchial region, long slow taper to a slender peduncle.
_PROFILE_U = np.array(
    [0.00, 0.03, 0.08, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.93, 1.00]
)
_PROFILE_F = np.array(
    [0.12, 0.38, 0.60, 0.80, 0.95, 1.00, 0.98, 0.90, 0.78, 0.60, 0.42, 0.30, 0.24]
)

# Fin construction parameters, in tube fraction u and radians of phi.
# phi is measured from +Z (dorsal) toward +Y (the animal's left), matching the
# tube chart in mesh3d: phi = atan2(v . B, v . N) with N = +Z, B = +Y.
#
# Anatomy encoded here (binding for the whole prototype):
#   - ONE dorsal fin, far posterior, over/behind the pelvics.  No second dorsal.
#   - pectorals lateroventral and anterior, just behind the seventh gill slit.
#   - strongly heterocercal caudal: upper lobe ~3x the lower lobe and swept far
#     posterior, which is why ``sweep`` carries it past the end of the body tube.
FIN_SPECS = {
    #                u0    u1    phi(deg) span   sweep  taper  n_span
    "pectoral_L":  (0.26, 0.35,  110.0,  0.130, 0.075, 0.60, 10),
    "pectoral_R":  (0.26, 0.35, -110.0,  0.130, 0.075, 0.60, 10),
    "pelvic_L":    (0.62, 0.70,  140.0,  0.070, 0.045, 0.55, 8),
    "pelvic_R":    (0.62, 0.70, -140.0,  0.070, 0.045, 0.55, 8),
    "dorsal":      (0.66, 0.76,    0.0,  0.075, 0.060, 0.65, 9),
    "anal":        (0.76, 0.83,  180.0,  0.050, 0.035, 0.55, 7),
    "caudal_upper": (0.90, 1.00,   0.0,  0.140, 0.300, 0.55, 14),
    "caudal_lower": (0.93, 1.00, 180.0,  0.075, 0.120, 0.55, 9),
}

# Seven gill slits, in tube fraction; drawn into the texture and reported in
# metadata so the rig can bind ``gill_slit_1``/``gill_slit_7`` stations.
_GILL_U = np.linspace(0.170, 0.268, 7)


def _body_radius(u):
    """Body tube radius at tube fraction ``u``, in units of the max radius."""
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    return np.interp(u, _PROFILE_U, _PROFILE_F)


def _axis_x(u, total_length, tube_length):
    """X of the tube axis at (possibly extrapolated) tube fraction ``u``."""
    return 0.5 * total_length - np.asarray(u, dtype=float) * tube_length


def _surface_point(u, phi, total_length, tube_length, r_max, ellipticity):
    """Body surface point at tube fraction ``u`` and angle ``phi`` (radians).

    ``u`` may exceed 1 (used only as a *fin* root guide beyond the peduncle);
    the radius is then held at its terminal value.
    """
    r = r_max * _body_radius(u)
    x = _axis_x(u, total_length, tube_length)
    y = r * np.sin(phi)
    z = r * ellipticity * np.cos(phi)
    return np.stack([x, y, z], axis=-1)


def _radial_dir(u, phi, r_max, ellipticity):
    """Unit direction from the tube axis to the surface at (u, phi)."""
    r = np.maximum(r_max * _body_radius(u), 1e-9)
    v = np.stack(
        [np.zeros_like(np.asarray(u, dtype=float)),
         r * np.sin(phi),
         r * ellipticity * np.cos(phi)],
        axis=-1,
    )
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Fin cross-section
# ---------------------------------------------------------------------------
# Every fin is a closed two-sided loft of a symmetric section.  These constants
# are the whole of its shape; the numbers they produce on the default stand-in
# are printed by ``fin_section_report``.

#: Maximum section thickness as a fraction of the local chord, at the root.
#: 10-14% is the range measured on the Meshy sevengill's pectoral and dorsal.
FIN_THICKNESS_RATIO = 0.12
#: Tip thickness as a fraction of the root's, *before* the chord taper is
#: applied on top of it, so the blade thins faster than it narrows.
FIN_TIP_THICKNESS_SCALE = 0.35
#: Nose closure, as a fraction of the section's maximum half-thickness.  The
#: leading edge is a rounded ``sqrt(x)`` nose truncated by this facet: two
#: coincident vertices would be a duplicate and a zero-area triangle.
FIN_LE_FRACTION = 0.20
#: Trailing-edge closure, same units.  Thin, but deliberately not zero.
FIN_TE_FRACTION = 0.03
#: Floor on the half-thickness in units of ``total_length``, so the thinnest
#: sliver of the thinnest fin tip is still well clear of float32 export noise.
FIN_MIN_HALF_THICKNESS = 1.5e-4
#: A root section may not be thicker than this fraction of the first span step,
#: or the blade would be born inside the body.  Inactive on the default fins;
#: it is what keeps a 0.3x-span sliver fin from turning inside out.
FIN_ROOT_CLEARANCE = 0.75
#: Body columns each side of the root slit that are slid aside to make room for
#: it.  The slit needs ``ceil(half-thickness / column)`` columns of clearance;
#: this many more keeps the displaced quads from collapsing.
FIN_WARP_MARGIN = 2


def _section_shape(xc):
    """Symmetric thickness law across the chord, normalised to a maximum of 1.

    NACA-00xx ``y_t`` -- a ``sqrt(x)`` nose, so the leading edge is *round*
    rather than a wedge -- plus two closures, ``FIN_LE_FRACTION`` at the nose
    and ``FIN_TE_FRACTION`` at the tail, which stop the two sides of the
    section from collapsing onto each other at the ends.  ``xc`` is 0 at the
    leading edge (anterior, +X) and 1 at the trailing edge.
    """
    x = np.clip(np.asarray(xc, dtype=float), 0.0, 1.0)
    naca = 10.0 * (
        0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
        + 0.2843 * x ** 3 - 0.1015 * x ** 4
    )
    f = naca + FIN_LE_FRACTION * (1.0 - x) ** 3 + FIN_TE_FRACTION * x ** 3
    return f / _SECTION_SHAPE_MAX


# Normaliser for the law above, found by sampling it with the normaliser at 1:
# the closures move the maximum off the NACA peak, so it is not 1 by algebra.
_SECTION_SHAPE_MAX = 1.0
_SECTION_SHAPE_MAX = float(np.max(_section_shape(np.linspace(0.0, 1.0, 4001))))


def _seam_columns(j, n_theta):
    """Grid columns holding the signed body column ``j``.

    The body grid carries one duplicated column so the UV wrap needs no
    degenerate parameterisation, so the column at ``phi = 0`` lives at two
    indices and both must move together.
    """
    j = int(j) % int(n_theta)
    return (0, int(n_theta)) if j == 0 else (j,)


def _fin_plan(spec, n_stations, n_theta, u_stations, total_length, tube_length,
              r_max, ellipticity):
    """Resolve one ``FIN_SPECS`` entry into everything the builder needs.

    Returns a dict with the station range and root column (unchanged from the
    zero-thickness construction, so the ``fins`` metadata contract is
    untouched), the per-station half-opening of the root slit in *column*
    units, the number of neighbouring columns to slide aside, and the section
    thickness law sampled at the chord stations.
    """
    u0, u1, phi_deg, span, sweep, taper, n_span = spec
    i0 = int(np.round(u0 * (n_stations - 1)))
    i1 = int(np.round(u1 * (n_stations - 1)))
    if i1 - i0 < 2:
        i1 = i0 + 2
    i1 = min(i1, int(n_stations) - 1)
    i0 = max(0, min(i0, i1 - 2))
    j_root = int(np.round((phi_deg % 360.0) / 360.0 * n_theta)) % int(n_theta)
    phi_root = 2.0 * np.pi * j_root / n_theta

    g_root = u_stations[i0:i1 + 1]
    n_chord = i1 - i0
    chord = float(g_root[-1] - g_root[0]) * tube_length
    u_mid = 0.5 * float(g_root[0] + g_root[-1])
    r_mid = r_max * float(_body_radius(u_mid))
    # Arc length swept on the body surface per radian of phi, at the root.
    arc_per_rad = max(
        r_mid * float(np.hypot(np.cos(phi_root), ellipticity * np.sin(phi_root))),
        1e-12,
    )

    half_max = 0.5 * FIN_THICKNESS_RATIO * chord
    half_max = min(half_max, FIN_ROOT_CLEARANCE * span * total_length / float(n_span))
    d_cols = (half_max / arc_per_rad) / (2.0 * np.pi / n_theta)
    n_warp = int(np.ceil(d_cols)) + int(FIN_WARP_MARGIN)

    xc = (g_root - g_root[0]) / max(float(g_root[-1] - g_root[0]), 1e-12)
    shape = _section_shape(xc)

    cells = set()
    for i in range(i0, i1 + 1):
        for k in range(-n_warp + 1, n_warp):
            for jc in _seam_columns(j_root + k, n_theta):
                cells.add((i, jc))

    return {
        "i0": i0, "i1": i1, "n_chord": n_chord, "j_root": j_root,
        "phi_root": phi_root, "g_root": g_root, "chord": chord,
        "half_max": half_max, "d_cols": d_cols * shape, "n_warp": n_warp,
        "shape": shape, "span": span, "sweep": sweep, "taper": taper,
        "n_span": int(n_span), "cells": cells,
    }


def _procedural_texture(size=256, seed=0):
    """Countershaded, spotted skin.  ``v`` is phi/2pi, so v=0 is dorsal."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    # meshgrid(..., indexing="xy") returns (column-varying, row-varying); the
    # image column is u and the row is v, so u comes first. (The earlier
    # ``vv, uu`` order put the countershading on the u axis - no dorsoventral
    # gradient on the mesh at all.)
    uu, vv = np.meshgrid(
        np.linspace(0.0, 1.0, size), np.linspace(0.0, 1.0, size), indexing="xy"
    )
    # Countershading: dark at v ~ 0 or 1 (dorsal), pale at v ~ 0.5 (ventral).
    dorsal = np.cos(2.0 * np.pi * vv) * 0.5 + 0.5
    shade = 0.30 + 0.55 * (1.0 - dorsal) ** 1.4
    img = np.stack([shade * 1.02, shade * 0.98, shade * 0.90], axis=-1)

    # Sparse dark spots, denser dorsally, as on a real sevengill.
    for _ in range(320):
        cu, cv = rng.random(), rng.random()
        if rng.random() > (1.0 - 0.75 * abs(np.cos(2.0 * np.pi * cv))):
            continue
        rad = rng.uniform(0.006, 0.020)
        d = np.hypot(uu - cu, np.minimum(np.abs(vv - cv), 1.0 - np.abs(vv - cv)))
        img *= (1.0 - 0.45 * np.exp(-(d / rad) ** 2))[..., None]

    # Seven gill slits: dark bands on both flanks (v ~ 0.25 and v ~ 0.75).
    for gu in _GILL_U:
        band = np.exp(-((uu - gu) / 0.0055) ** 2)
        flank = np.exp(-((vv - 0.25) / 0.075) ** 2) + np.exp(-((vv - 0.75) / 0.075) ** 2)
        img *= (1.0 - 0.7 * band * flank)[..., None]

    return Image.fromarray(np.clip(img * 255.0, 0, 255).astype(np.uint8), mode="RGB")


def make_sevengill(
    n_stations=112,
    n_theta=48,
    total_length=1.0,
    tube_fraction=0.80,
    radius_fraction=0.070,
    ellipticity=1.12,
    n_centerline=64,
    with_fins=True,
    textured=True,
    seed=0,
    solid_fins=True,
):
    """Build the canonical straight sevengill.

    Args:
        n_stations: body rings along the tube (>= 16).
        n_theta: samples around the tube (even, >= 12); one duplicated seam
            column is added so the UV wrap needs no degenerate parameterisation.
        total_length: snout tip to caudal upper-lobe tip, in metres.
        tube_fraction: body-tube (snout -> peduncle) length as a fraction of
            ``total_length``; the caudal lobes occupy the remainder.
        radius_fraction: maximum body radius as a fraction of ``total_length``.
        ellipticity: vertical/lateral radius ratio (>1 = laterally compressed).
        n_centerline: stations of the returned ground-truth centerline.
        with_fins: attach the eight fin sheets.  ``False`` gives the bare body
            tube, whose centerline is the control in the "fins must not divert
            the path" test.
        textured: attach the procedural texture and UVs.
        seed: texture RNG seed.
        solid_fins: build the fins as closed two-sided lofts welded into a slit
            in the body grid (the default).  ``False`` reverts to the
            pre-volumetric zero-thickness sheets, which is what the A/B numbers
            in the README are measured against.

    Returns:
        ``trimesh.Trimesh`` in the canonical pose, with ground truth in
        ``metadata`` (see the module docstring).
    """
    if n_stations < 16 or n_theta < 12 or n_theta % 2:
        raise ValueError("need n_stations >= 16 and an even n_theta >= 12")

    L = float(total_length)
    S = tube_fraction * L
    r_max = radius_fraction * L

    u_stations = np.linspace(0.0, 1.0, int(n_stations))
    n_theta = int(n_theta)
    n_stations = int(n_stations)
    dphi_col = 2.0 * np.pi / n_theta
    phi_cols = dphi_col * np.arange(n_theta + 1)      # +1 = UV seam copy

    # ---- fin plans, resolved first: the root slits are cut into the body ---
    plans = {}
    claimed = set()
    for name, spec in (FIN_SPECS.items() if with_fins else ()):
        plan = _fin_plan(spec, n_stations, n_theta, u_stations, L, S, r_max,
                         ellipticity)
        # Two fins cannot open the same strip of skin.  Only a hand-built spec
        # can ask for it -- the fin-merge regression fixture puts a second
        # caudal lobe on top of the first -- and the later fin then falls back
        # to the old zero-thickness sheet, flagged ``volumetric: False``.
        # A slit needs room: its own columns, plus the neighbours it slides
        # aside, must fit inside the ring without meeting round the back.
        fits = 2 * plan["n_warp"] < n_theta
        plan["volumetric"] = (bool(solid_fins) and fits
                              and not (plan["cells"] & claimed))
        if plan["volumetric"]:
            claimed |= plan["cells"]
        plans[name] = plan

    # ---- body grid, with the root slits opened -----------------------------
    # Each volumetric fin splits its root column into two lips separated by the
    # local section thickness, and slides its neighbours outward so the quads
    # between them stay well shaped.  ``v_grid`` follows the same displacement,
    # so the texture stays glued to the surface instead of shearing.
    uu = np.repeat(u_stations[:, None], n_theta + 1, axis=1)
    phi_grid = np.tile(phi_cols, (n_stations, 1))
    v_grid = np.tile(np.arange(n_theta + 1) / float(n_theta), (n_stations, 1))

    for plan in plans.values():
        if not plan["volumetric"]:
            continue
        j_root, n_warp = plan["j_root"], plan["n_warp"]
        for m, i in enumerate(range(plan["i0"], plan["i1"] + 1)):
            d = float(plan["d_cols"][m])
            for k in range(1, n_warp):
                # Linear re-spacing of the ring: the column that sat k out now
                # sits at d + k * (n_warp - d) / n_warp, so every gap out to
                # n_warp has the same width and nothing folds over.
                delta = d + k * (n_warp - d) / n_warp - k
                for jc in _seam_columns(j_root + k, n_theta):
                    phi_grid[i, jc] += delta * dphi_col
                    v_grid[i, jc] += delta / n_theta
                for jc in _seam_columns(j_root - k, n_theta):
                    phi_grid[i, jc] -= delta * dphi_col
                    v_grid[i, jc] -= delta / n_theta
            if j_root == 0:
                # The duplicated seam column already supplies two vertices per
                # station; the slit just pulls them apart.
                phi_grid[i, n_theta] = 2.0 * np.pi - d * dphi_col
                v_grid[i, n_theta] = 1.0 - d / n_theta
                phi_grid[i, 0] = d * dphi_col
                v_grid[i, 0] = d / n_theta
            else:
                phi_grid[i, j_root] = plan["phi_root"] - d * dphi_col
                v_grid[i, j_root] = (j_root - d) / float(n_theta)

    body = _surface_point(uu, phi_grid, L, S, r_max, ellipticity)
    grid = np.arange(n_stations * (n_theta + 1)).reshape(n_stations, n_theta + 1)

    verts = [body.reshape(-1, 3)]
    uvs = [np.stack([uu.ravel(), v_grid.ravel()], axis=-1)]
    labels = [np.full(grid.size, "body", dtype=object)]
    faces = []
    n_used = grid.size

    # Which vertex a body quad sees on each side of a column.  The two differ
    # only along a root slit: the left lip keeps the original id, the right lip
    # is a new vertex -- except at phi = 0, where the seam copy is already there.
    id_left_of = grid.copy()      # column j as the RIGHT edge of quad j-1
    id_right_of = grid.copy()     # column j as the LEFT edge of quad j

    for plan in plans.values():
        if not plan["volumetric"]:
            continue
        i0, i1, j_root = plan["i0"], plan["i1"], plan["j_root"]
        if j_root == 0:
            plan["left_ids"] = grid[i0:i1 + 1, n_theta]
            plan["right_ids"] = grid[i0:i1 + 1, 0]
            continue
        pts = _surface_point(u_stations[i0:i1 + 1],
                             plan["phi_root"] + plan["d_cols"] * dphi_col,
                             L, S, r_max, ellipticity)
        ids = np.arange(n_used, n_used + len(pts))
        n_used += len(pts)
        verts.append(pts)
        uvs.append(np.stack([u_stations[i0:i1 + 1],
                             (j_root + plan["d_cols"]) / float(n_theta)], axis=-1))
        labels.append(np.full(len(pts), "body", dtype=object))
        plan["left_ids"] = grid[i0:i1 + 1, j_root]
        plan["right_ids"] = ids
        id_right_of[i0:i1 + 1, j_root] = ids

    for i in range(n_stations - 1):
        for j in range(n_theta):
            a, b = id_right_of[i, j], id_left_of[i, j + 1]
            c, d = id_left_of[i + 1, j + 1], id_right_of[i + 1, j]
            faces.append((a, b, c))
            faces.append((a, c, d))

    # ---- snout and peduncle caps (fans to a single apex, no degenerate faces)
    cap = 0.35 * r_max * _body_radius(0.0)
    snout_apex = n_used
    verts.append(np.array([[0.5 * L + cap, 0.0, 0.0]]))
    uvs.append(np.array([[0.0, 0.5]]))
    labels.append(np.array(["body"], dtype=object))
    n_used += 1
    for j in range(n_theta):
        faces.append((snout_apex, id_left_of[0, j + 1], id_right_of[0, j]))

    tail_apex = n_used
    verts.append(np.array([[_axis_x(1.0, L, S) - cap, 0.0, 0.0]]))
    uvs.append(np.array([[1.0, 0.5]]))
    labels.append(np.array(["body"], dtype=object))
    n_used += 1
    for j in range(n_theta):
        faces.append((tail_apex, id_right_of[-1, j], id_left_of[-1, j + 1]))

    # ---- fins: closed two-sided lofts rising out of the root slits ---------
    fin_meta = {}
    for name, plan in plans.items():
        i0, i1, n_chord = plan["i0"], plan["i1"], plan["n_chord"]
        j_root, phi_root, g_root = plan["j_root"], plan["phi_root"], plan["g_root"]
        span, sweep, taper = plan["span"], plan["sweep"], plan["taper"]
        n_span = plan["n_span"]
        centre = 0.5 * (g_root[0] + g_root[-1])

        if plan["volumetric"]:
            # Thickness runs along the body tangent at the root, so the blade
            # stands proud of the surface rather than shearing along it.
            e_hat = np.array([0.0, np.cos(phi_root),
                              -ellipticity * np.sin(phi_root)])
            e_hat = e_hat / np.linalg.norm(e_hat)
            rows = [(plan["right_ids"], plan["left_ids"])]
            for k in range(1, n_span + 1):
                t = k / float(n_span)
                # Chord tapers toward the tip and sweeps posteriorly.
                g = centre + (g_root - centre) * (1.0 - taper * t) + sweep * t
                base = _surface_point(g, phi_root, L, S, r_max, ellipticity)
                rad = _radial_dir(np.minimum(g, 1.0), phi_root, r_max, ellipticity)
                mid = base + (span * L * t) * rad
                h = (plan["half_max"] * plan["shape"] * (1.0 - taper * t)
                     * (1.0 - (1.0 - FIN_TIP_THICKNESS_SCALE) * t))
                h = np.maximum(h, FIN_MIN_HALF_THICKNESS * L)
                pair = []
                for sign in (1.0, -1.0):
                    ids = np.arange(n_used, n_used + n_chord + 1)
                    n_used += n_chord + 1
                    verts.append(mid + (sign * h)[:, None] * e_hat)
                    uvs.append(np.stack([
                        np.clip(g, 0.0, 1.0),
                        (j_root / float(n_theta)
                         + sign * (0.12 * t + plan["d_cols"] / n_theta)) % 1.0,
                    ], axis=-1))
                    labels.append(np.full(n_chord + 1, name, dtype=object))
                    pair.append(ids)
                rows.append((pair[0], pair[1]))

            # One triangle closes the notch the slit leaves at each end of the
            # root; at the ends of the body that apex is the cap apex.
            head_col = j_root if j_root else n_theta
            a = snout_apex if i0 == 0 else int(id_left_of[i0 - 1, head_col])
            faces.append((a, int(rows[0][0][0]), int(rows[0][1][0])))
            b = (tail_apex if i1 == n_stations - 1
                 else int(id_left_of[i1 + 1, head_col]))
            faces.append((b, int(rows[0][1][-1]), int(rows[0][0][-1])))

            # Shell: quads between consecutive closed sections, walked as
            # right side leading edge -> trailing edge, then left side back.
            for k in range(n_span):
                (ra, la), (rb, lb) = rows[k], rows[k + 1]
                loop_in = np.concatenate([ra, la[::-1]])
                loop_out = np.concatenate([rb, lb[::-1]])
                n_loop = len(loop_in)
                for q in range(n_loop):
                    q2 = (q + 1) % n_loop
                    faces.append((loop_in[q], loop_in[q2], loop_out[q2]))
                    faces.append((loop_in[q], loop_out[q2], loop_out[q]))
            # Tip lid across the outermost section: no extra vertices, and it
            # uses every edge of that section exactly once.
            rt, lt = rows[-1]
            for m in range(n_chord):
                faces.append((rt[m], rt[m + 1], lt[m + 1]))
                faces.append((rt[m], lt[m + 1], lt[m]))
        else:
            # Fallback: the pre-volumetric zero-thickness sheet.
            root_ids = id_left_of[i0:i1 + 1, j_root if j_root else n_theta]
            sheet = [root_ids]
            for k in range(1, n_span + 1):
                t = k / float(n_span)
                g = centre + (g_root - centre) * (1.0 - taper * t) + sweep * t
                base = _surface_point(g, phi_root, L, S, r_max, ellipticity)
                rad = _radial_dir(np.minimum(g, 1.0), phi_root, r_max, ellipticity)
                ids = np.arange(n_used, n_used + n_chord + 1)
                n_used += n_chord + 1
                verts.append(base + (span * L * t) * rad)
                uvs.append(np.stack(
                    [np.clip(g, 0.0, 1.0),
                     np.full(n_chord + 1, (j_root / n_theta + 0.12 * t) % 1.0)],
                    axis=-1))
                labels.append(np.full(n_chord + 1, name, dtype=object))
                sheet.append(ids)
            for k in range(n_span):
                ra, rb = sheet[k], sheet[k + 1]
                for m in range(n_chord):
                    faces.append((ra[m], ra[m + 1], rb[m + 1]))
                    faces.append((ra[m], rb[m + 1], rb[m]))

        fin_meta[name] = {
            "u0": float(g_root[0]),
            "u1": float(g_root[-1]),
            "phi_root": float(((phi_root + np.pi) % (2.0 * np.pi)) - np.pi),
            "span": float(span * L),
            "station_range": (int(i0), int(i1)),
            "insertion_centroid": np.asarray(
                _surface_point(0.5 * (g_root[0] + g_root[-1]), phi_root, L, S,
                               r_max, ellipticity),
                dtype=float,
            ),
            # Added by the volumetric build; nothing downstream reads them.
            "volumetric": bool(plan["volumetric"]),
            "root_chord": float(plan["chord"]),
            "root_thickness": float(2.0 * plan["half_max"]),
            "root_thickness_ratio": float(
                2.0 * plan["half_max"] / max(plan["chord"], 1e-12)),
            # (span row, chord station, 2) -- each vertex of the blade paired
            # with its twin on the other side of the section, root row first.
            # This is the ground truth ``fin_section_report`` measures against;
            # it is absent on a fin that fell back to a sheet.
            "section_pairs": (
                np.stack([np.stack([np.asarray(ra), np.asarray(la)], axis=-1)
                          for ra, la in rows], axis=0)
                if plan["volumetric"] else None),
        }

    vertices = np.concatenate(verts, axis=0)
    uv = np.concatenate(uvs, axis=0)
    vertex_labels = np.concatenate(labels, axis=0).astype(str)
    faces = np.asarray(faces, dtype=np.int64)

    visual = None
    if textured:
        visual = trimesh.visual.TextureVisuals(
            uv=uv, image=_procedural_texture(seed=seed)
        )

    mesh = trimesh.Trimesh(
        vertices=vertices, faces=faces, visual=visual, process=False
    )
    centerline = np.stack(
        [
            _axis_x(np.linspace(0.0, 1.0, int(n_centerline)), L, S),
            np.zeros(int(n_centerline)),
            np.zeros(int(n_centerline)),
        ],
        axis=-1,
    )
    mesh.metadata.update(
        {
            "species": "Notorynchus cepedianus",
            "synthetic": True,
            "total_length": L,
            "tube_length": S,
            "max_radius": r_max,
            "ellipticity": float(ellipticity),
            "centerline": centerline,
            "vertex_labels": vertex_labels,
            "fins": fin_meta,
            "gill_u": _GILL_U.copy(),
            "gill_x": _axis_x(_GILL_U, L, S),
            "with_fins": bool(with_fins),
            "solid_fins": bool(solid_fins),
            "seed": int(seed),
        }
    )
    return mesh


def fin_section_report(mesh):
    """Measure every solid fin's cross-section, on the mesh as it was built.

    Reads ``metadata['fins'][name]['section_pairs']`` -- the (span row, chord
    station) grid pairing each vertex on one side of a blade with its twin on
    the other -- and measures the vertices themselves, so what comes back is
    what was built rather than what was asked for.  Fins that fell back to a
    zero-thickness sheet are absent from the result.

    Returns ``{name: {...}}`` in world units except where noted:

    ``root_chord``
        Chord of the root section (its leading to trailing edge along the body).
    ``root_thickness``, ``root_ratio``
        Widest point of the root section, and that over the chord -- the
        ``FIN_THICKNESS_RATIO`` the builder was asked for, as delivered.
    ``le_closure``, ``te_closure``
        Nose and tail facets of the root section, as fractions of the chord:
        how blunt the rounded leading edge is, and how thin the trailing edge.
    ``tip_thickness``
        Widest point of the outermost section (the tip lid).
    ``min_thickness``
        Thinnest place anywhere on the fin.  Nonzero by construction
        (``FIN_MIN_HALF_THICKNESS``); this is the number that says so.
    """
    v = np.asarray(mesh.vertices, dtype=float)
    out = {}
    for name, fin in mesh.metadata.get("fins", {}).items():
        pairs = fin.get("section_pairs")
        if pairs is None:
            continue
        p = np.asarray(pairs, dtype=np.int64)
        t = np.linalg.norm(v[p[..., 0]] - v[p[..., 1]], axis=-1)
        chord = float(fin["root_chord"])
        out[name] = {
            "root_chord": chord,
            "root_thickness": float(t[0].max()),
            "root_ratio": float(t[0].max() / chord),
            "le_closure": float(t[0, 0] / chord),
            "te_closure": float(t[0, -1] / chord),
            "tip_thickness": float(t[-1].max()),
            "min_thickness": float(t.min()),
        }
    return out


# ---------------------------------------------------------------------------
# Target centerlines for the forward bend.
# ---------------------------------------------------------------------------

def c_curve(total_length, turn_deg=120.0, n=64, plane="xy"):
    """Constant-curvature arc of arc length ``total_length``.

    ``turn_deg`` is the total tangent turn from head to tail (the owner's Meshy
    rest pose is roughly a 120 degree lateral C).  Returned head-first, starting
    at the origin with tangent -X so it overlays the canonical straight pose
    at the snout, and bending within ``plane`` ('xy' = lateral, 'xz' = sagittal).
    """
    turn = np.deg2rad(float(turn_deg))
    if abs(turn) < 1e-9:
        raise ValueError("turn_deg must be non-zero; use a straight centerline instead")
    radius = float(total_length) / turn
    th = np.linspace(0.0, turn, int(n))
    # Arc starting at the origin, initial tangent -X, curving toward +Y.
    along = -radius * np.sin(th)
    side = radius * (1.0 - np.cos(th))
    zero = np.zeros_like(th)
    if plane == "xy":
        return np.stack([along, side, zero], axis=-1)
    if plane == "xz":
        return np.stack([along, zero, side], axis=-1)
    raise ValueError("plane must be 'xy' or 'xz'")


def s_curve(total_length, turn_deg=90.0, n=64, plane="xy"):
    """Two opposed constant-curvature arcs (an S), total arc length preserved."""
    half = int(n) // 2 + 1
    first = c_curve(0.5 * float(total_length), turn_deg, half, plane)
    # Reflect the second half about the end tangent to reverse the curvature.
    t_end = first[-1] - first[-2]
    t_end = t_end / np.linalg.norm(t_end)
    second = c_curve(0.5 * float(total_length), -turn_deg, half, plane)
    r0 = second[1] - second[0]
    r0 = r0 / np.linalg.norm(r0)
    axis = np.cross(r0, t_end)
    ang = np.arctan2(np.linalg.norm(axis), float(np.dot(r0, t_end)))
    if np.linalg.norm(axis) > 1e-12:
        rot = trimesh.transformations.rotation_matrix(ang, axis / np.linalg.norm(axis))[:3, :3]
    else:
        rot = np.eye(3)
    second = (second - second[0]) @ rot.T + first[-1]
    return np.concatenate([first, second[1:]], axis=0)


def bend(mesh, curve=None, source_centerline=None, up=(0.0, 0.0, 1.0)):
    """Transport a straight mesh onto ``curve`` through tube coordinates.

    Args:
        mesh: canonical straight mesh (``make_sevengill`` output, or any mesh
            with a straight ground-truth centerline).
        curve: (m, 3) target centerline, or None for the default ~120 degree
            lateral C-curve.  Resampled to the source station count and rescaled
            to the source arc length, so the map is an isometry in ``s`` for body
            vertices; fin islands ride their insertion frame as rigid plates
            (``mesh3d.map_mesh``), and the records are stored in
            ``metadata["rigid_islands"]``.
        source_centerline: (n, 3) straight centerline; defaults to
            ``mesh.metadata['centerline']``.
        up: seed normal for the target rotation-minimising frames (dorsal).

    Returns:
        ``(bent_mesh, info)`` where ``info`` has ``centerline``, ``frames``
        (tangents, normals, binormals), ``source_centerline`` and
        ``source_frames`` -- the exact ground truth ``mesh3d.debend`` must
        recover.
    """
    import mesh3d

    src = np.asarray(
        mesh.metadata["centerline"] if source_centerline is None else source_centerline,
        dtype=float,
    )
    n = len(src)
    total = float(mesh3d.arc_length(src)[-1])
    if curve is None:
        curve = c_curve(total, 120.0, n)
    dst = mesh3d.resample_polyline(np.asarray(curve, dtype=float), n)
    dst = dst * (total / float(mesh3d.arc_length(dst)[-1]))

    src_frames = mesh3d.tube_frames(src, up=up)
    dst_frames = mesh3d.tube_frames(dst, up=up)

    # Body through the chart, fins as rigid plates hinged at their base -- the
    # same transport ``mesh3d.debend`` applies in the other direction.
    out, records = mesh3d.map_mesh(mesh, src, src_frames, dst, dst_frames)
    out.metadata["centerline"] = dst
    out.metadata["rigid_islands"] = records
    out.metadata["straight_centerline"] = src
    return out, {
        "centerline": dst,
        "frames": dst_frames,
        "source_centerline": src,
        "source_frames": src_frames,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def export_glb(mesh, path):
    """Write a binary glTF.  Metadata is dropped; geometry/UV/texture survive."""
    out = mesh.copy()
    out.metadata.clear()
    out.export(str(path))
    return str(path)


def preview_png(mesh, path, axes=(0, 1), size=(900, 380), light=(0.4, 0.5, 0.75),
                background=(250, 250, 248)):
    """Orthographic painter's-algorithm preview (no OpenGL in this environment).

    ``axes`` selects the two world axes drawn as (horizontal, vertical); the
    remaining axis is depth.  Faces are Lambert-shaded and drawn back to front.
    """
    from PIL import Image, ImageDraw

    v = np.asarray(mesh.vertices, dtype=float)
    f = np.asarray(mesh.faces, dtype=np.int64)
    h_ax, v_ax = axes
    d_ax = 3 - h_ax - v_ax

    pts = v[:, [h_ax, v_ax]].copy()
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    scale = 0.94 * min(size[0] / span[0], size[1] / span[1])
    origin = np.array(size, dtype=float) * 0.5 - 0.5 * (lo + hi) * scale
    px = pts * scale + origin
    px[:, 1] = size[1] - px[:, 1]

    tri = v[f]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    lam = np.abs(nrm @ (np.asarray(light, dtype=float) / np.linalg.norm(light)))
    shade = np.clip(0.28 + 0.72 * lam, 0.0, 1.0)
    depth = tri[:, :, d_ax].mean(axis=1)

    img = Image.new("RGB", tuple(size), tuple(background))
    draw = ImageDraw.Draw(img)
    for k in np.argsort(depth):
        c = int(255 * shade[k])
        draw.polygon(
            [tuple(px[i]) for i in f[k]],
            fill=(int(c * 0.72), int(c * 0.76), int(c * 0.80)),
        )
    img.save(str(path))
    return str(path)
