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
    peduncle, mild dorsoventral ellipticity) with eight fin sheets **welded** to
    it -- their root row *is* a row of body-grid vertices, so the mesh graph is
    connected exactly as a scanned/photogrammetric mesh would be.  Fins are
    zero-thickness sheets; the real Meshy fins are thin solids, and both are
    handled identically downstream because fin detection thresholds *radius*,
    not thickness.  Carries UVs and a procedural texture.
    ``mesh.metadata`` holds the ground truth: ``centerline`` (N,3),
    ``vertex_labels`` (V,) of str, ``fins`` (per-fin construction parameters),
    ``total_length``, ``tube_length``, ``gill_u``.

``bend(mesh, curve=None, ...) -> (bent_mesh, info)``
    Forward tube-coordinate transport of a *straight* mesh onto ``curve``.  Uses
    exactly the machinery ``mesh3d.debend`` inverts -- ``mesh3d.map_points`` --
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


def _procedural_texture(size=256, seed=0):
    """Countershaded, spotted skin.  ``v`` is phi/2pi, so v=0 is dorsal."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    vv, uu = np.meshgrid(
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
    phi_cols = 2.0 * np.pi * np.arange(n_theta + 1) / n_theta  # +1 = UV seam copy

    # ---- body grid -------------------------------------------------------
    uu, pp = np.meshgrid(u_stations, phi_cols, indexing="ij")
    body = _surface_point(uu, pp, L, S, r_max, ellipticity)
    grid = np.arange(int(n_stations) * (n_theta + 1)).reshape(int(n_stations), n_theta + 1)

    verts = [body.reshape(-1, 3)]
    uvs = [np.stack([uu.ravel(), (pp / (2.0 * np.pi)).ravel()], axis=-1)]
    labels = [np.full(body.shape[0] * body.shape[1], "body", dtype=object)]
    faces = []

    for i in range(int(n_stations) - 1):
        for j in range(n_theta):
            a, b, c, d = grid[i, j], grid[i, j + 1], grid[i + 1, j + 1], grid[i + 1, j]
            faces.append((a, b, c))
            faces.append((a, c, d))

    n_used = grid.size

    # ---- snout and peduncle caps (fans to a single apex, no degenerate faces)
    cap = 0.35 * r_max * _body_radius(0.0)
    snout_apex = n_used
    verts.append(np.array([[0.5 * L + cap, 0.0, 0.0]]))
    uvs.append(np.array([[0.0, 0.5]]))
    labels.append(np.array(["body"], dtype=object))
    n_used += 1
    for j in range(n_theta):
        faces.append((snout_apex, grid[0, j + 1], grid[0, j]))

    tail_apex = n_used
    verts.append(np.array([[_axis_x(1.0, L, S) - cap, 0.0, 0.0]]))
    uvs.append(np.array([[1.0, 0.5]]))
    labels.append(np.array(["body"], dtype=object))
    n_used += 1
    for j in range(n_theta):
        faces.append((tail_apex, grid[-1, j], grid[-1, j + 1]))

    # ---- fins: welded sheets whose root row is a column of body vertices ---
    fin_meta = {}
    for name, (u0, u1, phi_deg, span, sweep, taper, n_span) in (
        FIN_SPECS.items() if with_fins else ()
    ):
        i0 = int(np.round(u0 * (n_stations - 1)))
        i1 = int(np.round(u1 * (n_stations - 1)))
        if i1 - i0 < 2:
            i1 = i0 + 2
        j_root = int(np.round((phi_deg % 360.0) / 360.0 * n_theta)) % n_theta
        phi_root = 2.0 * np.pi * j_root / n_theta

        n_chord = i1 - i0
        root_ids = grid[i0:i1 + 1, j_root]
        # Continuous chord parameter of the root row, in tube fraction.
        g_root = u_stations[i0:i1 + 1]

        rows = [root_ids]
        for k in range(1, n_span + 1):
            t = k / float(n_span)
            # Chord tapers toward the tip and sweeps posteriorly (u increasing).
            centre = 0.5 * (g_root[0] + g_root[-1])
            g = centre + (g_root - centre) * (1.0 - taper * t) + sweep * t
            base = _surface_point(g, phi_root, L, S, r_max, ellipticity)
            d = _radial_dir(np.minimum(g, 1.0), phi_root, r_max, ellipticity)
            pts = base + (span * L * t) * d
            ids = np.arange(n_used, n_used + n_chord + 1)
            n_used += n_chord + 1
            verts.append(pts)
            uvs.append(
                np.stack(
                    [np.clip(g, 0.0, 1.0),
                     np.full(n_chord + 1, (j_root / n_theta + 0.12 * t) % 1.0)],
                    axis=-1,
                )
            )
            labels.append(np.full(n_chord + 1, name, dtype=object))
            rows.append(ids)

        for k in range(n_span):
            ra, rb = rows[k], rows[k + 1]
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
            "seed": int(seed),
        }
    )
    return mesh


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
            to the source arc length, so the map is an isometry in ``s``.
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

    out = mesh.copy()
    out.vertices = mesh3d.map_points(
        np.asarray(mesh.vertices, dtype=float), src, src_frames, dst, dst_frames
    )
    out.metadata["centerline"] = dst
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
