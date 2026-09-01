"""Self-contained fixtures for the chart <-> texture <-> photo modules.

Nothing here depends on prototype 04.  ``make_uv_tube`` builds a UV-unwrapped
tube mesh AND the per-vertex ground-truth ``(s, phi)`` that prototype 04's
``mesh3d.tube_coords`` will supply for real meshes, so bake/unbake can be
developed and tested against exact answers.

Everything is deterministic given ``seed``.

WHAT IS FIXTURE-ONLY.  ``make_test_chart``, ``render_lateral_tube`` and
``detect_chart_spots`` exist to give the tests ground truth.  They are NOT the
identity-pattern generator (module P, ``pattern.py``) and must not be used as
one -- their spot model is a placeholder circle process with no growth,
resighting-drift or occlusion semantics.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import trimesh
from scipy import ndimage

from bake import sample_chart, wrap_to_pi

__all__ = [
    "UVTube",
    "shark_radius_profile",
    "make_uv_tube",
    "make_seam_offset_tube",
    "tube_surface_points",
    "make_test_chart",
    "render_lateral_tube",
    "detect_chart_spots",
    "SPOT_DIAMETER_FRAC",
]

# [DERIVED from prototypes/02-centerline-chart/strain_demo.py PARAMS:
# spot_radius 4.5 px on L = 560 px]  Spot diameter as a fraction of body
# length.  Used only to size fixture spots so they are the same scale as the
# rest of the programme's synthetic patterns.
# [EVIDENCE GRADE: derived from this programme's own synthetic model; no
# measured sevengill freckle size distribution was available.]
SPOT_DIAMETER_FRAC = 0.016


class UVTube(NamedTuple):
    """A UV-unwrapped tube plus exact chart ground truth.

    mesh:        ``trimesh.Trimesh`` with ``visual.uv`` in ``[0, 1]^2``.
    vertex_s:    ``(V,)`` arc-length fraction, 0 at the snout end, 1 at caudal.
    vertex_phi:  ``(V,)`` circumferential angle in ``(-pi, pi]``, 0 = dorsal
                 (+Z), ``+pi/2`` = the animal's left (+Y) -- prototype 04's
                 ``TubeCoords`` convention.
    centerline:  ``(n_stations, 3)`` axis polyline, head first.
    radius:      ``(n_stations,)`` cross-section radius at each station.
    seam_phi:    the angle at which the UV atlas is cut.
    grid_shape:  ``(n_stations, n_around + 1)`` vertex grid shape.
    dorsal:      ``(n_stations, 3)`` unit dorsal direction per station (phi=0).
    left:        ``(n_stations, 3)`` unit left direction per station (phi=+pi/2).
    """

    mesh: trimesh.Trimesh
    vertex_s: np.ndarray
    vertex_phi: np.ndarray
    centerline: np.ndarray
    radius: np.ndarray
    seam_phi: float
    grid_shape: tuple
    dorsal: np.ndarray
    left: np.ndarray


def _profile_shape(u):
    return np.sin(np.pi * np.clip(u, 0.0, 1.0) ** 0.65) ** 0.6 * (1.0 - 0.45 * np.clip(u, 0.0, 1.0))


# Normaliser so the profile peaks at exactly 1.0 regardless of how many
# stations the caller asks for (a per-call ``f.max()`` would make the radius
# depend on the sampling, which would silently change the fixture's geometry
# between a 32-station and a 512-station tube).
_PROFILE_PEAK = float(_profile_shape(np.linspace(0.0, 1.0, 20001)).max())


def shark_radius_profile(s, r_max=1.0, floor=0.06):
    """Blunt-headed, tapering half-width profile r(s), peaking at ``r_max``.

    Shaped like ``strain_demo.half_width_profile`` -- anterior noticeably wider
    than posterior, so a silhouette of it orients head-first under
    ``centerline.extract_centerline``'s widest-end rule -- but never reaching
    zero (``floor``, as a fraction of ``r_max``), because a zero-radius station
    makes ``phi`` undefined.

    Shape only: this is a fixture, not a measured Notorynchus girth profile.
    [EVIDENCE GRADE: none -- geometric placeholder.]
    """
    u = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
    return r_max * np.maximum(_profile_shape(u) / _PROFILE_PEAK, floor)


def _resolve_radius(radius_profile, s_vals, r_max):
    if radius_profile is None:
        return shark_radius_profile(s_vals, r_max=r_max)
    if callable(radius_profile):
        return np.asarray(radius_profile(s_vals), dtype=float) * np.ones_like(s_vals)
    arr = np.asarray(radius_profile, dtype=float)
    if arr.ndim == 0:
        return np.full_like(s_vals, float(arr))
    if len(arr) != len(s_vals):
        raise ValueError(
            "radius_profile array has %d entries, need %d (one per station)"
            % (len(arr), len(s_vals))
        )
    return arr


def make_uv_tube(
    n_stations=48,
    n_around=32,
    length=1.0,
    radius_profile=None,
    seed=0,
    seam_phi=np.pi,
    r_max=0.12,
    bend=0.0,
):
    """Build a UV-unwrapped tube with exact per-vertex ``(s, phi)``.

    The mesh is a ``n_stations x (n_around + 1)`` vertex grid.  The extra
    column DUPLICATES the first one geometrically but carries ``v = 1`` instead
    of ``v = 0``.  That duplication is the standard way to give a closed tube a
    cut-free UV atlas: every quad is monotone in ``v`` and no face straddles
    the atlas border.

    UV layout: ``u = s`` (column = along the body), ``v`` = fraction of the way
    round the girth STARTING FROM ``seam_phi``.

    Args:
        n_stations: vertex rings along the body (>= 2).
        n_around: vertices per ring (>= 3); the ring is closed by duplication.
        length: total centerline length.
        radius_profile: ``None`` (shark-like default), a scalar, a callable
            ``s -> radius``, or an ``(n_stations,)`` array.
        seed: accepted for API stability; construction is deterministic and no
            step here is stochastic.
        seam_phi: where the atlas is cut.  ``pi`` (default) cuts at the ventral
            midline, which puts the ``+-pi`` wrap of the STORED ``phi`` values
            exactly on the atlas border -- the easy case.  Any other value
            moves the wrap INTO the middle of the atlas, so interior faces have
            corners like ``+3.1`` and ``-3.1`` and a bake that interpolates
            ``phi`` naively tears there.  See :func:`make_seam_offset_tube`.
        bend: if non-zero, the centerline is a constant-curvature arc of this
            total turn angle (radians) in the XY plane, so the fixture is not
            trivially axis-aligned.  ``phi`` is still measured in the
            rotation-minimizing cross-section frame.

    Returns:
        :class:`UVTube`.
    """
    n_stations = int(n_stations)
    n_around = int(n_around)
    if n_stations < 2:
        raise ValueError("n_stations must be >= 2")
    if n_around < 3:
        raise ValueError("n_around must be >= 3 for a closed ring")

    s_vals = np.linspace(0.0, 1.0, n_stations)
    radius = _resolve_radius(radius_profile, s_vals, r_max)

    # Centerline + a rotation-minimizing-ish frame.  For a planar arc the
    # binormal is constant (+Z out of plane is NOT what we want: we want the
    # animal's dorsal to stay dorsal), so: tangent in XY, "left" = tangent
    # rotated +90 deg in XY, "dorsal" = +Z.  That is exactly rotation-minimizing
    # for a planar curve.
    if abs(bend) < 1e-9:
        centerline = np.column_stack(
            [s_vals * length, np.zeros(n_stations), np.zeros(n_stations)]
        )
        tang = np.tile(np.array([1.0, 0.0, 0.0]), (n_stations, 1))
    else:
        turn = float(bend)
        R = length / turn
        ang = s_vals * turn
        centerline = np.column_stack(
            [R * np.sin(ang), R * (1.0 - np.cos(ang)), np.zeros(n_stations)]
        )
        tang = np.column_stack([np.cos(ang), np.sin(ang), np.zeros(n_stations)])
    dorsal = np.tile(np.array([0.0, 0.0, 1.0]), (n_stations, 1))
    left = np.cross(dorsal, tang)          # +Y when straight
    left /= np.linalg.norm(left, axis=1, keepdims=True)

    j = np.arange(n_around + 1)
    phi_ring = seam_phi + j * (2.0 * np.pi / n_around)      # unwrapped
    cos_p, sin_p = np.cos(phi_ring), np.sin(phi_ring)

    verts = (
        centerline[:, None, :]
        + radius[:, None, None] * (
            cos_p[None, :, None] * dorsal[:, None, :]
            + sin_p[None, :, None] * left[:, None, :]
        )
    ).reshape(-1, 3)

    vertex_s = np.repeat(s_vals, n_around + 1)
    vertex_phi = np.tile(wrap_to_pi(phi_ring), n_stations)
    uv = np.column_stack([
        np.repeat(s_vals, n_around + 1),
        np.tile(j / float(n_around), n_stations),
    ])

    stride = n_around + 1
    i0 = np.arange(n_stations - 1)[:, None] * stride + np.arange(n_around)[None, :]
    a = i0.ravel()
    b = a + stride
    c = b + 1
    d = a + 1
    faces = np.concatenate([
        np.column_stack([a, b, c]),
        np.column_stack([a, c, d]),
    ], axis=0)

    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        visual=trimesh.visual.TextureVisuals(uv=uv),
        process=False,
    )
    return UVTube(
        mesh=mesh,
        vertex_s=vertex_s,
        vertex_phi=vertex_phi,
        centerline=centerline,
        radius=radius,
        seam_phi=float(seam_phi),
        grid_shape=(n_stations, n_around + 1),
        dorsal=dorsal,
        left=left,
    )


def make_seam_offset_tube(seam_phi=-2.0, **kwargs):
    """A UV tube whose atlas seam is NOT at the ``phi`` wrap point.

    With the seam at ``phi = pi`` the atlas border and the ``+-pi`` wrap of the
    stored ``phi`` coincide, so even a naive interpolator never sees a face
    with a 2*pi jump across it.  Moving the seam (default ``-2.0`` rad, on the
    right ventro-lateral flank) puts that jump on an INTERIOR column of the
    atlas: the face spanning ``phi = +pi`` now lives at UV column
    ``v = (pi - seam_phi) / (2 pi)``.  This is the fixture the seam test needs.
    """
    return make_uv_tube(seam_phi=seam_phi, **kwargs)


def tube_surface_points(tube, s, phi):
    """3D points on a :class:`UVTube`'s surface at chart coordinates.

    Linear in the station index, so it agrees with the mesh's own faceting to
    O(1/n_stations^2).  Used to check that ``(s, phi)`` really is the surface
    parameterisation the mesh carries.
    """
    n_stations = tube.grid_shape[0]
    s_arr = np.atleast_1d(np.asarray(s, dtype=float))
    phi_arr = np.atleast_1d(np.asarray(phi, dtype=float))
    idx = np.clip(s_arr * (n_stations - 1), 0, n_stations - 1)
    i0 = np.clip(np.floor(idx).astype(int), 0, n_stations - 2)
    f = (idx - i0)[:, None]

    cen = tube.centerline[i0] * (1 - f) + tube.centerline[i0 + 1] * f
    rad = tube.radius[i0] * (1 - f[:, 0]) + tube.radius[i0 + 1] * f[:, 0]
    dor = tube.dorsal[i0] * (1 - f) + tube.dorsal[i0 + 1] * f
    lef = tube.left[i0] * (1 - f) + tube.left[i0 + 1] * f
    return cen + rad[:, None] * (
        np.cos(phi_arr)[:, None] * dor + np.sin(phi_arr)[:, None] * lef
    )


# ---------------------------------------------------------------------------
# fixture chart + fixture renderer
# ---------------------------------------------------------------------------

def make_test_chart(
    n_s=128,
    n_phi=256,
    n_spots=60,
    seed=0,
    spot_value=0.25,
    s_range=(0.08, 0.92),
    spot_diameter=SPOT_DIAMETER_FRAC,
    min_sep_frac=0.045,
):
    """A fixture chart of dark round spots on unmarked skin.

    Returns ``(chart, spots)``: ``chart`` is ``(n_s, n_phi)`` float ALBEDO
    MULTIPLIER (1 = unmarked, ``spot_value`` inside a spot) and ``spots`` is
    ``(k, 2)`` of ground-truth ``(s, phi)`` centres.

    Spot placement is a seeded dart-throw with a minimum separation, matching
    ``strain_demo.make_spots`` in spirit; separation is measured with ``phi``
    scaled to arc length at a nominal radius so spacing is isotropic ON THE
    SKIN, not in chart index space.

    NOT an identity generator.  Real per-individual patterns, growth scaling,
    resighting drift and occlusions are module P's (``pattern.py``) job.
    """
    rng = np.random.default_rng([int(seed), 0x5A17])
    # phi -> arc length uses a nominal girth radius; only the RATIO matters.
    nominal_r = 0.09
    sep = float(min_sep_frac)
    spots = []
    for _ in range(20000):
        if len(spots) >= int(n_spots):
            break
        s = rng.uniform(*s_range)
        phi = rng.uniform(-np.pi, np.pi)
        ok = True
        for (s2, p2) in spots:
            dphi = float(wrap_to_pi(phi - p2)) * nominal_r
            if np.hypot(s - s2, dphi) < sep:
                ok = False
                break
        if ok:
            spots.append((s, phi))
    spots = np.asarray(spots, dtype=float).reshape(-1, 2)

    s_ax = (np.arange(n_s) + 0.5) / n_s
    phi_ax = -np.pi + (np.arange(n_phi) + 0.5) * (2 * np.pi / n_phi)
    S, P = np.meshgrid(s_ax, phi_ax, indexing="ij")
    chart = np.ones((n_s, n_phi))
    rad = 0.5 * float(spot_diameter)
    for (s0, p0) in spots:
        d = np.hypot(S - s0, wrap_to_pi(P - p0) * nominal_r)
        soft = np.clip((rad - d) / (0.35 * rad) + 0.5, 0.0, 1.0)
        chart = chart * (1 - soft) + float(spot_value) * soft
    return chart, spots


def render_lateral_tube(
    chart,
    n_px=(220, 700),
    length=1.0,
    r_max=0.12,
    radius_profile=None,
    side="L",
    base_rgb=(0.42, 0.40, 0.34),
    shading=0.25,
    countershading=0.35,
    noise=0.004,
    seed=0,
    margin_px=14,
):
    """Analytic orthographic render of a straight spotted tube, seen laterally.

    Not a rasteriser: for every pixel the visible surface point is solved in
    closed form, so the ground-truth ``(s, phi)`` of each pixel is exact and
    the unbake test measures unbake's error, not a renderer's.

    Geometry.  The tube axis runs along world +X, dorsal is +Z, the animal's
    left is +Y.  For ``side="L"`` the camera is at ``+Y`` looking along ``-Y``,
    so the visible half is ``phi in (0, pi)`` and the surface point above image
    height ``z`` is ``phi = arccos(z / R(s))``.  For ``side="R"`` the camera is
    at ``-Y``, the visible half is ``phi in (-pi, 0)``, and the image is
    mirrored horizontally so the animal faces the other way -- exactly what a
    real right-side photograph looks like.

    Shading: Lambert against a fixed above-and-forward light, plus
    countershading (dorsum darker than ventrum), which is the real sevengill
    tone -- "dark speckling on grey-brown dorsum, lighter ventrally"
    [species anatomy, cited in the prototype brief].  Both are LOW FREQUENCY,
    which is what unbake's normalisation is supposed to divide out.

    NOTE that the two fight each other: the overhead key light brightens the
    dark dorsum.  At the defaults the dorsum still reads darker overall, but
    raise ``shading`` or lower ``countershading`` and it stops doing so -- which
    is precisely the regime in which ``unbake._infer_dorsal_sign``'s
    countershading heuristic must refuse to guess, and is used as such in the
    tests.

    Returns:
        ``(rgb, mask, info)``.  ``rgb`` is ``(H, W, 3)`` float ``[0, 1]``,
        ``mask`` is bool, ``info`` carries the exact per-pixel ``s``/``phi``,
        the pixel scale, and the radius profile used.
    """
    if side not in ("L", "R"):
        raise ValueError("side must be 'L' or 'R', got %r" % (side,))
    h, w = (int(n_px[0]), int(n_px[1])) if not isinstance(n_px, int) else (int(n_px), int(n_px))
    rng = np.random.default_rng([int(seed), 0xBEE5])

    usable_w = w - 2 * margin_px
    px_per_unit = usable_w / float(length)
    xs = (np.arange(w) - margin_px) / px_per_unit          # world X per column
    zs = ((h - 1) / 2.0 - np.arange(h)) / px_per_unit      # world Z per row, +Z up
    X, Z = np.meshgrid(xs, zs)

    s = np.clip(X / float(length), -0.1, 1.1)
    s_stations = np.linspace(0.0, 1.0, 256)
    r_stations = _resolve_radius(radius_profile, s_stations, r_max)
    R = np.interp(np.clip(s, 0.0, 1.0), s_stations, r_stations)

    inside = (X >= 0.0) & (X <= float(length)) & (np.abs(Z) <= R)
    cosphi = np.clip(np.where(R > 0, Z / np.maximum(R, 1e-9), 0.0), -1.0, 1.0)
    phi_abs = np.arccos(cosphi)                            # in [0, pi]
    phi = phi_abs if side == "L" else -phi_abs

    # surface normal in world coords, and a fixed light
    sinphi = np.sqrt(np.maximum(1.0 - cosphi ** 2, 0.0))
    view_sign = 1.0 if side == "L" else -1.0
    n_world = np.stack([np.zeros_like(cosphi), view_sign * sinphi, cosphi], axis=-1)
    light = np.array([0.30, 0.55 * view_sign, 0.78])
    light /= np.linalg.norm(light)
    lam = np.clip(np.einsum("...i,i->...", n_world, light), 0.0, 1.0)
    shade = (1.0 - shading) + shading * lam

    # countershading: dorsal (phi ~ 0) darker, ventral (|phi| ~ pi) lighter
    counter = 1.0 - countershading * (1.0 + cosphi) / 2.0

    pattern = sample_chart(np.asarray(chart, dtype=float), np.clip(s, 0.0, 1.0), phi)
    if pattern.ndim == 2:
        pattern = pattern[..., None]
    base = np.asarray(base_rgb, dtype=float).reshape(1, 1, 3)
    rgb = base * (shade * counter)[..., None] * pattern
    rgb = rgb + rng.normal(0.0, float(noise), size=rgb.shape)
    rgb = np.clip(np.where(inside[..., None], rgb, 0.06), 0.0, 1.0)

    if side == "R":
        rgb = rgb[:, ::-1].copy()
        inside = inside[:, ::-1].copy()
        s = s[:, ::-1].copy()
        phi = phi[:, ::-1].copy()

    info = {
        "s": np.where(inside, s, np.nan),
        "phi": np.where(inside, phi, np.nan),
        "px_per_unit": px_per_unit,
        "radius_stations": (s_stations, r_stations),
        "side": side,
        "length": float(length),
    }
    return rgb, inside, info


def detect_chart_spots(chart, threshold=0.86, min_cells=6, background_frac=0.08):
    """Detect dark spots in a chart image.  Fixture/eval helper.

    The chart is first divided by its own large-scale median (radius
    ``background_frac`` of the chart in ``s``), which removes any residual
    low-frequency term, then thresholded and connected-component labelled with
    the ``phi`` axis WRAPPED, so a spot sitting on the ``+-pi`` boundary is one
    component and not two.

    Returns ``(k, 2)`` of ``(s, phi)`` centroids, ``NaN``-free.
    """
    arr = np.asarray(chart, dtype=float)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    n_s, n_phi = arr.shape
    valid = np.isfinite(arr)
    filled = np.where(valid, arr, np.nanmedian(arr[valid]) if valid.any() else 1.0)

    size_s = max(3, int(round(background_frac * n_s)) | 1)
    size_p = max(3, int(round(background_frac * n_phi)) | 1)
    bg = ndimage.median_filter(filled, size=(size_s, size_p), mode="nearest")
    # wrap-aware background along phi: redo with a rolled copy at the border
    bg_wrap = np.roll(
        ndimage.median_filter(np.roll(filled, n_phi // 2, axis=1),
                              size=(size_s, size_p), mode="nearest"),
        -(n_phi // 2), axis=1,
    )
    edge = np.zeros(n_phi, dtype=bool)
    edge[: size_p] = True
    edge[-size_p:] = True
    bg[:, edge] = bg_wrap[:, edge]

    norm = filled / np.maximum(bg, 1e-6)
    dark = (norm < float(threshold)) & valid

    # label with phi wrap by doubling the array and folding labels back
    doubled = np.concatenate([dark, dark], axis=1)
    lab, n = ndimage.label(doubled)
    out = []
    seen = set()
    for k in range(1, n + 1):
        cells = np.argwhere(lab == k)
        cols = cells[:, 1]
        if cols.min() >= n_phi:
            continue
        if len(cells) < int(min_cells):
            continue
        rows = cells[:, 0]
        key = (int(round(rows.mean())), int(round(cols.mean())) % n_phi)
        if key in seen:
            continue
        seen.add(key)
        s_c = (rows.mean() + 0.5) / n_s
        phi_c = float(wrap_to_pi(
            -np.pi + ((cols.mean() % n_phi) + 0.5) * (2 * np.pi / n_phi)
        ))
        out.append((s_c, phi_c))
    return np.asarray(out, dtype=float).reshape(-1, 2)
