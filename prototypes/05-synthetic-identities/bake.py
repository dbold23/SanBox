"""Canonical chart space <-> mesh UV texture.

WHAT THIS MODULE IS FOR
-----------------------
Prototype 05 generates identity patterns in CANONICAL CHART SPACE ``(s, phi)``,
never in a mesh's UV atlas.  A chart pattern is mesh-agnostic: it survives
re-topology, it maps onto any pose through the rig, and every rendered pixel
carries exact ground-truth chart coordinates.  Baking that chart onto a
particular mesh's UV texture is a SEPARATE, LATER step -- this module.

BINDING CHART CONVENTION (shared by every module in prototype 05)
-----------------------------------------------------------------
``s``    arc-length fraction along the body centerline, 0 = snout, 1 = caudal.
``phi``  circumferential angle in ``(-pi, pi]``, 0 = dorsal midline,
         ``+pi/2`` = the animal's LEFT flank, ``-pi/2`` = right flank,
         ``+-pi`` = ventral midline.

This is exactly prototype 04's ``mesh3d.TubeCoords`` convention ("angle in the
cross-section, from the normal (+Z when straight, i.e. dorsal) toward the
binormal (+Y, the animal's left), in (-pi, pi]"), with ``s`` divided by
``TubeCoords.total_length`` to make it a fraction.  The adapter is one line:

    tc = mesh3d.tube_coords(mesh, centerline)
    vertex_s, vertex_phi = tc.s / tc.total_length, tc.phi

A chart IMAGE is an array of shape ``(n_s, n_phi)`` or ``(n_s, n_phi, C)``.
Cells are CELL-CENTRED::

    s_i   = (i + 0.5) / n_s                      i = 0 .. n_s - 1
    phi_j = -pi + (j + 0.5) * 2*pi / n_phi       j = 0 .. n_phi - 1

``s`` is CLAMPED at the ends (the body does not wrap), ``phi`` is PERIODIC
(the body does).  Cell centring is what makes the periodic axis exact: cell 0
and cell ``n_phi-1`` are separated by exactly one cell width across the
``+-pi`` boundary, with no half-cell fencepost error.

CHART VALUE SEMANTICS
---------------------
By default a chart image is an ALBEDO MULTIPLIER in ``[0, 1]``: 1.0 = unmarked
skin, lower = a darker mark.  "Composite as a multiplicative darkening" is then
literally ``texture = base_albedo * chart``.  Generators that emit a DARKNESS
map (0 = unmarked, 1 = fully dark) pass ``chart_semantics="darkness"`` and are
converted by :func:`darkness_to_multiplier`.  Nothing else in this module cares
which one you used.

ARRAY LAYOUT, AND THE INTEROP WITH ``pattern`` / ``exclusions``
---------------------------------------------------------------
This module lays charts out ``(n_s, n_phi)`` -- s-major -- because it is the
direct continuation of ``prototypes/02-centerline-chart/chart.rectify``, whose
strip is ``(n_s, n_r)``, and because ``unbake`` builds the chart row by row up
the body.  Prototype 05's pattern generator (``pattern.py`` / ``exclusions.py``)
lays them out ``(H_phi, W_s)`` -- phi-major -- and its charts are DARKNESS maps
(0 = unmarked) rather than albedo multipliers.

The ``(s, phi)`` SEMANTICS are identical on both sides (cell-centred, ``phi``
periodic on ``[-pi, pi)``, 0 dorsal, ``+pi/2`` the animal's left), and
``test_bake.py`` asserts that against ``exclusions.chart_axes`` so the two
cannot drift.  Only the transpose and the value convention differ, and they are
bridged explicitly rather than by guesswork:

    mult  = from_pattern_chart(darkness)          # (H_phi, W_s) -> (n_s, n_phi)
    dark  = to_pattern_chart(mult)                # the inverse

``pattern.exclusion_mask`` already defaults to ``axis_order="s_major"``, i.e.
it hands back a mask in THIS module's layout, and ``pattern.copy_from_chart``
accepts ``axis_order="s_major", chart_semantics="albedo"``.  ``unbake`` passes
both explicitly instead of relying on the "auto" heuristics.

UV CONVENTION
-------------
``u`` indexes texture COLUMNS and ``v`` indexes texture ROWS, both increasing,
with texel centres at ``((x + 0.5) / W, (y + 0.5) / H)``.  There is no OpenGL
v-flip here; flip your UVs on export if your renderer wants ``v`` up.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
from scipy import ndimage

__all__ = [
    "chart_axes",
    "chart_indices",
    "sample_chart",
    "splat_to_chart",
    "darkness_to_multiplier",
    "from_pattern_chart",
    "to_pattern_chart",
    "wrap_to_pi",
    "RasterUV",
    "rasterize_uv",
    "texel_chart_coords",
    "bake_chart_to_texture",
    "mesh_texture_to_chart",
    "estimate_lowfreq_luminance",
    "luminance",
    "masked_gaussian",
    "LUMA_WEIGHTS",
    "DELIGHT_SIGMA_S",
    "DELIGHT_SIGMA_PHI",
    "DELIGHT_HARMONICS",
    "DELIGHT_POLY_ORDER",
]

# Rec.709 luma weights.  Used only to build a scalar shading estimate; the
# de-lighting gain is applied to all three channels equally, so the choice of
# weights shifts the estimate by a few percent and never introduces a hue.
LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)

# De-lighting low-frequency cutoff, in chart units.
#
# [DERIVED from prototypes/02-centerline-chart/strain_demo.py PARAMS] the spot
# model there is spot_radius = 4.5 px on a body of L = 560 px, i.e. a spot
# DIAMETER of 0.016 body lengths, with min_sep = 30 px = 0.054 body lengths
# between centres.  A low-frequency cutoff of 0.10 body lengths therefore sits
# ~6x above the spot diameter and ~2x above the inter-spot spacing, which is
# the margin that lets the divide remove shading without eating spots.
# [EVIDENCE GRADE: derived from this programme's own synthetic spot model, NOT
# from measured sevengill freckle morphometry -- no published sevengill
# speckle size distribution was available.  Re-fit both constants once real
# charts exist.]
DELIGHT_SIGMA_S = 0.10        # body-length fractions
DELIGHT_SIGMA_PHI = 0.80      # radians (~46 deg of girth)

# Basis size for the default (method="basis") low-frequency fit.
# n_harmonics = 3 spans girth periods down to 2*pi/3 rad (~120 deg), which
# covers a dorsal key light, a ventral bounce and a flank falloff and stops
# well short of the ~0.05-0.15 rad angular width of a spot.  poly_order = 4
# spans fore-aft trends down to ~1/4 body length, well above the 0.016
# body-length spot diameter derived above.
# [BRACKET n_harmonics 2-5, poly_order 3-6] [EVIDENCE GRADE: derived from the
# spot scale above plus the frequency argument in estimate_lowfreq_luminance;
# not fitted to real capture lighting, which does not yet exist for this
# programme.]
DELIGHT_HARMONICS = 3
DELIGHT_POLY_ORDER = 4


# ---------------------------------------------------------------------------
# chart-space sampling primitives
# ---------------------------------------------------------------------------

def wrap_to_pi(angle):
    """Wrap angles into ``[-pi, pi)``.  Elementwise, shape-preserving."""
    return (np.asarray(angle, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def chart_axes(n_s, n_phi):
    """Cell-centre coordinates of a chart grid.

    Returns ``(s, phi)`` with shapes ``(n_s,)`` and ``(n_phi,)``.
    """
    s = (np.arange(int(n_s), dtype=float) + 0.5) / float(n_s)
    phi = -np.pi + (np.arange(int(n_phi), dtype=float) + 0.5) * (2.0 * np.pi / float(n_phi))
    return s, phi


def chart_indices(s, phi, n_s, n_phi):
    """Fractional chart indices for chart coordinates.

    ``s`` maps to ``s * n_s - 0.5`` (clamping happens at sample time), ``phi``
    maps to ``(phi + pi) / (2 pi) * n_phi - 0.5`` and is periodic mod ``n_phi``.
    """
    si = np.asarray(s, dtype=float) * float(n_s) - 0.5
    pj = (wrap_to_pi(phi) + np.pi) / (2.0 * np.pi) * float(n_phi) - 0.5
    return si, pj


def _bilinear_corners(si, pj, n_s, n_phi):
    """Corner indices and weights for bilinear chart lookup / splat."""
    i0 = np.floor(si).astype(np.int64)
    j0 = np.floor(pj).astype(np.int64)
    fi = si - i0
    fj = pj - j0
    corners = []
    for di in (0, 1):
        wi = fi if di else (1.0 - fi)
        ii = np.clip(i0 + di, 0, n_s - 1)          # s clamps: body has ends
        for dj in (0, 1):
            wj = fj if dj else (1.0 - fj)
            jj = (j0 + dj) % n_phi                  # phi wraps: body is closed
            corners.append((ii, jj, wi * wj))
    return corners


def sample_chart(chart, s, phi):
    """Bilinearly sample a chart image at chart coordinates.

    Args:
        chart: ``(n_s, n_phi)`` or ``(n_s, n_phi, C)`` array.
        s, phi: broadcastable arrays of chart coordinates.

    Returns:
        Array of shape ``broadcast(s, phi).shape`` (+ ``(C,)`` if the chart has
        channels).  ``s`` is clamped to the body ends, ``phi`` wraps.
    """
    arr = np.asarray(chart, dtype=float)
    n_s, n_phi = arr.shape[0], arr.shape[1]
    s_b, phi_b = np.broadcast_arrays(np.asarray(s, float), np.asarray(phi, float))
    si, pj = chart_indices(s_b, phi_b, n_s, n_phi)

    out = np.zeros(s_b.shape + arr.shape[2:], dtype=float)
    for ii, jj, w in _bilinear_corners(si, pj, n_s, n_phi):
        v = arr[ii, jj]
        out += v * (w[..., None] if arr.ndim == 3 else w)
    return out


def splat_to_chart(s, phi, values, n_s, n_phi, weights=None):
    """Accumulate scattered samples into a chart grid (bilinear splat).

    The adjoint of :func:`sample_chart`: mass is spread over the four
    surrounding cells, wrapping in ``phi`` and clamping in ``s``.

    Args:
        s, phi: ``(k,)`` chart coordinates.
        values: ``(k,)`` or ``(k, C)`` sample values.
        weights: optional ``(k,)`` per-sample confidence.

    Returns:
        ``(accum, weight)`` where ``accum`` has the chart's channel layout and
        ``weight`` is ``(n_s, n_phi)``.  The normalised chart is
        ``accum / weight`` wherever ``weight > 0``.
    """
    s = np.asarray(s, float).ravel()
    phi = np.asarray(phi, float).ravel()
    vals = np.asarray(values, float)
    vals = vals.reshape(len(s), -1)
    w = np.ones(len(s)) if weights is None else np.asarray(weights, float).ravel()

    si, pj = chart_indices(s, phi, n_s, n_phi)
    n_cells = n_s * n_phi
    accum = np.zeros((n_cells, vals.shape[1]))
    wsum = np.zeros(n_cells)
    for ii, jj, cw in _bilinear_corners(si, pj, n_s, n_phi):
        flat = ii * n_phi + jj
        cw = cw * w
        wsum += np.bincount(flat, weights=cw, minlength=n_cells)
        for c in range(vals.shape[1]):
            accum[:, c] += np.bincount(flat, weights=cw * vals[:, c], minlength=n_cells)
    accum = accum.reshape(n_s, n_phi, vals.shape[1])
    if np.ndim(values) == 1:
        accum = accum[..., 0]
    return accum, wsum.reshape(n_s, n_phi)


def darkness_to_multiplier(darkness, amplitude=1.0):
    """Convert a darkness map (0 = unmarked, 1 = fully dark) to a multiplier.

    ``multiplier = 1 - amplitude * darkness``, clipped to ``[0, 1]``.
    ``amplitude`` is the identity-signal strength knob -- the same idea as
    ``head_signal`` / ``body_signal`` in
    ``prototypes/01-melops-ablation/melops_data.make_synthetic``, where the
    layer amplitude (not the spot layout) is what a region's identity content
    is dialled with.
    """
    d = np.clip(np.asarray(darkness, dtype=float), 0.0, None)
    return np.clip(1.0 - float(amplitude) * d, 0.0, 1.0)


def from_pattern_chart(chart, semantics="darkness", amplitude=1.0):
    """Adapt a ``pattern``/``exclusions`` chart into this module's layout.

    Args:
        chart: ``(H_phi, W_s)`` array as ``pattern.render_chart`` emits it.
        semantics: ``"darkness"`` (0 = unmarked, the pattern module's
            convention) or ``"multiplier"`` (already 1 = unmarked).
        amplitude: identity-signal strength, applied when converting darkness.

    Returns:
        ``(n_s, n_phi)`` albedo multiplier, ready for
        :func:`bake_chart_to_texture`.

    The transpose is the whole geometric content: both modules agree on what
    ``s`` and ``phi`` MEAN, they disagree only on which one is axis 0.  Doing
    it here, once, by name, is the alternative to a bare ``.T`` scattered
    through callers -- where it is invisible on a square chart and silently
    rotates every pattern by a quarter turn.
    """
    arr = np.asarray(chart, dtype=float)
    if arr.ndim == 3:
        arr = np.moveaxis(arr, 0, 1)
    elif arr.ndim == 2:
        arr = arr.T
    else:
        raise ValueError("pattern chart must be 2-D or 3-D, got ndim=%d" % arr.ndim)
    if semantics == "darkness":
        return darkness_to_multiplier(arr, amplitude)
    if semantics == "multiplier":
        return arr
    raise ValueError("semantics must be 'darkness' or 'multiplier', got %r" % (semantics,))


def to_pattern_chart(chart, semantics="darkness"):
    """Inverse of :func:`from_pattern_chart`: ``(n_s, n_phi)`` mult -> ``(H_phi, W_s)``."""
    arr = np.asarray(chart, dtype=float)
    arr = np.moveaxis(arr, 0, 1) if arr.ndim == 3 else arr.T
    if semantics == "darkness":
        return np.clip(1.0 - arr, 0.0, 1.0)
    if semantics == "multiplier":
        return arr
    raise ValueError("semantics must be 'darkness' or 'multiplier', got %r" % (semantics,))


def luminance(rgb):
    """Rec.709 luma of an ``(..., >=3)`` array."""
    arr = np.asarray(rgb, dtype=float)
    w = np.asarray(LUMA_WEIGHTS)
    return arr[..., :3] @ w


# ---------------------------------------------------------------------------
# UV-space rasterisation
# ---------------------------------------------------------------------------

class RasterUV(NamedTuple):
    """Per-texel rasterisation of a mesh's UV atlas.

    face: ``(H, W)`` int32, index of the covering face, ``-1`` where uncovered.
    bary: ``(H, W, 3)`` barycentric weights of the texel centre in that face.
    covered: ``(H, W)`` bool, ``face >= 0``.
    """

    face: np.ndarray
    bary: np.ndarray
    covered: np.ndarray


def _mesh_uv(mesh):
    uv = getattr(getattr(mesh, "visual", None), "uv", None)
    if uv is None:
        raise ValueError(
            "mesh has no UV coordinates (mesh.visual.uv is None); bake needs a "
            "UV-unwrapped mesh -- see fixtures.make_uv_tube for the layout this "
            "module expects"
        )
    return np.asarray(uv, dtype=float)


def rasterize_uv(mesh, tex_size, uv=None):
    """Rasterise every face of a mesh into its UV atlas.

    Texel centres are ``((x + 0.5) / W, (y + 0.5) / H)``; a texel belongs to a
    face when its centre is inside the face's UV triangle.  Later faces
    overwrite earlier ones on overlap, so the result is deterministic (and
    overlap means a broken atlas anyway).

    Args:
        mesh: ``trimesh.Trimesh`` carrying ``visual.uv``.
        tex_size: int or ``(height, width)``.
        uv: optional ``(V, 2)`` override for ``mesh.visual.uv``.

    Returns:
        :class:`RasterUV`.

    Degenerate faces (zero UV area) are skipped; they cover no texel centre in
    any case and their barycentrics are undefined.
    """
    uv = _mesh_uv(mesh) if uv is None else np.asarray(uv, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if isinstance(tex_size, (int, np.integer)):
        h = w = int(tex_size)
    else:
        h, w = (int(v) for v in tex_size)

    face_id = np.full((h, w), -1, dtype=np.int32)
    bary = np.zeros((h, w, 3), dtype=float)

    # UV -> texel-centre coordinates: u * W - 0.5, v * H - 0.5.
    px = uv[:, 0] * w - 0.5
    py = uv[:, 1] * h - 0.5

    for fi in range(len(faces)):
        a, b, c = faces[fi]
        x0, y0 = px[a], py[a]
        x1, y1 = px[b], py[b]
        x2, y2 = px[c], py[c]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-12:
            continue
        xmin = max(0, int(np.floor(min(x0, x1, x2))))
        xmax = min(w - 1, int(np.ceil(max(x0, x1, x2))))
        ymin = max(0, int(np.floor(min(y0, y1, y2))))
        ymax = min(h - 1, int(np.ceil(max(y0, y1, y2))))
        if xmin > xmax or ymin > ymax:
            continue
        xs = np.arange(xmin, xmax + 1, dtype=float)
        ys = np.arange(ymin, ymax + 1, dtype=float)
        gx, gy = np.meshgrid(xs, ys)
        l0 = ((x1 - gx) * (y2 - gy) - (x2 - gx) * (y1 - gy)) / area
        l1 = ((x2 - gx) * (y0 - gy) - (x0 - gx) * (y2 - gy)) / area
        l2 = 1.0 - l0 - l1
        eps = -1e-9
        inside = (l0 >= eps) & (l1 >= eps) & (l2 >= eps)
        if not inside.any():
            continue
        sub_face = face_id[ymin:ymax + 1, xmin:xmax + 1]
        sub_bary = bary[ymin:ymax + 1, xmin:xmax + 1]
        sub_face[inside] = fi
        sub_bary[inside] = np.stack([l0, l1, l2], axis=-1)[inside]

    return RasterUV(face=face_id, bary=bary, covered=face_id >= 0)


def texel_chart_coords(mesh, vertex_s, vertex_phi, tex_size, raster=None, uv=None):
    """Per-texel ``(s, phi)`` of a UV atlas, with correct phi seam handling.

    ``s`` interpolates linearly.  ``phi`` CANNOT: a face whose corners read
    ``+3.10`` and ``-3.10`` rad is 0.08 rad wide, not 6.20 rad wide, and naive
    barycentric interpolation of the wrapped values sweeps the whole animal.
    Each face is therefore UNWRAPPED against its first corner (every other
    corner is shifted by the multiple of ``2 pi`` that brings it within ``pi``
    of corner 0), interpolated in that unwrapped frame, and re-wrapped.  This
    is exact for a chart that is linear in ``phi`` across the face, which holds
    whenever faces subtend less than ``pi`` of girth -- i.e. for any tube with
    at least 3 vertices around, and comfortably for the >= 16 that a usable
    mesh has.

    (The alternative -- interpolate ``(cos phi, sin phi)`` and take ``atan2``
    -- is also seam-correct but is a slerp-shaped, not linear, interpolant; it
    disagrees with the exact answer by up to ~1% of a face's angular width.
    Unwrapping is used because it is exact.)

    Returns:
        ``(s_tex, phi_tex, raster)``.  Uncovered texels carry ``NaN``.
    """
    if raster is None:
        raster = rasterize_uv(mesh, tex_size, uv=uv)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vs = np.asarray(vertex_s, dtype=float)
    vp = np.asarray(vertex_phi, dtype=float)
    n_v = len(mesh.vertices)
    if len(vs) != n_v or len(vp) != n_v:
        raise ValueError(
            "vertex_s/vertex_phi must be per-vertex: got %d/%d for %d vertices"
            % (len(vs), len(vp), n_v)
        )

    fs = vs[faces]                       # (F, 3)
    fp = vp[faces]                       # (F, 3)
    ref = fp[:, :1]
    fp_unwrapped = ref + wrap_to_pi(fp - ref)

    cov = raster.covered
    fid = raster.face[cov]
    bw = raster.bary[cov]
    s_tex = np.full(cov.shape, np.nan)
    phi_tex = np.full(cov.shape, np.nan)
    s_tex[cov] = np.einsum("kc,kc->k", bw, fs[fid])
    phi_tex[cov] = wrap_to_pi(np.einsum("kc,kc->k", bw, fp_unwrapped[fid]))
    return s_tex, phi_tex, raster


# ---------------------------------------------------------------------------
# de-lighting
# ---------------------------------------------------------------------------

def masked_gaussian(field, mask, sigma, wrap_axis=None):
    """Gaussian blur that ignores masked-out cells (normalised convolution).

    Returns ``(blurred, weight)``.  ``weight`` is the blurred mask, so cells
    the kernel barely reached are identifiable; ``blurred`` is meaningless
    where ``weight`` is near zero.  ``wrap_axis`` makes one axis periodic --
    always axis 1 for a chart, whose ``phi`` wraps.
    """
    m = mask.astype(float)
    f = np.where(mask, field, 0.0)
    modes = ["nearest", "nearest"]
    if wrap_axis is not None:
        modes[wrap_axis] = "wrap"
    if modes[0] == modes[1]:
        num = ndimage.gaussian_filter(f, sigma, mode=modes[0])
        den = ndimage.gaussian_filter(m, sigma, mode=modes[0])
    else:
        num, den = f, m
        for ax, (sg, md) in enumerate(zip(np.atleast_1d(sigma) * np.ones(2), modes)):
            num = ndimage.gaussian_filter1d(num, sg, axis=ax, mode=md)
            den = ndimage.gaussian_filter1d(den, sg, axis=ax, mode=md)
    return num / np.maximum(den, 1e-9), den


def _lowfreq_basis(s, phi, n_harmonics, poly_order):
    """Tensor-product design matrix: Legendre in ``s`` x Fourier in ``phi``.

    Columns are ``P_m(2s - 1) * {1, cos k phi, sin k phi}`` for
    ``m <= poly_order``, ``1 <= k <= n_harmonics``.  The Fourier half is
    periodic BY CONSTRUCTION, so the fit cannot invent a discontinuity at
    ``+-pi`` the way a non-wrapping smoother does.
    """
    s = np.asarray(s, float).ravel()
    phi = np.asarray(phi, float).ravel()
    x = np.clip(2.0 * s - 1.0, -1.0, 1.0)
    polys = [np.ones_like(x)]
    if poly_order >= 1:
        polys.append(x)
    for m in range(2, int(poly_order) + 1):     # Legendre recurrence
        polys.append(((2 * m - 1) * x * polys[m - 1] - (m - 1) * polys[m - 2]) / m)
    angs = [np.ones_like(phi)]
    for k in range(1, int(n_harmonics) + 1):
        angs.append(np.cos(k * phi))
        angs.append(np.sin(k * phi))
    return np.column_stack([p * a for p in polys for a in angs])


def _fit_lowfreq_log(s, phi, values, weights, n_harmonics, poly_order, robust_iters):
    """Weighted, spot-robust least squares of ``log(values)`` on the low-freq basis.

    Fitted in LOG space because shading is multiplicative.  Two asymmetric
    reweighting passes follow: cells whose residual is strongly NEGATIVE are
    downweighted, because a dark outlier in a shark's skin is a SPOT -- the
    signal -- not a shading feature.  Without that, dense speckling drags the
    "unmarked skin level" down and de-lighting eats part of the pattern.
    """
    A = _lowfreq_basis(s, phi, n_harmonics, poly_order)
    y = np.log(np.maximum(np.asarray(values, float).ravel(), 1e-6))
    w = np.asarray(weights, float).ravel().copy()
    coef = None
    for it in range(int(robust_iters) + 1):
        sw = np.sqrt(np.maximum(w, 0.0))[:, None]
        coef, *_ = np.linalg.lstsq(A * sw, y * sw[:, 0], rcond=None)
        if it == int(robust_iters):
            break
        res = y - A @ coef
        mad = float(np.median(np.abs(res - np.median(res)))) * 1.4826
        if mad < 1e-9:
            break
        w = np.where(res < -1.5 * mad, w * 0.05, w)
    return coef


def estimate_lowfreq_luminance(
    base_albedo,
    raster,
    s_tex,
    phi_tex,
    space="chart",
    method="basis",
    sigma_s=DELIGHT_SIGMA_S,
    sigma_phi=DELIGHT_SIGMA_PHI,
    chart_shape=(96, 192),
    n_harmonics=DELIGHT_HARMONICS,
    poly_order=DELIGHT_POLY_ORDER,
    robust_iters=2,
):
    """Low-frequency luminance of a texture, for de-lighting.

    Args:
        base_albedo: ``(H, W, >=3)`` texture, float in ``[0, 1]``.
        raster, s_tex, phi_tex: output of :func:`texel_chart_coords`.
        space: ``"chart"`` (default) or ``"uv"``.
        method: ``"basis"`` (default) fits a low-order Legendre x Fourier
            surface in log space; ``"blur"`` uses a masked Gaussian.
            ``space="uv"`` always uses a blur.
        sigma_s, sigma_phi: chart-space cutoff, for ``method="blur"``.
        chart_shape: resolution of the intermediate chart.
        n_harmonics, poly_order: basis size for ``method="basis"``.
        robust_iters: asymmetric reweighting passes that stop dark spots from
            dragging the fitted skin level down.

    Returns:
        ``(H, W)`` low-frequency luminance; ``NaN`` on uncovered texels.

    WHY CHART SPACE.  A blur in UV space mixes texels that are adjacent in the
    atlas but far apart on the animal -- across an atlas seam a dorsal texel
    can average with a ventral one, and the "shading" estimate picks up a
    discontinuity exactly where the atlas was cut.  Working in ``(s, phi)``
    stays on the SURFACE and wraps correctly around the girth.  ``space="uv"``
    is kept for comparison and for meshes whose ``(s, phi)`` is unavailable.

    WHY A BASIS FIT AND NOT A BLUR.  The dominant lighting term on a horizontal
    animal is a dorsal-to-ventral gradient, i.e. ``cos(phi)`` -- the LOWEST
    non-constant frequency there is around the girth.  A Gaussian with
    ``sigma_phi = 0.8`` rad attenuates it to ``exp(-0.5 * 0.8^2) ~ 0.73``, so a
    blur-based estimate leaves ~27% of the very term it exists to remove
    (measured on this prototype's fixtures: correlation to the clean pattern
    recovers to 0.67, not to ~1).  A basis containing ``cos phi`` and ``sin phi``
    represents that gradient EXACTLY and removes it completely, while a spot --
    a few cells wide -- is not representable at ``n_harmonics = 3`` and survives
    untouched.  The cost is that the basis is global: one badly blown-out
    region tilts the whole fit, which the robust passes only partly contain.
    """
    cov = raster.covered
    lum = luminance(base_albedo)
    if space == "uv":
        h, w = cov.shape
        sig = 0.5 * (sigma_s * h + (sigma_phi / (2 * np.pi)) * w)
        low, _ = masked_gaussian(lum, cov, sig)
        return np.where(cov, low, np.nan)
    if space != "chart":
        raise ValueError("space must be 'chart' or 'uv', got %r" % (space,))

    n_s, n_phi = int(chart_shape[0]), int(chart_shape[1])
    acc, wgt = splat_to_chart(s_tex[cov], phi_tex[cov], lum[cov], n_s, n_phi)
    have = wgt > 0
    chart_lum = np.zeros_like(acc)
    chart_lum[have] = acc[have] / wgt[have]

    if method == "basis":
        s_ax, phi_ax = chart_axes(n_s, n_phi)
        S, P = np.meshgrid(s_ax, phi_ax, indexing="ij")
        usable = have & (chart_lum > 1e-6)
        if usable.sum() < 4 * (poly_order + 1) * (2 * n_harmonics + 1):
            method = "blur"                      # not enough data to fit; fall back
        else:
            coef = _fit_lowfreq_log(
                S[usable], P[usable], chart_lum[usable], wgt[usable],
                n_harmonics, poly_order, robust_iters,
            )
            out = np.full(cov.shape, np.nan)
            A = _lowfreq_basis(s_tex[cov], phi_tex[cov], n_harmonics, poly_order)
            out[cov] = np.exp(A @ coef)
            return out
    if method != "blur":
        raise ValueError("method must be 'basis' or 'blur', got %r" % (method,))

    sig = (max(sigma_s * n_s, 0.5), max(sigma_phi / (2 * np.pi) * n_phi, 0.5))
    low_chart, _ = masked_gaussian(chart_lum, have, sig, wrap_axis=1)
    out = np.full(cov.shape, np.nan)
    out[cov] = sample_chart(low_chart, s_tex[cov], phi_tex[cov])
    return out


def _delight(base_albedo, raster, s_tex, phi_tex, space, method, sigma_s,
             sigma_phi, chart_shape, n_harmonics, poly_order):
    cov = raster.covered
    low = estimate_lowfreq_luminance(
        base_albedo, raster, s_tex, phi_tex,
        space=space, method=method, sigma_s=sigma_s, sigma_phi=sigma_phi,
        chart_shape=chart_shape, n_harmonics=n_harmonics, poly_order=poly_order,
    )
    finite = cov & np.isfinite(low) & (low > 1e-6)
    if not finite.any():
        return np.array(base_albedo, dtype=float), np.ones(cov.shape)
    ref = float(np.median(low[finite]))
    gain = np.ones(cov.shape)
    gain[finite] = ref / low[finite]
    # Guard rail: a x4 correction is no longer de-lighting, it is inventing
    # albedo out of a black pixel.  Clamp and let the caller see it in tests.
    gain = np.clip(gain, 0.25, 4.0)
    out = np.array(base_albedo, dtype=float).copy()
    out[..., :3] = np.clip(out[..., :3] * gain[..., None], 0.0, 1.0)
    return out, gain


# ---------------------------------------------------------------------------
# the two public conversions
# ---------------------------------------------------------------------------

def _warn_if_inverted(chart, semantics):
    """Shout when a chart looks like the other value convention.

    A speckled animal is mostly unmarked skin, so a MULTIPLIER chart has mean
    near 1 and a DARKNESS chart mean near 0; the two populations are far apart.
    Getting this wrong inverts the pattern -- dark skin with light spots -- and
    the bake still succeeds, so it is exactly the failure that reaches the
    dataset unnoticed.  This never overrides the caller, it only complains.
    """
    arr = np.asarray(chart, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return
    mean = float(finite.mean())
    if semantics == "multiplier" and mean < 0.35:
        warnings.warn(
            "chart_semantics='multiplier' but the chart's mean is %.3f; a "
            "multiplier chart is mostly unmarked skin (mean near 1). This "
            "looks like a darkness map -- pass chart_semantics='darkness' or "
            "bake.from_pattern_chart()." % mean,
            RuntimeWarning, stacklevel=3,
        )
    elif semantics == "darkness" and mean > 0.65:
        warnings.warn(
            "chart_semantics='darkness' but the chart's mean is %.3f; a "
            "darkness map is mostly zero. This looks like an albedo "
            "multiplier -- pass chart_semantics='multiplier'." % mean,
            RuntimeWarning, stacklevel=3,
        )


def _as_float_rgb(img, shape):
    """Coerce a base albedo argument to ``(H, W, 3)`` float in [0, 1]."""
    if img is None:
        return np.ones(shape + (3,), dtype=float)
    arr = np.asarray(img)
    if arr.ndim == 0 or (arr.ndim == 1 and arr.size in (1, 3)):
        triple = np.asarray(arr, dtype=float).reshape(-1) * np.ones(3)
        return np.tile(triple[None, None, :], shape + (1,))
    if arr.dtype == np.uint8:
        arr = arr.astype(float) / 255.0
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[:2] != shape:
        raise ValueError(
            "base_albedo shape %r does not match tex_size %r" % (arr.shape[:2], shape)
        )
    return arr[..., :3].copy()


def bake_chart_to_texture(
    mesh,
    vertex_s,
    vertex_phi,
    chart_image,
    tex_size,
    base_albedo=None,
    delight=True,
    chart_semantics="multiplier",
    amplitude=1.0,
    delight_space="chart",
    delight_method="basis",
    delight_sigma_s=DELIGHT_SIGMA_S,
    delight_sigma_phi=DELIGHT_SIGMA_PHI,
    delight_chart_shape=(96, 192),
    delight_harmonics=DELIGHT_HARMONICS,
    delight_poly_order=DELIGHT_POLY_ORDER,
    gutter=2,
    return_debug=False,
    raster=None,
):
    """Bake a canonical-chart pattern onto a mesh's UV texture.

    Args:
        mesh: ``trimesh.Trimesh`` with ``visual.uv``.
        vertex_s: ``(V,)`` arc-length fraction per vertex, 0 snout -> 1 caudal.
        vertex_phi: ``(V,)`` circumferential angle per vertex in ``(-pi, pi]``,
            0 dorsal, ``+pi/2`` the animal's left.  For a real mesh these come
            from prototype 04: ``tc = mesh3d.tube_coords(mesh, centerline)``,
            then ``vertex_s = tc.s / tc.total_length``, ``vertex_phi = tc.phi``.
        chart_image: ``(n_s, n_phi)`` or ``(n_s, n_phi, 3)`` chart, see
            ``chart_semantics``.
        tex_size: int or ``(height, width)`` of the output texture.
        base_albedo: ``(H, W, 3)`` texture, an RGB triple, or ``None`` for
            white (in which case the output IS the chart, resampled into UV).
        delight: divide out ``base_albedo``'s low-frequency luminance first.
        chart_semantics: ``"multiplier"`` (1 = unmarked) or ``"darkness"``
            (0 = unmarked), the latter scaled by ``amplitude``.
        gutter: dilate colour this many texels past the covered region so
            bilinear texture filtering does not sample background at UV island
            edges.  Alpha stays at true coverage, so gutter texels are
            identifiable as ``alpha == 0``.
        raster: optional :class:`RasterUV` from :func:`rasterize_uv` for this
            mesh at this ``tex_size``.  Purely a cache -- it depends on the UV
            atlas and the texture size and on nothing else, so reusing it
            across bakes of different charts onto the same mesh is exact.

    Returns:
        ``(H, W, 4)`` float32 RGBA in ``[0, 1]``.  Alpha is TRUE COVERAGE
        (1 inside the atlas, 0 outside), not opacity of the pattern.
        With ``return_debug=True``, returns ``(texture, dict)`` where the dict
        carries ``s``, ``phi``, ``raster``, ``gain`` and ``albedo``.

    DE-LIGHTING -- what it is and what it cannot do.
    This is the owner's "excluding shadows", applied at the TEXTURE level.  A
    photogrammetry or AI-generated texture has the capture's lighting baked in:
    a bright dorsal highlight, a dark ventral occlusion, a cast shadow from a
    fin.  Multiplying an identity chart onto that texture entangles identity
    with that particular capture's lighting, and every render of the individual
    then carries a lighting fingerprint that a re-ID model can latch onto -- a
    catastrophic shortcut in a synthetic corpus.  So the low-frequency
    luminance is estimated on the surface (see
    :func:`estimate_lowfreq_luminance`) and divided out.

    LIMITS, stated plainly:
      * The divide CANNOT distinguish low-frequency SHADING from low-frequency
        ALBEDO.  Sevengill countershading -- dark grey-brown dorsum, lighter
        ventrum -- is exactly such a low-frequency albedo term, and de-lighting
        flattens it along with the shading.  If you want countershading in the
        output, put it back in chart space AFTER the bake (it belongs to the
        species, not to the capture).  This is a deliberate separation, not a
        bug: it is what keeps identity, species tone and capture lighting in
        three separable layers.
      * It is a scalar gain, so it does not correct coloured illuminants.
      * Hard shadow EDGES are not low frequency and survive.  Only shadows
        softer than ``delight_sigma_*`` are removed.
      * The gain is clamped to ``[0.25, 4]``; a fully black region cannot be
        recovered, and clipping there is reported in the debug dict.
      * Cast shadows are removed as a 2D field on the surface.  Nothing here
        knows the light direction, so a genuinely 3D de-lighting (inverse
        rendering) will beat it whenever multiple views of the same surface
        exist.
    """
    if isinstance(tex_size, (int, np.integer)):
        shape = (int(tex_size), int(tex_size))
    else:
        shape = (int(tex_size[0]), int(tex_size[1]))

    # ``raster`` is an OPTIONAL cache: rasterize_uv depends only on the atlas
    # and the texture size, never on the chart, so a caller baking many
    # patterns onto ONE mesh (make_dataset.py) can compute it once and pass
    # it in. Omitting it is the original behaviour, bit for bit.
    s_tex, phi_tex, raster = texel_chart_coords(mesh, vertex_s, vertex_phi,
                                                shape, raster=raster)
    cov = raster.covered
    albedo = _as_float_rgb(base_albedo, shape)

    gain = np.ones(shape)
    if delight and base_albedo is not None:
        albedo, gain = _delight(
            albedo, raster, s_tex, phi_tex, delight_space, delight_method,
            delight_sigma_s, delight_sigma_phi, delight_chart_shape,
            delight_harmonics, delight_poly_order,
        )

    chart = np.asarray(chart_image, dtype=float)
    if chart_semantics == "darkness":
        _warn_if_inverted(chart, "darkness")
        chart = darkness_to_multiplier(chart, amplitude)
    elif chart_semantics == "multiplier":
        _warn_if_inverted(chart, "multiplier")
    else:
        raise ValueError(
            "chart_semantics must be 'multiplier' or 'darkness', got %r" % (chart_semantics,)
        )

    mult = np.ones(shape + (3,))
    sampled = sample_chart(chart, s_tex[cov], phi_tex[cov])
    if sampled.ndim == 1:
        sampled = sampled[:, None]
    mult[cov] = sampled if sampled.shape[1] == 3 else np.repeat(sampled[:, :1], 3, axis=1)

    rgb = np.clip(albedo * mult, 0.0, 1.0)
    # Every uncovered atlas texel takes the nearest COVERED texel's colour (the
    # alpha channel below still marks it as a gutter). Writing black there put
    # dark slivers on every sub-texel face - the fin-root slits and the atlas
    # seam rendered as stripes - and a base albedo can itself be NaN off-atlas
    # (de-lighting leaves it undefined there), so copying the base is not safe.
    if cov.any() and not cov.all():
        _, (iy_all, ix_all) = ndimage.distance_transform_edt(~cov, return_indices=True)
        rgb[~cov] = rgb[iy_all[~cov], ix_all[~cov]]
    rgb = np.nan_to_num(rgb, nan=0.0)
    tex = np.zeros(shape + (4,), dtype=np.float32)
    tex[..., :3] = rgb
    tex[..., 3] = cov.astype(np.float32)

    if gutter > 0 and cov.any() and not cov.all():
        _, (iy, ix) = ndimage.distance_transform_edt(~cov, return_indices=True)
        near = ndimage.binary_dilation(cov, iterations=int(gutter)) & ~cov
        tex[..., :3][near] = rgb[iy[near], ix[near]]

    if return_debug:
        debug = {
            "s": s_tex, "phi": phi_tex, "raster": raster,
            "gain": gain, "albedo": albedo,
            "gain_clipped_frac": float(
                np.mean((gain[cov] <= 0.2501) | (gain[cov] >= 3.9999)) if cov.any() else 0.0
            ),
            "coverage_frac": float(cov.mean()),
        }
        return tex, debug
    return tex


def mesh_texture_to_chart(
    mesh,
    uv_texture,
    vertex_s,
    vertex_phi,
    chart_shape=(128, 256),
    alpha_threshold=0.5,
    fill_holes=True,
    return_coverage=False,
):
    """Read an existing textured mesh INTO canonical chart space (the inverse).

    Every covered texel is bilinearly splatted into the chart at its own
    ``(s, phi)``, then normalised by the accumulated weight.  This is the
    adjoint of the forward bake, so ``bake -> mesh_texture_to_chart`` is a
    resampling round trip: it recovers the source chart up to (a) the texel
    density relative to ``chart_shape`` and (b) two bilinear filterings.

    Args:
        mesh: ``trimesh.Trimesh`` with ``visual.uv``.
        uv_texture: ``(H, W, 3)`` or ``(H, W, 4)`` texture, float ``[0, 1]`` or
            uint8.  If 4-channel, ``alpha < alpha_threshold`` texels (gutter
            and background) are dropped.
        vertex_s, vertex_phi: as in :func:`bake_chart_to_texture`.
        chart_shape: ``(n_s, n_phi)`` of the output.
        fill_holes: fill cells no texel reached by iterated masked blur.  Cells
            still empty afterwards are left at ``NaN``.

    Returns:
        ``(n_s, n_phi, 3)`` chart, or ``(chart, coverage)`` with
        ``return_coverage=True`` where ``coverage`` is the splat weight per
        cell (0 = nothing landed there).

    SAMPLING GUIDANCE.  Choose ``chart_shape`` so that cells are not finer than
    the texture: for a tube of ``tex_size`` T covering the whole atlas, roughly
    ``n_s <= T`` and ``n_phi <= T``.  Finer than that and the chart holds
    interpolation, not measurement, and ``coverage`` will show it.
    """
    tex = np.asarray(uv_texture)
    if tex.dtype == np.uint8:
        tex = tex.astype(float) / 255.0
    tex = np.asarray(tex, dtype=float)
    shape = tex.shape[:2]

    s_tex, phi_tex, raster = texel_chart_coords(mesh, vertex_s, vertex_phi, shape)
    cov = raster.covered.copy()
    if tex.ndim == 3 and tex.shape[2] == 4:
        cov &= tex[..., 3] >= alpha_threshold
    rgb = tex[..., :3] if tex.ndim == 3 else np.repeat(tex[..., None], 3, axis=2)

    n_s, n_phi = int(chart_shape[0]), int(chart_shape[1])
    acc, wgt = splat_to_chart(s_tex[cov], phi_tex[cov], rgb[cov], n_s, n_phi)
    have = wgt > 0
    chart = np.full((n_s, n_phi, 3), np.nan)
    chart[have] = acc[have] / wgt[have][:, None]

    if fill_holes and not have.all():
        filled = np.where(have[..., None], np.nan_to_num(chart), 0.0)
        m = have.copy()
        for _ in range(8):
            if m.all():
                break
            blurred = np.empty_like(filled)
            den = None
            for c in range(3):
                blurred[..., c], den = masked_gaussian(
                    filled[..., c], m, (1.0, 1.0), wrap_axis=1
                )
            newly = (den > 1e-3) & ~m
            if not newly.any():
                break
            filled[newly] = blurred[newly]
            m |= newly
        chart = np.where(m[..., None], filled, np.nan)

    if return_coverage:
        return chart, wgt
    return chart
