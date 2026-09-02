"""Turn the scan's own skin texture into catalogue individual #0.

The owner's Meshy GLB carries a *photo-projected* texture of a real sevengill.
De-bending (``mesh3d.debend``) moves vertices and leaves ``faces``, ``visual``
(UVs, material, image) untouched, so that photograph is still glued to the
straightened body afterwards.  This module reads it back off the surface into
prototype 05's canonical ``(s, phi)`` chart, removes the capture's lighting,
and fits a spot table to what is left.  The result is an ordinary
``pattern.Individual`` -- so the real animal enters the synthetic catalogue as
**individual #0** with no photograph, no annotation and no detection step, and
``drift.resight`` can then age it exactly like a synthetic one.

Pipeline (one call, :func:`run`)::

    GLB -> load_mesh -> extract_centerline_3d -> tube_frames -> tube_coords
        -> debend                         (04: the straight, still-textured mesh)
        -> detect_fins                    (which texels are body skin)
        -> chart_coords                   (04 (s, r, phi) -> 05 (s in [0,1], phi))
        -> de-light (bake's own path)     chart_delighted.png
        -> bake.mesh_texture_to_chart     chart_real.png, fin texels dropped
        -> pattern.copy_from_chart        individual0.json
        -> drift.resight x N              resight_NN.glb
        -> pattern.randomize x M          random_NN.glb
        -> contact_sheet.png

CONVENTION BRIDGE -- the one thing that must be got right.
Prototype 04 measures ``phi = atan2(v.B, v.N)`` from dorsal ``+Z`` toward the
animal's left ``+Y``, in the half-open interval ``(-pi, pi]``: ventral is
``+pi``.  Prototype 05 uses the same zero and the same sign but the OTHER
half-open interval, ``[-pi, pi)``: ventral is ``-pi``.  The two agree
everywhere except on the ventral midline itself, where 04 says ``+pi`` and 05
says ``-pi`` -- and a vertex exactly on that seam is not rare, it is every
vertex of the ventral seam column of a UV-unwrapped tube.  :func:`chart_coords`
converts with ``bake.wrap_to_pi`` and nothing else, which maps ``+pi -> -pi``
and is the identity elsewhere; the conversion is asserted in the tests rather
than left to the reader.

``s`` is the other half of the bridge.  04's ``TubeCoords.s`` is an arc length
in metres along the *chart*, and it deliberately runs outside ``[0, L]``:
``extract_centerline_3d`` stops at the peduncle, so the caudal fin overhangs
the far end and the snout cap overhangs the near one.  05 wants a fraction
with ``0 = snout tip`` and ``1 = caudal terminus``.  The default
``normalize="extent"`` therefore rescales the *observed* range of ``s`` over
the mesh's vertices onto ``[0, 1]``, which puts the two anatomical ends where
05's chart convention says they are.  ``normalize="chart"`` divides by the
chart length instead (the recipe in ``bake.bake_chart_to_texture``'s
docstring); it is exact on the body and clamps the overhang onto the chart
ends, which is wrong for a heterocercal tail.

DE-LIGHTING is the load-bearing step, and it is bake's, not this module's:
``bake_chart_to_texture(..., base_albedo=real_texture, delight=True)`` with a
unit multiplier chart returns the de-lighted albedo and nothing else, so the
low-frequency luminance divide used here is *literally* the one 05 uses when
it bakes a synthetic pattern.  What it removes and what it cannot is stated in
that function's docstring and repeated in the README section; the short form is
that it removes smooth shading and leaves hard shadow edges and speculars.

Run::

    python texture_identity.py --glb IN.glb --out DIR \\
        [--n-resights 4] [--years 3] [--n-random 3] [--seed 0]

With no ``--glb`` the module builds ``synth.make_sevengill(textured=True)``,
whose procedural skin stands in for the Meshy photo texture.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from typing import NamedTuple

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gltf_export  # noqa: E402
import mesh3d  # noqa: E402
import synth  # noqa: E402


# ---------------------------------------------------------------------------
# Prototype 05, imported read-only through a sys.path shim
# ---------------------------------------------------------------------------

DEFAULT_P05_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "05-synthetic-identities"
    )
)


def _add_p05_to_path(path=None):
    """Put prototype 05 on ``sys.path`` and return the directory used.

    05 is a sibling prototype, not a package: its modules import each other by
    bare name (``from exclusions import ...``), so the directory itself has to
    be importable.  ``$SEVENGILL_P05_DIR`` overrides the default location.
    Nothing under that directory is ever written by this module.

    APPENDED, not prepended: 05 contributes half a dozen very ordinary
    top-level names (``render``, ``pattern``, ``fixtures``, ...) to whatever
    process imports this, and putting them ahead of the caller's own path would
    let them shadow the caller's modules.  At the end they can only be shadowed,
    which fails loudly at import rather than quietly at run time.
    """
    d = os.path.abspath(path or os.environ.get("SEVENGILL_P05_DIR", DEFAULT_P05_DIR))
    if not os.path.isdir(d):
        raise ImportError(
            "prototype 05 not found at %r; set $SEVENGILL_P05_DIR to the "
            "directory holding bake.py / pattern.py / drift.py" % d
        )
    if d not in sys.path:
        sys.path.append(d)
    return d


P05_DIR = _add_p05_to_path()

import bake  # noqa: E402
import drift  # noqa: E402
import exclusions  # noqa: E402
import pattern  # noqa: E402
import render as p05_render  # noqa: E402


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Chart girth resolution.  128 rows over 2 pi is ~2.8 deg per row; the
#: matching arc-length resolution comes from ``pattern.isotropic_resolution``
#: so that a chart pixel is square in the scaled chart metric.
CHART_H_PHI = 128

#: Working texture size, in texels.  A Meshy texture is commonly 2048 or 4096;
#: rasterising a 12k-face atlas at 4096 costs minutes and buys nothing, because
#: the chart it feeds is 240 x 128.  Source textures larger than this are
#: box-resampled down to it (reported in the summary).
DEFAULT_TEX_SIZE = 1024

#: Side-view render size for the contact sheet, ``(H, W)``.
PREVIEW_RESOLUTION = (200, 500)

DEFAULT_START_DATE = "2020-01-01"
DEFAULT_LENGTH_CM = 250.0

#: A resight GLB must keep the atlas bit-for-bit; this is the tolerance the
#: round trip through trimesh's GLB writer actually achieves (measured ~4e-8).
UV_ROUND_TRIP_TOL = 1e-6

DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "demo", "identity"
)


def default_chart_shape(h_phi=CHART_H_PHI, phi_scale=0.085):
    """``(n_s, n_phi)`` in BAKE layout, isotropic in the scaled chart metric.

    Two layouts are in play and they are transposes of one another:

    * **bake layout** ``(n_s, n_phi)`` -- ``bake.py``, ``sample_chart``,
      ``bake_chart_to_texture``, ``mesh_texture_to_chart``;
    * **pattern layout** ``(H_phi, W_s)`` -- ``pattern.render_chart``,
      ``exclusions.build_exclusion_mask``, ``drift.similarity``.

    Every function in this module states which one it takes.  The default is
    ``(240, 128)``: 240 cells along the body, 128 around the girth.
    """
    h, w = pattern.isotropic_resolution(int(h_phi), float(phi_scale))
    return (int(w), int(h))


# ---------------------------------------------------------------------------
# 04 -> 05 convention bridge
# ---------------------------------------------------------------------------

def chart_coords(mesh_or_coords, centerline=None, frames=None, normalize="extent"):
    """Per-vertex ``(vertex_s, vertex_phi)`` in prototype 05's chart convention.

    Args:
        mesh_or_coords: a ``trimesh.Trimesh`` (charted here) or an existing
            ``mesh3d.TubeCoords``.
        centerline: required when a mesh is passed.
        frames: optional ``mesh3d.tube_frames`` output.
        normalize: ``"extent"`` (default) rescales the observed ``s`` range
            onto ``[0, 1]`` so ``0`` is the snout tip and ``1`` the caudal
            terminus, which is what 05's ``CHART_CONVENTION`` means; ``"chart"``
            divides by the centerline length and clips, i.e. the recipe in
            ``bake.bake_chart_to_texture``'s docstring, which pins the body
            correctly and folds the caudal overhang onto ``s = 1``.

    Returns:
        ``(vertex_s, vertex_phi)``, both ``(V,)`` float64.  ``vertex_s`` is in
        ``[0, 1]``; ``vertex_phi`` is in ``[-pi, pi)`` -- note the half-open
        end, see the module docstring: 04 hands out ``(-pi, pi]`` and the
        ventral seam sits exactly on the disagreement.
    """
    if isinstance(mesh_or_coords, mesh3d.TubeCoords):
        coords = mesh_or_coords
    else:
        if centerline is None:
            raise ValueError("centerline is required when a mesh is passed")
        coords = mesh3d.tube_coords(mesh_or_coords, centerline, frames)

    s = np.asarray(coords.s, dtype=float)
    if normalize == "extent":
        lo, hi = float(s.min()), float(s.max())
        span = hi - lo
        if span <= 1e-12:
            raise ValueError("degenerate arc-length range on this mesh")
        s_frac = (s - lo) / span
    elif normalize == "chart":
        s_frac = np.clip(s / max(float(coords.total_length), 1e-12), 0.0, 1.0)
    else:
        raise ValueError("normalize must be 'extent' or 'chart', got %r" % (normalize,))

    # THE SEAM.  04's phi is in (-pi, pi]; 05's is in [-pi, pi).  wrap_to_pi is
    # the identity on (-pi, pi) and sends the single disputed value +pi to -pi,
    # which is where 05 puts the ventral midline.  Doing this by hand (or not
    # at all) is the bug that puts a stripe of ventral vertices at the wrong
    # end of the chart's phi axis.
    phi = bake.wrap_to_pi(np.asarray(coords.phi, dtype=float))
    return np.ascontiguousarray(s_frac), np.ascontiguousarray(phi)


def texture_image(mesh, tex_size=None):
    """The mesh's base-colour texture as ``(H, W, 3)`` float in ``[0, 1]``.

    Handles both material classes trimesh produces: ``PBRMaterial`` (a GLB,
    ``baseColorTexture``) and ``SimpleMaterial`` (an OBJ or a mesh built in
    memory, ``image``).  ``tex_size`` box-resamples to at most that many texels
    on the longer side -- see :data:`DEFAULT_TEX_SIZE` for why.
    """
    from PIL import Image

    visual = getattr(mesh, "visual", None)
    material = getattr(visual, "material", None)
    img = getattr(material, "baseColorTexture", None)
    if img is None:
        img = getattr(material, "image", None)
    if img is None:
        raise ValueError(
            "mesh carries no base-colour texture (material=%r); this pipeline "
            "reads identity off the scan's own skin, so an untextured mesh has "
            "nothing to read" % (type(material).__name__,)
        )
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img))
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    if tex_size is not None:
        cap = int(tex_size)
        longest = max(img.size)
        if longest > cap:
            scale = cap / float(longest)
            img = img.resize(
                (max(1, int(round(img.size[0] * scale))),
                 max(1, int(round(img.size[1] * scale)))),
                Image.BOX,
            )
    arr = np.asarray(img, dtype=np.float64)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.ascontiguousarray(arr[..., :3])


# ---------------------------------------------------------------------------
# Step 1: load, straighten, chart
# ---------------------------------------------------------------------------

class Straightened(NamedTuple):
    """The de-bent, still-textured mesh plus everything the bake needs.

    mesh: the straight ``trimesh.Trimesh`` (snout ``+X``, dorsal ``+Z``).
    centerline: its straight centerline.
    coords: ``mesh3d.TubeCoords`` of ``mesh`` on that centerline.
    vertex_s, vertex_phi: 05-convention chart coordinates (see
        :func:`chart_coords`).
    fins: ``mesh3d.FinDetection`` on the straight pose.
    texture: ``(H, W, 3)`` float working copy of the source texture.
    info: dict from ``extract_centerline_3d`` plus timings and sizes.
    """

    mesh: trimesh.Trimesh
    centerline: np.ndarray
    coords: object
    vertex_s: np.ndarray
    vertex_phi: np.ndarray
    fins: object
    texture: np.ndarray
    info: dict


def straighten(source, n_stations=64, tex_size=DEFAULT_TEX_SIZE, seed=0,
               normalize="extent", report=False, detect_fins=True):
    """Load (or accept) a mesh, de-bend it and chart it.

    ``source`` is a path, a ``trimesh.Trimesh`` or ``None`` (which builds
    ``synth.make_sevengill(textured=True, seed=seed)``).  The centerline is
    always re-extracted from the geometry -- never taken from ``metadata`` --
    so the synthetic and the real GLB go down exactly the same path.
    """
    t0 = time.time()
    if source is None:
        mesh = synth.make_sevengill(textured=True, seed=int(seed))
    elif isinstance(source, trimesh.Trimesh):
        mesh = source
    else:
        mesh = mesh3d.load_mesh(str(source), report=report)

    cl, cl_info = mesh3d.extract_centerline_3d(
        mesh, n_stations=int(n_stations), seed=seed
    )
    frames = mesh3d.tube_frames(cl)
    straight, straight_cl = mesh3d.debend(mesh, cl, frames=frames, check_roll=False)
    coords = mesh3d.tube_coords(straight, straight_cl)
    vs, vphi = chart_coords(coords, normalize=normalize)

    fins = None
    if detect_fins:
        with warnings.catch_warnings():
            # An unnamed island or a missing fin is a diagnostic, not a reason
            # to stop: nothing downstream of here binds fins by name.
            warnings.simplefilter("ignore", RuntimeWarning)
            fins = mesh3d.detect_fins(straight, coords, check=False)

    texture = texture_image(mesh, tex_size=tex_size)
    info = dict(cl_info) if isinstance(cl_info, dict) else {"centerline": str(cl_info)}
    info.update({
        "n_vertices": int(len(straight.vertices)),
        "n_faces": int(len(straight.faces)),
        "texture_shape": [int(texture.shape[0]), int(texture.shape[1])],
        "s_normalize": normalize,
        "chart_length_m": float(coords.total_length),
        "s_raw_range": [float(coords.s.min()), float(coords.s.max())],
        "fins_found": ([] if fins is None else
                       sorted(k for k in fins.fins
                              if not k.startswith("unassigned_island_"))),
        "straighten_seconds": round(time.time() - t0, 3),
    })
    return Straightened(
        mesh=straight, centerline=straight_cl, coords=coords,
        vertex_s=vs, vertex_phi=vphi, fins=fins, texture=texture, info=info,
    )


# ---------------------------------------------------------------------------
# Step 2: read the real texture into the chart, de-light it, fit individual #0
# ---------------------------------------------------------------------------

class IdentityContext(NamedTuple):
    """Schema-derived pieces shared by every individual in one run.

    Sharing them is what makes individual #0 and the random individuals
    commensurable: the same exclusion geometry, the same head/flank/tail
    signal breakpoints, so a similarity between them measures pattern and not
    a difference in bookkeeping.
    """

    schema_path: str
    stations: dict
    regions: tuple
    params: object


def identity_context(schema_path=None, params=None, stations=None):
    """Build the shared :class:`IdentityContext`.

    Mirrors what ``pattern.Individual.generate`` does internally when it is
    handed no regions, so a copied individual and a randomised one end up with
    identical geometry and identical region breakpoints.
    """
    path = schema_path or pattern.DEFAULT_SCHEMA_PATH
    schema = exclusions.load_schema(path)
    st = exclusions.default_stations(schema) if stations is None else stations
    regions = tuple(exclusions.exclusion_regions(schema, stations=st))
    params = params or pattern.PatternParams()
    if params.head_s_max is None:
        params = params.replace(head_s_max=float(st["gill_slit_7_dorsal_origin"]))
    if params.flank_s_max is None:
        params = params.replace(flank_s_max=float(st["precaudal_pit"]))
    return IdentityContext(schema_path=path, stations=st, regions=regions,
                           params=params)


def exclusion_mask(chart_shape, schema_path=None, layout="bake", **kw):
    """Boolean chart mask, ``True`` where no identity pattern may exist.

    ``chart_shape`` is in BAKE layout ``(n_s, n_phi)``.  ``layout="bake"``
    returns ``(n_s, n_phi)``; ``layout="pattern"`` returns the transpose,
    ``(H_phi, W_s)``, which is what ``pattern.render_chart`` and
    ``drift.similarity`` want.  The regions themselves come from
    ``exclusions.build_exclusion_mask`` -- eyes, nares, the mouth/jaw band and
    the seven gill slits.
    """
    n_s, n_phi = int(chart_shape[0]), int(chart_shape[1])
    mask = exclusions.build_exclusion_mask(
        schema_path or pattern.DEFAULT_SCHEMA_PATH, resolution=(n_phi, n_s), **kw
    )
    if layout == "pattern":
        return mask
    if layout == "bake":
        return np.ascontiguousarray(mask.T)
    raise ValueError("layout must be 'bake' or 'pattern', got %r" % (layout,))


def body_texel_alpha(mesh, fins, raster):
    """``(H, W)`` alpha: 1 on texels of a wholly-``body`` face, 0 elsewhere.

    ``mesh_texture_to_chart`` splats EVERY covered texel into the chart cell at
    its own ``(s, phi)``, and a fin's texels have an ``(s, phi)`` too -- the
    body cell the blade happens to project onto.  So a fin's albedo is averaged
    into the skin underneath it, which on a real scan is a different colour
    (thin, translucent, often shadowed).  Handing the read a 4-channel texture
    with this as alpha makes ``mesh_texture_to_chart`` drop those texels: the
    chart then holds body skin only, and the cells the fins were hiding come
    back as ``NaN`` -- UNOBSERVED, which is the truth, rather than "the colour
    of a pelvic fin".

    ``fins`` is a ``mesh3d.FinDetection``; ``None`` returns all-ones (nothing
    excluded).  A face is kept only if all three of its vertices are body, so
    the fin-root ring is excluded with the blade.
    """
    cov = np.asarray(raster.covered)
    alpha = np.zeros(cov.shape, dtype=float)
    if fins is None:
        alpha[cov] = 1.0
        return alpha
    body_vertex = np.asarray(fins.labels) == "body"
    face_is_body = body_vertex[np.asarray(mesh.faces, dtype=np.int64)].all(axis=1)
    alpha[cov] = face_is_body[np.asarray(raster.face)[cov]].astype(float)
    return alpha


def read_chart(mesh, texture, vertex_s, vertex_phi, chart_shape=None,
               alpha=None, return_coverage=False, fill_holes=True):
    """Read a UV texture off the surface into a chart, in BAKE layout.

    Thin wrapper over ``bake.mesh_texture_to_chart`` that fixes the default
    chart shape and lets a caller pass a per-texel ``alpha`` (see
    :func:`body_texel_alpha`) to drop texels from the read.  Returns
    ``(n_s, n_phi, 3)`` float, ``NaN`` where no texel reached the cell.
    """
    shape = tuple(chart_shape or default_chart_shape())
    tex = np.asarray(texture, dtype=float)
    if alpha is not None:
        tex = np.dstack([tex[..., :3], np.asarray(alpha, dtype=float)])
    return bake.mesh_texture_to_chart(
        mesh, tex, vertex_s, vertex_phi, chart_shape=shape,
        return_coverage=return_coverage, fill_holes=fill_holes,
    )


class Delit(NamedTuple):
    """The de-lighting result.

    albedo: ``(H, W, 3)`` de-lighted texture (the capture's smooth shading
        divided out).
    skin: ``(H, W, 3)`` de-lighted texture with the IDENTITY layer flattened
        too -- see :func:`delight_texture`.
    gain: ``(H, W)`` the multiplicative correction that was applied.
    raster: ``bake.RasterUV`` for this atlas at this texture size (a cache:
        every later bake reuses it).
    s_tex, phi_tex: per-texel chart coordinates, ``NaN`` off-atlas.
    stats: dict with ``gain_clipped_frac``, ``coverage_frac`` and the
        dorsal/ventral contrast before and after.
    """

    albedo: np.ndarray
    skin: np.ndarray
    gain: np.ndarray
    raster: object
    s_tex: np.ndarray
    phi_tex: np.ndarray
    stats: dict


def delight_texture(mesh, vertex_s, vertex_phi, texture, raster=None, **delight_kw):
    """Divide the capture's lighting out of a texture, using bake's own path.

    The de-lighting here is not a re-implementation: it is
    ``bake.bake_chart_to_texture`` driven with a UNIT multiplier chart, so the
    only thing it does is the ``base_albedo`` de-light that a normal bake does
    before it multiplies a pattern on.  Bit for bit the same estimator,
    the same ``[0.25, 4]`` gain clamp, the same warnings.

    Two textures come back, and the difference matters:

    * ``albedo`` -- de-lit.  Shading gone, FRECKLES STILL THERE.  This is what
      individual #0 is read from.
    * ``skin`` -- de-lit AND flattened at the spot scale, using
      ``bake.estimate_lowfreq_luminance``'s spot-robust fit (its asymmetric
      reweighting exists precisely to keep dark marks out of the smooth
      level).  This is the base a SYNTHETIC individual is baked onto.  Baking
      a random individual onto ``albedo`` instead would give every synthetic
      animal in the corpus the real animal's freckles on top of its own -- a
      shared, identity-free feature across the whole dataset, which is exactly
      the shortcut this programme exists to avoid.
    """
    shape = texture.shape[:2]
    unit = np.ones((8, 16))
    _, dbg = bake.bake_chart_to_texture(
        mesh, vertex_s, vertex_phi, unit, shape, base_albedo=texture,
        delight=True, chart_semantics="multiplier", raster=raster,
        return_debug=True, **delight_kw
    )
    albedo = np.asarray(dbg["albedo"], dtype=float)
    raster = dbg["raster"]
    s_tex, phi_tex = dbg["s"], dbg["phi"]
    cov = raster.covered

    low = bake.estimate_lowfreq_luminance(albedo, raster, s_tex, phi_tex)
    lum = bake.luminance(albedo)
    ratio = np.ones(shape)
    ok = cov & np.isfinite(low) & (lum > 1e-6)
    ratio[ok] = np.clip(low[ok] / lum[ok], 0.25, 4.0)
    skin = np.clip(albedo * ratio[..., None], 0.0, 1.0)
    skin[~cov] = 0.0

    gain = np.asarray(dbg["gain"], dtype=float)
    stats = {
        "gain_clipped_frac": float(dbg["gain_clipped_frac"]),
        "coverage_frac": float(dbg["coverage_frac"]),
        # ``gain = ref / low_before``, so 1/gain IS the low-frequency luminance
        # up to the constant ``ref`` -- its normalised swing costs nothing to
        # report and is the honest "how much shading was there" number.
        "lowfreq_swing_before": _relative_swing(
            np.where(cov, 1.0 / np.maximum(gain, 1e-9), np.nan), cov),
        "lowfreq_swing_after": _relative_swing(low, cov),
        "dorsoventral_contrast_before": _dorsoventral_contrast(
            texture, phi_tex, cov),
        "dorsoventral_contrast_after": _dorsoventral_contrast(
            albedo, phi_tex, cov),
    }
    return Delit(albedo=albedo, skin=skin, gain=gain,
                 raster=raster, s_tex=s_tex, phi_tex=phi_tex, stats=stats)


def _relative_swing(field, covered):
    """Peak-to-peak (5th-95th percentile) of a field over its median.

    Applied to the low-frequency luminance this is the size of the smooth
    shading term relative to the skin level -- one orientation-free number for
    "how much lighting is baked in", which is what de-lighting is measured by.
    """
    vals = np.asarray(field, dtype=float)[covered]
    vals = vals[np.isfinite(vals)]
    if vals.size < 8:
        return float("nan")
    lo, hi = np.percentile(vals, [5.0, 95.0])
    med = float(np.median(vals))
    return float((hi - lo) / max(abs(med), 1e-9))


def highfreq_contrast(chart, sigma=6.0):
    """Standard deviation of a chart's SPOT-SCALE detail, ignoring ``NaN``.

    The identity layer is the high-frequency part of the skin, so this is the
    number that says whether an operation kept it.  A masked Gaussian is used
    (``bake.masked_gaussian``, wrapping in ``phi``) so that cells the atlas
    never covered neither contribute nor manufacture an edge -- filling them
    with a constant first and then high-passing measures the fill, not the
    skin, which is a factor-of-two mistake on the caudal end of these charts.

    ``chart`` is in BAKE layout, ``(n_s, n_phi)`` or ``(n_s, n_phi, 3)``;
    ``sigma`` is in chart cells and sits above the spot scale.
    """
    lum = np.asarray(chart, dtype=float)
    if lum.ndim == 3:
        lum = bake.luminance(lum)
    ok = np.isfinite(lum)
    if ok.sum() < 16:
        return float("nan")
    low, den = bake.masked_gaussian(np.where(ok, lum, 0.0), ok,
                                    (float(sigma), float(sigma)), wrap_axis=1)
    use = ok & (den > 0.3)
    if use.sum() < 16:
        return float("nan")
    return float((lum[use] - low[use]).std())


def _dorsoventral_contrast(texture, phi_tex, covered, band=0.6):
    """Mean-luminance ratio dorsal / ventral, the term de-lighting targets.

    The dominant lighting term on a horizontal animal is a ``cos(phi)``
    gradient; this reduces it to one number so the README and the tests can
    quote what the divide bought on the girth specifically.  NOTE that on the
    synthetic stand-in this number stays near 1: ``synth._procedural_texture``
    lays its countershading out along the atlas's OTHER axis, so the fixture's
    smooth term runs fore-aft rather than dorsoventrally.  That is why
    ``lowfreq_swing_*`` -- which does not care which way the gradient runs --
    is the number the tests assert on.
    """
    lum = bake.luminance(np.asarray(texture, dtype=float))
    dorsal = covered & (np.abs(phi_tex) < band)
    ventral = covered & (np.abs(phi_tex) > np.pi - band)
    if not dorsal.any() or not ventral.any():
        return float("nan")
    return float(lum[dorsal].mean() / max(lum[ventral].mean(), 1e-9))


def fit_individual(chart, context=None, identity="individual0",
                   date=DEFAULT_START_DATE, length_cm=DEFAULT_LENGTH_CM,
                   threshold=None, min_area_px=4, radius_gain=1.0):
    """Fit a ``pattern.Individual`` to a chart read off a real texture.

    ``chart`` is in BAKE layout, ``(n_s, n_phi)`` or ``(n_s, n_phi, 3)``, and
    holds ALBEDO (1 = unmarked skin), which is what
    ``bake.mesh_texture_to_chart`` returns.  Three things are pinned rather
    than left to ``copy_from_chart``'s auto-detection, because each of them
    silently produces a plausible-looking wrong answer:

    * ``axis_order="s_major"`` -- auto guesses from the aspect ratio, and a
      chart with more girth cells than length cells guesses wrong;
    * ``chart_semantics="albedo"`` -- auto guesses from the mean, and a
      de-lit dark-dorsum sevengill chart can sit below the 0.5 cut and be
      read as a darkness map, inverting the pattern;
    * ``regions`` -- so individual #0 carries the same exclusion geometry as
      every randomised individual, and ``render_chart`` masks it identically.

    ``NaN`` cells (girth the atlas never covered) are passed through: 05 reads
    them as UNOBSERVED, not as unmarked skin.
    """
    ctx = context or identity_context()
    arr = np.asarray(chart, dtype=float)
    if arr.ndim == 3:
        arr = bake.luminance(arr)
    if arr.ndim != 2:
        raise ValueError("chart must be (n_s, n_phi[, 3]), got %r" % (arr.shape,))
    return pattern.copy_from_chart(
        arr, mask=None, params=ctx.params, threshold=threshold,
        identity=identity, date=date, length_cm=length_cm,
        regions=ctx.regions, min_area_px=int(min_area_px),
        radius_gain=float(radius_gain),
        chart_semantics="albedo", axis_order="s_major",
    )


def identity_round_trip(individual, chart_shape=None, threshold=None,
                        context=None):
    """How much of a fitted individual survives render -> re-fit.

    Fitting a spot table to a chart and rendering it back is lossy in a way
    05 states plainly: touching marks merge, sub-pixel marks vanish, and
    ``render_chart`` fades a spot by region signal and countershading so that
    a mark on the pale ventrum can drop below the segmentation threshold
    entirely.  ``pattern.recoverable_spot_count`` is the honest denominator for
    that last effect, and this returns it alongside the count a re-fit actually
    gets, so a test can assert on the ratio instead of on a bare number.

    Returns a dict with ``n_spots``, ``recoverable``, ``refit``, ``threshold``
    and ``area_fraction`` (of the rendered chart above the threshold).
    """
    ctx = context or identity_context()
    n_s, n_phi = tuple(chart_shape or default_chart_shape())
    thr = float(individual.provenance.get("threshold", 0.5)
                if threshold is None else threshold)
    img, rendered = pattern.render_chart(individual, resolution=(n_phi, n_s))
    refit = pattern.copy_from_chart(
        1.0 - img.T, mask=None, params=ctx.params, threshold=thr,
        identity=individual.identity + "_refit", regions=ctx.regions,
        chart_semantics="albedo", axis_order="s_major",
    )
    return {
        "n_spots": len(individual),
        "recoverable": int(pattern.recoverable_spot_count(rendered, thr)),
        "refit": len(refit),
        "threshold": thr,
        "area_fraction": float((img >= thr).mean()),
    }


def individual_as_dict(individual):
    """JSON-ready dict: the spot table plus enough provenance to re-derive it."""
    spots = individual.spots
    rows = []
    for k in range(len(spots)):
        rows.append({
            "id": int(spots["id"][k]),
            "s": float(spots["s"][k]),
            "phi": float(spots["phi"][k]),
            "radius": float(spots["radius"][k]),
            "eccentricity": float(spots["eccentricity"][k]),
            "angle": float(spots["angle"][k]),
            "darkness": float(spots["darkness"][k]),
            "birth_date": str(spots["birth_date"][k]),
        })
    return {
        "identity": individual.identity,
        "seed": int(individual.seed),
        "date": str(individual.date),
        "length_cm": float(individual.length_cm),
        "n_spots": len(spots),
        "chart_convention": exclusions.CHART_CONVENTION,
        "params": individual.params.as_dict(),
        "provenance": _jsonable(individual.provenance),
        "regions": [r.name for r in individual.regions],
        "spots": rows,
        "scars": [
            {
                "id": int(sc.id), "s": float(sc.s), "phi": float(sc.phi),
                "length": float(sc.length), "width": float(sc.width),
                "angle": float(sc.angle), "darkness": float(sc.darkness),
                "birth_date": str(sc.birth_date),
            }
            for sc in individual.scars
        ],
    }


def _jsonable(obj):
    """Recursively coerce numpy scalars/arrays and dates to JSON types."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.datetime64):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# Step 3/4: identity -> chart -> texture -> GLB
# ---------------------------------------------------------------------------

def bake_individual(mesh, vertex_s, vertex_phi, individual, base_albedo,
                    chart_shape=None, date=None, raster=None, amplitude=1.0):
    """Render an individual to a chart and bake it onto ``base_albedo``.

    Returns ``(texture_rgba, chart_darkness)`` where ``texture_rgba`` is
    ``(H, W, 4)`` float (alpha = atlas coverage, not opacity) and
    ``chart_darkness`` is the PATTERN-layout ``(H_phi, W_s)`` image that
    ``render_chart`` produced -- kept so the contact sheet can show the
    identity layer next to the surface it was baked onto.

    ``delight`` is OFF here: ``base_albedo`` is already the de-lit skin from
    :func:`delight_texture`, and de-lighting it twice would fit a second
    low-frequency surface to an image that no longer has one.
    """
    n_s, n_phi = tuple(chart_shape or default_chart_shape())
    img, _ = pattern.render_chart(individual, resolution=(n_phi, n_s), date=date)
    tex = bake.bake_chart_to_texture(
        mesh, vertex_s, vertex_phi, img.T, base_albedo.shape[:2],
        base_albedo=base_albedo, delight=False, chart_semantics="darkness",
        amplitude=float(amplitude), raster=raster,
    )
    return tex, img


def write_textured_glb(mesh, texture, path, validate=True):
    """Write ``mesh`` with ``texture`` as its base colour, and validate it.

    Geometry, faces and UVs are the straightened mesh's, untouched -- only
    ``visual`` is replaced -- so every GLB a run emits is the same body
    wearing a different skin, which is what makes them comparable.

    trimesh's own GLB writer is used rather than
    ``gltf_export.write_skinned_glb``: there is no skeleton here, and trimesh's
    writer already round-trips the UV V-axis flip (load -> write -> load is a
    UV identity to ~4e-8, asserted in the tests).  The file is then put through
    the Khronos validator, which must report zero errors.
    """
    from PIL import Image

    arr = np.asarray(texture, dtype=float)
    rgb = (np.clip(arr[..., :3], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    out = mesh.copy()
    out.metadata.clear()
    out.visual = trimesh.visual.TextureVisuals(
        uv=np.asarray(mesh.visual.uv, dtype=np.float64).copy(),
        image=Image.fromarray(rgb, mode="RGB"),
    )
    out.export(str(path))
    if validate:
        gltf_export.validate_glb(str(path))
    return str(path)


def resight_series(individual, n_resights, years, growth_model=None,
                   params=None, seed=0):
    """Chain ``n_resights`` resightings of ``individual`` over ``years``.

    The chain is sequential -- each resighting drifts from the previous one's
    date, not from t0 -- so drift accumulates the way a real capture history
    does and ``spots["id"]`` still tracks the same physical mark throughout.
    ``rng`` is left to ``drift.resight``, which seeds it from the individual's
    seed and the elapsed days, so the series is reproducible from
    ``(individual, years, n_resights)`` alone.
    """
    n = int(n_resights)
    if n < 0:
        raise ValueError("n_resights must be >= 0")
    out = []
    current = individual
    t_start = pattern.as_date(individual.date)
    for k in range(1, n + 1):
        days = int(round(k * float(years) * drift.DAYS_PER_YEAR / max(n, 1)))
        t_next = t_start + np.timedelta64(days, "D")
        current = drift.resight(current, current.date, t_next,
                                growth_model=growth_model, params=params)
        out.append(current)
    return out


def random_individuals(n, context=None, seed=0, date=DEFAULT_START_DATE,
                       length_cm=DEFAULT_LENGTH_CM, n_spots_target=None):
    """``n`` fresh synthetic individuals sharing ``context``'s geometry.

    ``n_spots_target`` overrides the density.  :func:`run` passes individual
    #0's fitted spot count, so the synthetic siblings sit at the density
    actually recovered from the scan rather than at the module default -- which
    matters because a density difference is itself an identity-free cue that a
    re-ID model would happily learn to separate "the real one" from "the fakes".
    """
    ctx = context or identity_context()
    params = ctx.params
    if n_spots_target is not None:
        params = params.replace(n_spots_target=int(n_spots_target))
    out = []
    for k in range(int(n)):
        out.append(pattern.randomize(
            seed=int(seed) * 1000 + 101 + k, params=params,
            identity="random%02d" % (k + 1), date=date, length_cm=length_cm,
            regions=ctx.regions,
        ))
    return out


# ---------------------------------------------------------------------------
# Step 5: previews and the contact sheet
# ---------------------------------------------------------------------------

def render_side(mesh, texture, vertex_s, vertex_phi, resolution=PREVIEW_RESOLUTION,
                from_left=False, ambient=0.55):
    """Orthographic side view of the straightened mesh wearing ``texture``.

    Uses prototype 05's rasteriser read-only (``render.render`` with
    ``exclusion=None`` and shadows off), so what the contact sheet shows is the
    same sampler the synthetic corpus will render with -- not a second,
    differently-behaved previewer.  ``from_left`` puts the camera on ``+Y``
    (the animal's left, snout to the left of frame); the default views the
    right flank with the snout to the right.

    Returns ``(H, W, 3)`` uint8.
    """
    verts = np.asarray(mesh.vertices, dtype=float)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    centre = 0.5 * (lo + hi)
    span = np.maximum(hi - lo, 1e-9)
    h, w = int(resolution[0]), int(resolution[1])
    ortho_h = 1.06 * max(span[2], span[0] * h / float(w))
    dist = 4.0 * float(span.max())
    sign = 1.0 if from_left else -1.0

    tex = np.asarray(texture, dtype=float)[..., :3]
    inst = p05_render.Instance(
        vertices=verts, faces=np.asarray(mesh.faces),
        uv=np.asarray(mesh.visual.uv, dtype=float),
        texture=np.clip(tex, 0.0, 1.0),
        vertex_s=vertex_s, vertex_phi=vertex_phi, name="sevengill",
    )
    cam = p05_render.Camera(
        eye=centre + np.array([0.0, sign * dist, 0.0]), target=centre,
        up=(0.0, 0.0, 1.0), resolution=(h, w), kind="ortho",
        ortho_height=ortho_h,
    )
    light = p05_render.DirectionalLight(direction=(0.25, sign * 0.4, -0.8),
                                        ambient=float(ambient))
    frame = p05_render.render([inst], cam, light=light, exclusion=None,
                              shadows=False)
    return (np.clip(frame["rgb"], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def chart_to_image(chart, semantics="albedo"):
    """A chart (BAKE layout) as a displayable ``(n_phi, n_s, 3)`` uint8 image.

    Transposed on the way out so the body runs left-to-right and the girth
    top-to-bottom, which is how every chart figure in this programme is drawn.
    ``semantics="darkness"`` inverts, so an identity layer is shown as dark
    marks on light skin like the textures it will be multiplied into.
    """
    arr = np.asarray(chart, dtype=float)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    arr = np.nan_to_num(arr, nan=1.0 if semantics == "albedo" else 0.0)
    if semantics == "darkness":
        arr = 1.0 - arr
    elif semantics != "albedo":
        raise ValueError("semantics must be 'albedo' or 'darkness'")
    return (np.clip(arr.transpose(1, 0, 2), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def save_png(image, path):
    """Write a float ``[0, 1]`` or uint8 array as PNG; returns the path."""
    from PIL import Image

    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = (np.clip(np.nan_to_num(arr.astype(float), nan=0.0), 0.0, 1.0)
               * 255.0 + 0.5).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    Image.fromarray(arr[..., :3], mode="RGB").save(str(path))
    return str(path)


def contact_sheet(panels, path, cell_height=200, pad=8, label_height=18,
                  background=(250, 250, 248)):
    """One row per stage: ``label | chart | textured side view``.

    ``panels`` is a sequence of ``(label, chart_image_or_None, side_image)``
    with images as uint8 arrays.  Rows are the identity stages the run
    produced, so reading down the sheet is reading the pipeline: the real skin,
    what de-lighting left of it, the individual fitted from that, how it drifts,
    and what an unrelated animal looks like on the same body.
    """
    from PIL import Image, ImageDraw

    rows = list(panels)
    if not rows:
        raise ValueError("contact_sheet needs at least one panel")

    def _fit(img, height):
        pil = Image.fromarray(np.asarray(img, dtype=np.uint8))
        w = max(1, int(round(pil.size[0] * height / float(pil.size[1]))))
        return pil.resize((w, height), Image.NEAREST)

    charts, sides = [], []
    for _, chart, side in rows:
        charts.append(None if chart is None else _fit(chart, cell_height))
        sides.append(_fit(side, cell_height))
    chart_w = max((c.size[0] for c in charts if c is not None), default=0)
    side_w = max(s.size[0] for s in sides)

    width = pad + chart_w + pad + side_w + pad
    row_h = label_height + cell_height + pad
    sheet = Image.new("RGB", (width, pad + row_h * len(rows)), tuple(background))
    draw = ImageDraw.Draw(sheet)
    y = pad
    for (label, _, _), chart, side in zip(rows, charts, sides):
        draw.text((pad, y + 3), str(label), fill=(30, 30, 30))
        y += label_height
        if chart is not None:
            sheet.paste(chart, (pad, y))
        sheet.paste(side, (pad + chart_w + pad, y))
        y += cell_height + pad
    sheet.save(str(path))
    return str(path)


# ---------------------------------------------------------------------------
# The whole thing
# ---------------------------------------------------------------------------

def run(glb=None, out_dir=DEFAULT_OUT, n_resights=4, years=3.0, n_random=3,
        seed=0, n_stations=64, tex_size=DEFAULT_TEX_SIZE, chart_shape=None,
        schema_path=None, start_date=DEFAULT_START_DATE,
        length_cm=DEFAULT_LENGTH_CM, validate=True, sheet_panels=2,
        match_random_density=True, exclude_fins=True, report=True):
    """The whole pipeline; returns a dict of paths and objects.

    Args:
        glb: input mesh path, a ``Trimesh``, or ``None`` for the synthetic
            stand-in ``synth.make_sevengill(textured=True)``.
        out_dir: written here (created if absent).  Nothing outside it, and
            nothing under prototype 05, is written.
        n_resights, years: the resighting chain of individual #0.
        n_random: unrelated synthetic individuals baked on the same body.
        sheet_panels: how many resights and how many randoms go on the contact
            sheet (2 each, per the brief).
        match_random_density: give the random individuals individual #0's
            fitted spot count as their target density -- see
            :func:`random_individuals`.
        exclude_fins: drop fin texels from the chart READ (not from the bake),
            so a blade's albedo is not averaged into the body skin under it --
            see :func:`body_texel_alpha`.

    Outputs in ``out_dir``:
        ``chart_real.png``, ``chart_delighted.png``, ``chart_skin.png``,
        ``individual0.json``, ``individual0.glb``, ``resight_NN.glb``,
        ``random_NN.glb``, ``textures/*.png``, ``contact_sheet.png``,
        ``summary.json``.
    """
    t_start = time.time()
    out_dir = str(out_dir)
    tex_dir = os.path.join(out_dir, "textures")
    os.makedirs(tex_dir, exist_ok=True)
    shape = tuple(chart_shape or default_chart_shape())

    # 1. straighten and chart -------------------------------------------------
    st = straighten(glb, n_stations=n_stations, tex_size=tex_size, seed=seed,
                    report=report)
    mesh, vs, vphi = st.mesh, st.vertex_s, st.vertex_phi
    if report:
        print("straighten: %d verts, chart %.4f m, fins %s, %.2fs"
              % (st.info["n_vertices"], st.info["chart_length_m"],
                 ",".join(st.info["fins_found"]) or "none",
                 st.info["straighten_seconds"]))

    # 2. real texture -> chart -> de-light -> individual #0 --------------------
    ctx = identity_context(schema_path=schema_path)
    delit = delight_texture(mesh, vs, vphi, st.texture)
    alpha = (body_texel_alpha(mesh, st.fins, delit.raster)
             if exclude_fins else None)
    chart_real = read_chart(mesh, st.texture, vs, vphi, chart_shape=shape,
                            alpha=alpha)
    chart_delighted = read_chart(mesh, delit.albedo, vs, vphi, chart_shape=shape,
                                 alpha=alpha)
    chart_skin = read_chart(mesh, delit.skin, vs, vphi, chart_shape=shape,
                            alpha=alpha)

    # Fit on the UNFILLED read: hole-filling diffuses neighbours into cells no
    # texel reached (the caudal overhang, dropped fin texels), and that grey
    # smear thresholded as one giant "spot". NaN there means unobserved, and
    # copy_from_chart treats it so. The filled chart stays for display/bake.
    chart_fit = read_chart(mesh, delit.albedo, vs, vphi, chart_shape=shape,
                           alpha=alpha, fill_holes=False)
    ind0 = fit_individual(chart_fit, context=ctx, identity="individual0",
                          date=start_date, length_cm=length_cm)
    if report:
        print("de-light: low-frequency swing %.3f -> %.3f, dorsoventral "
              "contrast %.3f -> %.3f, gain clipped on %.2f%% of texels"
              % (delit.stats["lowfreq_swing_before"],
                 delit.stats["lowfreq_swing_after"],
                 delit.stats["dorsoventral_contrast_before"],
                 delit.stats["dorsoventral_contrast_after"],
                 100.0 * delit.stats["gain_clipped_frac"]))
        print("individual0: %d spots (threshold %.3f, %d components)"
              % (len(ind0), ind0.provenance["threshold"],
                 ind0.provenance["components_found"]))

    paths = {
        "chart_real": save_png(chart_to_image(chart_real),
                               os.path.join(out_dir, "chart_real.png")),
        "chart_delighted": save_png(chart_to_image(chart_delighted),
                                    os.path.join(out_dir, "chart_delighted.png")),
        "chart_skin": save_png(chart_to_image(chart_skin),
                               os.path.join(out_dir, "chart_skin.png")),
        "texture_real": save_png(st.texture,
                                 os.path.join(tex_dir, "real.png")),
        "texture_delighted": save_png(delit.albedo,
                                      os.path.join(tex_dir, "delighted.png")),
        "texture_skin": save_png(delit.skin,
                                 os.path.join(tex_dir, "skin.png")),
    }
    with open(os.path.join(out_dir, "individual0.json"), "w") as fh:
        json.dump(individual_as_dict(ind0), fh, indent=2, sort_keys=True)
    paths["individual0_json"] = os.path.join(out_dir, "individual0.json")

    # 3./4. bake individual #0, its resightings and the random individuals ----
    resights = resight_series(ind0, n_resights, years)
    randoms = random_individuals(
        n_random, context=ctx, seed=seed, date=start_date, length_cm=length_cm,
        n_spots_target=(len(ind0) if match_random_density and len(ind0) else None),
    )

    baked = []
    for name, ind in ([("individual0", ind0)]
                      + [("resight_%02d" % (k + 1), r) for k, r in enumerate(resights)]
                      + [("random_%02d" % (k + 1), r) for k, r in enumerate(randoms)]):
        tex, chart_img = bake_individual(
            mesh, vs, vphi, ind, delit.skin, chart_shape=shape,
            raster=delit.raster,
        )
        glb_path = write_textured_glb(mesh, tex, os.path.join(out_dir, name + ".glb"),
                                      validate=validate)
        save_png(tex, os.path.join(tex_dir, name + ".png"))
        baked.append({"name": name, "individual": ind, "texture": tex,
                      "chart": chart_img, "glb": glb_path})
        paths[name + "_glb"] = glb_path
        if report:
            print("baked %-14s %3d spots  date %s  -> %s"
                  % (name, len(ind), ind.date, os.path.basename(glb_path)))

    by_name = {b["name"]: b for b in baked}

    # 5. contact sheet --------------------------------------------------------
    k = int(sheet_panels)
    stages = [
        ("real texture (chart | body)", chart_to_image(chart_real), st.texture),
        ("de-lighted", chart_to_image(chart_delighted), delit.albedo),
        ("individual #0 (fitted, re-baked)",
         chart_to_image(by_name["individual0"]["chart"].T, semantics="darkness"),
         by_name["individual0"]["texture"]),
    ]
    for nm in (["resight_%02d" % (i + 1) for i in range(min(k, len(resights)))]
               + ["random_%02d" % (i + 1) for i in range(min(k, len(randoms)))]):
        b = by_name[nm]
        stages.append((
            "%s  (%s)" % (nm.replace("_", " "), b["individual"].date),
            chart_to_image(b["chart"].T, semantics="darkness"),
            b["texture"],
        ))
    panels = [
        (label, chart, render_side(mesh, tex, vs, vphi))
        for label, chart, tex in stages
    ]
    paths["contact_sheet"] = contact_sheet(
        panels, os.path.join(out_dir, "contact_sheet.png"))

    # summary -----------------------------------------------------------------
    summary = {
        "input": (None if glb is None else str(glb)),
        "out_dir": out_dir,
        "seed": int(seed),
        "chart_shape_bake": [int(shape[0]), int(shape[1])],
        "texture_size": [int(st.texture.shape[0]), int(st.texture.shape[1])],
        "straighten": _jsonable(st.info),
        "exclude_fins": bool(exclude_fins),
        "chart_finite_frac": float(np.isfinite(chart_delighted[..., 0]).mean()),
        "delight": dict(
            _jsonable(delit.stats),
            highfreq_chart_real=highfreq_contrast(chart_real),
            highfreq_chart_delighted=highfreq_contrast(chart_delighted),
            highfreq_chart_skin=highfreq_contrast(chart_skin),
        ),
        "individual0": {
            "n_spots": len(ind0),
            "threshold": float(ind0.provenance["threshold"]),
            "components_found": int(ind0.provenance["components_found"]),
            "round_trip": identity_round_trip(ind0, chart_shape=shape,
                                              context=ctx),
        },
        "resights": [{"name": "resight_%02d" % (i + 1), "date": str(r.date),
                      "n_spots": len(r), "length_cm": float(r.length_cm),
                      "similarity_to_individual0": float(
                          drift.similarity(ind0, r))}
                     for i, r in enumerate(resights)],
        "randoms": [{"name": "random_%02d" % (i + 1), "n_spots": len(r),
                     "similarity_to_individual0": float(drift.similarity(ind0, r))}
                    for i, r in enumerate(randoms)],
        "paths": paths,
        "seconds": round(time.time() - t_start, 2),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    paths["summary"] = os.path.join(out_dir, "summary.json")
    if report:
        print("wrote %s in %.1fs" % (out_dir, summary["seconds"]))

    return {
        "straightened": st, "context": ctx, "individual0": ind0,
        "resights": resights, "randoms": randoms, "baked": baked,
        "by_name": by_name, "delit": delit, "chart_real": chart_real,
        "chart_delighted": chart_delighted, "chart_skin": chart_skin,
        "chart_shape": shape, "paths": paths, "summary": summary,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--glb", default=None,
                    help="input textured GLB; omit for the synthetic stand-in")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n-resights", type=int, default=4)
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-n", "--n-stations", type=int, default=64)
    ap.add_argument("--tex-size", type=int, default=DEFAULT_TEX_SIZE)
    ap.add_argument("--chart-h-phi", type=int, default=CHART_H_PHI)
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)
    return run(
        glb=args.glb, out_dir=args.out, n_resights=args.n_resights,
        years=args.years, n_random=args.n_random, seed=args.seed,
        n_stations=args.n_stations, tex_size=args.tex_size,
        chart_shape=default_chart_shape(args.chart_h_phi),
        validate=not args.no_validate, report=not args.quiet,
    )


if __name__ == "__main__":
    main()
