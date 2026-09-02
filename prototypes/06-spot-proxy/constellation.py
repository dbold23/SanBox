"""Non-RGB spot-constellation matcher on the OSEA detection contract.

The matcher never sees pixels. Its whole input is one *detection dict* in the
shape the OSEA tagger already stores per photo
(``spot_detector/scripts/infer_pipeline.run_image`` ->
``reid/scripts/pipeline_worker.stage_detect`` -> ``images`` columns):

    {"width": int, "height": int,
     "body_polygon":        [[x, y], ...],                 # ONE polygon, original px
     "obstruction_polygons": [[[x, y], ...], ...] | None,  # 0..n polygons
     "spots": [{"x","y","w","h","cx","cy","conf"}, ...]}   # original px

Because that is the *only* input, a synthetic render and a real photograph are
interchangeable here as long as the same detector ran on both -- which is the
whole point of prototype 06.

Pipeline
--------
1. ``build_body_mask``   polygon (minus obstructions) -> boolean mask at a
   working resolution <= ``MAX_SIDE`` px on the long side; spots scaled to match.
   ``extract_spotset`` then runs ``_largest_filled``, which **refills** any
   obstruction that lies wholly inside the silhouette (see its docstring): the
   punch-out survives only where an occluder crosses the outline.  That is
   deliberate -- the chart's local half width must describe the *body*, and it
   must describe the same mask prototype 02 charted -- and the refilled area is
   logged as ``obstruction_refilled`` in ``SpotSet.meta["drops"]``.
2. ``extract_spotset``   prototype 02 (``extract_centerline`` +
   ``chart.image_to_chart``) rectifies each spot centre to
      s in [0, 1]   arc length along the medial axis, 0 = widest end,
      r in [-1, 1]  signed offset along the station normal, normalised by the
                    *local* half width.
   A PCA frame is the fallback when 02 fails (logged in ``SpotSet.meta``).
3. ``descriptors``       per-spot shape context over the K nearest neighbours,
   orientation fixed by the body axis (no per-point rotation normalisation --
   the chart has already removed rotation), scale normalised by the set's
   median nearest-neighbour distance.
4. ``match_score``       chi2 descriptor cost -> Hungarian assignment ->
   RANSAC on a low-DOF geometric model -> inlier count inside a gate of
   ``GATE_FRAC`` x median NN distance. ``score = inliers / min(n_a, n_b)``.
   The registration itself is *directed* (b is mapped onto a, residuals are
   measured in a's metric, the least-squares refit regresses one way round), so
   the raw directed score is NOT symmetric -- measured mean |score(a,b) -
   score(b,a)| = 0.018 and max 0.10 on a 24-set toy sample, and it does not
   shrink with the RANSAC budget. ``match_score`` therefore scores BOTH
   directions by default (``symmetric=True``) and keeps the better one, which
   makes the public score exactly symmetric at 2x the cost; the direction that
   won is reported as ``result["direction"]``.
5. ``rank``              query vs gallery, sorted.

Orientation is ambiguous on a real photo (there is no head detector in the
shipped weights, and ``extract_centerline``'s widest-end-first rule can pick the
wrong end on a head-and-forebody crop). Every match therefore tries the four
``s -> 1-s`` x ``r -> -r`` flips of the *gallery* set and keeps the best.

Local half width
----------------
The task brief says "use the distance transform for the local half width". The
distance transform is used to *size the chart* (the global ``half_width`` passed
to ``chart.image_to_chart``) and as the floor for the per-station width, but it
is NOT used directly as the local half width: the EDT at a station is the
distance to the *nearest* boundary, which near the snout/tail tips is the
distance to the tip, not the lateral half width -- normalising by it pushes
legitimate spots outside the body (measured |r| up to 1.33 on a test ellipse
whose spots reach 98% of the rim, against 0.98 for the ray-marched width). ``_station_half_widths`` instead marches
the mask along each station normal and takes the contiguous inside run, which is
the true lateral half width *at that station*.
Set ``r_norm="edt"`` to get the naive behaviour (and the clipping it needs).

That is a better normaliser, but it is **not** a guarantee that |r| <= 1 for a
spot inside the body, because the numerator is not measured at the same place:
``chart.image_to_chart`` assigns a point to the NEAREST centerline segment and
measures r along *that* segment's normal.  Where the medial axis turns sharply
-- a snout or tail tip, where it hooks through ~90 degrees -- a spot 60 px
*along* the body is charted as a 60 px *lateral* offset at a station whose
lateral half width is 2 px, giving |r| = 30.  Measured on the real corpus: 40 of
the 1030 charted images produce max |r| > 1, up to 81.3 (image 854).  So
``_chart_is_unusable`` tests both the *fraction* of spots with |r| > 1 and its
*magnitude*, and whatever survives is clipped to +-1 with the clipped count and
``max_abs_r`` logged in ``meta["drops"]``.

The PCA fallback frame normalises r the same way -- by a local half width read
off the mask in 64 longitudinal bins, one value each side of the axis -- and NOT
by the body's global minor extent.  This matters: the two frames are compared
against each other by ``match_score``, and a global normaliser makes a spot at
85% of the local half width read as r = 0.85 in the chart frame and 0.85 * 40 /
150 = 0.23 in the PCA frame of a tapering body, i.e. two incomparable numbers.

Geometric model (justification)
-------------------------------
The chart has already removed rotation (the body axis *is* the s axis) and has
normalised r by the local half width. What is left between two sightings of one
animal is
  (i)  a different *visible extent* along the body -- the OSEA photos are
       head-and-forebody crops, so sighting A may cover s in [0.0, 0.6] of the
       animal and sighting B s in [0.1, 0.9]; after per-image renormalisation
       that is exactly ``s' = alpha*s + beta``;
  (ii) possibly a different r frame, ``r' = gamma*r + delta``, if the mask that
       set the local half width was truncated differently in the two photos.
Three models are implemented and compared head-to-head on the toy
(``eval_constellation.py --toy --compare-models``):
  ``s_affine``  alpha, beta along s; r rigid              -- 3 free numbers
  ``axis``      alpha, beta along s; gamma, delta on r    -- 4
  ``sim``       full 2D similarity (adds a rotation)      -- 4
``axis`` is the DEFAULT. Every measured table in this module is regenerated by
``eval_constellation.py --ablate`` under the SHIPPED defaults -- run it after
touching any default and paste the output back, because these numbers have
drifted before: they were once measured under ``s_affine`` while the shipped
default was already ``axis``, and none of them reproduced. Protocol: 2% jitter /
20% dropout / 20% clutter, 5 seeds x 20 identities on the brief's generator.

    model      mean Rank-1   per-seed Rank-1              AUROC    diff score
    s_affine   0.980         1.00 1.00 0.90 1.00 1.00     0.9978   0.223
    axis       0.980         0.95 1.00 0.95 1.00 1.00     0.9932   0.263
    sim        0.920         0.90 1.00 0.90 0.95 0.85     0.9846   0.268

``sim`` loses outright: its rotation tilts the body axis the chart just removed,
and the extra freedom is one more way for RANSAC to align two *different*
animals. Between the other two, ``s_affine`` has the cleaner separation when the
r frame is stable, but it holds r rigid, so it collapses as soon as the r frame
moves -- and it does move in practice, because r is normalised by a half width
read off a hand-occluded mask. On the ``--toy-hard`` generator, which perturbs
it (r -> gamma*r + delta, sd 0.05 / 0.04, plus a 75-100% visible-extent crop),
at 2% jitter / 20% dropout (``--ablate`` group "model_hard"): ``axis`` Rank-1
0.640 and AUROC 0.9142 against ``s_affine``'s 0.450 and 0.7598, and at 0.5%
jitter / 0% dropout 1.000 against 0.630. ``axis`` also has the better worst seed
on the clean generator. Use ``s_affine`` when the r frame is trustworthy --
whole-body synthetic renders with a clean silhouette are the case for it.

The chance floor
----------------
The score has an irreducible false-positive floor that is worth stating up
front. A gate of ``GATE_FRAC`` x the median nearest-neighbour distance is a disc
holding, on average, ``0.25 * pi * GATE_FRAC^2`` = 0.28 spots of ANY constellation
of the same density, whatever the scale -- so two different animals with equally
dense patterns score ~0.2-0.3 before anything is learned, and the matcher has to
win by pushing the true pair's inlier fraction well above that. Measured mean
different-individual score on the toy: 0.390 with no descriptor gate, 0.263 with
the default one. This is why the toy table degrades sharply once the per-spot
jitter approaches the spot spacing (2% of body length is ~0.36 of the median
spacing at 40-120 spots; 4% is ~0.7 and the statistic is at chance).

Numpy / scipy / sklearn / PIL only.
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

# --- prototype 02 (imported, never modified) -------------------------------
_P02 = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "02-centerline-chart"
)
if _P02 not in sys.path:
    sys.path.insert(0, _P02)

import centerline as cl02  # noqa: E402
import chart as chart02  # noqa: E402
import frames as fr02  # noqa: E402

__all__ = [
    "SpotSet",
    "build_body_mask",
    "extract_spotset",
    "descriptors",
    "match_score",
    "rank",
    "MAX_SIDE",
    "N_S",
    "N_R",
    "DEFAULT_ASPECT",
    "GATE_FRAC",
]

MAX_SIDE = 1024        # working resolution, long side, px
N_S = 256              # chart stations along the medial axis
N_R = 257              # chart columns (odd -> r = 0 lands on an exact index)
DEFAULT_ASPECT = 1.0   # s is stretched by this before any metric is taken; see below
GATE_FRAC = 0.6        # inlier gate, in units of the median NN distance (per the brief)
K_NN = 20              # shape-context neighbourhood
N_RADIAL = 4
N_ANGULAR = 8
_RHO_MIN = 0.25        # inner log-radial edge, in median-NN units
_RHO_MAX = 4.0         # outer log-radial edge
_EPS = 1e-12
_MAX_R_OUT_FRAC = 0.10  # spots outside the body that condemn the chart frame
_MAX_R_ABS = 2.0        # a single |r| this far outside the body condemns it too
_PCA_N_BINS = 64        # longitudinal bins for the PCA frame's local half width
_MIN_LEN_FRAC = 0.60    # centerline length / mask principal extent, below which it curled

# DEFAULT_ASPECT selects the metric in which distances, the k-NN neighbourhood
# and the inlier gate are all measured: xy() = (s * aspect, r).
#   aspect = L/W (~3 for the OSEA head-and-forebody crops, bbox w/h 1.5-2.0)
#     makes the metric physically isotropic -- correct if the dominant per-spot
#     error is the detector's box-centre error, which is isotropic in pixels.
#   aspect = 1 makes the metric isotropic in CHART units -- correct if the
#     dominant error is rectification error, i.e. an imperfect centerline or an
#     imperfect local half width, which displaces r by a fraction of the half
#     width and s by a fraction of the length.
# The second dominates in this pipeline (the masks come from a body detector on
# hand-occluded photos; the box centres are good to ~1 px). `--ablate` group
# "aspect" (2% jitter, 20% dropout, 20% clutter, 5 seeds x 20 identities):
#     aspect 1 -> Rank-1 0.980, AUROC 0.9932
#     aspect 2 -> Rank-1 0.840, AUROC 0.9770
#     aspect 3 -> Rank-1 0.830, AUROC 0.9605
# The per-image measurement is still reported in SpotSet.meta["aspect_measured"];
# it is deliberately NOT used as the metric, because two images must share one
# metric for their descriptors to be comparable.
# K_NN = 20 rather than the 12 of the brief: top-1 descriptor recall 0.23 (k=12)
# vs 0.29 (k=20) vs 0.35 (k=30) at 2% jitter -- a descriptor-only number, so not
# part of --ablate -- and end-to-end (`--ablate` group "k_nn") Rank-1 / AUROC
# 0.930 / 0.9898 (k=12), 0.980 / 0.9932 (k=20), 0.940 / 0.9886 (k=30). 20 is the
# knee, and k=30 loses at both ends: its different-individual score rises to
# 0.310 against 0.263.


# --------------------------------------------------------------------------- #
# 1. mask                                                                      #
# --------------------------------------------------------------------------- #
def build_body_mask(body_polygon, obstruction_polygons=None, width=None, height=None,
                    max_side=MAX_SIDE):
    """Rasterise the body polygon minus the obstruction polygons.

    Args:
        body_polygon: list of [x, y] in original (exif-transposed) pixels.
        obstruction_polygons: list of polygons punched out of the body, or None.
        width, height: original image size in px. If None they are inferred from
            the polygon extent (+1 px), which is enough for the mask but makes
            the working scale image-relative rather than absolute.
        max_side: working resolution cap on the long side.

    Returns:
        (mask, scale): mask is a bool (H, W) array at the working resolution,
        ``scale`` is the factor original px -> working px (multiply original
        coordinates by it). Raises ValueError if the polygon is degenerate.
    """
    poly = np.asarray(body_polygon, dtype=float)
    if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2:
        raise ValueError("body_polygon must be >= 3 points of [x, y]")
    if width is None or height is None:
        width = int(np.ceil(poly[:, 0].max())) + 1
        height = int(np.ceil(poly[:, 1].max())) + 1
    width, height = int(width), int(height)
    if width < 2 or height < 2:
        raise ValueError("image size %dx%d is degenerate" % (width, height))

    scale = min(1.0, float(max_side) / float(max(width, height)))
    w_work = max(2, int(round(width * scale)))
    h_work = max(2, int(round(height * scale)))

    img = Image.new("L", (w_work, h_work), 0)
    draw = ImageDraw.Draw(img)
    draw.polygon([(float(x) * scale, float(y) * scale) for x, y in poly], fill=255)
    n_obstr = 0
    for ob in (obstruction_polygons or []):
        ob = np.asarray(ob, dtype=float)
        if ob.ndim == 2 and ob.shape[0] >= 3:
            draw.polygon([(float(x) * scale, float(y) * scale) for x, y in ob], fill=0)
            n_obstr += 1
    mask = np.asarray(img) > 127
    if not mask.any():
        raise ValueError("body polygon rasterised to an empty mask")
    return mask, scale


def _largest_filled(mask):
    """Largest 8-connected component of ``mask``, holes filled.

    Mirrors what ``centerline.extract_centerline`` does internally, so the half
    widths measured here describe exactly the mask it charted. Anything else
    would be a bug: we would be normalising r by a half width read off a
    different mask than the one that produced the stations.

    **This undoes the obstruction punch-out for interior occluders**, and that
    is intended. ``binary_fill_holes`` cannot tell an occluder-shaped hole from
    a rasterisation hole, and prototype 02 fills both before charting. It is
    also the behaviour we want: an occluder does not make the animal narrower,
    so the *body* half width -- not the visible half width -- is the right
    normaliser, and the spots under the occluder were already dropped upstream
    by ``run_image``'s not-inside-an-obstruction filter, which is dropout, not a
    change of frame. The asymmetry that remains is real and worth knowing: an
    occluder that crosses the silhouette DOES cut the mask, because the cut is
    then not a hole. ``extract_spotset`` logs the refilled area as
    ``obstruction_refilled``. Measured on the real corpus: 40 of 40 sampled
    obstructed records had area punched out and 28 of those were fully
    restored.
    """
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        raise ValueError("mask is empty")
    if n > 1:
        sizes = ndimage.sum_labels(np.ones_like(labels), labels, index=np.arange(1, n + 1))
        mask = labels == (1 + int(np.argmax(sizes)))
    return ndimage.binary_fill_holes(mask)


# --------------------------------------------------------------------------- #
# 2. the chart frame                                                           #
# --------------------------------------------------------------------------- #
def _station_half_widths(mask, stations, normals, max_reach, step=0.5):
    """True lateral half width each side of every station.

    March the mask along +normal and -normal in ``step`` px increments and take
    the length of the contiguous inside run. Returns (hw_pos, hw_neg), each
    (n_s,) in working px, floored at 1 px.
    """
    n_t = int(np.ceil(max_reach / step)) + 1
    t = (np.arange(1, n_t + 1) * step)                       # (n_t,) skip t=0
    out = []
    for sign in (+1.0, -1.0):
        pts = stations[:, None, :] + sign * t[None, :, None] * normals[:, None, :]
        inside = ndimage.map_coordinates(
            mask.astype(np.float32),
            [pts[..., 1].ravel(), pts[..., 0].ravel()],
            order=0, mode="constant", cval=0.0,
        ).reshape(stations.shape[0], n_t) > 0.5
        run = np.cumprod(inside, axis=1).sum(axis=1) * step   # contiguous run only
        out.append(np.maximum(run, 1.0))
    return out[0], out[1]


def _pca_frame(mask, pts_xy, n_bins=_PCA_N_BINS):
    """Fallback frame: (u, v) from the mask's principal axes.

    ``s`` is u rescaled to the mask's u-extent. ``r`` is v divided by the
    **local** half width -- the largest |v| among the mask pixels in the same
    one of ``n_bins`` longitudinal bins, measured separately on each side of the
    axis -- so it means the same thing as the chart frame's r and the two frames
    can be compared. Orientation follows the same widest-end-first rule as
    ``extract_centerline``.

    Normalising by the mask's *global* minor half extent (what this did before)
    silently broke every chart-vs-pca comparison on a tapering body: with the
    half width running 150 -> 40 px, a spot at 85% of the local half width is
    r = 0.85 in the chart frame and 0.85 * 40 / 150 = 0.23 here. Measured on the
    real corpus, the 546 cross-frame pairs scored their positives *below* their
    negatives (Cohen's d -0.46, AUROC 0.351) while the same-frame pairs gave
    d 1.23 / AUROC 0.777.

    ``r`` is returned unclipped; ``extract_spotset`` clips it and logs what it
    clipped, the same as for the chart frame.
    """
    yy, xx = np.nonzero(mask)
    body = np.column_stack([xx, yy]).astype(float)
    mu = body.mean(axis=0)
    cov = np.cov((body - mu).T)
    evals, evecs = np.linalg.eigh(cov)
    e1 = evecs[:, int(np.argmax(evals))]
    e2 = np.array([-e1[1], e1[0]])

    # explicit dot products, not ``@``: numpy's matmul on the macOS Accelerate
    # BLAS raises spurious invalid/divide-by-zero RuntimeWarnings on arrays this
    # size, and this runs inside a fallback path that must stay quiet.
    def _proj(q, e):
        return q[:, 0] * e[0] + q[:, 1] * e[1]

    u_b = _proj(body - mu, e1)
    v_b = _proj(body - mu, e2)
    u0, u1 = float(u_b.min()), float(u_b.max())
    span = max(u1 - u0, 1e-6)

    # widest-end-first: compare mean |v| in the first vs last decile of u
    frac = (u_b - u0) / span
    lo = np.abs(v_b[frac < 0.1]).mean() if (frac < 0.1).any() else 0.0
    hi = np.abs(v_b[frac > 0.9]).mean() if (frac > 0.9).any() else 0.0
    if hi > lo:
        e1, e2 = -e1, -e2
        u_b = _proj(body - mu, e1)
        v_b = _proj(body - mu, e2)
        u0, u1 = float(u_b.min()), float(u_b.max())
        span = max(u1 - u0, 1e-6)

    # ---- local half width, per longitudinal bin, one value per side --------
    n_bins = max(2, int(n_bins))
    b_idx = np.minimum(((u_b - u0) / span * n_bins).astype(int), n_bins - 1)
    b_idx = np.maximum(b_idx, 0)
    hw_pos = np.zeros(n_bins)
    hw_neg = np.zeros(n_bins)
    up = v_b >= 0
    np.maximum.at(hw_pos, b_idx[up], v_b[up])
    np.maximum.at(hw_neg, b_idx[~up], -v_b[~up])
    centres = (np.arange(n_bins) + 0.5) / n_bins
    occupied = np.bincount(b_idx, minlength=n_bins) > 0
    if not occupied.all() and occupied.any():
        # a bin with no body pixels at all cannot be measured; interpolate it
        # from its neighbours rather than let it collapse to the 1 px floor.
        hw_pos = np.interp(centres, centres[occupied], hw_pos[occupied])
        hw_neg = np.interp(centres, centres[occupied], hw_neg[occupied])
    hw_pos = np.maximum(hw_pos, 1.0)
    hw_neg = np.maximum(hw_neg, 1.0)

    u_p = _proj(pts_xy - mu, e1)
    v_p = _proj(pts_xy - mu, e2)
    s = np.clip((u_p - u0) / span, 0.0, 1.0)
    hw_at = np.where(v_p >= 0, np.interp(s, centres, hw_pos), np.interp(s, centres, hw_neg))
    r = v_p / np.maximum(hw_at, 1e-6)

    hw_mean = float(0.5 * (hw_pos + hw_neg).mean())
    return s, r, {"length_px": span, "half_width_px": hw_mean,
                  "half_width_global_px": float(max(np.abs(v_b).max(), 1e-6)),
                  "aspect_measured": float(span / max(hw_mean, 1e-6)),
                  "n_bins": n_bins}


def _chart_is_unusable(mask, meta, r_raw, max_out_frac=_MAX_R_OUT_FRAC,
                       min_len_frac=_MIN_LEN_FRAC, max_abs_r=_MAX_R_ABS):
    """Reasons the prototype-02 chart should be thrown away for this mask, or [].

    Prototype 02 charts an *elongate* mask. The OSEA photos are close-ups of a
    small shark held in a hand or a tub, so the body silhouette is a fat wedge
    of aspect ~1.5-2 that fills the frame, not a tube -- and on a fat blob the
    medial-weighted longest path curls inside the widest region instead of
    spanning the body (measured on 8 tagged photos: extracted lengths of 239 to
    1018 px against area-derived expectations of 1303 to 9890 px, with spots
    landing at |r| up to 73). The checks are:

      ``centerline_warning``  prototype 02 itself flagged the mask as non-tubular.
      ``centerline_too_short``  the path spans less than ``min_len_frac`` of the
          mask's principal-axis extent -- it is not going end to end.
      ``spots_outside_body``  more than ``max_out_frac`` of the spots come out
          with |r| > 1, i.e. outside a body they were detected inside.
      ``spot_far_outside_body``  a *single* spot with |r| > ``max_abs_r``. The
          fraction test alone is not enough: at a tip where the medial axis
          hooks round, one spot can be charted at |r| = 30 (real example: image
          271, 4 of 97 spots out, fraction 0.041, well under ``max_out_frac``,
          max |r| 30.1) and clipping it to +-1 turns it into a fake rim inlier.
          Measured over the whole corpus, 40 of 1030 charted images exceeded 1
          and the worst reached 81.3.

    Any one of them means the chart's (s, r) is not a body frame, and the PCA
    frame -- which cannot curl, because it is a straight axis -- is used instead.
    """
    reasons = []
    if meta.get("centerline_warnings"):
        reasons.append("centerline_warning")
    yy, xx = np.nonzero(mask)
    body = np.column_stack([xx, yy]).astype(float)
    mu = body.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov((body - mu).T))
    e1 = evecs[:, int(np.argmax(evals))]
    u = (body[:, 0] - mu[0]) * e1[0] + (body[:, 1] - mu[1]) * e1[1]
    major = float(u.max() - u.min())
    length = float(meta.get("length_px", 0.0))
    if major > 0 and length < min_len_frac * major:
        reasons.append("centerline_too_short(%.2f)" % (length / major))
    if len(r_raw):
        abs_r = np.abs(r_raw)
        out = float(np.mean(abs_r > 1.0))
        if out > max_out_frac:
            reasons.append("spots_outside_body(%.2f)" % out)
        mx = float(abs_r.max())
        if mx > max_abs_r:
            reasons.append("spot_far_outside_body(%.1f)" % mx)
    return reasons


@dataclass
class SpotSet:
    """Spot constellation in a body-normalised frame.

    Attributes:
        s: (n,) arc-length position along the medial axis, 0 = widest end, 1 = other end.
        r: (n,) signed lateral offset, normalised by the local half width, in [-1, 1].
        size: (n,) sqrt(w*h) of the spot box, as a fraction of the body length.
        conf: (n,) detector confidence.
        frame: "chart" (prototype 02) or "pca" (fallback) or "toy".
        aspect: the s-stretch used to build a metric from (s, r).
        meta: diagnostics; ``meta["drops"]`` lists everything discarded.
    """

    s: np.ndarray
    r: np.ndarray
    size: np.ndarray
    conf: np.ndarray
    frame: str = "chart"
    aspect: float = DEFAULT_ASPECT
    meta: dict = field(default_factory=dict)
    cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        self.s = np.asarray(self.s, dtype=float).ravel()
        self.r = np.asarray(self.r, dtype=float).ravel()
        self.size = np.asarray(self.size, dtype=float).ravel()
        self.conf = np.asarray(self.conf, dtype=float).ravel()
        n = len(self.s)
        if not (len(self.r) == len(self.size) == len(self.conf) == n):
            raise ValueError("SpotSet arrays must be the same length")

    def __len__(self):
        return len(self.s)

    def xy(self):
        """(n, 2) metric coordinates ``(s * aspect, r)``."""
        return np.column_stack([self.s * self.aspect, self.r])

    def flipped(self, flip_s=False, flip_r=False):
        """A new SpotSet with ``s -> 1-s`` and/or ``r -> -r`` (cached)."""
        key = (bool(flip_s), bool(flip_r))
        if key == (False, False):
            return self
        store = self.cache.setdefault("_flips", {})
        if key not in store:
            store[key] = SpotSet(
                s=(1.0 - self.s) if flip_s else self.s.copy(),
                r=(-self.r) if flip_r else self.r.copy(),
                size=self.size.copy(), conf=self.conf.copy(),
                frame=self.frame, aspect=self.aspect,
                meta=dict(self.meta, flip=key),
            )
        return store[key]

    def to_dict(self):
        return {"s": self.s.tolist(), "r": self.r.tolist(),
                "size": self.size.tolist(), "conf": self.conf.tolist(),
                "frame": self.frame, "aspect": self.aspect, "meta": self.meta}

    @staticmethod
    def from_dict(d):
        return SpotSet(np.array(d["s"]), np.array(d["r"]), np.array(d["size"]),
                       np.array(d["conf"]), d.get("frame", "chart"),
                       float(d.get("aspect", DEFAULT_ASPECT)), dict(d.get("meta", {})))


def extract_spotset(detection, max_side=MAX_SIDE, n_s=N_S, n_r=N_R,
                    aspect=DEFAULT_ASPECT, r_norm="ray", min_spots=1):
    """Rectify one detection dict into a :class:`SpotSet`.

    Args:
        detection: dict with keys ``body_polygon`` (required),
            ``obstruction_polygons``, ``spots``, ``width``, ``height``. Extra
            keys (``image_id``, ``individual_code``, ...) are copied to
            ``meta["record"]``.
        r_norm: ``"ray"`` (default; contiguous mask run along the station
            normal) or ``"edt"`` (distance transform at the station -- the
            medial radius, which over-normalises near the tips).
        min_spots: below this many usable spots the function returns None.

    Returns:
        SpotSet, or None when the detection has no usable body polygon or too
        few spots. ``SpotSet.frame`` says which frame was actually used and
        ``SpotSet.meta["drops"]`` says why: prototype 02 is tried first, and its
        chart is thrown away for the PCA frame when ``_chart_is_unusable``
        objects (which it does for most real OSEA photos -- their body masks are
        fat wedges, not tubes).
    """
    drops = []
    body = detection.get("body_polygon")
    if not body or len(body) < 3:
        return None
    spots = detection.get("spots") or []
    if len(spots) < min_spots:
        return None

    mask_punched, scale = build_body_mask(
        body, detection.get("obstruction_polygons"),
        detection.get("width"), detection.get("height"), max_side=max_side,
    )
    mask = _largest_filled(mask_punched)
    # _largest_filled fills holes, which restores every obstruction that lies
    # wholly inside the silhouette (see its docstring -- deliberate, because the
    # chart must describe the mask prototype 02 charts). Say so rather than let
    # the punch-out look effective.
    n_refilled = int(np.count_nonzero(mask & ~mask_punched))
    if n_refilled:
        drops.append({"reason": "obstruction_refilled", "n_px": n_refilled,
                      "frac_of_mask": float(n_refilled) / float(max(int(mask.sum()), 1))})

    cx = np.array([float(sp["cx"]) for sp in spots]) * scale
    cy = np.array([float(sp["cy"]) for sp in spots]) * scale
    sw = np.array([float(sp.get("w", 0.0)) for sp in spots]) * scale
    sh = np.array([float(sp.get("h", 0.0)) for sp in spots]) * scale
    conf = np.array([float(sp.get("conf", 1.0)) for sp in spots])
    pts = np.column_stack([cx, cy])

    # spot centres that fall outside the (filled, largest-component) mask are
    # dropped -- run_image already filters on the raw polygon, so this only
    # catches the rasterisation/obstruction edge cases.
    h_m, w_m = mask.shape
    ii = np.clip(np.round(cy).astype(int), 0, h_m - 1)
    jj = np.clip(np.round(cx).astype(int), 0, w_m - 1)
    keep = mask[ii, jj]
    if not keep.all():
        drops.append({"reason": "spot_centre_outside_mask", "n": int((~keep).sum())})
    if keep.sum() < min_spots:
        return None
    pts, sw, sh, conf = pts[keep], sw[keep], sh[keep], conf[keep]

    frame = "chart"
    meta = {}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cl = cl02.extract_centerline(mask, n_stations=n_s)
        for wmsg in caught:
            meta.setdefault("centerline_warnings", []).append(str(wmsg.message)[:200])

        stations = cl02.resample_polyline(cl, n_s)
        _, normals = fr02.tangents_normals_2d(stations)
        edt = ndimage.distance_transform_edt(mask)
        half_width = max(float(edt.max()), 1.0)

        idx = chart02.image_to_chart(cl, half_width, n_s, n_r, pts)
        s_idx = idx[:, 0]
        r_px = -half_width + idx[:, 1] * (2.0 * half_width / (n_r - 1))

        hw_edt = ndimage.map_coordinates(
            edt, [stations[:, 1], stations[:, 0]], order=1, mode="nearest")
        if r_norm == "edt":
            hw_pos = hw_neg = np.maximum(hw_edt, 1.0)
        elif r_norm == "ray":
            hw_pos, hw_neg = _station_half_widths(
                mask, stations, normals, max_reach=half_width * 1.6)
            hw_pos = np.maximum(hw_pos, np.maximum(hw_edt, 1.0))
            hw_neg = np.maximum(hw_neg, np.maximum(hw_edt, 1.0))
        else:
            raise ValueError("r_norm must be 'ray' or 'edt'")

        f = np.clip(s_idx, 0.0, n_s - 1.0)
        hw_at = np.where(r_px >= 0, np.interp(f, np.arange(n_s), hw_pos),
                         np.interp(f, np.arange(n_s), hw_neg))
        r_raw = r_px / np.maximum(hw_at, 1e-6)
        s = np.clip(s_idx / float(n_s - 1), 0.0, 1.0)

        length_px = float(cl02.arc_length(stations)[-1])
        hw_mean = float(0.5 * (hw_pos + hw_neg).mean())
        meta.update({
            "length_px": length_px,
            "half_width_px": hw_mean,
            "half_width_edt_px": float(hw_edt.mean()),
            "aspect_measured": float(length_px / max(hw_mean, 1e-6)),
            "r_norm": r_norm,
        })
    except Exception as exc:  # noqa: BLE001 -- any 02 failure falls back
        drops.append({"reason": "centerline_failed", "detail": "%s: %s"
                      % (type(exc).__name__, str(exc)[:160])})
        frame = "pca"

    if frame == "chart":
        bad = _chart_is_unusable(mask, meta, r_raw)
        if bad:
            drops.append({"reason": "chart_frame_rejected", "detail": bad})
            meta["chart_meta"] = {k: meta[k] for k in
                                  ("length_px", "half_width_px", "aspect_measured")
                                  if k in meta}
            frame = "pca"
    if frame == "pca":
        s, r_raw, pca_meta = _pca_frame(mask, pts)
        meta.update(pca_meta)
        meta["r_norm"] = "pca_minor_extent"

    n_out = int((np.abs(r_raw) > 1.0).sum())
    if n_out:
        drops.append({"reason": "r_clipped_to_unit", "n": n_out,
                      "max_abs_r": float(np.abs(r_raw).max())})
    meta["max_abs_r_raw"] = float(np.abs(r_raw).max()) if len(r_raw) else 0.0
    r = np.clip(r_raw, -1.0, 1.0)

    length_px = float(meta.get("length_px", 1.0))
    size = np.sqrt(np.maximum(sw * sh, 0.0)) / max(length_px, 1e-6)

    meta["n_spots_in"] = len(spots)
    meta["n_spots_out"] = len(s)
    meta["scale"] = scale
    meta["mask_area_px"] = int(mask.sum())
    meta["drops"] = drops
    meta["record"] = {k: v for k, v in detection.items()
                      if k not in ("body_polygon", "obstruction_polygons", "spots")}
    return SpotSet(s=s, r=r, size=size, conf=conf, frame=frame, aspect=aspect, meta=meta)


# --------------------------------------------------------------------------- #
# 3. shape-context descriptors                                                 #
# --------------------------------------------------------------------------- #
def descriptors(spotset, k=K_NN, n_radial=N_RADIAL, n_angular=N_ANGULAR):
    """Per-spot shape context. Returns (hist (n, n_radial*n_angular), median_nn).

    The neighbourhood is the ``k`` nearest spots in the metric space
    ``(s * aspect, r)``; radii are divided by the *set's* median
    nearest-neighbour distance (scale normalisation) and binned in log space
    between ``_RHO_MIN`` and ``_RHO_MAX``; angles are measured from the +s
    direction with no per-point rotation normalisation, because the chart has
    already fixed the orientation. Histograms are L1-normalised.
    """
    key = ("desc", k, n_radial, n_angular)
    if key in spotset.cache:
        return spotset.cache[key]

    p = spotset.xy()
    n = len(p)
    n_bins = n_radial * n_angular
    if n < 2:
        out = (np.zeros((n, n_bins)), 1.0)
        spotset.cache[key] = out
        return out

    diff = p[:, None, :] - p[None, :, :]                    # (n, n, 2) i minus j
    dist = np.hypot(diff[..., 0], diff[..., 1])
    np.fill_diagonal(dist, np.inf)
    median_nn = float(np.median(dist.min(axis=1)))
    if not np.isfinite(median_nn) or median_nn <= 0:
        median_nn = 1.0

    kk = min(k, n - 1)
    nbr = np.argsort(dist, axis=1)[:, :kk]                  # (n, kk)
    rows = np.repeat(np.arange(n), kk)
    cols = nbr.ravel()
    v = p[cols] - p[rows]                                   # neighbour minus centre
    rho = np.hypot(v[:, 0], v[:, 1]) / median_nn
    theta = np.arctan2(v[:, 1], v[:, 0])

    lo, hi = np.log(_RHO_MIN), np.log(_RHO_MAX)
    lr = (np.log(np.maximum(rho, _EPS)) - lo) / (hi - lo) * n_radial
    rb = np.clip(np.floor(lr).astype(int), 0, n_radial - 1)
    ab = np.floor((theta + np.pi) / (2 * np.pi) * n_angular).astype(int) % n_angular

    flat = rows * n_bins + rb * n_angular + ab
    hist = np.bincount(flat, minlength=n * n_bins).astype(float).reshape(n, n_bins)
    hist /= np.maximum(hist.sum(axis=1, keepdims=True), 1.0)

    out = (hist, median_nn)
    spotset.cache[key] = out
    return out


def _chi2(ha, hb):
    """Chi2 cost matrix between two L1-normalised histogram stacks, in [0, 1]."""
    a = ha[:, None, :]
    b = hb[None, :, :]
    num = (a - b) ** 2
    den = a + b
    return 0.5 * np.where(den > _EPS, num / np.maximum(den, _EPS), 0.0).sum(axis=2)


# --------------------------------------------------------------------------- #
# 4. geometric verification                                                    #
# --------------------------------------------------------------------------- #
# Admissible scales. After per-image body normalisation the residual s-scale
# between two sightings is just the ratio of visible extents, so it lives near
# 1. The bounds are a hard prior, not a measured optimum: they exist so that
# RANSAC cannot compress one set onto the other, which would multiply the local
# density and inflate the chance inlier count.
_ALPHA_LO, _ALPHA_HI = 0.72, 1.40   # admissible s scale
_GAMMA_LO, _GAMMA_HI = 0.80, 1.25   # admissible r scale
_MIN_BASELINE = 0.3                 # min metric separation of a minimal sample
_SEED_FRAC = 0.5                    # fraction of the assignment used as RANSAC seeds
_TAU_Q = 0.15                       # descriptor gate: quantile of the chi2 matrix
_N_REFINE = 6                       # hypotheses carried to the consensus stage
_BIG = 1e9                          # sentinel cost for pairs excluded by the gates
_N_ICP = 4                          # consensus re-fit / re-assign iterations
_N_ITERS = 400                      # RANSAC minimal samples per flip
_SEED_MODE = "hungarian"            # RANSAC seed pool; see _seed_pairs
_DESC_GATE = "quantile"             # descriptor admissibility; see _descriptor_gate
_RANK_T = 3                         # descriptor rank depth for the rank gates


def _fit_apply(model, pa, pb, i, j, aspect):
    """Vectorised minimal-sample fit b -> a. i, j are (m,) index arrays.

    Returns (mapped (m, n, 2) or None, valid (m,) bool) where ``mapped[k]`` is
    all of ``pb`` under the k-th hypothesis, in the metric space of ``pa``.
    """
    sb, rb = pb[:, 0] / aspect, pb[:, 1]
    sa, ra = pa[:, 0] / aspect, pa[:, 1]
    valid = np.ones(len(i), dtype=bool)

    if model in ("axis", "s_affine"):
        dsb = sb[i] - sb[j]
        valid &= np.abs(dsb) > 1e-6
        alpha = np.where(valid, (sa[i] - sa[j]) / np.where(valid, dsb, 1.0), 1.0)
        beta = sa[i] - alpha * sb[i]
        valid &= (alpha > _ALPHA_LO) & (alpha < _ALPHA_HI)
        s_map = alpha[:, None] * sb[None, :] + beta[:, None]
        if model == "s_affine":
            r_map = np.repeat(rb[None, :], len(i), axis=0)
        else:
            # gamma (the r scale) is NOT estimated from the minimal sample:
            # two points separated by dr give gamma = dra/drb, an unbounded
            # ratio when the pair happens to share an r. The sample fixes
            # gamma = 1 and takes delta as the mean r offset of its two points
            # (well conditioned); the least-squares refit in ``_refit`` recovers
            # gamma from the whole consensus set. Measured: this changed nothing
            # end-to-end on the r-frame-perturbed toy (the --toy-hard generator
            # at 2% jitter / 20% dropout, where the shipped code scores Rank-1
            # 0.640 -- the same either way),
            # so it is kept for the unbounded-ratio guarantee, not for a score.
            delta = 0.5 * ((ra[i] - rb[i]) + (ra[j] - rb[j]))
            r_map = rb[None, :] + delta[:, None]
        return np.stack([s_map * aspect, r_map], axis=2), valid

    if model == "sim":
        za = pa[:, 0] + 1j * pa[:, 1]
        zb = pb[:, 0] + 1j * pb[:, 1]
        dzb = zb[i] - zb[j]
        valid &= np.abs(dzb) > 1e-9
        c = np.where(valid, (za[i] - za[j]) / np.where(valid, dzb, 1.0), 1.0)
        valid &= (np.abs(c) > _ALPHA_LO) & (np.abs(c) < _ALPHA_HI)
        t = za[i] - c * zb[i]
        z_map = c[:, None] * zb[None, :] + t[:, None]
        return np.stack([z_map.real, z_map.imag], axis=2), valid

    raise ValueError("unknown model %r" % (model,))


def _refit(model, pa, pb, aspect):
    """Least-squares fit of ``model`` on the given (already-inlier) pairs.

    ``pa``/``pb`` are (m, 2) metric coordinates of corresponding points.
    Returns a callable mapping (n, 2) metric points from b's frame into a's.
    """
    sb, rb = pb[:, 0] / aspect, pb[:, 1]
    sa, ra = pa[:, 0] / aspect, pa[:, 1]

    def _lin(x, y):
        vx = float(np.var(x))
        if vx < 1e-9:
            return 1.0, float(np.mean(y) - np.mean(x))
        a = float(np.cov(x, y, bias=True)[0, 1] / vx)
        return a, float(np.mean(y) - a * np.mean(x))

    if model == "sim":
        mua, mub = pa.mean(axis=0), pb.mean(axis=0)
        qa, qb = pa - mua, pb - mub
        den = float((qb ** 2).sum())
        if den < 1e-12:
            return lambda q: q + (mua - mub)
        cc = float((qa[:, 0] * qb[:, 0] + qa[:, 1] * qb[:, 1]).sum()) / den
        ss = float((qa[:, 1] * qb[:, 0] - qa[:, 0] * qb[:, 1]).sum()) / den
        c = complex(cc, ss)
        if not (_ALPHA_LO < abs(c) < _ALPHA_HI):
            c = c / max(abs(c), 1e-9)
        t = complex(*mua) - c * complex(*mub)

        def _apply_sim(q):
            z = c * (q[:, 0] + 1j * q[:, 1]) + t
            return np.column_stack([z.real, z.imag])
        return _apply_sim

    alpha, beta = _lin(sb, sa)
    alpha = float(np.clip(alpha, _ALPHA_LO, _ALPHA_HI))
    if model == "s_affine":
        gamma, delta = 1.0, 0.0
    else:
        gamma, delta = _lin(rb, ra)
        gamma = float(np.clip(gamma, _GAMMA_LO, _GAMMA_HI))

    def _apply_axis(q):
        return np.column_stack([(alpha * (q[:, 0] / aspect) + beta) * aspect,
                                gamma * q[:, 1] + delta])
    return _apply_axis


def _descriptor_gate(cost, mode, tau_q, rank_t):
    """Boolean (n_a, n_b) mask of pairs whose descriptors agree well enough.

    ``mode``:
      ``"quantile"``   cost < the ``tau_q`` quantile of the whole matrix. A
                       *global* threshold, so a random pair passes it ``tau_q``
                       of the time (0.15 at the default) -- far too generous.
      ``"rank_union"`` j is among a's ``rank_t`` cheapest partners, or i is
                       among b's. A random pair passes at ~2*rank_t/n (0.09 at
                       rank_t=3, n=66) while a true correspondence passes at its
                       top-3 recall (0.54 measured at 2% jitter), which is a
                       sharper per-pair contrast than the quantile gate's.
      ``"rank_both"``  the intersection: ~(rank_t/n)^2 for a random pair.
      ``"none"``       no descriptor gate (geometry only).

    The sharper gate does NOT win end-to-end, which is why ``quantile`` is the
    default. ``--ablate`` group "desc_gate", Rank-1 / AUROC / mean-different
    score at 2% jitter: 0.980 / 0.9932 / 0.263 (quantile), 0.930 / 0.9842 /
    0.199 (rank_union t=3), 0.920 / 0.9812 / 0.236 (rank_union t=5), 0.810 /
    0.9608 / 0.390 (none). The rank gates do suppress different-individual
    scores hardest, but they throw away true inliers at nearly the same rate, so
    the *ordering* gets no better; a gate is still essential -- dropping it costs
    17 points of Rank-1 and lifts the chance floor from 0.263 to 0.390.
    """
    if mode == "none":
        return np.ones(cost.shape, dtype=bool)
    if mode == "quantile":
        return cost < float(np.quantile(cost, tau_q))
    t = max(1, int(rank_t))
    na, nb = cost.shape
    row = np.zeros(cost.shape, dtype=bool)
    row[np.arange(na)[:, None], np.argsort(cost, axis=1)[:, :t]] = True
    col = np.zeros(cost.shape, dtype=bool)
    col[np.argsort(cost, axis=0)[:t, :], np.arange(nb)[None, :]] = True
    if mode == "rank_union":
        return row | col
    if mode == "rank_both":
        return row & col
    raise ValueError("unknown desc_gate %r" % (mode,))


def _seed_pairs(cost, seed_mode):
    """Candidate correspondences that seed RANSAC. Returns (rows, cols).

    Measured seed precision on the toy at the 2%-jitter operating point (the
    fraction of seeds that are the true correspondence, k=20 descriptors):
    ``hungarian`` 0.19, ``mutual`` 0.25, ``top1`` 0.29 (a seed-pool number, not
    part of ``--ablate``). A minimal sample needs BOTH of its pairs to be right,
    so those are 0.036 / 0.063 / 0.084 good samples per draw. End-to-end the
    difference largely washes out once the RANSAC budget is big enough to find a
    good sample either way -- ``--ablate`` group "seed_mode" at ``n_iters=400``:
    Rank-1 0.980 / 0.940 / 0.970 and AUROC 0.9932 / 0.9799 / 0.9912 for
    hungarian / mutual / top1. ``hungarian`` is the default because it is what
    the brief specifies and it has the best measured Rank-1 and AUROC; ``top1``
    is the one to use if the budget is cut, since the pool has no one-to-one
    constraint forcing unmatchable spots onto a partner.
    """
    if seed_mode == "hungarian":
        ri, ci = linear_sum_assignment(cost)
        return np.asarray(ri), np.asarray(ci)
    best_b = np.argmin(cost, axis=1)
    rows = np.arange(cost.shape[0])
    if seed_mode == "top1":
        return rows, best_b
    if seed_mode == "mutual":
        best_a = np.argmin(cost, axis=0)
        keep = best_a[best_b] == rows
        return rows[keep], best_b[keep]
    raise ValueError("unknown seed_mode %r" % (seed_mode,))


def _score_one(a, b, model, k, gate_frac, n_iters, rng, tau_q=_TAU_Q, n_icp=_N_ICP,
               seed_mode=_SEED_MODE, desc_gate=_DESC_GATE, rank_t=_RANK_T):
    """Match ``b`` (already flipped) onto ``a``. Returns (score, inliers, gate).

    Three stages:

    1. *Correspondence hypotheses.* Chi2 descriptor cost -> ``_seed_pairs``
       (Hungarian assignment by default); the cheapest ``_SEED_FRAC`` of those
       pairs are the RANSAC seed pool. The assignment is only ~19% correct at
       the operating point (measured, k=20, 2% jitter) because dropout and
       clutter leave part of one set with no partner at all and a one-to-one
       assignment must pair them with *something*; taking the cheap half raises
       the precision of the pool.
    2. *RANSAC.* All minimal samples are fitted and scored in one vectorised
       pass; the ``_N_REFINE`` best hypotheses go on.
    3. *Consensus (ICP).* Each surviving hypothesis is refitted by least squares
       on its consensus set and applied to *all* of b; the inlier set is a
       Hungarian assignment on the chi2 cost restricted to pairs that pass BOTH
       gates -- geometric distance < ``gate`` and the descriptor gate. That
       re-fit / re-assign step is iterated up to ``n_icp`` times, stopping as
       soon as the count stops growing (``--ablate`` group "n_icp": Rank-1 0.960
       at n_icp=1, 0.970 at 2, 0.980 at 4, 0.980 at 6).
       Geometry alone is not enough: the gate disc holds ~0.28 random spots at
       any density, so a purely geometric consensus scores different individuals
       at 0.390 (``--ablate`` group "desc_gate", row ``none``); adding the
       descriptor gate drops that to 0.263 while the same-individual score falls
       only from 0.586 to 0.521.
    """
    na, nb = len(a), len(b)
    if na < 3 or nb < 3:
        return 0.0, 0, 0.0
    ha, mnn_a = descriptors(a, k=k)
    hb, mnn_b = descriptors(b, k=k)
    gate = gate_frac * 0.5 * (mnn_a + mnn_b)
    if gate <= 0:
        return 0.0, 0, 0.0

    cost = _chi2(ha, hb)
    desc_ok = _descriptor_gate(cost, desc_gate, tau_q, rank_t)
    ri, ci = _seed_pairs(cost, seed_mode)
    if len(ri) < 2:
        return 0.0, 0, gate
    order = np.argsort(cost[ri, ci], kind="mergesort")
    m = max(4, int(_SEED_FRAC * len(order)))
    ri, ci = ri[order[:m]], ci[order[:m]]
    m = len(ri)
    if m < 2:
        return 0.0, 0, gate

    pa_all, pb_all = a.xy(), b.xy()
    pa, pb = pa_all[ri], pb_all[ci]

    i = rng.integers(0, m, size=n_iters * 4)
    j = rng.integers(0, m, size=n_iters * 4)
    ok = (i != j) & (np.hypot(*(pb[i] - pb[j]).T) > _MIN_BASELINE)
    i, j = i[ok][:n_iters], j[ok][:n_iters]
    if len(i) == 0:
        return 0.0, 0, gate

    mapped, valid = _fit_apply(model, pa, pb, i, j, a.aspect)
    res = np.hypot(mapped[..., 0] - pa[None, :, 0], mapped[..., 1] - pa[None, :, 1])
    inl = (res < gate) & valid[:, None]
    counts = inl.sum(axis=1)

    n_in = 0
    for h in np.argsort(-counts, kind="mergesort")[:_N_REFINE]:
        sel = inl[h]
        if sel.sum() < 2:
            continue
        src_a, src_b = pa[sel], pb[sel]
        best_h = 0
        for it in range(max(1, n_icp)):
            apply = _refit(model, src_a, src_b, a.aspect)
            pm = apply(pb_all)
            dist = np.hypot(pa_all[:, None, 0] - pm[None, :, 0],
                            pa_all[:, None, 1] - pm[None, :, 1])
            ok2 = (dist < gate) & desc_ok
            if not ok2.any():
                break
            gated = np.where(ok2, cost, _BIG)
            rr, cc = linear_sum_assignment(gated)
            keep = gated[rr, cc] < _BIG
            cnt = int(keep.sum())
            if cnt < 2 or (it and cnt <= best_h):
                best_h = max(best_h, cnt)
                break
            best_h = cnt
            src_a, src_b = pa_all[rr[keep]], pb_all[cc[keep]]
        n_in = max(n_in, best_h)

    return float(n_in) / float(min(na, nb)), n_in, gate


def _directed_score(a, b, model, k, gate_frac, n_iters, seed, try_flips,
                    tau_q, n_icp, seed_mode, desc_gate, rank_t):
    """Best over the four flips of ``b``, registering b onto a. One direction.

    The RNG is created here, not by the caller, so the stream a direction sees
    depends only on ``seed`` -- which is what makes ``match_score``'s two
    directions independent of the order they are evaluated in.
    """
    rng = np.random.default_rng(seed)
    flips = [(False, False), (True, False), (False, True), (True, True)] \
        if try_flips else [(False, False)]
    best = {"score": 0.0, "n_inliers": 0, "n_a": len(a), "n_b": len(b),
            "flip": (False, False), "gate": 0.0, "model": model}
    for fs, fr in flips:
        bb = b.flipped(fs, fr)
        sc, nin, gate = _score_one(a, bb, model, k, gate_frac, n_iters, rng,
                                   tau_q=tau_q, n_icp=n_icp, seed_mode=seed_mode,
                                   desc_gate=desc_gate, rank_t=rank_t)
        if sc > best["score"] or (sc == best["score"] and nin > best["n_inliers"]):
            best = {"score": sc, "n_inliers": nin, "n_a": len(a), "n_b": len(b),
                    "flip": (fs, fr), "gate": gate, "model": model}
    return best


def match_score(a, b, model="axis", k=K_NN, gate_frac=GATE_FRAC, n_iters=_N_ITERS,
                seed=0, try_flips=True, tau_q=_TAU_Q, n_icp=_N_ICP,
                seed_mode=_SEED_MODE, desc_gate=_DESC_GATE, rank_t=_RANK_T,
                symmetric=True):
    """Score two constellations.

    Returns:
        dict(score, n_inliers, n_a, n_b, flip, gate, model, direction) where
        ``flip`` is the (flip_s, flip_r) that brings the two into register (it
        is the same relative flip whichever set it was applied to, because the
        admissible s scale is strictly positive) and ``direction`` is ``"ab"``
        when b was mapped onto a and ``"ba"`` when a was mapped onto b.
        ``score = n_inliers / min(n_a, n_b)`` in [0, 1].

    Symmetry. One directed registration is *not* symmetric: ``_fit_apply`` maps
    b into a's metric and measures the residual there, ``_MIN_BASELINE`` gates
    the minimal sample on b's separation only, and ``_refit`` regresses b on a
    rather than the reverse -- so swapping the arguments changes the score, by a
    measured mean of 0.018 and a max of 0.10 on a scale whose different-
    individual mean is 0.25, and the gap does not close as ``n_iters`` grows
    (mean 0.029 / 0.022 / 0.024 at 400 / 2000 / 8000 iterations). With
    ``symmetric=True`` (the default) both directions are scored and the better
    is returned, which makes ``match_score(a, b) == match_score(b, a)``
    exactly, at twice the cost. Pass ``symmetric=False`` for the raw directed
    score -- and then remember that a pairwise evaluation's numbers depend on
    the order its records happen to sit in.

    Why the *better* and not the mean. Measured at 2% jitter / 20% dropout /
    20% clutter, 5 seeds x 20 identities: directed Rank-1 0.980 AUROC 0.9940
    diff 0.253; max 0.980 / 0.9932 / 0.263; mean 0.980 / 0.9948 / 0.253. The
    mean is marginally ahead on AUROC and does not lift the chance floor, but it
    breaks the identity ``score == n_inliers / min(n_a, n_b)`` -- there is no
    inlier set behind an averaged score -- so ``score`` is the max, which is a
    real hypothesis with a real ``flip``, ``gate`` and inlier count, and the
    mean is reported alongside it as ``score_mean`` for anyone who wants it.
    ``score_ab`` / ``score_ba`` are the two directed scores.
    """
    ab = _directed_score(a, b, model, k, gate_frac, n_iters, seed, try_flips,
                         tau_q, n_icp, seed_mode, desc_gate, rank_t)
    ab["direction"] = "ab"
    if not symmetric:
        ab.update({"score_ab": ab["score"], "score_ba": None,
                   "score_mean": ab["score"]})
        return ab
    ba = _directed_score(b, a, model, k, gate_frac, n_iters, seed, try_flips,
                         tau_q, n_icp, seed_mode, desc_gate, rank_t)
    # n_a / n_b name the arguments the CALLER passed, not the direction that won
    ba.update({"n_a": len(a), "n_b": len(b), "direction": "ba"})
    best = ba if (ba["score"], ba["n_inliers"]) > (ab["score"], ab["n_inliers"]) else ab
    best.update({"score_ab": ab["score"], "score_ba": ba["score"],
                 "score_mean": 0.5 * (ab["score"] + ba["score"])})
    return best


def rank(query, gallery, ids=None, **kw):
    """Score ``query`` against every SpotSet in ``gallery``, best first.

    Args:
        query: SpotSet.
        gallery: sequence of SpotSet.
        ids: optional labels, one per gallery entry (default: the index).
    Returns:
        list of dicts (the ``match_score`` dict plus ``id`` and ``index``),
        sorted by score then inlier count, descending.
    """
    out = []
    for idx, g in enumerate(gallery):
        res = match_score(query, g, **kw)
        res["index"] = idx
        res["id"] = ids[idx] if ids is not None else idx
        out.append(res)
    out.sort(key=lambda d: (d["score"], d["n_inliers"]), reverse=True)
    return out
