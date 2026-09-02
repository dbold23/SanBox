"""The OSEA detection contract, wrapped so real photos and synthetic renders
go through *exactly* the same door.

Prototype 06 ("spot-proxy") never lets the matcher see RGB.  Both domains are
reduced to the same non-RGB representation -- a YOLO body polygon plus YOLO
spot boxes -- by the *same* detector, and identity is matched on the spot
constellation alone.  This module is that door.  It has three jobs:

  1. ``load_models``  -- resolve the OSEA weights (with the documented v2->v1
     spot fallback) and hand back the two ultralytics models.
  2. ``detect``       -- run ``spot_detector/scripts/infer_pipeline.run_image``
     and reshape its output into the *catalog.db column shapes* that
     ``reid/scripts/pipeline_worker.stage_detect`` writes.  Whatever comes out
     of here is byte-for-byte what the OSEA tagger would have stored.
  3. ``features``     -- turn one detection into the non-RGB feature dict that
     the rest of prototype 06 (distribution matching, constellation matching)
     consumes.

Nothing here writes to the database, and nothing here modifies the OSEA tree.

--------------------------------------------------------------------------
Contract notes (verified by reading the OSEA source, not assumed)
--------------------------------------------------------------------------
* ``run_image`` is imported, not re-implemented.  ``infer_pipeline`` is
  import-safe: its module level only defines paths and constants.
* Weights.  ``DEFAULT_BODY_OBSTR`` (runs/body_obstr/v1) exists.
  ``DEFAULT_SPOTS`` (runs/spots/v2) does **not** exist on this checkout, so we
  fall back to ``DEFAULT_SPOTS_FALLBACK`` (runs/spots/v1), exactly as both
  ``infer_pipeline.main`` and ``stage_detect`` do.  The head weights
  (runs/head/v1) do not exist either, so the head model is ``None`` and the
  three head fields are always ``None``.
* Confidences: body 0.40, head 0.40, spot 0.25 -- the OSEA defaults.
* **The spot count is capped.**  ``run_image`` calls ``predict`` without a
  ``max_det``, so ultralytics' default of 300 applies and NMS silently returns
  only its 300 best boxes.  The centre-inside-body / not-in-obstruction filter
  then runs *after* the cap, so a truncated image can report any ``spot_count``
  at or below 300 -- on this catalog 24 images are truncated and only 6 of them
  show ``spot_count == 300``.  We keep the cap (it is what the tagger stores)
  and report it: :func:`detect` returns ``spots_raw_count`` /
  ``spots_max_det`` / ``spots_truncated`` alongside the DB columns, and
  ``real_features.py`` counts the truncated images into ``summary.json``.
  Treat ``n_spots`` and ``density`` on a truncated image as lower bounds.
* Image sizes: the body model runs at imgsz=640 on the full frame, the spot
  model at imgsz=1280 on the body crop padded by PAD_FRAC=0.05.  Both are
  inside ``run_image``; we do not touch them.
* **Channel order.**  The OSEA pipeline hands ultralytics an *RGB* array
  straight from ``PIL.ImageOps.exif_transpose`` while ultralytics documents
  numpy input as BGR.  That quirk is part of the deployed contract, so we
  replicate it: ``detect`` takes an RGB uint8 array and passes it through
  unchanged.  Synthetic renders must be fed the same way (RGB, uint8,
  H x W x 3) or they will not sit in the same detector input distribution.

--------------------------------------------------------------------------
The PCA frame and its sign ambiguity
--------------------------------------------------------------------------
``features`` puts every spot into a body-attached frame built from the second
moments of the *filled* body polygon, measured by rasterising it (see
``raster_moments``), so the frame follows the region rather than the vertex
sampling density and survives the self-intersecting contours the body
segmenter occasionally emits:

    origin  = area centroid of the body polygon
    e_major = principal eigenvector of the central second-moment matrix
    e_minor = the other eigenvector
    L_major = extent of the polygon along e_major, in px
    D_minor = extent of the polygon along e_minor, in px   (the length unit)
    u = ((p - origin) . e_major) / D_minor
    v = ((p - origin) . e_minor) / D_minor

An eigenvector is only defined up to sign, so the frame is defined up to the
four-element group {u -> +-u} x {v -> +-v}.  Two things follow, and both are
enforced by ``tests/test_contract.py``:

* **Every scalar in ``feats``** is a function of sign-invariant quantities
  only -- counts, areas, extents, pairwise distances, |quantile| pairs are not
  used at all.  Sizes, confidences, nearest-neighbour distances and densities
  are invariant under any orthogonal map, so the scalar block is identical
  under both flips (and under rotation, translation and mirroring of the whole
  scene).
* **The exported ``u``/``v`` arrays** cannot be made canonical from geometry
  alone (a photographed shark has no detectable head-vs-tail or
  dorsal-vs-ventral cue in a polygon).  We therefore pin the sign with an
  explicit, documented *tie-break*: the third central moment.  The sign of
  each axis is chosen so that the spot skewness along it is >= 0, falling back
  to the polygon skewness and finally to a fixed sign when the skewness is
  numerically zero.  This is deterministic and rotation-equivariant but has no
  biological meaning; downstream matchers must still be flip-invariant (try
  all four sign combinations, or use flip-invariant descriptors).
  ``feats["frame"]["sign_rule"]`` records which rule fired.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

__all__ = [
    "main_root",
    "weight_paths",
    "load_models",
    "detect",
    "features",
    "DB_COLUMNS",
    "BODY_CONF",
    "HEAD_CONF",
    "SPOT_CONF",
    "spot_max_det",
]

BODY_CONF = 0.40
HEAD_CONF = 0.40
SPOT_CONF = 0.25

#: Last-resort value for the ultralytics detection cap, used only when the
#: installed ultralytics does not expose a default (see :func:`spot_max_det`).
SPOT_MAX_DET_FALLBACK = 300

#: The catalog.db columns that ``detect`` mirrors, in the order
#: ``pipeline_worker.stage_detect`` writes them.
DB_COLUMNS = (
    "body_polygon_json",
    "body_bbox_json",
    "body_conf",
    "head_polygon_json",
    "head_bbox_json",
    "head_conf",
    "obstruction_polygon_json",
    "obstruction_count",
    "spots_json",
    "spot_count",
)

_MARKER = Path("spot_detector") / "scripts" / "infer_pipeline.py"


# --------------------------------------------------------------------------- #
# Locating the OSEA checkout                                                   #
# --------------------------------------------------------------------------- #
def main_root() -> Path:
    """Absolute path of the OSEA main checkout (the one holding spot_detector/).

    Resolution order:
      1. ``$SEVENGILL_MAIN_ROOT`` if set (must contain the marker file).
      2. Walk up from this file until a directory contains
         ``spot_detector/scripts/infer_pipeline.py``.  This prototype lives in
         ``MAIN/.claude/worktrees/<name>/prototypes/06-spot-proxy``, so the walk
         finds MAIN five levels up.
      3. Raise ``FileNotFoundError`` with both attempted routes spelled out.
    """
    env = os.environ.get("SEVENGILL_MAIN_ROOT")
    if env:
        cand = Path(env).expanduser().resolve()
        if (cand / _MARKER).is_file():
            return cand
        raise FileNotFoundError(
            "SEVENGILL_MAIN_ROOT={!r} does not contain {}".format(env, _MARKER)
        )
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / _MARKER).is_file():
            return parent
    raise FileNotFoundError(
        "could not locate the 7Gill main checkout: no ancestor of {} contains {}. "
        "Set SEVENGILL_MAIN_ROOT to the directory that holds spot_detector/, "
        "reid/ and tagger/.".format(here, _MARKER)
    )


def _infer_pipeline():
    """Import ``spot_detector/scripts/infer_pipeline`` (cached by sys.modules).

    The module is import-safe: its top level only builds Path constants.  We
    import rather than copy so the polygon/bbox/spot conversion can never drift
    from what OSEA actually runs.
    """
    root = main_root()
    scripts = str(root / "spot_detector" / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import infer_pipeline  # noqa: E402  (path set above)

    return infer_pipeline


def weight_paths() -> Dict[str, Optional[Path]]:
    """Resolve the four OSEA weight roles, applying the documented fallbacks.

    Returns a dict with keys ``body_obstr``, ``body_only``, ``head``, ``spots``;
    a value is ``None`` when no usable file exists for that role.
    """
    ip = _infer_pipeline()
    body_obstr = ip.DEFAULT_BODY_OBSTR if Path(ip.DEFAULT_BODY_OBSTR).is_file() else None
    body_only = ip.DEFAULT_BODY_ONLY if Path(ip.DEFAULT_BODY_ONLY).is_file() else None
    head = ip.DEFAULT_HEAD if Path(ip.DEFAULT_HEAD).is_file() else None
    spots = ip.DEFAULT_SPOTS if Path(ip.DEFAULT_SPOTS).is_file() else None
    if spots is None and Path(ip.DEFAULT_SPOTS_FALLBACK).is_file():
        spots = ip.DEFAULT_SPOTS_FALLBACK
    return {"body_obstr": body_obstr, "body_only": body_only, "head": head, "spots": spots}


_MODEL_CACHE = {}  # type: Dict[str, Any]


def load_models(device: str = "cpu"):
    """Load and cache the OSEA models.  Returns ``(body_obstr, spots)``.

    ``body_obstr`` is the two-class body+obstruction segmenter (class 0 body,
    class 1 obstruction); if its weights are missing we fall back to the older
    body-only segmenter, and ``detect`` then reports no obstructions -- the same
    degradation ``infer_pipeline.main`` accepts.  ``spots`` is the v2 spot
    detector when present, else v1.  Either may be ``None``; ``detect`` raises
    only when *no* body model exists.

    ``device`` is stashed on each model (``model.to(device)`` via ultralytics'
    predict arg is not used because ``run_image`` does not forward it -- see
    the note in ``detect``).
    """
    key = str(device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    ip = _infer_pipeline()
    paths = weight_paths()
    body = ip.safe_load(paths["body_obstr"], "body+obstruction")
    if body is None:
        body = ip.safe_load(paths["body_only"], "body-only fallback")
    spots = ip.safe_load(paths["spots"], "spots")
    if body is not None:
        try:
            body.to(device)
        except Exception:  # pragma: no cover - device juggling is best effort
            pass
    if spots is not None:
        try:
            spots.to(device)
        except Exception:  # pragma: no cover
            pass
    bundle = (body, spots)
    _MODEL_CACHE[key] = bundle
    return bundle


# --------------------------------------------------------------------------- #
# detect                                                                       #
# --------------------------------------------------------------------------- #
def spot_max_det(spot_model=None) -> int:
    """The detection cap that will apply to ``run_image``'s spot ``predict`` call.

    ``infer_pipeline.run_image`` calls ``spot_model.predict(crop, imgsz=1280,
    conf=spot_conf)`` with **no** ``max_det``, so the effective cap is the
    model's own override if it has one and otherwise ultralytics' default
    (``DEFAULT_CFG.max_det``, 300 on every release this prototype has seen).
    That cap is silent: NMS simply returns its ``max_det`` highest-confidence
    boxes and nothing downstream can tell a truncated frame from a complete one.

    Prototype 06 keeps the cap -- raising it would break contract fidelity with
    what the OSEA tagger stores -- and *reports* it instead; see the
    ``spots_raw_count`` / ``spots_truncated`` keys on :func:`detect`.
    """
    override = getattr(spot_model, "overrides", None) or {}
    try:
        val = override.get("max_det")
    except AttributeError:                      # pragma: no cover - odd model
        val = None
    if val is None:
        try:
            from ultralytics.utils import DEFAULT_CFG
            val = DEFAULT_CFG.max_det
        except Exception:                       # pragma: no cover - no ultralytics
            val = None
    try:
        return int(val)
    except (TypeError, ValueError):             # pragma: no cover
        return int(SPOT_MAX_DET_FALLBACK)


class _CountingPredict(object):
    """Context manager that records how many boxes the spot model returned.

    ``run_image`` filters the spot boxes (centre inside the body, centre not
    inside an obstruction) *after* NMS has already truncated them, so the stored
    ``spot_count`` can be anything from ``max_det`` down to zero on a truncated
    frame and the truncation is invisible in the output.  The only place the
    pre-filter count exists is the ultralytics ``Results`` object inside
    ``run_image``, so we borrow the model's ``predict`` for the duration of the
    call and read ``len(result.boxes)`` off it.

    The wrapper is installed as an *instance* attribute and removed again in
    ``__exit__``, so the model object is byte-identical afterwards and the
    prediction itself is untouched.
    """

    def __init__(self, model):
        self.model = model
        self.count = None                       # type: Optional[int]
        self._prev = None
        self._had_own = False

    def __enter__(self):
        model = self.model
        if model is None:
            return self
        self._had_own = "predict" in vars(model)
        self._prev = vars(model).get("predict")
        orig = model.predict

        def _recording_predict(*args, **kwargs):
            res = orig(*args, **kwargs)
            try:
                boxes = res[0].boxes
                self.count = 0 if boxes is None else int(len(boxes))
            except Exception:                   # pragma: no cover - defensive
                pass
            return res

        model.predict = _recording_predict
        return self

    def __exit__(self, *exc):
        model = self.model
        if model is None:
            return False
        if self._had_own:
            model.predict = self._prev
        else:
            try:
                del model.predict
            except AttributeError:              # pragma: no cover - defensive
                pass
        return False


def detect(
    img_rgb,
    models=None,
    device: str = "cpu",
    body_conf: float = BODY_CONF,
    head_conf: float = HEAD_CONF,
    spot_conf: float = SPOT_CONF,
) -> Dict[str, Any]:
    """Run the OSEA detector on one RGB uint8 image.

    Parameters
    ----------
    img_rgb : (H, W, 3) uint8 ndarray
        EXIF-transposed RGB, exactly as ``pipeline_worker.stage_detect`` builds
        it (``np.array(ImageOps.exif_transpose(Image.open(path)))``).  Passed to
        ultralytics unchanged -- see the channel-order note in the module
        docstring.
    models : (body_model, spot_model) or None
        Pre-loaded pair from :func:`load_models`; loaded on demand if omitted.

    Returns
    -------
    dict with the catalog.db column shapes:
        body_polygon         list of ``[x, y]`` floats (1 dp), or None
        body_bbox            ``{"x","y","w","h"}`` ints, or None
        body_conf            float, or None
        obstruction_polygons list of polygons (possibly empty)
        obstruction_count    int
        head_polygon/head_bbox/head_conf   always None (no head weights)
        spots                list of ``{"x","y","w","h","cx","cy","conf"}``
                             (coords 1 dp, conf 3 dp), original pixels
        spot_count           int
    plus these non-DB conveniences (never written to catalog.db):
        image_width, image_height   ints
        spots_raw_count   boxes the spot model returned *before* run_image's
                          centre-inside-body / not-in-obstruction filter, or
                          None when the spot model never ran (no body, or no
                          spot weights)
        spots_max_det     the ultralytics detection cap in force (300)
        spots_truncated   True when ``spots_raw_count >= spots_max_det``, i.e.
                          NMS hit the cap and an unknown number of real spots
                          was silently discarded.  ``spot_count`` on such an
                          image is a floor, not a count -- 24 catalog images are
                          affected and only 6 of them show ``spot_count == 300``,
                          because the filter runs after the cap.

    Note on ``device``: ``infer_pipeline.run_image`` calls ``model.predict``
    without a ``device=`` argument, so the device is whatever the model object
    already sits on.  :func:`load_models` moves the models once, up front.
    """
    img = np.asarray(img_rgb)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("detect() expects (H, W, 3) RGB, got shape {}".format(img.shape))
    if img.dtype != np.uint8:
        raise ValueError("detect() expects uint8, got {}".format(img.dtype))
    img = np.ascontiguousarray(img)

    if models is None:
        models = load_models(device=device)
    body_model, spot_model = models
    if body_model is None:
        raise RuntimeError(
            "no body model available; expected {}".format(weight_paths()["body_obstr"])
        )
    ip = _infer_pipeline()
    max_det = spot_max_det(spot_model)
    # The body model always goes in the ``body_obstr_model`` slot: for the
    # 1-class body-only fallback the two branches of ``run_image`` issue the
    # identical predict() call, and ``all_polygons_of_class(r, 1)`` simply
    # returns [] because no class-1 instance exists.  Passing it here keeps one
    # code path instead of two.
    with _CountingPredict(spot_model) as counter:
        raw = ip.run_image(
            img,
            body_model,
            None,
            None,  # head weights do not exist on this checkout
            spot_model,
            body_conf,
            head_conf,
            spot_conf,
        )
    raw_spots = counter.count
    obstructions = list(raw.get("obstructions") or [])
    return {
        "body_polygon": raw["body"],
        "body_bbox": raw["body_bbox"],
        "body_conf": raw["body_conf"],
        "obstruction_polygons": obstructions,
        "obstruction_count": len(obstructions),
        "head_polygon": raw["head"],
        "head_bbox": raw["head_bbox"],
        "head_conf": raw["head_conf"],
        "spots": list(raw["spots"]),
        "spot_count": int(raw["spot_count"]),
        "image_width": int(img.shape[1]),
        "image_height": int(img.shape[0]),
        "spots_raw_count": raw_spots,
        "spots_max_det": int(max_det),
        "spots_truncated": bool(raw_spots is not None and raw_spots >= max_det),
    }


def to_db_row(det: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-serialise ``det`` into the ten catalog.db columns.

    Mirrors ``pipeline_worker.stage_detect``'s UPDATE exactly, including its
    "empty means NULL" behaviour.  Provided so a caller can diff this wrapper
    against a real DB row; prototype 06 never writes to catalog.db.
    """
    import json

    return {
        "body_polygon_json": json.dumps(det["body_polygon"]) if det["body_polygon"] else None,
        "body_bbox_json": json.dumps(det["body_bbox"]) if det["body_bbox"] else None,
        "body_conf": det["body_conf"],
        "head_polygon_json": json.dumps(det["head_polygon"]) if det["head_polygon"] else None,
        "head_bbox_json": json.dumps(det["head_bbox"]) if det["head_bbox"] else None,
        "head_conf": det["head_conf"],
        "obstruction_polygon_json": (
            json.dumps(det["obstruction_polygons"]) if det["obstruction_polygons"] else None
        ),
        "obstruction_count": det["obstruction_count"] or 0,
        "spots_json": json.dumps(det["spots"]) if det["spots"] else None,
        "spot_count": det["spot_count"],
    }


# --------------------------------------------------------------------------- #
# geometry helpers                                                             #
# --------------------------------------------------------------------------- #
def polygon_area(poly) -> float:
    """Shoelace area (always positive) of a closed polygon given as [[x, y], ...]."""
    p = np.asarray(poly, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _project(rel: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Scalar projection of (N, 2) offsets onto a unit 2-vector, without BLAS."""
    return rel[:, 0] * axis[0] + rel[:, 1] * axis[1]


RASTER_LONG_SIDE = 1024
DEGENERATE_AREA_TOL = 0.05


def raster_moments(poly, long_side: int = RASTER_LONG_SIDE):
    """Area centroid and central second moments of the *filled* polygon.

    Computed by rasterising, not by Green's theorem, because the YOLO body
    contour is sometimes **self-intersecting**: ``mask.xy`` can hand back a
    single contour that snakes across the frame and doubles back (real example:
    catalog image 799, a 1744-vertex contour whose shoelace area nearly cancels
    to zero, which sent the analytic ``density`` to 23159 -- two orders of
    magnitude above the corpus median).  A raster fill has no such failure mode:
    every pixel is either inside or outside.

    The polygon is fitted to a canvas whose long side is ``long_side`` px, so
    the centroid is quantised to ~0.1% of the body extent; the axis *extents*
    are still measured on the original float vertices, in original pixels.

    Known bias: ``cv2.fillPoly`` paints the boundary pixels, so the area comes
    out roughly half a perimeter-pixel high -- about +0.3% for a body-shaped
    outline at 1024 px.  It is a systematic bias of the *shared* extractor, so
    real photos and synthetic renders carry the identical offset and it cancels
    in every real-vs-synthetic comparison this prototype makes.

    Returns ``(area_px2, centroid_xy, cov_2x2)`` in ORIGINAL pixel units, or
    ``None`` when the fill is empty.
    """
    p = np.asarray(poly, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 3:
        return None
    x0, y0 = p.min(axis=0)
    x1, y1 = p.max(axis=0)
    w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    scale = float(long_side) / max(w, h)
    W = max(int(np.ceil(w * scale)) + 2, 3)
    H = max(int(np.ceil(h * scale)) + 2, 3)
    if W * H > 16_000_000:                      # pathological aspect guard
        return None
    q = np.round((p - np.array([x0, y0])) * scale).astype(np.int32).reshape(-1, 1, 2)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [q], 1)
    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] <= 0:
        return None
    inv = 1.0 / scale
    area = float(m["m00"]) * inv * inv
    cx = m["m10"] / m["m00"] * inv + x0
    cy = m["m01"] / m["m00"] * inv + y0
    cov = np.array(
        [[m["mu20"] / m["m00"], m["mu11"] / m["m00"]],
         [m["mu11"] / m["m00"], m["mu02"] / m["m00"]]],
        dtype=np.float64,
    ) * (inv * inv)
    return area, np.array([cx, cy], dtype=np.float64), cov


def _skew(values: np.ndarray) -> float:
    """Unnormalised third central moment; 0.0 for fewer than 3 samples."""
    v = np.asarray(values, dtype=np.float64)
    if v.size < 3:
        return 0.0
    d = v - v.mean()
    return float((d ** 3).mean())


def pca_frame(poly, spots_uv: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Body-attached frame from the second moments of the *filled* polygon.

    Moments come from :func:`raster_moments`, i.e. the *filled region*, so the
    frame is immune both to uneven vertex sampling and to the self-intersecting
    contours the body segmenter occasionally emits.  Degenerate polygons (< 3
    vertices, empty fill, zero extent) return ``None``.

    ``spots_uv`` -- optional (N, 2) array of spot centres in *pixels*; used only
    for the sign tie-break described in the module docstring.

    Returns dict: origin (2,), e_major (2,), e_minor (2,), L_major, D_minor,
    area, aspect, theta_deg, sign_rule, degenerate_contour, area_shoelace.

    ``degenerate_contour`` is True when the shoelace area disagrees with the
    rasterised area by more than ``DEGENERATE_AREA_TOL``, which is a reliable
    signature of a self-intersecting contour.  The frame is still returned (it
    is well defined); the flag lets callers drop or down-weight the image.
    """
    p = np.asarray(poly, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 3:
        return None
    rm = raster_moments(p)
    if rm is None:
        return None
    area, origin, cov = rm
    if area <= 0.0:
        return None
    area_shoelace = polygon_area(p)
    degenerate = abs(area_shoelace - area) > DEGENERATE_AREA_TOL * area

    evals, evecs = np.linalg.eigh(cov)          # ascending
    e_major = evecs[:, 1] / np.linalg.norm(evecs[:, 1])
    e_minor = evecs[:, 0] / np.linalg.norm(evecs[:, 0])

    rel = p - origin
    # np.errstate: running ultralytics/torch first can leave FPU exception flags
    # set, and numpy then reports spurious divide/overflow warnings on the very
    # next BLAS call.  We suppress them and check finiteness explicitly instead.
    with np.errstate(all="ignore"):
        proj_major = _project(rel, e_major)
        proj_minor = _project(rel, e_minor)
    if not (np.isfinite(proj_major).all() and np.isfinite(proj_minor).all()):
        return None
    L_major = float(proj_major.max() - proj_major.min())
    D_minor = float(proj_minor.max() - proj_minor.min())
    if not np.isfinite(L_major) or not np.isfinite(D_minor):
        return None
    if D_minor <= 0 or L_major <= 0:
        return None

    # ---- sign tie-break (see module docstring; NOT biologically meaningful) --
    rule = "polygon_skew"
    su = _skew(proj_major)
    sv = _skew(proj_minor)
    if spots_uv is not None and len(spots_uv) >= 3:
        srel = np.asarray(spots_uv, dtype=np.float64) - origin
        with np.errstate(all="ignore"):
            s_su = _skew(_project(srel, e_major))
            s_sv = _skew(_project(srel, e_minor))
        if s_su != 0.0 or s_sv != 0.0:
            su, sv = s_su, s_sv
            rule = "spot_skew"
    if su == 0.0 and sv == 0.0:
        rule = "fixed"
    if su < 0:
        e_major = -e_major
    if sv < 0:
        e_minor = -e_minor
    # Keep a right-handed pair only if it does not fight the tie-break: we do
    # NOT force handedness, because forcing it would couple the two signs and
    # break the "identical under u->-u and v->-v independently" guarantee.

    theta = float(np.degrees(np.arctan2(e_major[1], e_major[0])))
    return {
        "origin": origin,
        "e_major": e_major,
        "e_minor": e_minor,
        "L_major": L_major,
        "D_minor": D_minor,
        "area": area,
        "aspect": float(L_major / D_minor),
        "theta_deg": theta,
        "sign_rule": rule,
        "eig": (float(evals[1]), float(evals[0])),
        "degenerate_contour": bool(degenerate),
        "area_shoelace": float(area_shoelace),
    }


def _quantiles(values, qs=(0.05, 0.25, 0.50, 0.75, 0.95)) -> Dict[str, Optional[float]]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(("q{:02d}".format(int(round(q * 100))), None) for q in qs)
    out = {}
    for q in qs:
        out["q{:02d}".format(int(round(q * 100)))] = float(np.quantile(v, q))
    out["mean"] = float(v.mean())
    return out


def _nn_distances(pts: np.ndarray) -> np.ndarray:
    """Nearest-neighbour distance for each point (empty if fewer than 2)."""
    if pts.shape[0] < 2:
        return np.zeros(0, dtype=np.float64)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def _filled_area(poly) -> float:
    """Area of the *filled* polygon, by raster fill, falling back to shoelace.

    Same measurement convention as :func:`raster_moments` (which is where the
    fill happens), so areas produced here are directly comparable to
    ``feats["frame"]["area_px2"]`` and carry the identical boundary-pixel bias.
    The shoelace fallback only fires when the raster fill is empty or the
    polygon is degenerate enough that ``raster_moments`` refuses it.
    """
    rm = raster_moments(poly)
    if rm is None:
        return polygon_area(poly)
    return float(rm[0])


def _obstruction_overlap(body_poly, obstr_polys, max_side: int = 512) -> Tuple[float, float]:
    """(fraction of the body polygon covered by obstructions, raw area ratio).

    The covered fraction is measured by rasterising both onto a canvas whose
    long side is ``max_side`` px -- obstruction polygons routinely extend past
    the animal, so summing their areas alone overstates the occlusion.  The raw
    ratio (sum of obstruction areas / body area, uncapped) is returned too.

    **Both** areas are measured by raster fill (:func:`_filled_area`), never by
    the shoelace formula: the body contour self-intersects on 117 of the 1030
    catalog bodies, and on those the shoelace area nearly cancels, which is the
    whole reason :func:`raster_moments` exists.  Dividing by it inflated this
    ratio by the same factor -- up to 70x on catalog image 799, which reported
    ``obstruction_area_ratio`` 74.9 for an obstruction covering 26% of the body.
    """
    if not obstr_polys:
        return 0.0, 0.0
    body = np.asarray(body_poly, dtype=np.float64)
    body_area = _filled_area(body)
    if body_area <= 0:
        return 0.0, 0.0
    raw = sum(_filled_area(o) for o in obstr_polys) / body_area
    x0, y0 = body.min(axis=0)
    x1, y1 = body.max(axis=0)
    w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    scale = float(max_side) / max(w, h)
    W = max(int(round(w * scale)) + 2, 3)
    H = max(int(round(h * scale)) + 2, 3)

    def _to_canvas(poly):
        q = (np.asarray(poly, dtype=np.float64) - np.array([x0, y0])) * scale
        return np.round(q).astype(np.int32).reshape(-1, 1, 2)

    body_mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(body_mask, [_to_canvas(body)], 1)
    ob_mask = np.zeros((H, W), np.uint8)
    for o in obstr_polys:
        arr = np.asarray(o, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 3:
            continue
        cv2.fillPoly(ob_mask, [_to_canvas(arr)], 1)
    body_px = int(body_mask.sum())
    if body_px == 0:
        return 0.0, float(raw)
    inter = int((body_mask & ob_mask).sum())
    return float(inter) / float(body_px), float(raw)


# --------------------------------------------------------------------------- #
# features                                                                     #
# --------------------------------------------------------------------------- #
def features(det: Dict[str, Any], image_size: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Non-RGB per-image features from one :func:`detect` result.

    ``image_size`` -- optional ``(width, height)``; defaults to the
    ``image_width``/``image_height`` keys that :func:`detect` adds.  Only used
    for ``bbox_width_frac``.

    Returns
    -------
    dict with

    ``ok``            False when there is no usable body polygon; every other
                      geometric field is then None and ``spots_uv`` is empty.
    ``frame``         origin/axes/extents from :func:`pca_frame`, JSON-safe.
    ``spots_uv``      (N, 4) float list: ``[u, v, size, conf]`` per spot, with
                      u along e_major and v across, both divided by D_minor,
                      and size = sqrt(w*h)/D_minor.
    ``spots_raw``     the untouched detector spot dicts (original pixels).
    ``scalars``       the flip-invariant scalar block (see below).

    Scalars
    -------
    n_spots, density (= n_spots / (area / D_minor**2)), size_q* / size_mean,
    nn_median (median nearest-neighbour distance / D_minor) and nn_q*,
    conf_q* / conf_mean, bbox_width_frac, body_conf, obstruction_count,
    obstruction_area_frac (rasterised body-covered fraction),
    obstruction_area_ratio (raw), degenerate_contour (1 when the body contour
    self-intersects -- see :func:`raster_moments`), plus the frame descriptors
    L_major, D_minor, area_px2, aspect and area_norm (= area / D_minor**2).

    Sign invariance: every scalar above is built from counts, areas, extents,
    pairwise distances or per-spot magnitudes, all of which are unchanged when
    e_major -> -e_major and/or e_minor -> -e_minor.  ``spots_uv`` is *not*
    invariant; its sign is pinned by the documented skewness tie-break and
    downstream code must treat the four sign combinations as equivalent.
    """
    body_poly = det.get("body_polygon")
    spots = list(det.get("spots") or [])
    if image_size is None:
        iw = det.get("image_width")
        ih = det.get("image_height")
    else:
        iw, ih = int(image_size[0]), int(image_size[1])

    bbox = det.get("body_bbox")
    bbox_width_frac = None
    if bbox is not None and iw:
        bbox_width_frac = float(bbox["w"]) / float(iw)

    base = {
        "ok": False,
        "frame": None,
        "spots_uv": [],
        "spots_raw": spots,
        "scalars": {
            "n_spots": len(spots),
            "density": None,
            "area_px2": None,
            "area_norm": None,
            "L_major": None,
            "D_minor": None,
            "aspect": None,
            "size": _quantiles([]),
            "nn": _quantiles([]),
            "nn_median": None,
            "conf": _quantiles([s["conf"] for s in spots]) if spots else _quantiles([]),
            "bbox_width_frac": bbox_width_frac,
            "body_conf": det.get("body_conf"),
            "obstruction_count": int(det.get("obstruction_count") or 0),
            "obstruction_area_frac": None,
            "obstruction_area_ratio": None,
            "degenerate_contour": None,
        },
    }
    if not body_poly:
        return base

    centres_px = (
        np.array([[float(s["cx"]), float(s["cy"])] for s in spots], dtype=np.float64)
        if spots
        else np.zeros((0, 2), dtype=np.float64)
    )
    frame = pca_frame(body_poly, centres_px if len(centres_px) else None)
    if frame is None:
        return base

    D = frame["D_minor"]
    origin = frame["origin"]
    e_major = frame["e_major"]
    e_minor = frame["e_minor"]

    if len(centres_px):
        rel = centres_px - origin
        with np.errstate(all="ignore"):
            u = _project(rel, e_major) / D
            v = _project(rel, e_minor) / D
        sizes = np.array(
            [np.sqrt(max(float(s["w"]), 0.0) * max(float(s["h"]), 0.0)) / D for s in spots],
            dtype=np.float64,
        )
        confs = np.array([float(s["conf"]) for s in spots], dtype=np.float64)
        uv = np.column_stack([u, v])
        nn = _nn_distances(uv)
    else:
        u = v = sizes = confs = np.zeros(0, dtype=np.float64)
        uv = np.zeros((0, 2), dtype=np.float64)
        nn = np.zeros(0, dtype=np.float64)

    area_norm = frame["area"] / (D * D)
    ob_frac, ob_ratio = _obstruction_overlap(body_poly, det.get("obstruction_polygons") or [])

    scalars = {
        "n_spots": int(len(spots)),
        "density": float(len(spots) / area_norm) if area_norm > 0 else None,
        "area_px2": float(frame["area"]),
        "area_norm": float(area_norm),
        "L_major": float(frame["L_major"]),
        "D_minor": float(D),
        "aspect": float(frame["aspect"]),
        "size": _quantiles(sizes),
        "nn": _quantiles(nn),
        "nn_median": float(np.median(nn)) if nn.size else None,
        "conf": _quantiles(confs),
        "bbox_width_frac": bbox_width_frac,
        "body_conf": det.get("body_conf"),
        "obstruction_count": int(det.get("obstruction_count") or 0),
        "obstruction_area_frac": float(ob_frac),
        "obstruction_area_ratio": float(ob_ratio),
        "degenerate_contour": int(bool(frame["degenerate_contour"])),
    }
    return {
        "ok": True,
        "frame": {
            "origin": [float(origin[0]), float(origin[1])],
            "e_major": [float(e_major[0]), float(e_major[1])],
            "e_minor": [float(e_minor[0]), float(e_minor[1])],
            "L_major": float(frame["L_major"]),
            "D_minor": float(D),
            "area_px2": float(frame["area"]),
            "aspect": float(frame["aspect"]),
            "theta_deg": float(frame["theta_deg"]),
            "sign_rule": frame["sign_rule"],
            "degenerate_contour": bool(frame["degenerate_contour"]),
            "area_shoelace_px2": float(frame["area_shoelace"]),
        },
        "spots_uv": [
            [float(u[i]), float(v[i]), float(sizes[i]), float(confs[i])]
            for i in range(len(spots))
        ],
        "spots_raw": spots,
        "scalars": scalars,
    }


#: Flat names of every scalar ``features`` produces, for tabulation.
SCALAR_KEYS = (
    "n_spots",
    "density",
    "area_px2",
    "area_norm",
    "L_major",
    "D_minor",
    "aspect",
    "nn_median",
    "bbox_width_frac",
    "body_conf",
    "obstruction_count",
    "obstruction_area_frac",
    "obstruction_area_ratio",
    "degenerate_contour",
)


def flat_scalars(feats: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Flatten ``feats['scalars']`` (including the quantile sub-dicts) to one level."""
    s = feats["scalars"]
    out = {}  # type: Dict[str, Optional[float]]
    for k in SCALAR_KEYS:
        out[k] = s.get(k)
    for group in ("size", "nn", "conf"):
        for q, val in (s.get(group) or {}).items():
            out["{}_{}".format(group, q)] = val
    return out
