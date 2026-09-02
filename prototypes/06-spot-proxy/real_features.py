"""Run the OSEA detection contract over every real photo in catalog.db and
record the non-RGB feature distribution.

This is the "real" half of prototype 06: the target distribution that the
synthetic renders must be pushed onto.  Nothing here looks at colour beyond
handing the raw RGB array to the detector -- everything downstream of
``osea_contract.detect`` is polygons, boxes and scalars.

The database is opened **read-only** (``file:...?mode=ro``); this script never
writes to catalog.db.

Outputs, under ``--out`` (default ``results/real``):

  detections.jsonl   one JSON object per *processed* image::

                       {image_id, filename, rel_path, individual_code|null,
                        encounter_id, exif_ts, width, height, det, feats}

                     ``width``/``height`` are the EXIF-transposed pixel sizes
                     actually fed to the detector, which can differ from the
                     catalog's stored width/height (those are pre-transpose).
                     32.4 MB over the full catalog (11.2 MB of body polygons,
                     10.1 MB of spot boxes, 10.2 MB of feats), so it is
                     **gitignored**.

  detections_slim.jsonl  the same records with the two bulky fields shrunk:
                     ``det.body_polygon`` dropped entirely and each spot
                     written as ``[cx, cy, w, h, conf]`` instead of a
                     seven-key object.  ``det.body_bbox`` and the whole
                     ``feats`` block are kept, so every scalar in
                     ``summary.json`` can be recomputed from it.  This is the
                     variant that IS tracked in git: 15.3 MB against 32.4,
                     measured on the full catalog.  Anything that needs the
                     body outline -- the constellation matcher, any
                     re-rectification -- needs the full file, so re-run this
                     script.

  skipped.jsonl      one JSON object per skipped image, with a ``reason``.

  summary.json       counts, per-scalar quantiles, pooled per-spot histograms
                     (size / nearest-neighbour distance / confidence) with
                     explicit bin edges, and the run's throughput.

                     Two things to know before quoting its numbers.  (1) The
                     per-image scalar quantiles are conditioned on a usable body
                     polygon: an image with no body has no spot field to
                     measure, so its structural ``n_spots = 0`` is recorded as
                     missing rather than as a zero (``n`` and ``n_missing`` per
                     scalar say how many).  (2) ``counts.spots_truncated`` lists
                     the images where ultralytics' ``max_det`` cap of 300 cut
                     the spot detections off; their ``n_spots`` and ``density``
                     are lower bounds and they should be dropped before the real
                     distribution is used as a synthetic target.

  features_contact.png   histogram contact sheet of the same numbers.

  detections_contact.png (optional, ``--overlay-n``) montage of zoomed
                     detection overlays -- body polygon (green), obstruction
                     polygons (orange) and spot boxes (magenta, labelled with
                     confidence).  The full-resolution singles land in
                     ``overlays/`` (>= 900 px on the short side, for eyeballing)
                     and are gitignored; only the montage is kept.

Example::

    "MAIN/.venv/bin/python" real_features.py --limit 20 --overlay-n 6
    "MAIN/.venv/bin/python" real_features.py            # full catalog, ~5 min
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))

import osea_contract as oc  # noqa: E402

QUANTILES = (0.0, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0)

#: pooled per-spot histograms: (name, bin edges)
POOLED_BINS = {
    "spot_size": np.linspace(0.0, 0.20, 41),        # sqrt(w*h) / D_minor
    "spot_nn": np.linspace(0.0, 0.40, 41),          # NN distance / D_minor
    "spot_conf": np.linspace(0.25, 1.0, 31),        # detector floor is 0.25
    "spot_u": np.linspace(-1.5, 1.5, 61),           # body-frame position
    "spot_v": np.linspace(-1.0, 1.0, 41),
}


# --------------------------------------------------------------------------- #
# db                                                                           #
# --------------------------------------------------------------------------- #
def open_ro(db_path: Path) -> sqlite3.Connection:
    """Open catalog.db strictly read-only."""
    conn = sqlite3.connect("file:{}?mode=ro".format(db_path), uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_rows(conn: sqlite3.Connection, limit: Optional[int]) -> List[sqlite3.Row]:
    """Every non-video image, with its individual code resolved through the FK."""
    q = (
        "SELECT i.id, i.filename, i.rel_path, i.width, i.height, i.encounter_id, "
        "       i.exif_ts, i.side, ind.individual_id AS individual_code "
        "  FROM images i LEFT JOIN individuals ind ON i.individual_id = ind.id "
        " WHERE i.file_type != 'mp4' "
        " ORDER BY i.id"
    )
    if limit:
        q += " LIMIT {:d}".format(int(limit))
    return conn.execute(q).fetchall()


def resolve_path(rel_path: str, tagger_root: Path) -> Path:
    """catalog.db rel_path is normally relative to MAIN/tagger; absolute is honoured."""
    p = Path(rel_path)
    return p if p.is_absolute() else tagger_root / rel_path


# --------------------------------------------------------------------------- #
# accumulation                                                                 #
# --------------------------------------------------------------------------- #
#: The scalars that ``features()`` still reports as a number when there is no
#: body polygon, i.e. the ones :meth:`Accumulator.add`'s body conditioning
#: actually changes.  ``n_spots`` is 0 because ``run_image`` never runs the spot
#: model without a body, and ``obstruction_count`` is whatever the body model
#: emitted on class 1 with no class 0; both are artefacts of the failure, not
#: measurements.  Every other scalar is already None on that path.  Recording
#: these two as numbers put a 61-image spike at zero into a distribution that is
#: meant to describe spot fields: it moved n_spots q05 from 39 to 0, q25 from 69
#: to 64 and q50 from 112 to 107.
BODY_CONDITIONED_COUNTS = ("n_spots", "obstruction_count")


class Accumulator(object):
    """Streaming collector: per-image scalars plus pooled per-spot arrays.

    Every per-image scalar is conditioned on a usable body polygon: on the
    ``feats["ok"] is False`` path all of them go in as NaN and drop out of the
    quantile table, which reports ``n`` and ``n_missing`` so the exclusion is
    visible.  ``n_images`` / ``n_body`` / ``n_no_body`` still count every image.
    """

    def __init__(self):
        self.scalars = {}       # type: Dict[str, List[float]]
        self.pooled = {k: [] for k in POOLED_BINS}
        self.n_images = 0
        self.n_body = 0
        self.n_no_body = 0
        self.n_degenerate = 0
        self.degenerate_ids = []    # type: List[int]
        self.n_truncated = 0
        self.truncated_ids = []     # type: List[int]

    def add(self, feats: Dict[str, Any]) -> None:
        self.n_images += 1
        flat = oc.flat_scalars(feats)
        ok = bool(feats["ok"])
        for k, v in flat.items():
            if not ok:
                v = None            # no body -> nothing was measured on this image
            self.scalars.setdefault(k, []).append(np.nan if v is None else float(v))
        if feats["ok"]:
            self.n_body += 1
            if feats["frame"]["degenerate_contour"]:
                self.n_degenerate += 1
            uv = np.asarray(feats["spots_uv"], dtype=np.float64).reshape(-1, 4)
            if uv.shape[0]:
                self.pooled["spot_u"].append(uv[:, 0])
                self.pooled["spot_v"].append(uv[:, 1])
                self.pooled["spot_size"].append(uv[:, 2])
                self.pooled["spot_conf"].append(uv[:, 3])
                if uv.shape[0] >= 2:
                    d = np.linalg.norm(uv[:, None, :2] - uv[None, :, :2], axis=2)
                    np.fill_diagonal(d, np.inf)
                    self.pooled["spot_nn"].append(d.min(axis=1))
        else:
            self.n_no_body += 1

    def pooled_array(self, key: str) -> np.ndarray:
        chunks = self.pooled[key]
        return np.concatenate(chunks) if chunks else np.zeros(0)

    def quantile_table(self) -> Dict[str, Dict[str, Any]]:
        out = {}
        for k in sorted(self.scalars):
            v = np.asarray(self.scalars[k], dtype=np.float64)
            finite = v[np.isfinite(v)]
            entry = {"n": int(finite.size), "n_missing": int(v.size - finite.size)}
            if finite.size:
                for q in QUANTILES:
                    entry["q{:03d}".format(int(round(q * 100)))] = float(np.quantile(finite, q))
                entry["mean"] = float(finite.mean())
                entry["std"] = float(finite.std())
            out[k] = entry
        return out

    def histograms(self) -> Dict[str, Dict[str, Any]]:
        out = {}
        for key, edges in POOLED_BINS.items():
            arr = self.pooled_array(key)
            counts, _ = np.histogram(arr, bins=edges)
            out[key] = {
                "n": int(arr.size),
                "bin_edges": [float(e) for e in edges],
                "counts": [int(c) for c in counts],
                "n_below": int((arr < edges[0]).sum()),
                "n_above": int((arr > edges[-1]).sum()),
                "quantiles": (
                    {"q{:03d}".format(int(round(q * 100))): float(np.quantile(arr, q))
                     for q in QUANTILES}
                    if arr.size else {}
                ),
            }
        return out


# --------------------------------------------------------------------------- #
# the tracked slim record                                                      #
# --------------------------------------------------------------------------- #
#: Order of the five numbers a slim spot is written as.
SLIM_SPOT_KEYS = ("cx", "cy", "w", "h", "conf")


def slim_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Shrink one detections.jsonl record for the version-controlled variant.

    Two changes, both to ``det``:

    * ``body_polygon`` is dropped.  It is 11.2 MB of the 32.4 (hundreds to
      thousands of ``[x, y]`` pairs per image) and nothing that reads the slim
      file needs it; ``body_bbox`` stays, so the framing is still there.
    * every spot becomes ``[cx, cy, w, h, conf]`` (:data:`SLIM_SPOT_KEYS`)
      instead of a seven-key object.  ``x``/``y`` are dropped because they are
      exactly ``cx - w/2`` / ``cy - h/2``.

    ``feats`` is untouched, so every number in ``summary.json`` is
    recomputable from the slim file.  The full file is what the constellation
    matcher needs (it rectifies against the body outline) and is regenerated by
    re-running this script.
    """
    det = dict(rec["det"])
    det["body_polygon"] = None
    det["body_polygon_dropped"] = True
    det["spots"] = [[s["cx"], s["cy"], s["w"], s["h"], s["conf"]]
                    for s in (rec["det"].get("spots") or [])]
    det["spots_format"] = list(SLIM_SPOT_KEYS)
    out = dict(rec)
    out["det"] = det
    return out


# --------------------------------------------------------------------------- #
# figures                                                                      #
# --------------------------------------------------------------------------- #
def contact_sheet(acc: Accumulator, path: Path) -> None:
    """Histogram contact sheet of the real-photo feature distribution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("n_spots", "spots per image", np.linspace(0, 320, 41)),
        ("density", "spots / (area / D^2)", np.linspace(0, 400, 41)),
        ("nn_median", "median NN dist / D", np.linspace(0, 0.15, 41)),
        ("size_q50", "median spot size / D", np.linspace(0, 0.10, 41)),
        ("aspect", "body L/D", np.linspace(0.4, 6.0, 41)),
        ("area_norm", "body area / D^2", np.linspace(0, 4, 41)),
        ("bbox_width_frac", "bbox w / image w", np.linspace(0, 1.05, 43)),
        ("body_conf", "body confidence", np.linspace(0.4, 1.0, 41)),
        ("obstruction_count", "obstructions per image", np.arange(-0.5, 9.5, 1.0)),
        ("obstruction_area_frac", "body area occluded", np.linspace(0, 0.5, 41)),
        ("D_minor", "D_minor (px)", np.linspace(0, 5000, 41)),
        ("conf_q50", "median spot conf", np.linspace(0.25, 0.8, 41)),
    ]
    pooled = [
        ("spot_size", "per-spot size / D"),
        ("spot_nn", "per-spot NN dist / D"),
        ("spot_conf", "per-spot confidence"),
        ("spot_u", "per-spot u (along body)"),
        ("spot_v", "per-spot v (across body)"),
    ]
    ncol = 4
    nrow = int(np.ceil((len(panels) + len(pooled)) / float(ncol)))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.7 * nrow))
    axes = np.atleast_1d(axes).ravel()

    def _panel(ax, values, bins, title, color):
        """Histogram with the median marked; out-of-range values are counted,
        not silently dropped, so a clipped axis can never read as an empty tail."""
        v = np.asarray(values, dtype=np.float64)
        v = v[np.isfinite(v)]
        n_out = int(((v < bins[0]) | (v > bins[-1])).sum())
        ax.hist(np.clip(v, bins[0], bins[-1]), bins=bins, color=color, edgecolor="none")
        cap = "{}  (n={}{})".format(title, v.size,
                                    ", {} clipped".format(n_out) if n_out else "")
        ax.set_title(cap, fontsize=9)
        ax.tick_params(labelsize=7)
        if v.size:
            ax.axvline(float(np.median(v)), color="#c0392b", lw=1.2)

    for ax, (key, title, bins) in zip(axes, panels):
        _panel(ax, acc.scalars.get(key, []), bins, title, "#3b6ea5")

    for ax, (key, title) in zip(axes[len(panels):], pooled):
        _panel(ax, acc.pooled_array(key), POOLED_BINS[key], title, "#4a8c5c")

    for ax in axes[len(panels) + len(pooled):]:
        ax.axis("off")
    fig.suptitle(
        "Prototype 06 -- real OSEA photos: non-RGB detection features "
        "({} images, {} with a body polygon, {} self-intersecting contours)".format(
            acc.n_images, acc.n_body, acc.n_degenerate),
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(str(path), dpi=110)
    plt.close(fig)


def draw_overlay(img: np.ndarray, det: Dict[str, Any], label: str = "",
                 min_side: int = 900) -> Optional[np.ndarray]:
    """Zoomed detection overlay for visual verification; returns a BGR canvas.

    Crops to the padded body bbox, upscales so the shorter side is at least
    ``min_side`` px (the "look at it before saying it looks right" rule), and
    draws the body polygon (green), obstruction polygons (orange) and spot
    boxes (magenta) with per-spot confidence.
    """
    import cv2

    bbox = det.get("body_bbox")
    H, W = img.shape[:2]
    if bbox:
        x0 = max(0, bbox["x"]); y0 = max(0, bbox["y"])
        x1 = min(W, bbox["x"] + bbox["w"]); y1 = min(H, bbox["y"] + bbox["h"])
    else:
        x0, y0, x1, y1 = 0, 0, W, H
    crop = np.ascontiguousarray(img[y0:y1, x0:x1])
    if crop.size == 0:
        return None
    scale = max(1.0, float(min_side) / max(1, min(crop.shape[0], crop.shape[1])))
    if scale > 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    canvas = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

    def to_c(pts):
        q = (np.asarray(pts, dtype=np.float64) - np.array([x0, y0])) * scale
        return np.round(q).astype(np.int32).reshape(-1, 1, 2)

    if det.get("body_polygon"):
        cv2.polylines(canvas, [to_c(det["body_polygon"])], True, (60, 220, 60), 3)
    for ob in det.get("obstruction_polygons") or []:
        if len(ob) >= 3:
            cv2.polylines(canvas, [to_c(ob)], True, (30, 160, 255), 3)
    for s in det.get("spots") or []:
        p0 = to_c([[s["x"], s["y"]]])[0, 0]
        p1 = to_c([[s["x"] + s["w"], s["y"] + s["h"]]])[0, 0]
        cv2.rectangle(canvas, tuple(p0), tuple(p1), (230, 60, 230), 2)
        cv2.putText(canvas, "{:.2f}".format(s["conf"]), (int(p0[0]), int(p0[1]) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 60, 230), 1, cv2.LINE_AA)
    cap = "{}  spots={}  body_conf={:.2f}  obstr={}".format(
        label, det["spot_count"], det["body_conf"] or 0.0, det["obstruction_count"])
    cv2.putText(canvas, cap, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(canvas, cap, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def montage(tiles: List[np.ndarray], path: Path, ncol: int = 3,
            tile_w: int = 470) -> None:
    """Grid the overlay canvases into one kept contact sheet."""
    import cv2

    if not tiles:
        return
    scaled = []
    for t in tiles:
        h = int(round(t.shape[0] * tile_w / float(t.shape[1])))
        scaled.append(cv2.resize(t, (tile_w, h), interpolation=cv2.INTER_AREA))
    nrow = int(np.ceil(len(scaled) / float(ncol)))
    rows = []
    for r in range(nrow):
        chunk = scaled[r * ncol:(r + 1) * ncol]
        hmax = max(t.shape[0] for t in chunk)
        padded = [
            cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0,
                               cv2.BORDER_CONSTANT, value=(255, 255, 255))
            for t in chunk
        ]
        while len(padded) < ncol:
            padded.append(np.full((hmax, tile_w, 3), 255, np.uint8))
        rows.append(np.hstack(padded))
    sheet = np.vstack(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_PNG_COMPRESSION, 6])


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    root = oc.main_root()
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", type=Path, default=root / "tagger" / "data" / "catalog.db")
    ap.add_argument("--out", type=Path, default=here / "results" / "real")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-width", type=int, default=800,
                    help="skip images whose longest catalog side is below this")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--overlay-n", type=int, default=0,
                    help="write this many zoomed detection overlays for eyeballing")
    ap.add_argument("--overlay-every", type=int, default=0,
                    help="if set, take overlays every Nth processed image instead "
                         "of the first --overlay-n")
    args = ap.parse_args(argv)

    if not args.db.is_file():
        print("ERROR: no catalog at {}".format(args.db), file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    tagger_root = root / "tagger"

    print("main root : {}".format(root))
    print("catalog   : {} (read-only)".format(args.db))
    print("out       : {}".format(args.out))
    wp = oc.weight_paths()
    for role, p in sorted(wp.items()):
        print("weights   : {:11s} {}".format(role, p if p else "MISSING"))
    models = oc.load_models(args.device)
    if models[0] is None:
        print("ERROR: no body model available", file=sys.stderr)
        return 1
    if models[1] is None:
        print("WARNING: no spot model; every image will report spot_count=0")

    conn = open_ro(args.db)
    rows = fetch_rows(conn, args.limit)
    print("\n{} candidate rows (file_type != 'mp4')".format(len(rows)))

    acc = Accumulator()
    skipped = []            # type: List[Dict[str, Any]]
    det_path = args.out / "detections.jsonl"
    slim_path = args.out / "detections_slim.jsonl"
    skip_path = args.out / "skipped.jsonl"
    n_overlay = 0
    overlay_tiles = []      # type: List[np.ndarray]
    t0 = time.time()
    detect_seconds = 0.0

    with open(str(det_path), "w") as fh, open(str(slim_path), "w") as slim_fh:
        for idx, row in enumerate(rows):
            rec_id = int(row["id"])
            longest = max(int(row["width"] or 0), int(row["height"] or 0))
            if longest < args.min_width:
                skipped.append({"image_id": rec_id, "filename": row["filename"],
                                "rel_path": row["rel_path"], "reason": "below_min_width",
                                "detail": "longest catalog side {} < {}".format(
                                    longest, args.min_width)})
                continue
            path = resolve_path(row["rel_path"], tagger_root)
            if not path.is_file():
                skipped.append({"image_id": rec_id, "filename": row["filename"],
                                "rel_path": row["rel_path"], "reason": "file_missing",
                                "detail": str(path)})
                continue
            try:
                img = np.array(ImageOps.exif_transpose(Image.open(str(path))).convert("RGB"))
            except Exception as e:
                skipped.append({"image_id": rec_id, "filename": row["filename"],
                                "rel_path": row["rel_path"], "reason": "decode_error",
                                "detail": "{}: {}".format(type(e).__name__, e)})
                continue
            if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
                skipped.append({"image_id": rec_id, "filename": row["filename"],
                                "rel_path": row["rel_path"], "reason": "bad_array",
                                "detail": "{} {}".format(img.shape, img.dtype)})
                continue
            try:
                t = time.time()
                det = oc.detect(img, models)
                detect_seconds += time.time() - t
                feats = oc.features(det)
            except Exception as e:
                skipped.append({"image_id": rec_id, "filename": row["filename"],
                                "rel_path": row["rel_path"], "reason": "detect_error",
                                "detail": "{}: {}".format(type(e).__name__, e)})
                continue

            acc.add(feats)
            if feats["ok"] and feats["frame"]["degenerate_contour"]:
                acc.degenerate_ids.append(rec_id)
            # NMS hit ultralytics' max_det cap: n_spots and density are lower
            # bounds on this image and it must not be used as a distribution
            # target.  See osea_contract.spot_max_det.
            if det.get("spots_truncated"):
                acc.n_truncated += 1
                acc.truncated_ids.append(rec_id)
            # ``feats["spots_raw"]`` is byte-identical to ``det["spots"]``; keep
            # one copy on disk (it is ~10 MB of the record over the corpus).
            feats_out = {k: v for k, v in feats.items() if k != "spots_raw"}
            record = {
                "image_id": rec_id,
                "filename": row["filename"],
                "rel_path": row["rel_path"],
                "individual_code": row["individual_code"],
                "encounter_id": row["encounter_id"],
                "exif_ts": row["exif_ts"],
                "side": row["side"],
                "width": int(img.shape[1]),
                "height": int(img.shape[0]),
                "det": det,
                "feats": feats_out,
            }
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
            slim_fh.write(json.dumps(slim_record(record), separators=(",", ":")) + "\n")

            want_overlay = (
                n_overlay < args.overlay_n
                and (args.overlay_every <= 0 or acc.n_images % args.overlay_every == 1)
            )
            if want_overlay and det["body_polygon"]:
                canvas = draw_overlay(img, det, label="#{}".format(rec_id))
                if canvas is not None:
                    # full-resolution single overlays are gitignored working
                    # files (no "_contact" in the name); only the montage below
                    # is version-controlled.
                    import cv2 as _cv2
                    op = args.out / "overlays"
                    op.mkdir(parents=True, exist_ok=True)
                    _cv2.imwrite(str(op / "{:04d}_{}.png".format(
                        rec_id, Path(row["filename"]).stem)), canvas)
                    overlay_tiles.append(canvas)
                    n_overlay += 1

            if acc.n_images % 100 == 0:
                el = time.time() - t0
                rate = acc.n_images / el
                print("  {}/{}  {:.2f} img/s  ETA {:.0f}s  no-body={}".format(
                    acc.n_images, len(rows), rate,
                    (len(rows) - idx - 1) / max(rate, 1e-9), acc.n_no_body))

    with open(str(skip_path), "w") as fh:
        for s in skipped:
            fh.write(json.dumps(s, separators=(",", ":")) + "\n")

    elapsed = time.time() - t0
    from collections import Counter
    reasons = Counter(s["reason"] for s in skipped)
    summary = {
        "run": {
            "db": str(args.db),
            "main_root": str(root),
            "device": args.device,
            "min_width": args.min_width,
            "limit": args.limit,
            "weights": {k: (str(v) if v else None) for k, v in wp.items()},
            "body_conf": oc.BODY_CONF, "spot_conf": oc.SPOT_CONF,
            "spot_max_det": oc.spot_max_det(models[1]),
            "elapsed_s": round(elapsed, 2),
            "detect_s": round(detect_seconds, 2),
            "throughput_img_per_s": round(acc.n_images / max(elapsed, 1e-9), 3),
            "detect_s_per_image": round(detect_seconds / max(acc.n_images, 1), 4),
        },
        "counts": {
            "candidate_rows": len(rows),
            "processed": acc.n_images,
            "with_body": acc.n_body,
            "without_body": acc.n_no_body,
            "degenerate_contour": acc.n_degenerate,
            "degenerate_contour_image_ids": acc.degenerate_ids,
            "spots_truncated": acc.n_truncated,
            "spots_truncated_image_ids": acc.truncated_ids,
            "skipped": len(skipped),
            "skipped_by_reason": dict(reasons),
            "total_spots": int(acc.pooled_array("spot_size").size),
        },
        "scalar_quantiles": acc.quantile_table(),
        "pooled_histograms": acc.histograms(),
    }
    with open(str(args.out / "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    contact_sheet(acc, args.out / "features_contact.png")
    if overlay_tiles:
        montage(overlay_tiles, args.out / "detections_contact.png")

    print("\nprocessed {} / {} rows in {:.1f}s ({:.2f} img/s, detect {:.3f} s/img)".format(
        acc.n_images, len(rows), elapsed,
        acc.n_images / max(elapsed, 1e-9), detect_seconds / max(acc.n_images, 1)))
    print("with body {}   without body {}   skipped {} {}".format(
        acc.n_body, acc.n_no_body, len(skipped), dict(reasons)))
    print("self-intersecting body contours (degenerate_contour): {}".format(
        acc.n_degenerate))
    print("spot detections truncated at max_det={}: {} images {}".format(
        summary["run"]["spot_max_det"], acc.n_truncated,
        acc.truncated_ids[:12] + (["..."] if len(acc.truncated_ids) > 12 else [])))
    print("total spots {}".format(summary["counts"]["total_spots"]))
    print("\nscalar quantiles (q05 / q50 / q95):")
    for k in ("n_spots", "density", "nn_median", "size_q50", "conf_q50",
              "aspect", "area_norm", "bbox_width_frac", "body_conf",
              "obstruction_count", "obstruction_area_frac", "D_minor"):
        e = summary["scalar_quantiles"].get(k, {})
        if "q050" in e:
            print("  {:24s} {:10.4f} {:10.4f} {:10.4f}   (n={})".format(
                k, e["q005"], e["q050"], e["q095"], e["n"]))
    print("\npooled per-spot quantiles (q05 / q50 / q95):")
    for k in ("spot_size", "spot_nn", "spot_conf", "spot_u", "spot_v"):
        q = summary["pooled_histograms"][k]["quantiles"]
        if q:
            print("  {:12s} {:10.4f} {:10.4f} {:10.4f}   (n={})".format(
                k, q["q005"], q["q050"], q["q095"],
                summary["pooled_histograms"][k]["n"]))
    wrote = [det_path.name, slim_path.name, skip_path.name, "summary.json",
             "features_contact.png"]
    if overlay_tiles:
        wrote.append("detections_contact.png ({} tiles)".format(len(overlay_tiles)))
    print("\nwrote " + ", ".join(wrote))
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
