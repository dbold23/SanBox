"""Run the OSEA detection contract over a synthetic corpus and score the
detector against the render ground truth.

This is the *bridge* between the generator (``synth_render.py``) and the
detector (``osea_contract.py``).  Every synthetic frame goes through exactly
the same door the real photos went through in ``real_features.py`` -- PIL open,
``exif_transpose``, ``convert("RGB")``, ``osea_contract.detect`` on the raw RGB
uint8 array -- so the two ``detections.jsonl`` files are directly comparable by
``compare_features.py``.

Because the renders carry per-spot ground truth (``gt/<id>_spots.json``: the
projected centre and radius of every rendered spot, with a ``visible`` flag),
this script can also say what the *detector* costs on the synthetic domain:

    a detected spot is a TRUE POSITIVE if its centre ``(cx, cy)`` lies within
    ``max(radius_px, --min-tol-px)`` of the centre of a visible ground-truth
    spot, matched ONE-TO-ONE, greedily in order of descending detector
    confidence (each detection takes the nearest still-unclaimed GT spot that
    is inside its own tolerance).

That number is *not* comparable to a real-photo precision (there is no real
per-spot ground truth), but it is the only handle on "did the renderer draw
spots the deployed detector can see".

Outputs, under ``--corpus`` (default ``results/synth_smoke``):

  detections.jsonl     one record per truth row, in the SAME shape as
                       ``results/real/detections.jsonl``
                       (image_id, filename, rel_path, individual_code,
                       encounter_id, exif_ts, side, width, height, det, feats)
                       plus the truth fields identity, sighting, date, pose,
                       camera.  ``individual_code``/``encounter_id`` are
                       aliases of ``identity``/``date`` so that
                       ``eval_constellation.py --synth`` reads the file
                       unchanged.

  detector_summary.json  per-image TP/FP/FN with precision and recall and the
                       TP-vs-FP confidence quantiles, pooled precision/recall
                       at spot-confidence thresholds 0.25/0.40/0.50/0.60, and
                       pooled TP/FP confidence histograms.

  detector_contact.png   two frames, cropped to the body and upscaled to at
                       least 900 px on the short side, with true positives in
                       GREEN, false positives in RED and missed ground-truth
                       spots in BLUE.

Example::

    MAIN=/Volumes/External\\ Dive\\ 2TB/projects/marine-cv/7Gill
    "$MAIN/.venv/bin/python" -W ignore synth_features.py --corpus results/synth_smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))

import osea_contract as oc  # noqa: E402

#: pooled precision/recall is reported at these spot-confidence thresholds.
CONF_THRESHOLDS = (0.25, 0.40, 0.50, 0.60)

#: quantiles reported for the TP / FP confidence distributions
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)

#: bin edges for the pooled TP / FP confidence histograms
CONF_BINS = np.linspace(0.25, 1.0, 31)

#: truth fields copied verbatim onto every detection record
TRUTH_FIELDS = ("identity", "sighting", "date", "side", "pose", "camera")


# --------------------------------------------------------------------------- #
# ground-truth matching                                                        #
# --------------------------------------------------------------------------- #
def match_detections(det_spots: Sequence[Dict[str, Any]],
                     gt_spots: Sequence[Dict[str, Any]],
                     min_tol_px: float = 6.0) -> Dict[str, Any]:
    """One-to-one greedy match of detected spots to ground-truth spots.

    Parameters
    ----------
    det_spots
        detector spots, each with ``cx``, ``cy`` and ``conf`` (the shape
        ``osea_contract.detect`` emits).
    gt_spots
        ground-truth spots, each with ``cx``, ``cy`` and ``radius_px``.  The
        caller is responsible for having filtered to *visible* spots with a
        centre; this function matches everything it is handed.
    min_tol_px
        floor on the per-GT match radius, so a sub-pixel rendered spot is still
        matchable.  The tolerance for GT spot ``j`` is
        ``max(radius_px_j, min_tol_px)``.

    Returns
    -------
    dict with

    ``pairs``      ``[(det_index, gt_index, distance_px), ...]``, in the order
                   the greedy pass claimed them (descending detector conf).
    ``fp``         detector indices that claimed no GT spot.
    ``missed``     GT indices nothing claimed.
    ``tp``/``n_det``/``n_gt``  counts.
    ``precision``  ``tp / n_det`` (``None`` when there are no detections).
    ``recall``     ``tp / n_gt``  (``None`` when there is no GT).
    ``tp_conf`` / ``fp_conf``   the detector confidences of each group.

    Greedy-by-confidence is deliberate: the detector's own ranking decides who
    gets first pick, so a low-confidence duplicate on top of a real spot is
    counted as the false positive rather than displacing the confident one.
    """
    n_det = len(det_spots)
    n_gt = len(gt_spots)
    if n_gt:
        gt_xy = np.array([[float(g["cx"]), float(g["cy"])] for g in gt_spots],
                         dtype=np.float64)
        gt_tol = np.array([max(float(g.get("radius_px") or 0.0), float(min_tol_px))
                           for g in gt_spots], dtype=np.float64)
    else:
        gt_xy = np.zeros((0, 2), dtype=np.float64)
        gt_tol = np.zeros(0, dtype=np.float64)

    order = sorted(range(n_det),
                   key=lambda i: (-float(det_spots[i].get("conf", 0.0)), i))
    claimed = np.zeros(n_gt, dtype=bool)
    pairs = []          # type: List[Tuple[int, int, float]]
    fp = []             # type: List[int]
    for i in order:
        s = det_spots[i]
        if n_gt == 0:
            fp.append(i)
            continue
        p = np.array([float(s["cx"]), float(s["cy"])], dtype=np.float64)
        d = np.linalg.norm(gt_xy - p, axis=1)
        eligible = (~claimed) & (d <= gt_tol)
        if not eligible.any():
            fp.append(i)
            continue
        d_masked = np.where(eligible, d, np.inf)
        j = int(np.argmin(d_masked))
        claimed[j] = True
        pairs.append((i, j, float(d[j])))

    missed = [j for j in range(n_gt) if not claimed[j]]
    tp = len(pairs)
    tp_set = set(p[0] for p in pairs)
    return {
        "pairs": pairs,
        "fp": sorted(fp),
        "missed": missed,
        "tp": tp,
        "n_det": n_det,
        "n_gt": n_gt,
        "precision": (tp / float(n_det)) if n_det else None,
        "recall": (tp / float(n_gt)) if n_gt else None,
        "tp_conf": [float(det_spots[i].get("conf", 0.0)) for i in sorted(tp_set)],
        "fp_conf": [float(det_spots[i].get("conf", 0.0)) for i in sorted(fp)],
    }


def visible_gt(gt_spots: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The ground-truth spots that were actually rendered into the image."""
    return [g for g in gt_spots
            if g.get("visible") and g.get("cx") is not None and g.get("cy") is not None]


def _quantiles(values: Sequence[float]) -> Dict[str, Optional[float]]:
    v = np.asarray(list(values), dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(("q{:02d}".format(int(round(q * 100))), None) for q in QUANTILES)
    out = {"q{:02d}".format(int(round(q * 100))): float(np.quantile(v, q))
           for q in QUANTILES}
    out["mean"] = float(v.mean())
    return out


# --------------------------------------------------------------------------- #
# corpus io                                                                    #
# --------------------------------------------------------------------------- #
def load_truth(corpus: Path) -> List[Dict[str, Any]]:
    """Every row of ``<corpus>/truth.jsonl``, in file order."""
    rows = []
    with open(str(corpus / "truth.jsonl")) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def image_path(corpus: Path, truth_row: Dict[str, Any]) -> Path:
    rel = (truth_row.get("paths") or {}).get("image")
    if not rel:
        rel = "body/{}.jpg".format(truth_row["image_id"])
    return corpus / rel


def gt_spots_path(corpus: Path, truth_row: Dict[str, Any]) -> Path:
    rel = (truth_row.get("paths") or {}).get("spots")
    if not rel:
        rel = "gt/{}_spots.json".format(truth_row["image_id"])
    return corpus / rel


def load_image(path: Path) -> np.ndarray:
    """Exactly the ingest path used by ``real_features.py``."""
    return np.array(ImageOps.exif_transpose(Image.open(str(path))).convert("RGB"))


# --------------------------------------------------------------------------- #
# figures                                                                      #
# --------------------------------------------------------------------------- #
def draw_match_overlay(img: np.ndarray, det: Dict[str, Any],
                       gt: Sequence[Dict[str, Any]], match: Dict[str, Any],
                       label: str = "", min_side: int = 900) -> Optional[np.ndarray]:
    """Zoomed TP/FP/missed-GT overlay; returns a BGR canvas.

    green  = true positive detection box
    red    = false positive detection box
    blue   = ground-truth spot nothing detected (circle at its match radius)
    """
    import cv2

    bbox = det.get("body_bbox")
    H, W = img.shape[:2]
    if bbox:
        x0 = max(0, bbox["x"]); y0 = max(0, bbox["y"])
        x1 = min(W, bbox["x"] + bbox["w"]); y1 = min(H, bbox["y"] + bbox["h"])
    else:
        x0, y0, x1, y1 = 0, 0, W, H
    if x1 <= x0 or y1 <= y0:
        return None
    crop = np.ascontiguousarray(img[y0:y1, x0:x1])
    if crop.size == 0:
        return None
    scale = max(1.0, float(min_side) / max(1, min(crop.shape[0], crop.shape[1])))
    if scale > 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    canvas = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

    def to_c(x, y):
        return (int(round((x - x0) * scale)), int(round((y - y0) * scale)))

    spots = det.get("spots") or []
    tp_idx = set(p[0] for p in match["pairs"])
    for i, s in enumerate(spots):
        colour = (60, 220, 60) if i in tp_idx else (50, 50, 235)   # BGR
        p0 = to_c(s["x"], s["y"])
        p1 = to_c(s["x"] + s["w"], s["y"] + s["h"])
        cv2.rectangle(canvas, p0, p1, colour, 2)
    for j in match["missed"]:
        g = gt[j]
        c = to_c(float(g["cx"]), float(g["cy"]))
        r = max(4, int(round(max(float(g.get("radius_px") or 0.0), 6.0) * scale)))
        cv2.circle(canvas, c, r, (235, 140, 40), 2)

    cap = "{}  TP={} FP={} miss={}  P={:.2f} R={:.2f}".format(
        label, match["tp"], len(match["fp"]), len(match["missed"]),
        match["precision"] if match["precision"] is not None else float("nan"),
        match["recall"] if match["recall"] is not None else float("nan"))
    for thick, col in ((4, (255, 255, 255)), (1, (20, 20, 20))):
        cv2.putText(canvas, cap, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    col, thick, cv2.LINE_AA)
    legend = "green=TP  red=FP  blue=missed GT"
    for thick, col in ((4, (255, 255, 255)), (1, (20, 20, 20))):
        cv2.putText(canvas, legend, (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    col, thick, cv2.LINE_AA)
    return canvas


def montage(tiles: List[np.ndarray], path: Path, ncol: int = 2,
            tile_w: int = 760) -> None:
    """Grid the overlay canvases into one contact sheet."""
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
        padded = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0,
                                     cv2.BORDER_CONSTANT, value=(255, 255, 255))
                  for t in chunk]
        while len(padded) < ncol:
            padded.append(np.full((hmax, tile_w, 3), 255, np.uint8))
        rows.append(np.hstack(padded))
    sheet = np.vstack(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_PNG_COMPRESSION, 6])


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def run(corpus: Path, device: str = "cpu", min_tol_px: float = 6.0,
        contact_n: int = 2, quiet: bool = False) -> Dict[str, Any]:
    """Detect over the corpus, score against GT, write the three outputs."""
    truth = load_truth(corpus)
    models = oc.load_models(device)
    if models[0] is None:
        raise RuntimeError("no body model available")

    det_path = corpus / "detections.jsonl"
    per_image = []          # type: List[Dict[str, Any]]
    dropped = []            # type: List[Dict[str, Any]]
    # per-threshold pooled tallies, plus the raw spot lists for the histograms
    pooled = {t: {"tp": 0, "fp": 0, "n_det": 0, "n_gt": 0, "missed": 0}
              for t in CONF_THRESHOLDS}
    tp_conf_all = []        # type: List[float]
    fp_conf_all = []        # type: List[float]
    tiles = []              # type: List[np.ndarray]
    t0 = time.time()
    detect_seconds = 0.0

    with open(str(det_path), "w") as fh:
        for row in truth:
            image_id = str(row["image_id"])
            ipath = image_path(corpus, row)
            gpath = gt_spots_path(corpus, row)
            if not ipath.is_file():
                dropped.append({"image_id": image_id, "reason": "image_missing",
                                "detail": str(ipath)})
                continue
            img = load_image(ipath)
            if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
                dropped.append({"image_id": image_id, "reason": "bad_array",
                                "detail": "{} {}".format(img.shape, img.dtype)})
                continue
            t = time.time()
            det = oc.detect(img, models)
            detect_seconds += time.time() - t
            feats = oc.features(det)
            feats_out = {k: v for k, v in feats.items() if k != "spots_raw"}

            rec = {
                "image_id": image_id,
                "filename": ipath.name,
                "rel_path": str(ipath.relative_to(corpus)),
                # aliases so eval_constellation.py --synth reads this unchanged
                "individual_code": row.get("identity"),
                "encounter_id": row.get("date"),
                "exif_ts": None,
                "width": int(img.shape[1]),
                "height": int(img.shape[0]),
                "det": det,
                "feats": feats_out,
            }
            for k in TRUTH_FIELDS:
                rec[k] = row.get(k)
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")

            gt_all = json.load(open(str(gpath))) if gpath.is_file() else []
            gt = visible_gt(gt_all)
            spots = det.get("spots") or []
            m = match_detections(spots, gt, min_tol_px=min_tol_px)
            tp_conf_all.extend(m["tp_conf"])
            fp_conf_all.extend(m["fp_conf"])

            by_thr = {}
            for thr in CONF_THRESHOLDS:
                keep = [s for s in spots if float(s["conf"]) >= thr]
                mt = match_detections(keep, gt, min_tol_px=min_tol_px)
                by_thr["{:.2f}".format(thr)] = {
                    "n_det": mt["n_det"], "tp": mt["tp"], "fp": len(mt["fp"]),
                    "fn": len(mt["missed"]),
                    "precision": mt["precision"], "recall": mt["recall"]}
                pooled[thr]["tp"] += mt["tp"]
                pooled[thr]["fp"] += len(mt["fp"])
                pooled[thr]["n_det"] += mt["n_det"]
                pooled[thr]["n_gt"] += mt["n_gt"]
                pooled[thr]["missed"] += len(mt["missed"])

            sc = feats["scalars"]
            per_image.append({
                "image_id": image_id,
                "identity": row.get("identity"),
                "sighting": row.get("sighting"),
                "n_gt_rendered": len(gt_all),
                "n_gt_visible": len(gt),
                "n_det": m["n_det"],
                "tp": m["tp"], "fp": len(m["fp"]), "fn": len(m["missed"]),
                "precision": m["precision"], "recall": m["recall"],
                "match_dist_px_median": (float(np.median([p[2] for p in m["pairs"]]))
                                         if m["pairs"] else None),
                "tp_conf": _quantiles(m["tp_conf"]),
                "fp_conf": _quantiles(m["fp_conf"]),
                "body_ok": bool(feats["ok"]),
                "body_conf": det.get("body_conf"),
                "D_minor": sc.get("D_minor"),
                "aspect": sc.get("aspect"),
                "area_norm": sc.get("area_norm"),
                "density": sc.get("density"),
                "nn_median": sc.get("nn_median"),
                "size_q50": (sc.get("size") or {}).get("q50"),
                "conf_q50": (sc.get("conf") or {}).get("q50"),
                "bbox_width_frac": sc.get("bbox_width_frac"),
                "by_threshold": by_thr,
            })
            if len(tiles) < contact_n:
                canvas = draw_match_overlay(img, det, gt, m, label=image_id)
                if canvas is not None:
                    tiles.append(canvas)
            if not quiet:
                print("  {:12s} det={:4d} gt={:4d} TP={:4d} FP={:4d} miss={:4d} "
                      "P={} R={}".format(
                          image_id, m["n_det"], m["n_gt"], m["tp"], len(m["fp"]),
                          len(m["missed"]),
                          "n/a" if m["precision"] is None else "{:.3f}".format(m["precision"]),
                          "n/a" if m["recall"] is None else "{:.3f}".format(m["recall"])))

    elapsed = time.time() - t0
    pooled_out = {}
    for thr in CONF_THRESHOLDS:
        p = pooled[thr]
        pooled_out["{:.2f}".format(thr)] = {
            "n_det": p["n_det"], "n_gt": p["n_gt"], "tp": p["tp"], "fp": p["fp"],
            "fn": p["missed"],
            "precision": (p["tp"] / float(p["n_det"])) if p["n_det"] else None,
            "recall": (p["tp"] / float(p["n_gt"])) if p["n_gt"] else None,
        }

    tp_hist, _ = np.histogram(np.asarray(tp_conf_all, dtype=np.float64), bins=CONF_BINS)
    fp_hist, _ = np.histogram(np.asarray(fp_conf_all, dtype=np.float64), bins=CONF_BINS)
    summary = {
        "run": {
            "corpus": str(corpus),
            "device": device,
            "min_tol_px": float(min_tol_px),
            "n_truth_rows": len(truth),
            "n_images": len(per_image),
            "n_dropped": len(dropped),
            "seconds": round(elapsed, 2),
            "detect_seconds": round(detect_seconds, 2),
            "seconds_per_image": round(elapsed / max(1, len(per_image)), 3),
        },
        "dropped": dropped,
        "match_rule": ("detection centre within max(gt radius_px, {:g} px) of a "
                       "visible GT centre; one-to-one, greedy by descending "
                       "detector confidence".format(min_tol_px)),
        "pooled": pooled_out,
        "conf_hist": {
            "bin_edges": [float(e) for e in CONF_BINS],
            "tp_counts": [int(c) for c in tp_hist],
            "fp_counts": [int(c) for c in fp_hist],
            "tp_quantiles": _quantiles(tp_conf_all),
            "fp_quantiles": _quantiles(fp_conf_all),
            "n_tp": len(tp_conf_all),
            "n_fp": len(fp_conf_all),
        },
        "per_image": per_image,
    }
    with open(str(corpus / "detector_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    montage(tiles, corpus / "detector_contact.png")
    return summary


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=here / "results" / "synth_smoke",
                    help="corpus directory holding truth.jsonl, body/ and gt/")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min-tol-px", type=float, default=6.0,
                    help="floor on the per-GT-spot match radius")
    ap.add_argument("--contact-n", type=int, default=2,
                    help="frames drawn into detector_contact.png")
    args = ap.parse_args(argv)

    corpus = args.corpus if args.corpus.is_absolute() else (here / args.corpus)
    if not (corpus / "truth.jsonl").is_file():
        print("ERROR: no truth.jsonl under {}".format(corpus), file=sys.stderr)
        return 1
    print("corpus : {}".format(corpus))
    s = run(corpus, device=args.device, min_tol_px=args.min_tol_px,
            contact_n=args.contact_n)
    print("\npooled precision / recall")
    for thr in sorted(s["pooled"]):
        p = s["pooled"][thr]
        print("  conf>={}  n_det={:5d} n_gt={:5d} TP={:5d} FP={:5d} FN={:5d} "
              "P={} R={}".format(
                  thr, p["n_det"], p["n_gt"], p["tp"], p["fp"], p["fn"],
                  "n/a" if p["precision"] is None else "{:.3f}".format(p["precision"]),
                  "n/a" if p["recall"] is None else "{:.3f}".format(p["recall"])))
    print("\nwrote {}".format(args.corpus / "detections.jsonl"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
