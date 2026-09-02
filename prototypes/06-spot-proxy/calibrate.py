#!/usr/bin/env python
"""Drive the render -> detect -> compare loop and search the render config.

One CONFIG here is an override dict for :data:`synth_render.DEFAULT_CONFIG`.
:func:`run_config` renders ``n_individuals x sightings`` frames from it with a
fixed seed, runs the deployed OSEA detector over them with
:func:`synth_features.run`, compares the resulting per-image / pooled feature
distributions against the real ingested ones with :mod:`compare_features`, and
returns one flat record carrying

``objective``           ``compare_features.objective`` -- the spot half:
                        ``0.5 * mean KS D over per-image {density, size_q50,
                        nn_median, conf_q50} + 0.5 * mean KS D over pooled
                        {size, nn, conf}``, at spot conf >= 0.25.
``geometry_objective``  ``compare_features.geometry_objective`` -- mean KS D
                        over per-image ``{aspect, area_norm,
                        bbox_width_frac}``, which the CAMERA block moves.
``objective_040``       the same spot objective recomputed at spot conf >=
                        0.40.  A config that improves at 0.25 by drowning the
                        real detector's false-positive mixture in true spots
                        gets worse here; both are logged for every config.
``detector``            precision / recall of the deployed detector against the
                        renderer's own visible ground truth, per threshold.
``body``                body_conf quantiles and how many frames lost their body
                        to the OSEA 0.40 floor.

Every record is appended to ``results/calibration/grid.jsonl`` as it lands, so
a killed search keeps what it measured.

COMMON RANDOM NUMBERS.  Every config in a stage renders with the same ``seed``,
so the identity timelines and the nuisance draws are drawn from the same
generator stream.  Two configs that differ in one knob therefore differ by that
knob plus whatever the knob itself changes about the draw, not by fresh noise.
With 10 frames the per-image KS D still has a sampling floor near 0.25 (n=10
against n=1030), so per-image D values separate configs only when they differ
by more than that; the pooled half, over thousands of spots, is far tighter and
is what actually ranks the search.

PARALLELISM.  ``--workers N`` runs N configs at once in separate processes
(spawn), each rendering its own corpus.  A frame is ~8 s of single-threaded
numpy rasterisation, so this is near-linear until the core count runs out.
Each worker reloads the YOLO weights once and caches them
(``osea_contract._MODEL_CACHE``), and reads the real side from a pickle cache
built once by the parent, not from the 32 MB ``detections.jsonl``.  The start
method is SPAWN (the macOS default), so a caller that drives :func:`run_many`
from its own script file must put the call under ``if __name__ ==
"__main__":`` -- without it every worker re-executes the script on import and
forks again.

Run::

    P=".../7Gill/.venv/bin/python"
    "$P" -W ignore calibrate.py --before                 # pre-fix baseline
    "$P" -W ignore calibrate.py --stage1 --workers 5     # camera / framing
    "$P" -W ignore calibrate.py --stage2 --workers 5     # pattern / appearance
    "$P" -W ignore calibrate.py --final --frames 40      # the winner, bigger n
    "$P" -W ignore calibrate.py --contact                # calibration_contact.png
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import compare_features as cf          # noqa: E402
import synth_features                  # noqa: E402
import synth_render as sr              # noqa: E402

__all__ = [
    "DEFAULT_ROOT",
    "REAL_DETECTIONS",
    "BEFORE_OVERRIDE",
    "real_side",
    "merge_overrides",
    "config_from",
    "score_corpus",
    "run_config",
    "run_many",
    "record_row",
    "finalise",
    "write_deliverables",
    "calibration_contact",
]

DEFAULT_ROOT = os.path.join(_HERE, "results", "calibration")
REAL_DETECTIONS = os.path.join(_HERE, "results", "real", "detections.jsonl")

#: The renderer as it stood BEFORE this module's fixes, expressed as an
#: override so the baseline is measurable with the fixed code.  The only fix
#: that is not a config value is the ``frame_camera`` elevation sign, and a
#: NEGATIVE elevation range under the fixed sign is exactly the old positive
#: range under the old sign (``direction_z = -sin(elev)`` then,
#: ``+sin(elev)`` now).  The range is written ``[0, -50]``, not ``[-50, 0]``:
#: ``synth_render._u`` is ``low + (high-low)*u``, so a REVERSED range negates
#: each drawn value rather than mirroring the sample, and BEFORE replays the
#: old renderer frame for frame under the corrected sign.  ``s_target`` was the
#: scalar 0.25.
BEFORE_OVERRIDE = {
    "camera": {"elevation_deg": [0.0, -50.0], "s_target": 0.25},
}


# --------------------------------------------------------------------------- #
# the real side, loaded once                                                   #
# --------------------------------------------------------------------------- #
def real_side(path: str = REAL_DETECTIONS, cache: Optional[str] = None) -> Dict[str, Any]:
    """``compare_features.load_side`` of the real corpus, pickled to ``cache``.

    Parsing ``results/real/detections.jsonl`` costs ~5 s and 32 MB of JSON; the
    sample arrays it reduces to are ~15 MB of float64.  Every worker needs the
    same arrays, so they are built once and read back from the pickle.
    """
    cache = cache or os.path.join(DEFAULT_ROOT, "real_side.pkl")
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    side = cf.load_side(path)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump(side, fh, protocol=4)
    return side


# --------------------------------------------------------------------------- #
# configs                                                                      #
# --------------------------------------------------------------------------- #
def merge_overrides(*patches: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Deep-merge override dicts left to right; later wins.

    Returns a new dict and mutates nothing, so a base override can be reused
    across a whole stage of the search.
    """
    out = {}  # type: Dict[str, Any]
    for patch in patches:
        sr._deep_update(out, copy.deepcopy(patch or {}))
    return out


def config_from(overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The effective ``synth_render`` config for an override dict."""
    return sr.load_config(overrides=copy.deepcopy(overrides or {}))


# --------------------------------------------------------------------------- #
# scoring                                                                      #
# --------------------------------------------------------------------------- #
def _ks_block(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Just the KS D values, per threshold, small enough to log per config."""
    out = {}
    for thr, block in summary["thresholds"].items():
        out[thr] = {
            "per_image": {k: float(v["ks_D"]) for k, v in block["per_image"].items()},
            "pooled": {k: float(v["ks_D"]) for k, v in block["pooled"].items()},
            "n_synth_per_image": {k: int(v["n_synth"])
                                  for k, v in block["per_image"].items()},
        }
    return out


def _detector_block(det_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Precision / recall against the renderer's own visible GT, per threshold."""
    out = {}
    for key, row in (det_summary.get("pooled") or {}).items():
        out[key] = {
            "precision": row.get("precision"),
            "recall": row.get("recall"),
            "n_det": row.get("n_det"),
            "n_gt": row.get("n_gt"),
            "tp": row.get("tp"),
            "fp": row.get("fp"),
        }
    return out


def _body_block(det_path: str) -> Dict[str, Any]:
    """body_conf quantiles and the frames the OSEA 0.40 floor threw away."""
    confs = []
    n = 0
    n_ok = 0
    if not os.path.exists(det_path):
        return {"n_records": 0, "n_with_body": 0, "n_no_body": 0,
                "body_conf_q": [], "body_conf_median": None}
    for line in open(det_path):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        n += 1
        conf = (rec.get("det") or {}).get("body_conf")
        if (rec.get("feats") or {}).get("ok"):
            n_ok += 1
        if conf is not None:
            confs.append(float(conf))
    q = (np.percentile(confs, [5, 25, 50, 75, 95]).tolist() if confs else [])
    return {"n_records": n, "n_with_body": n_ok, "n_no_body": n - n_ok,
            "body_conf_q": [float(v) for v in q],
            "body_conf_median": float(np.median(confs)) if confs else None}


def score_corpus(corpus: str, real: Dict[str, Any],
                 out_dir: Optional[str] = None) -> Dict[str, Any]:
    """Compare an already-detected corpus against the real side.

    ``corpus`` must already hold ``detections.jsonl`` (written by
    :func:`synth_features.run`).  Writes ``summary.json`` / ``summary.md`` under
    ``out_dir`` when one is given.
    """
    synth = cf.load_side(os.path.join(corpus, "detections.jsonl"))
    summary = cf.compare(real, synth)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
        with open(os.path.join(out_dir, "summary.md"), "w") as fh:
            fh.write(cf.summary_markdown(summary))
    return summary


def run_config(name: str, overrides: Optional[Dict[str, Any]],
               root: str = DEFAULT_ROOT, seed: int = 0,
               n_individuals: int = 5, sightings: int = 2,
               real: Optional[Dict[str, Any]] = None,
               real_path: str = REAL_DETECTIONS,
               prune: bool = True, contact_n: int = 0,
               device: str = "cpu") -> Dict[str, Any]:
    """Render, detect and compare one config.  Returns the log record.

    ``prune`` deletes the per-frame ``gt/*.npz`` (9.6 MB each) once the
    detector has been scored against them; ``gt/*_spots.json``, the JPEGs and
    ``detections.jsonl`` stay, which is everything the contact sheets and the
    report need.
    """
    t0 = time.time()
    corpus = os.path.join(root, "corpora", name)
    if os.path.isdir(corpus):
        shutil.rmtree(corpus)
    os.makedirs(corpus, exist_ok=True)

    cfg = config_from(overrides)
    render_summary = sr.generate(corpus, n_individuals=int(n_individuals),
                                 sightings_per_individual=int(sightings),
                                 seed=int(seed), config=cfg, report=False)
    t_render = time.time() - t0

    t1 = time.time()
    det_summary = synth_features.run(_P(corpus), device=device,
                                     contact_n=int(contact_n), quiet=True)
    t_detect = time.time() - t1

    if real is None:
        real = real_side(real_path)
    t2 = time.time()
    summary = score_corpus(corpus, real, out_dir=os.path.join(root, "compare", name))
    t_compare = time.time() - t2

    if prune:
        for fn in os.listdir(os.path.join(corpus, "gt")):
            if fn.endswith(".npz"):
                os.remove(os.path.join(corpus, "gt", fn))

    record = record_row(name, overrides, summary, det_summary, render_summary,
                        corpus, seed=seed)
    record["seconds"] = {"render": round(t_render, 1), "detect": round(t_detect, 1),
                         "compare": round(t_compare, 1),
                         "total": round(time.time() - t0, 1)}
    return record


def _P(path):
    from pathlib import Path
    return Path(str(path))


def record_row(name: str, overrides: Optional[Dict[str, Any]],
               summary: Dict[str, Any], det_summary: Dict[str, Any],
               render_summary: Dict[str, Any], corpus: str,
               seed: int = 0) -> Dict[str, Any]:
    """One flat, JSON-safe row for ``grid.jsonl``."""
    return {
        "name": name,
        "seed": int(seed),
        "overrides": copy.deepcopy(overrides or {}),
        "corpus": corpus,
        "n_frames": int(render_summary.get("n_frames", 0)),
        "objective": float(cf.objective(summary)),
        "objective_040": float(cf.objective(summary, threshold="0.40")),
        "geometry_objective": float(cf.geometry_objective(summary)),
        "objective_detail": cf.objective(summary, detail=True),
        "geometry_detail": cf.geometry_objective(summary, detail=True),
        "ks": _ks_block(summary),
        "detector": _detector_block(det_summary),
        "body": _body_block(os.path.join(corpus, "detections.jsonl")),
        "visible_spots": render_summary.get("visible_spots"),
        "spot_radius_px": render_summary.get("spot_radius_px"),
    }


# --------------------------------------------------------------------------- #
# parallel driver                                                              #
# --------------------------------------------------------------------------- #
_WORKER_STATE = {}  # type: Dict[str, Any]


def _worker_init(real_path: str, cache: str) -> None:
    _WORKER_STATE["real"] = real_side(real_path, cache)


def _worker(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return run_config(real=_WORKER_STATE.get("real"), **job)
    except Exception as exc:                     # keep the search alive
        import traceback
        return {"name": job.get("name"), "overrides": job.get("overrides"),
                "error": "%s: %s" % (type(exc).__name__, exc),
                "traceback": traceback.format_exc()}


def run_many(jobs: Sequence[Dict[str, Any]], workers: int = 1,
             root: str = DEFAULT_ROOT, real_path: str = REAL_DETECTIONS,
             log: Optional[str] = None,
             report: bool = True) -> List[Dict[str, Any]]:
    """Run a list of ``run_config`` kwargs dicts; append each record to ``log``.

    ``workers <= 1`` runs in-process (which is also what the tests use).
    """
    log = log or os.path.join(root, "grid.jsonl")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    cache = os.path.join(root, "real_side.pkl")
    real_side(real_path, cache)                  # build it before forking
    out = []

    def _emit(rec):
        out.append(rec)
        with open(log, "a") as fh:
            fh.write(json.dumps(rec, sort_keys=True, default=float) + "\n")
        if report:
            if "error" in rec:
                print("%-22s ERROR %s" % (rec.get("name"), rec["error"]))
            else:
                print("%-22s obj %.4f (0.40 %.4f)  geom %.4f  bodies %d/%d  "
                      "conf50 %s  P/R@0.25 %s/%s  %ss"
                      % (rec["name"], rec["objective"], rec["objective_040"],
                         rec["geometry_objective"], rec["body"]["n_with_body"],
                         rec["body"]["n_records"],
                         _f(rec["body"]["body_conf_median"]),
                         _f((rec["detector"].get("0.25") or {}).get("precision")),
                         _f((rec["detector"].get("0.25") or {}).get("recall")),
                         rec.get("seconds", {}).get("total")))
            sys.stdout.flush()

    if int(workers) <= 1:
        real = real_side(real_path, cache)
        for job in jobs:
            try:
                _emit(run_config(real=real, **job))
            except Exception as exc:
                import traceback
                _emit({"name": job.get("name"), "error": "%s: %s" % (type(exc).__name__, exc),
                       "traceback": traceback.format_exc()})
        return out

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(int(workers), initializer=_worker_init,
                  initargs=(real_path, cache)) as pool:
        for rec in pool.imap_unordered(_worker, list(jobs)):
            _emit(rec)
    return out


def _f(v, nd=3):
    if v is None or (isinstance(float(v) if v is not None else 0.0, float)
                     and not np.isfinite(float(v))):
        return "-"
    return "{:.{}f}".format(float(v), nd)


# --------------------------------------------------------------------------- #
# reporting                                                                    #
# --------------------------------------------------------------------------- #
TABLE_KEYS_PER_IMAGE = ("density", "size_q50", "nn_median", "conf_q50",
                        "aspect", "area_norm", "bbox_width_frac", "body_conf",
                        "n_spots")
TABLE_KEYS_POOLED = ("size", "nn", "conf")


def table(records: Sequence[Dict[str, Any]], thresholds=("0.25", "0.40")) -> str:
    """A markdown table: one row per config, KS D per feature per threshold."""
    good = [r for r in records if "error" not in r]
    good = sorted(good, key=lambda r: r["objective"])
    head = ["config", "obj@.25", "obj@.40", "geom", "bodies"]
    for thr in thresholds:
        head += ["%s|%s" % (k, thr) for k in TABLE_KEYS_PER_IMAGE]
        head += ["%s*|%s" % (k, thr) for k in TABLE_KEYS_POOLED]
    head += ["P@.25", "R@.25", "P@.40", "R@.40"]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    for r in good:
        row = [r["name"], "%.4f" % r["objective"], "%.4f" % r["objective_040"],
               "%.4f" % r["geometry_objective"],
               "%d/%d" % (r["body"]["n_with_body"], r["body"]["n_records"])]
        for thr in thresholds:
            blk = r["ks"].get(thr, {})
            row += ["%.3f" % blk["per_image"][k] for k in TABLE_KEYS_PER_IMAGE]
            row += ["%.3f" % blk["pooled"][k] for k in TABLE_KEYS_POOLED]
        for thr in ("0.25", "0.40"):
            d = r["detector"].get(thr) or {}
            row += [_f(d.get("precision")), _f(d.get("recall"))]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


#: Where ``results/real/detections.jsonl`` rel_paths resolve from.
TAGGER_ROOT = os.path.normpath(os.path.join(
    _HERE, "..", "..", "..", "..", "..", "tagger"))

#: The seven features ``compare_features.objective`` is built from.
OBJECTIVE_FEATURES = [(k, "per_image") for k in cf.CAL_PER_IMAGE] + \
                     [(k, "pooled") for k in cf.CAL_POOLED]


def body_crop(image, feats, size=900, span=1.35):
    """A ``size``-square crop around the detected body, at MATCHED SCALE.

    The crop side is ``span * D_minor`` in the source image, so a spot whose
    ``size`` (``sqrt(w*h) / D_minor``) matches the real one is the same number
    of pixels across on the page whatever the camera distance or the source
    resolution was.  That is the only property the sheet needs, and it is the
    reason a 2016-wide render can be compared with a 4032-wide photograph.
    """
    from PIL import Image

    img = image if hasattr(image, "size") else Image.open(str(image))
    img = img.convert("RGB")
    w, h = img.size
    frame = feats["frame"]
    side = max(int(round(float(span) * float(frame["D_minor"]))), 32)
    side = min(side, min(w, h))
    cx, cy = float(frame["origin"][0]), float(frame["origin"][1])
    x0 = int(round(min(max(cx - side / 2.0, 0), w - side)))
    y0 = int(round(min(max(cy - side / 2.0, 0), h - side)))
    return img.crop((x0, y0, x0 + side, y0 + side)).resize((int(size), int(size)),
                                                           Image.LANCZOS)


def _pick(records, n, key):
    """``n`` records spread over the ``key`` order, best-lit first."""
    ranked = sorted(records, key=key, reverse=True)
    if len(ranked) <= n:
        return ranked
    idx = np.linspace(0, len(ranked) - 1, n).round().astype(int)
    return [ranked[i] for i in idx]


def calibration_contact(synth_corpus: str, summary: Dict[str, Any], path: str,
                        real_path: str = REAL_DETECTIONS,
                        tagger_root: str = TAGGER_ROOT,
                        real_side_arrays: Optional[Dict[str, Any]] = None,
                        synth_side_arrays: Optional[Dict[str, Any]] = None,
                        n_tiles: int = 4, size: int = 900,
                        title: str = "") -> str:
    """Real crops over synthetic crops, then the seven objective histograms.

    Both crop rows are ``size`` px and scaled by the detected ``D_minor``, so
    the two rows are directly comparable by eye.  The histogram row is the
    SEVEN features ``compare_features.objective`` is made of, at spot
    confidence >= 0.25, with the KS D of each in its title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real_rows = []
    for line in open(real_path):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if not (rec.get("feats") or {}).get("ok"):
            continue
        src = rec["rel_path"]
        src = src if os.path.isabs(src) else os.path.join(tagger_root, src)
        if os.path.exists(src) and rec.get("width", 0) >= 2000:
            rec["_src"] = src
            real_rows.append(rec)
    real_pick = _pick(real_rows, n_tiles,
                      key=lambda r: r["feats"]["scalars"]["body_conf"])

    synth_rows = []
    for line in open(os.path.join(synth_corpus, "detections.jsonl")):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if (rec.get("feats") or {}).get("ok"):
            rec["_src"] = os.path.join(synth_corpus, "body",
                                       "%s.jpg" % rec["image_id"])
            synth_rows.append(rec)
    synth_pick = _pick(synth_rows, n_tiles,
                       key=lambda r: r["feats"]["scalars"]["body_conf"])

    if real_side_arrays is None:
        real_side_arrays = cf.load_side(real_path)
    if synth_side_arrays is None:
        synth_side_arrays = cf.load_side(os.path.join(synth_corpus,
                                                      "detections.jsonl"))

    ncol = max(n_tiles, 4)
    tile_in = size / 100.0
    hist_in = 3.0
    fig = plt.figure(figsize=(ncol * tile_in, 2 * tile_in + 2 * hist_in),
                     dpi=100)
    gs = fig.add_gridspec(4, ncol,
                          height_ratios=[tile_in, tile_in, hist_in, hist_in])

    for row, (rows, label) in enumerate(((real_pick, "REAL"),
                                         (synth_pick, "SYNTH"))):
        for col in range(ncol):
            ax = fig.add_subplot(gs[row, col])
            ax.set_xticks([])
            ax.set_yticks([])
            if col >= len(rows):
                ax.axis("off")
                continue
            rec = rows[col]
            sc = rec["feats"]["scalars"]
            ax.imshow(np.asarray(body_crop(rec["_src"], rec["feats"], size)))
            ax.set_title("%s %s  n=%d size50=%.4f nn50=%s dens=%.0f conf50=%.2f "
                         "bconf=%.2f" % (
                             label, rec.get("filename") or rec.get("image_id"),
                             sc["n_spots"], sc["size"]["q50"],
                             _f(sc["nn_median"], 4), sc["density"] or 0.0,
                             sc["conf"]["q50"] or 0.0, sc["body_conf"]),
                         fontsize=8)

    block = summary["thresholds"][cf.OBJECTIVE_THRESHOLD]
    for i, (name, group) in enumerate(OBJECTIVE_FEATURES):
        ax = fig.add_subplot(gs[2 + i // ncol, i % ncol])
        lo, hi = cf.PLOT_RANGE.get(name, (0.0, 1.0))
        bins = np.linspace(lo, hi, 41)
        r = cf._finite(real_side_arrays[group][0.25][name])
        sy = cf._finite(synth_side_arrays[group][0.25][name])
        n_clip = int(((r < lo) | (r > hi)).sum() + ((sy < lo) | (sy > hi)).sum())
        for arr, colour, lab in ((r, "#3b6ea5", "real"), (sy, "#c0563a", "synth")):
            if arr.size:
                ax.hist(np.clip(arr, lo, hi), bins=bins, density=True,
                        color=colour, alpha=0.55, label=lab, edgecolor="none")
        e = block[group][name]
        ax.set_title("%s [%s]  D=%.3f\nnR=%d nS=%d%s" % (
            name, "img" if group == "per_image" else "spot", e["ks_D"],
            e["n_real"], e["n_synth"],
            ", %d clipped" % n_clip if n_clip else ""), fontsize=9)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")
    for i in range(len(OBJECTIVE_FEATURES), 2 * ncol):
        fig.add_subplot(gs[2 + i // ncol, i % ncol]).axis("off")

    fig.suptitle("prototype 06 calibration -- %s   objective=%.4f  "
                 "geometry=%.4f  (crops scaled by D_minor, %d px)"
                 % (title or os.path.basename(synth_corpus),
                    cf.objective(summary), cf.geometry_objective(summary), size),
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def finalise(overrides: Dict[str, Any], root: str = DEFAULT_ROOT,
             name: str = "best", seed: int = 0, n_individuals: int = 20,
             sightings: int = 2, real_path: str = REAL_DETECTIONS,
             size: int = 900) -> Dict[str, Any]:
    """Re-run the winning override on a bigger corpus and write the deliverables.

    Writes ``<root>/best.json`` (the override dict, ready for
    ``synth_render.py --config``), ``<root>/best_record.json`` (its full grid
    record), ``<root>/calibration_contact.png`` and ``<root>/report.md``.  The
    corpus is kept whole (``prune=False``) so the crops stay reproducible.
    """
    real = real_side(real_path, os.path.join(root, "real_side.pkl"))
    rec = run_config(name, overrides, root=root, seed=seed,
                     n_individuals=n_individuals, sightings=sightings,
                     real=real, real_path=real_path, prune=False, contact_n=4)
    with open(os.path.join(root, "grid.jsonl"), "a") as fh:
        fh.write(json.dumps(rec, sort_keys=True, default=float) + "\n")
    write_deliverables(rec, root=root, real=real, real_path=real_path, size=size)
    return rec


def write_deliverables(rec: Dict[str, Any], root: str = DEFAULT_ROOT,
                       real: Optional[Dict[str, Any]] = None,
                       real_path: str = REAL_DETECTIONS,
                       size: int = 900) -> Dict[str, str]:
    """``best.json``, ``best_record.json``, the contact sheet and ``report.md``.

    Split out of :func:`finalise` so the same writes can follow a parallel
    ``run_many`` (the final BEST and BEFORE corpora render at the same time).
    """
    os.makedirs(root, exist_ok=True)
    name = rec["name"]
    paths = {"best": os.path.join(root, "best.json"),
             "record": os.path.join(root, "best_record.json"),
             "contact": os.path.join(root, "calibration_contact.png"),
             "report": os.path.join(root, "report.md")}
    with open(paths["best"], "w") as fh:
        json.dump(rec["overrides"], fh, indent=2, sort_keys=True)
    with open(paths["record"], "w") as fh:
        json.dump(rec, fh, indent=2, sort_keys=True, default=float)
    summary = json.load(open(os.path.join(root, "compare", name, "summary.json")))
    calibration_contact(rec["corpus"], summary, paths["contact"],
                        real_path=real_path,
                        real_side_arrays=real or real_side(real_path),
                        n_tiles=4, size=size, title=name)
    with open(paths["report"], "w") as fh:
        fh.write(table(read_grid(root=root)) + "\n")
    return paths


def read_grid(log: Optional[str] = None, root: str = DEFAULT_ROOT) -> List[Dict[str, Any]]:
    log = log or os.path.join(root, "grid.jsonl")
    if not os.path.exists(log):
        return []
    return [json.loads(l) for l in open(log) if l.strip()]


# --------------------------------------------------------------------------- #
# search spaces                                                                #
# --------------------------------------------------------------------------- #
def _jobs(cands, base=None, **kw):
    return [dict(name=n, overrides=merge_overrides(base, o), **kw)
            for n, o in cands]


#: Camera / framing candidates -- the geometry objective.  The three geometry
#: features are set by where the camera is and how much animal it frames:
#: ``aspect`` and ``area_norm`` by the elevation (a near-dorsal view splays the
#: pectorals and turns the silhouette into a fat cross) and by how much length
#: is in frame, ``bbox_width_frac`` by whether the snout falls inside the
#: frame, which is ``s_target`` against ``s_frame_max * width_frac``.
STAGE1 = [
    ("s00_fixed", {}),
    ("s01_smax_short", {"camera": {"s_frame_max": [0.24, 0.34],
                                   "s_target": [0.11, 0.17]}}),
    ("s02_smax_long", {"camera": {"s_frame_max": [0.38, 0.54],
                                  "s_target": [0.17, 0.26]}}),
    ("s03_wf_tight", {"camera": {"width_frac": [0.84, 1.00]}}),
    ("s04_wf_loose", {"camera": {"width_frac": [0.62, 0.80]}}),
    ("s05_elev_low", {"camera": {"elevation_deg": [0.0, 22.0]}}),
    ("s06_elev_hi", {"camera": {"elevation_deg": [8.0, 45.0]}}),
    ("s07_starg_lo", {"camera": {"s_target": [0.09, 0.16]}}),
    ("s08_starg_hi", {"camera": {"s_target": [0.20, 0.30]}}),
    ("s09_roll_wide", {"camera": {"roll_deg": [-45.0, 45.0]}}),
    ("s10_az_wide", {"camera": {"azimuth_deg": [-40.0, 40.0]}}),
]

#: The camera block stage 1 settled on, as an override on the shipped default.
#: MEASURED (11 frames, seed 0, grid.jsonl): the shipped default already sits
#: at the geometry noise floor (D ~ 0.256 is what two identical distributions
#: give at n_synth = 11 against n_real = 1030), so this only nudges the two
#: features that were still above it -- ``area_norm`` (real median 1.175 against
#: 0.86 at the default; a slightly higher elevation fills the silhouette) and
#: ``bbox_width_frac`` (0.913 against 0.914 -- already right, and a tighter
#: ``width_frac`` keeps it there while ``s_frame_max`` grows).
STAGE1_BEST = {
    "camera": {"elevation_deg": [4.0, 38.0], "width_frac": [0.82, 0.98],
               "s_frame_max": [0.30, 0.46], "s_target": [0.14, 0.22]},
}

#: Pattern / appearance candidates -- the calibration objective.  Directions
#: are MEASURED off stage 1, not guessed: at the stage-1 configs the synthetic
#: spots are ~9% too SMALL (pooled size q50 0.0266 against a real 0.0291), ~11%
#: too far APART (pooled nn q50 0.0453 against 0.0410) and too CONFIDENT
#: (pooled conf q50 0.444 against 0.414 -- the real detector fires at low
#: confidence on real skin clutter the synthetic skin does not have).
STAGE2 = [
    ("p00_base", {}),
    ("p01_r115", {"pattern": {"radius_median": 0.00345}}),
    ("p02_r130", {"pattern": {"radius_median": 0.0039}}),
    ("p03_sep085", {"pattern": {"min_sep": 0.0081, "n_spots": 1150}}),
    ("p04_r_sep", {"pattern": {"radius_median": 0.00345, "min_sep": 0.0081,
                               "n_spots": 1150}}),
    ("p05_dense", {"pattern": {"n_spots": 1500, "min_sep": 0.0072}}),
    ("p06_spread", {"pattern": {"radius_log_sigma": 0.55}}),
    ("p07_soft", {"pattern": {"edge_softness": 0.55}}),
    ("p08_softer", {"pattern": {"edge_softness": 0.90}}),
    ("p09_darkvar", {"pattern": {"darkness_mean": 0.72, "darkness_sigma": 0.26,
                                 "darkness_min": 0.25}}),
    ("p10_ecc", {"pattern": {"ecc_sigma": 0.55, "ecc_max": 3.2}}),
    ("p11_amp", {"pattern": {"amplitude": 0.75}}),
    ("p12_mottle", {"skin": {"mottle": 0.10, "mottle_px": 5.0}}),
    ("p13_mottle2", {"skin": {"mottle": 0.17, "mottle_px": 8.0}}),
    ("p14_bgtex", {"background": {"gradient": 0.42, "noise": 0.07}}),
    ("p15_hands", {"occluders": {"count_probs": [0.15, 0.45, 0.40]}}),
    ("p16_combo", {"pattern": {"radius_median": 0.00345, "min_sep": 0.0081,
                               "n_spots": 1150, "edge_softness": 0.55,
                               "radius_log_sigma": 0.50,
                               "darkness_mean": 0.72, "darkness_sigma": 0.26,
                               "darkness_min": 0.25},
                   "skin": {"mottle": 0.10, "mottle_px": 5.0}}),
]

#: Three camera combinations carried into the stage-2 batch, so the geometry
#: choice is re-measured under the same 11 frames as the pattern candidates.
STAGE1B = [
    ("c01_best", STAGE1_BEST),
    ("c02_hi", {"camera": {"elevation_deg": [8.0, 45.0],
                           "width_frac": [0.84, 1.00]}}),
    ("c03_lo", {"camera": {"elevation_deg": [0.0, 26.0],
                           "width_frac": [0.82, 0.98],
                           "s_frame_max": [0.32, 0.48]}}),
]


#: The base stage 3 searches around, MEASURED off stage 2 (grid.jsonl, 11
#: frames each).  Raising ``radius_median`` is the single largest move in the
#: whole search: pooled ``size`` KS D 0.132 -> 0.068 and per-image ``size_q50``
#: 0.435 -> 0.234 at 1.30x (``p02_r130``), and 1.15x with a smaller
#: ``min_sep`` and more spots (``p04_r_sep``) gives the best POOLED half of the
#: objective, 0.061 against 0.094 at the base.  The camera keeps the shipped
#: framing but caps the elevation at 24 deg, which is where ``body_conf``
#: matches the real median exactly (0.802 at ``s05_elev_low``, against 0.72-0.75
#: for every candidate that went above 35 deg).
STAGE3_BASE = {
    "camera": {"elevation_deg": [0.0, 24.0]},
    "pattern": {"radius_median": 0.0039, "min_sep": 0.0088, "n_spots": 1050},
}

#: Third round: a coordinate descent around :data:`STAGE3_BASE`, plus the two
#: SPREAD candidates.  The per-image features are spread problems rather than
#: median problems -- the real corpus is 1030 photographs of different animals,
#: cameras, distances and water, so per-image ``size_q50`` spans 0.017-0.049
#: and ``conf_q50`` 0.34-0.51, while every synthetic frame is the same mesh,
#: the same skin chart and the same 2016x1512 sensor.  ``degrade`` is the only
#: per-frame knob that moves both (blur shrinks a detected box and lowers its
#: confidence), and a small ``min_sep`` lets spots merge into the irregular,
#: touching blotches the real skin has (10.9% of real spot pairs are closer
#: than 0.02 D_minor against 5.3% synthetic) instead of a blue-noise lattice.
STAGE3 = [
    ("x00_carry", {}),
    ("x01_r_lo", {"pattern": {"radius_median": 0.0035}}),
    ("x02_r_hi", {"pattern": {"radius_median": 0.0043}}),
    ("x03_n1300", {"pattern": {"n_spots": 1300, "min_sep": 0.0080}}),
    ("x04_n860", {"pattern": {"n_spots": 860, "min_sep": 0.0098}}),
    ("x05_cluster", {"pattern": {"min_sep": 0.0055}}),
    ("x06_degrade", {"degrade": {"blur_sigma": [0.0, 3.2],
                                 "jpeg_quality": [55, 95]}}),
    ("x07_spread", {"pattern": {"radius_log_sigma": 0.52}}),
    ("x08_bg", {"background": {"palette": [[0.88, 0.79, 0.76],
                                           [0.72, 0.80, 0.78],
                                           [0.66, 0.72, 0.79],
                                           [0.80, 0.80, 0.79],
                                           [0.42, 0.46, 0.46]],
                               "gradient": 0.34, "noise": 0.055}}),
    ("x09_spec", {"specular": {"strength": [0.10, 0.75],
                               "shininess": [12.0, 70.0]}}),
    # p12_mottle (skin grain 0.045 -> 0.10 at a 5 px correlation length) was
    # the best single change in stage 2 at threshold 0.25 (obj 0.1640) but the
    # worst of the leaders at 0.40 (0.2171): grain buys low-confidence
    # detections that look like the real detector's clutter mixture and can
    # then be thresholded away.  Both are carried into stage 3 and judged on
    # BOTH thresholds.
    ("x10_mottle", {"skin": {"mottle": 0.10, "mottle_px": 5.0}}),
    ("x11_mottle_r", {"skin": {"mottle": 0.10, "mottle_px": 5.0},
                      "pattern": {"radius_median": 0.0035}}),
]


#: Fourth round: the stage-3 winners combined.  Stage 3 leaders at 11 frames,
#: ``objective`` (``objective`` at conf >= 0.40 in brackets):
#: ``x03_n1300`` 0.1439 (0.1627), ``x07_spread`` 0.1590 (0.1696),
#: ``x05_cluster`` 0.1635 (0.1469), ``x10_mottle`` 0.1732 (0.1551).
#: ``x03`` is carried as the base of the round and the other three are added
#: to it one at a time and together, plus one elevation candidate: the only
#: per-image feature still clearly above the sampling floor is ``area_norm``
#: (real median 1.175 against 0.90), and the one knob that moves it is
#: elevation -- at 8-45 deg it reads 1.203, but ``body_conf`` then falls from
#: 0.802 (exactly real) to 0.751, so the trade has to be measured.
STAGE4_BASE = merge_overrides(STAGE3_BASE,
                              {"pattern": {"n_spots": 1300, "min_sep": 0.0080}})

STAGE4 = [
    ("y0_carry", {}),
    ("y1_spread", {"pattern": {"radius_log_sigma": 0.52}}),
    ("y2_mottle", {"skin": {"mottle": 0.10, "mottle_px": 5.0}}),
    ("y3_cluster", {"pattern": {"min_sep": 0.0055}}),
    ("y4_elev", {"camera": {"elevation_deg": [2.0, 34.0]}}),
    ("y5_all", {"pattern": {"radius_log_sigma": 0.52, "min_sep": 0.0062},
                "skin": {"mottle": 0.10, "mottle_px": 5.0}}),
    ("y6_all_elev", {"pattern": {"radius_log_sigma": 0.52, "min_sep": 0.0062},
                     "skin": {"mottle": 0.10, "mottle_px": 5.0},
                     "camera": {"elevation_deg": [2.0, 32.0]}}),
]


def stage4_jobs(base=None, **kw):
    """``run_config`` kwargs for every :data:`STAGE4` candidate."""
    return _jobs(STAGE4, merge_overrides(STAGE4_BASE, base), **kw)


def stage3_jobs(base=None, **kw):
    """``run_config`` kwargs for every :data:`STAGE3` candidate."""
    return _jobs(STAGE3, base, **kw)


def stage1b_jobs(base=None, **kw):
    """``run_config`` kwargs for every :data:`STAGE1B` camera combination."""
    return _jobs(STAGE1B, base, **kw)


def stage1_jobs(base=None, **kw):
    """``run_config`` kwargs for every :data:`STAGE1` candidate."""
    return _jobs(STAGE1, base, **kw)


def stage2_jobs(base=None, **kw):
    """``run_config`` kwargs for every :data:`STAGE2` candidate."""
    return _jobs(STAGE2, base, **kw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--real", default=REAL_DETECTIONS)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--identities", type=int, default=5)
    ap.add_argument("--sightings", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--config", default=None,
                    help="JSON file (or literal) of overrides to run as one config")
    ap.add_argument("--before", action="store_true",
                    help="render the pre-fix baseline (BEFORE_OVERRIDE)")
    ap.add_argument("--stage1", action="store_true")
    ap.add_argument("--stage1b", action="store_true")
    ap.add_argument("--stage2", action="store_true")
    ap.add_argument("--stage3", action="store_true")
    ap.add_argument("--stage4", action="store_true")
    ap.add_argument("--final", action="store_true",
                    help="re-run --base at --identities x --sightings and write "
                         "best.json + calibration_contact.png")
    ap.add_argument("--base", default=None,
                    help="JSON file (or literal) every stage candidate builds on")
    ap.add_argument("--table", action="store_true",
                    help="print the markdown table of everything in grid.jsonl")
    args = ap.parse_args(argv)

    if args.table:
        print(table(read_grid(root=args.root)))
        return 0

    kw = dict(root=args.root, seed=args.seed, n_individuals=args.identities,
              sightings=args.sightings, real_path=args.real)
    base = None
    if args.base:
        if os.path.exists(args.base):
            with open(args.base) as fh:
                base = json.load(fh)
        else:
            base = json.loads(args.base)

    jobs = []
    if args.before:
        jobs.append(dict(name=args.name or "before", overrides=BEFORE_OVERRIDE, **kw))
    if args.stage1:
        jobs.extend(stage1_jobs(base, **kw))
    if args.stage1b:
        jobs.extend(stage1b_jobs(base, **kw))
    if args.stage2:
        jobs.extend(stage2_jobs(base, **kw))
    if args.stage3:
        jobs.extend(stage3_jobs(base, **kw))
    if args.stage4:
        jobs.extend(stage4_jobs(base, **kw))
    if args.final:
        print(json.dumps(finalise(base or {}, root=args.root, seed=args.seed,
                                  n_individuals=args.identities,
                                  sightings=args.sightings,
                                  real_path=args.real,
                                  name=args.name or "best")["objective_detail"],
                         indent=2, sort_keys=True))
        return 0
    if args.config:
        if os.path.exists(args.config):
            with open(args.config) as fh:
                ov = json.load(fh)
        else:
            ov = json.loads(args.config)
        jobs.append(dict(name=args.name or "config", overrides=ov, **kw))
    if not jobs:
        ap.error("nothing to do: pass --before, --stage1, --stage2, --config "
                 "or --table")

    run_many(jobs, workers=args.workers, root=args.root, real_path=args.real)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
