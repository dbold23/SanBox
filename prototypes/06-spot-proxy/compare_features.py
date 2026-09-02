"""Compare the real-photo and synthetic non-RGB feature distributions.

Both sides are ``detections.jsonl`` files written through the same door
(``osea_contract.detect`` + ``osea_contract.features``): ``real_features.py``
for the catalog photos, ``synth_features.py`` for a rendered corpus.  This
module measures how far apart the two distributions are, feature by feature,
with a two-sample Kolmogorov-Smirnov statistic, and packages that into a single
scalar the next agent can minimise while tuning the renderer.

What is compared
----------------
*Per-image scalars* -- one value per image::

    n_spots  density  nn_median  size_q50  conf_q50        (spot statistics)
    aspect   area_norm  bbox_width_frac  body_conf         (geometry / framing)

``D_minor`` is deliberately excluded: it is an absolute pixel length, so it
compares the *camera resolution* of the two corpora, not the animal.

An image with no body polygon contributes *nothing* to the spot statistics
(there is no frame to normalise by), so it drops out of those samples rather
than entering as a zero.  This is why the real per-image quantiles here differ
slightly from ``results/real/summary.json``: that file averages over all 1091
processed images (61 of which have no body and are counted as ``n_spots = 0``,
pulling the median to 107), while this comparison uses the 1030 with a body
(median 112).  The geometry scalars follow the same rule.

*Pooled per-spot features* -- one value per detected spot, pooled over the
corpus::

    size   nn   conf   u   v

Confidence thresholds
---------------------
The v1 spot detector runs at a 0.25 floor and is low-precision there (46% of
real spots score below 0.40).  Every comparison is therefore repeated at spot
confidence >= 0.25 (all), >= 0.40 and >= 0.50, with the per-image scalars
*recomputed* from the surviving spots.  ``n_spots``, ``density``, ``nn_median``,
``size_q50`` and ``conf_q50`` all move with the threshold; the four geometry
scalars do not (they are properties of the body polygon) and are repeated
unchanged so each row of the table is self-contained.

The calibration objective
-------------------------
``objective(summary)`` -- what a renderer-tuning loop should minimise::

    0.5 * mean KS D over per-image {density, size_q50, nn_median, conf_q50}
  + 0.5 * mean KS D over pooled    {size, nn, conf}

both halves at threshold 0.25, weighted equally so that a corpus cannot win by
matching the per-image summaries while getting the per-spot distribution wrong
(or the reverse).  ``geometry_objective(summary)`` is the separate, unweighted
mean KS D over {aspect, area_norm, bbox_width_frac} -- framing and pose, which
are tuned with the camera block rather than the spot pattern.  Both are in
[0, 1]; 0 is a perfect distributional match.  A feature whose sample is empty
on either side scores the worst possible D = 1.0 and is flagged ``degenerate``,
so "the renderer produced no spots at all" cannot read as a good score.

Example::

    MAIN=/Volumes/External\\ Dive\\ 2TB/projects/marine-cv/7Gill
    "$MAIN/.venv/bin/python" -W ignore compare_features.py \\
        --real results/real/detections.jsonl \\
        --synth results/synth_smoke/detections.jsonl \\
        --out results/compare/synth_smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

#: spot-confidence thresholds every comparison is repeated at
THRESHOLDS = (0.25, 0.40, 0.50)

#: per-image scalars that move when the spot-confidence threshold moves
SPOT_SCALARS = ("n_spots", "density", "nn_median", "size_q50", "conf_q50")

#: per-image scalars that are properties of the body polygon alone
GEOM_SCALARS = ("aspect", "area_norm", "bbox_width_frac", "body_conf")

PER_IMAGE_FEATURES = SPOT_SCALARS + GEOM_SCALARS

#: pooled per-spot features
POOLED_FEATURES = ("size", "nn", "conf", "u", "v")

#: the two halves of the calibration objective (both at threshold 0.25)
CAL_PER_IMAGE = ("density", "size_q50", "nn_median", "conf_q50")
CAL_POOLED = ("size", "nn", "conf")
#: the separate geometry-only objective
GEOM_OBJECTIVE = ("aspect", "area_norm", "bbox_width_frac")

OBJECTIVE_THRESHOLD = "0.25"

QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)

#: histogram ranges for compare_contact.png (values outside are clipped, and
#: the clipped count is printed in the panel title -- never silently dropped)
PLOT_RANGE = {
    "n_spots": (0.0, 320.0),
    "density": (0.0, 400.0),
    "nn_median": (0.0, 0.15),
    "size_q50": (0.0, 0.10),
    "conf_q50": (0.25, 0.85),
    "aspect": (0.4, 6.0),
    "area_norm": (0.0, 4.0),
    "bbox_width_frac": (0.0, 1.05),
    "body_conf": (0.4, 1.0),
    "size": (0.0, 0.20),
    "nn": (0.0, 0.40),
    "conf": (0.25, 1.0),
    "u": (-1.5, 1.5),
    "v": (-1.0, 1.0),
}


# --------------------------------------------------------------------------- #
# feature extraction                                                           #
# --------------------------------------------------------------------------- #
def _nn_distances(pts: np.ndarray) -> np.ndarray:
    """Nearest-neighbour distance per point (same rule as osea_contract)."""
    if pts.shape[0] < 2:
        return np.zeros(0, dtype=np.float64)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def image_features(feats: Dict[str, Any], threshold: float) -> Tuple[Dict[str, Optional[float]],
                                                                    Dict[str, np.ndarray]]:
    """Per-image scalars and per-spot arrays for one record at one threshold.

    The spot statistics are recomputed from ``feats["spots_uv"]`` filtered by
    confidence, so they are *not* just the stored scalars once ``threshold``
    rises above the detector floor.  Geometry comes straight from the stored
    block.  A statistic with no support (no spots, fewer than two spots for a
    nearest-neighbour distance, a missing body polygon) is returned as ``None``
    / an empty array rather than a filled-in zero.
    """
    sc = feats.get("scalars") or {}
    out = dict((k, None) for k in PER_IMAGE_FEATURES)  # type: Dict[str, Optional[float]]
    for k in GEOM_SCALARS:
        v = sc.get(k)
        out[k] = None if v is None else float(v)

    uv = np.asarray(feats.get("spots_uv") or [], dtype=np.float64).reshape(-1, 4)
    if uv.shape[0]:
        uv = uv[uv[:, 3] >= threshold]
    spots = {"u": uv[:, 0] if uv.shape[0] else np.zeros(0),
             "v": uv[:, 1] if uv.shape[0] else np.zeros(0),
             "size": uv[:, 2] if uv.shape[0] else np.zeros(0),
             "conf": uv[:, 3] if uv.shape[0] else np.zeros(0),
             "nn": _nn_distances(uv[:, :2]) if uv.shape[0] else np.zeros(0)}

    if not feats.get("ok"):
        # no body frame -> no normalised spot statistics at all
        return out, {k: np.zeros(0) for k in POOLED_FEATURES}

    n = int(uv.shape[0])
    out["n_spots"] = float(n)
    area_norm = sc.get("area_norm")
    if area_norm:
        out["density"] = float(n) / float(area_norm)
    if n:
        out["size_q50"] = float(np.median(spots["size"]))
        out["conf_q50"] = float(np.median(spots["conf"]))
    if spots["nn"].size:
        out["nn_median"] = float(np.median(spots["nn"]))
    return out, spots


def load_side(path: Path, thresholds: Sequence[float] = THRESHOLDS,
              limit: Optional[int] = None) -> Dict[str, Any]:
    """Stream one detections.jsonl into per-threshold sample arrays.

    Returns ``{"n_records", "n_ok", "n_no_body", "per_image": {thr: {feat: [..]}},
    "pooled": {thr: {feat: ndarray}}}``.  Streaming matters: the real file is
    ~32 MB of polygons.
    """
    per_image = {t: dict((k, []) for k in PER_IMAGE_FEATURES) for t in thresholds}
    pooled_chunks = {t: dict((k, []) for k in POOLED_FEATURES) for t in thresholds}
    n_records = 0
    n_ok = 0
    with open(str(path)) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            feats = rec.get("feats")
            if feats is None:
                continue
            n_records += 1
            if feats.get("ok"):
                n_ok += 1
            for t in thresholds:
                scal, spots = image_features(feats, t)
                for k in PER_IMAGE_FEATURES:
                    per_image[t][k].append(np.nan if scal[k] is None else scal[k])
                for k in POOLED_FEATURES:
                    if spots[k].size:
                        pooled_chunks[t][k].append(spots[k])
            if limit and n_records >= limit:
                break
    pooled = {t: {k: (np.concatenate(v) if v else np.zeros(0))
                  for k, v in pooled_chunks[t].items()} for t in thresholds}
    per_image_arr = {t: {k: np.asarray(v, dtype=np.float64)
                         for k, v in per_image[t].items()} for t in thresholds}
    return {"path": str(path), "n_records": n_records, "n_ok": n_ok,
            "n_no_body": n_records - n_ok,
            "per_image": per_image_arr, "pooled": pooled}


# --------------------------------------------------------------------------- #
# statistics                                                                   #
# --------------------------------------------------------------------------- #
def _finite(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).ravel()
    return a[np.isfinite(a)]


def _quantiles(a: np.ndarray) -> Dict[str, Optional[float]]:
    a = _finite(a)
    if a.size == 0:
        return dict(("q{:02d}".format(int(round(q * 100))), None) for q in QUANTILES)
    out = {"q{:02d}".format(int(round(q * 100))): float(np.quantile(a, q))
           for q in QUANTILES}
    out["mean"] = float(a.mean())
    return out


def ks_entry(real: np.ndarray, synth: np.ndarray) -> Dict[str, Any]:
    """Two-sample KS between one real and one synthetic sample.

    An empty sample on either side scores the worst possible ``ks_D`` of 1.0 and
    is flagged ``degenerate`` -- a renderer that emits no spots must not be able
    to score well by having nothing to disagree about.
    """
    r = _finite(real)
    s = _finite(synth)
    entry = {
        "n_real": int(r.size), "n_synth": int(s.size),
        "real": _quantiles(r), "synth": _quantiles(s),
    }
    if r.size == 0 or s.size == 0:
        entry.update({"ks_D": 1.0, "ks_p": None, "degenerate": True})
        return entry
    res = stats.ks_2samp(r, s)
    entry.update({"ks_D": float(res.statistic), "ks_p": float(res.pvalue),
                  "degenerate": False})
    return entry


def compare(real: Dict[str, Any], synth: Dict[str, Any],
            thresholds: Sequence[float] = THRESHOLDS) -> Dict[str, Any]:
    """Full KS table for every feature at every threshold."""
    blocks = {}
    for t in thresholds:
        key = "{:.2f}".format(t)
        blocks[key] = {
            "per_image": {k: ks_entry(real["per_image"][t][k], synth["per_image"][t][k])
                          for k in PER_IMAGE_FEATURES},
            "pooled": {k: ks_entry(real["pooled"][t][k], synth["pooled"][t][k])
                       for k in POOLED_FEATURES},
        }
    summary = {
        "run": {
            "real": real["path"], "synth": synth["path"],
            "n_real_records": real["n_records"], "n_real_with_body": real["n_ok"],
            "n_real_no_body": real["n_no_body"],
            "n_synth_records": synth["n_records"], "n_synth_with_body": synth["n_ok"],
            "n_synth_no_body": synth["n_no_body"],
            "thresholds": [float(t) for t in thresholds],
        },
        "thresholds": blocks,
    }
    summary["objective"] = objective(summary, detail=True)
    summary["geometry_objective"] = geometry_objective(summary, detail=True)
    return summary


# --------------------------------------------------------------------------- #
# the calibration objective (import this)                                      #
# --------------------------------------------------------------------------- #
def _mean_D(block: Dict[str, Any], group: str, keys: Sequence[str]) -> Tuple[float, Dict[str, float]]:
    per = {}
    for k in keys:
        e = block[group][k]
        per[k] = float(e["ks_D"])
    return (float(np.mean([per[k] for k in keys])) if keys else 1.0), per


def objective(summary: Dict[str, Any], threshold: str = OBJECTIVE_THRESHOLD,
              detail: bool = False):
    """The scalar a renderer-calibration loop minimises.

    ``0.5 * mean KS D over per-image {density, size_q50, nn_median, conf_q50}``
    ``+ 0.5 * mean KS D over pooled {size, nn, conf}``, both at spot confidence
    >= 0.25.  Range [0, 1]; lower is a closer match to the real photos.

    With ``detail=True`` a dict is returned instead of the bare float, carrying
    the two half-means and the per-feature D values that produced them.
    """
    block = summary["thresholds"][threshold]
    a, a_per = _mean_D(block, "per_image", CAL_PER_IMAGE)
    b, b_per = _mean_D(block, "pooled", CAL_POOLED)
    value = 0.5 * a + 0.5 * b
    if not detail:
        return value
    return {"value": value, "threshold": threshold,
            "per_image_mean_D": a, "pooled_mean_D": b,
            "per_image_D": a_per, "pooled_D": b_per,
            "definition": ("0.5 * mean KS D over per-image "
                           "{density,size_q50,nn_median,conf_q50} + 0.5 * mean KS D "
                           "over pooled {size,nn,conf}, both at spot conf >= 0.25")}


def geometry_objective(summary: Dict[str, Any], threshold: str = OBJECTIVE_THRESHOLD,
                       detail: bool = False):
    """Mean KS D over the framing/pose scalars {aspect, area_norm, bbox_width_frac}.

    Reported separately from :func:`objective` because it is tuned by the camera
    and pose blocks of the render config, not by the spot pattern.
    """
    block = summary["thresholds"][threshold]
    value, per = _mean_D(block, "per_image", GEOM_OBJECTIVE)
    if not detail:
        return value
    return {"value": value, "threshold": threshold, "per_image_D": per,
            "definition": "mean KS D over per-image {aspect,area_norm,bbox_width_frac}"}


# --------------------------------------------------------------------------- #
# reporting                                                                    #
# --------------------------------------------------------------------------- #
def _fmt(v: Optional[float], nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return "{:.{}f}".format(v, nd)


def summary_markdown(summary: Dict[str, Any]) -> str:
    run = summary["run"]
    lines = []
    lines.append("# real vs synthetic: KS comparison of the OSEA contract features")
    lines.append("")
    lines.append("real  : `{}`  ({} records, {} with a body polygon)".format(
        run["real"], run["n_real_records"], run["n_real_with_body"]))
    lines.append("synth : `{}`  ({} records, {} with a body polygon)".format(
        run["synth"], run["n_synth_records"], run["n_synth_with_body"]))
    lines.append("")
    obj = summary["objective"]
    geo = summary["geometry_objective"]
    lines.append("**objective = {:.4f}**  (per-image half {:.4f}, pooled half {:.4f}; "
                 "lower is better, 0 = identical)".format(
                     obj["value"], obj["per_image_mean_D"], obj["pooled_mean_D"]))
    lines.append("")
    lines.append("**geometry objective = {:.4f}**  ({})".format(
        geo["value"], ", ".join("{} {:.3f}".format(k, geo["per_image_D"][k])
                                for k in GEOM_OBJECTIVE)))
    lines.append("")
    lines.append("`objective` = " + obj["definition"])
    lines.append("")
    for thr in sorted(summary["thresholds"]):
        block = summary["thresholds"][thr]
        lines.append("## spot confidence >= {}".format(thr))
        lines.append("")
        head = ("| feature | kind | KS D | p | n real | n synth | "
                "real q25 | real q50 | real q75 | synth q25 | synth q50 | synth q75 |")
        lines.append(head)
        lines.append("|" + "---|" * 12)
        for group, kind, keys in (("per_image", "per-image", PER_IMAGE_FEATURES),
                                  ("pooled", "per-spot", POOLED_FEATURES)):
            for k in keys:
                e = block[group][k]
                flag = " *(degenerate)*" if e.get("degenerate") else ""
                lines.append("| {}{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    k, flag, kind, _fmt(e["ks_D"]),
                    "-" if e["ks_p"] is None else "{:.2e}".format(e["ks_p"]),
                    e["n_real"], e["n_synth"],
                    _fmt(e["real"]["q25"], 4), _fmt(e["real"]["q50"], 4),
                    _fmt(e["real"]["q75"], 4),
                    _fmt(e["synth"]["q25"], 4), _fmt(e["synth"]["q50"], 4),
                    _fmt(e["synth"]["q75"], 4)))
        lines.append("")
    return "\n".join(lines) + "\n"


def contact_sheet(real: Dict[str, Any], synth: Dict[str, Any],
                  summary: Dict[str, Any], path: Path,
                  thresholds: Sequence[float] = THRESHOLDS) -> None:
    """Overlaid real/synth histograms, one panel per feature, grouped by threshold."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feats = [(k, "per_image") for k in PER_IMAGE_FEATURES]
    feats += [(k, "pooled") for k in POOLED_FEATURES]
    ncol = 7
    per_thr_rows = int(np.ceil(len(feats) / float(ncol)))
    nrow = per_thr_rows * len(thresholds)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.45 * nrow))
    axes = np.atleast_2d(axes)

    for ti, t in enumerate(thresholds):
        key = "{:.2f}".format(t)
        block = summary["thresholds"][key]
        for fi, (name, group) in enumerate(feats):
            ax = axes[ti * per_thr_rows + fi // ncol, fi % ncol]
            lo, hi = PLOT_RANGE.get(name, (0.0, 1.0))
            bins = np.linspace(lo, hi, 41)
            r = _finite(real[group][t][name])
            s = _finite(synth[group][t][name])
            n_clip = int(((r < lo) | (r > hi)).sum() + ((s < lo) | (s > hi)).sum())
            for arr, colour, lab in ((r, "#3b6ea5", "real"), (s, "#c0563a", "synth")):
                if arr.size:
                    ax.hist(np.clip(arr, lo, hi), bins=bins, density=True,
                            color=colour, alpha=0.55, label=lab, edgecolor="none")
            e = block[group][name]
            ax.set_title("{} [{}]  D={}{}\nnR={} nS={}{}".format(
                name, "img" if group == "per_image" else "spot",
                _fmt(e["ks_D"]), " (degen)" if e.get("degenerate") else "",
                e["n_real"], e["n_synth"],
                ", {} clipped".format(n_clip) if n_clip else ""), fontsize=8)
            ax.tick_params(labelsize=6)
            if fi == 0:
                ax.set_ylabel("conf >= {}".format(key), fontsize=9)
                ax.legend(fontsize=6, loc="upper right")
        for fi in range(len(feats), per_thr_rows * ncol):
            axes[ti * per_thr_rows + fi // ncol, fi % ncol].axis("off")

    fig.suptitle(
        "Prototype 06 -- real vs synthetic OSEA-contract features   "
        "objective={:.4f}   geometry={:.4f}".format(
            summary["objective"]["value"], summary["geometry_objective"]["value"]),
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=105)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--real", type=Path,
                    default=here / "results" / "real" / "detections.jsonl")
    ap.add_argument("--synth", type=Path,
                    default=here / "results" / "synth_smoke" / "detections.jsonl")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default results/compare/<synth corpus name>)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    for p in (args.real, args.synth):
        if not p.is_file():
            print("ERROR: missing {}".format(p), file=sys.stderr)
            return 1
    out = args.out or (here / "results" / "compare" / args.synth.parent.name)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    real = load_side(args.real, limit=args.limit)
    synth = load_side(args.synth, limit=args.limit)
    summary = compare(real, synth)
    summary["run"]["seconds"] = round(time.time() - t0, 2)

    with open(str(out / "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    md = summary_markdown(summary)
    with open(str(out / "summary.md"), "w") as fh:
        fh.write(md)
    if not args.no_plot:
        contact_sheet(real, synth, summary, out / "compare_contact.png")

    print(md)
    print("wrote {}".format(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
