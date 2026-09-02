"""Chart-space readout of a synthetic corpus -- the script behind every
chart-space number in README.md.

WHAT IT MEASURES
----------------
A render is a photograph of a curved animal from one arbitrary side, under one
arbitrary light.  Two renders of one shark do not line up pixel for pixel.  The
corpus, however, ships the renderer's own per-pixel chart ground truth
(``gt/<image_id>.npz``: ``chart_s`` / ``chart_phi``), so every image can be
UNWRAPPED back into the canonical ``(s, phi)`` chart it was painted in.  This
script does that and then asks four questions:

  (a) UNWRAP        scatter each selected pixel's albedo proxy into a chart
                    grid through its own ``(s, phi)``, average per cell, and
                    record which cells were covered at all;
  (b) SIMILARITY    NCC of two unwraps over the cells they JOINTLY cover and
                    that are not anatomically excluded, with a stated minimum
                    joint coverage below which the pair is UNDEFINED rather
                    than scored;
  (c) IDENTITY      one-shot open-set Rank-1 inside the corpus, using the SAME
                    split as ``prototypes/01-melops-ablation/protocol.py``
                    (imported, never reimplemented), matched within side;
  (d) DRIFT         the recapture-gap curve -- mean same-individual NCC per gap
                    bucket, its rank correlation with elapsed days, and its
                    correlation with the TRUE chart similarity
                    (``drift.similarity`` between the two generative states of
                    the animal, rebuilt deterministically from the run's seed).

READ THE HEADLINE AS A CEILING, AND AS A FAMILY OF NUMBERS
----------------------------------------------------------
This uses the ORACLE chart: the renderer's exact ``(s, phi)``, not an estimated
one.  It is what a perfect pose-and-chart estimator would reach, not what any
real pipeline reaches.

It is also not one number.  On the 40-animal demo corpus the Rank-1 this
reports spans 0.51-0.94 depending on choices that have nothing to do with the
corpus -- the readout
chart resolution, the minimum joint coverage, whether the identity mask is
honoured, and the chart-space high-pass radius -- so all four are CLI options
and ``--sensitivity`` sweeps them.  Quote the table, not a single cell.  (An earlier, unshipped version of
this measurement reported 0.807 with unstated settings; re-deriving it here is
exactly why the project rule says a README number must come from a script in
the repository.)

Usage
-----
    python chart_readout.py --data demo                       # default config
    python chart_readout.py --data demo --sensitivity         # + the sweep
    python chart_readout.py --data demo --chart-resolution 48x90
    python chart_readout.py --data demo --no-identity-mask
    python chart_readout.py --data demo --no-truth            # skip drift GT

Writes ``readout.json`` next to the corpus (override with ``--out``) and prints
markdown tables.  Deterministic: the only randomness is the split's tie-break
seed, which is an explicit ``--split-seed``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_MELOPS = os.path.abspath(os.path.join(_HERE, "..", "01-melops-ablation"))

import drift          # noqa: E402
import exclusions     # noqa: E402
import make_dataset   # noqa: E402
import pattern        # noqa: E402

try:
    from PIL import Image
except ImportError:                          # pragma: no cover - env has it
    Image = None


#: Default readout chart, ``(H_phi, W_s)``.  ``pattern.isotropic_resolution``
#: picks the width whose cells are square in the scaled chart metric, so a
#: round spot stays round; 128 rows puts the default spot diameter
#: (``PatternParams.radius_median`` * 2 ~ 0.011 s-units) across ~2.6 cells,
#: comfortably above Nyquist.
DEFAULT_CHART_RESOLUTION = pattern.isotropic_resolution(128)
#: A pair is scored only if the two unwraps jointly cover at least this
#: fraction of the non-excluded cells.  Below it the NCC is a few dozen cells
#: of one flank against another and is noise, so the pair is UNDEFINED.
DEFAULT_MIN_JOINT_COVERAGE = 0.05
#: Local-mean box radius for the chart-space high pass, as a fraction of the
#: chart WIDTH (cells are square, so the same cell radius is used on both
#: axes).  It exists to remove shading, veil and the countershading ramp --
#: which are low-frequency in chart space -- and NOT the spots, so the radius
#: must sit a small multiple above the spot diameter and well below the body:
#: 0.02 * 240 = 4.8 cells ~ 1.8 default spot diameters (2 *
#: ``PatternParams.radius_median`` = 0.011 s-units = 2.64 cells at 240).  It
#: moves the headline as much as the three flags do, so it is a CLI option and
#: it is swept in ``--sensitivity`` alongside them.
DEFAULT_HIGHPASS_FRAC = 0.02
#: Luminance weights (Rec. 709) -- the albedo proxy that gets unwrapped.
LUMA = (0.2126, 0.7152, 0.0722)
#: Recapture-gap buckets, in days.  Same edges as ``diagnose.py`` reports.
GAP_BUCKETS = ((0, 30), (31, 180), (181, 365), (366, 730), (731, 10 ** 9))


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

class ReadoutParams(object):
    """The three sensitivity flags plus the high pass, in one record."""

    __slots__ = ("chart_resolution", "min_joint_coverage", "use_identity_mask",
                 "highpass_frac")

    def __init__(self, chart_resolution=DEFAULT_CHART_RESOLUTION,
                 min_joint_coverage=DEFAULT_MIN_JOINT_COVERAGE,
                 use_identity_mask=True, highpass_frac=DEFAULT_HIGHPASS_FRAC):
        h, w = int(chart_resolution[0]), int(chart_resolution[1])
        if h < 4 or w < 4:
            raise ValueError("chart_resolution must be at least (4, 4), got %r"
                             % (chart_resolution,))
        if not (0.0 <= float(min_joint_coverage) <= 1.0):
            raise ValueError("min_joint_coverage must be in [0, 1], got %r"
                             % (min_joint_coverage,))
        if not (0.0 < float(highpass_frac) <= 0.5):
            raise ValueError("highpass_frac must be in (0, 0.5], got %r"
                             % (highpass_frac,))
        self.chart_resolution = (h, w)
        self.min_joint_coverage = float(min_joint_coverage)
        self.use_identity_mask = bool(use_identity_mask)
        self.highpass_frac = float(highpass_frac)

    def replace(self, **kw):
        cur = dict(chart_resolution=self.chart_resolution,
                   min_joint_coverage=self.min_joint_coverage,
                   use_identity_mask=self.use_identity_mask,
                   highpass_frac=self.highpass_frac)
        cur.update(kw)
        return ReadoutParams(**cur)

    def as_dict(self):
        return {"chart_resolution": [self.chart_resolution[0],
                                     self.chart_resolution[1]],
                "min_joint_coverage": self.min_joint_coverage,
                "use_identity_mask": self.use_identity_mask,
                "highpass_frac": self.highpass_frac}

    def label(self):
        return "%dx%d cells, min joint cov %.2f, hp %.2f, identity mask %s" % (
            self.chart_resolution[0], self.chart_resolution[1],
            self.min_joint_coverage, self.highpass_frac,
            "on" if self.use_identity_mask else "off")

    def __repr__(self):
        return "ReadoutParams(%s)" % self.label()


# ---------------------------------------------------------------------------
# (a) The unwrap
# ---------------------------------------------------------------------------

def _box_blur(values, radius):
    """Box mean of a chart array: periodic in phi (axis 0), clamped in s."""
    r = int(radius)
    if r <= 0:
        return np.array(values, dtype=np.float64)
    out = np.asarray(values, dtype=np.float64)
    h, w = out.shape
    # phi: wrap.  Tile enough to cover the kernel, then cumsum.
    k = min(r, h - 1)
    padded = np.concatenate([out[h - k:], out, out[:k]], axis=0)
    cs = np.cumsum(padded, axis=0)
    cs = np.concatenate([np.zeros((1, w)), cs], axis=0)
    idx = np.arange(h) + k
    out = cs[idx + k + 1] - cs[idx - k]
    # s: clamp (the chart is not periodic in s -- snout is not caudal).
    k = min(r, w - 1)
    padded = np.concatenate(
        [np.repeat(out[:, :1], k, axis=1), out, np.repeat(out[:, -1:], k, axis=1)],
        axis=1)
    cs = np.cumsum(padded, axis=1)
    cs = np.concatenate([np.zeros((h, 1)), cs], axis=1)
    idx = np.arange(w) + k
    return cs[:, idx + k + 1] - cs[:, idx - k]


def highpass(values, coverage, radius):
    """Subtract the local mean of the COVERED cells (normalised convolution).

    Shading, veil and the countershading ramp are low-frequency in chart
    space; the speckle is not.  Uncovered cells must not drag the local mean
    towards zero, so the box mean is computed as ``blur(v*c) / blur(c)`` and
    is only defined where a covered cell was seen inside the box.
    """
    cov = np.asarray(coverage, dtype=bool)
    v = np.where(cov, np.asarray(values, dtype=np.float64), 0.0)
    num = _box_blur(v, radius)
    den = _box_blur(cov.astype(np.float64), radius)
    local = np.zeros_like(num)
    ok = den > 0.0
    local[ok] = num[ok] / den[ok]
    return np.where(cov, np.asarray(values, dtype=np.float64) - local, 0.0)


def unwrap(rgb, chart_s, chart_phi, select, resolution, exclusion=None,
           highpass_frac=DEFAULT_HIGHPASS_FRAC):
    """Scatter selected pixels into a chart grid. Returns ``(values, coverage)``.

    ``values`` is the per-cell mean of the pixels that landed in it, high-passed
    in chart space; ``coverage`` is True where at least one pixel landed and the
    cell is not excluded.  A pixel's cell comes from its OWN ``(s, phi)`` ground
    truth, so no pose, camera or lighting model is involved -- this is the
    oracle unwrap.
    """
    h, w = int(resolution[0]), int(resolution[1])
    sel = np.asarray(select, dtype=bool) & np.isfinite(chart_s) & np.isfinite(chart_phi)
    values = np.zeros((h, w), dtype=np.float64)
    cov = np.zeros((h, w), dtype=bool)
    if sel.any():
        rgb = np.asarray(rgb, dtype=np.float64)
        luma = rgb[..., 0] * LUMA[0] + rgb[..., 1] * LUMA[1] + rgb[..., 2] * LUMA[2]
        s = np.clip(np.asarray(chart_s, dtype=np.float64)[sel], 0.0, 1.0 - 1e-9)
        phi = exclusions.wrap_phi(np.asarray(chart_phi, dtype=np.float64)[sel])
        si = np.minimum((s * w).astype(np.int64), w - 1)
        pi = np.minimum((((phi + math.pi) / (2.0 * math.pi)) * h).astype(np.int64), h - 1)
        flat = pi * w + si
        counts = np.bincount(flat, minlength=h * w)
        sums = np.bincount(flat, weights=luma[sel], minlength=h * w)
        counts = counts.reshape(h, w)
        cov = counts > 0
        values[cov] = sums.reshape(h, w)[cov] / counts[cov]
    if exclusion is not None:
        cov &= ~np.asarray(exclusion, dtype=bool)
        values = np.where(cov, values, 0.0)
    radius = max(1, int(round(float(highpass_frac) * w)))
    return highpass(values, cov, radius), cov


def exclusion_mask(resolution, stations=None):
    """The anatomical exclusion mask on the READOUT chart, dilated one cell.

    Same construction ``make_dataset`` uses at render time: the sampling mask
    grown by one cell so a nearest-neighbour lookup cannot leak an excluded
    pixel's chart coordinate into a scored cell.
    """
    schema = exclusions.load_schema(pattern.DEFAULT_SCHEMA_PATH)
    stations = exclusions.default_stations(schema) if stations is None else stations
    regions = exclusions.exclusion_regions(schema, stations=stations)
    return make_dataset.dilate_chart_mask(
        exclusions.mask_from_regions(regions, resolution), n_cells=1)


def band_masks(resolution, stations=None):
    """``{name: cell mask}`` for the whole body and the three anatomical bands.

    The cuts are the schema's own stations -- ``gill_slit_7_dorsal_origin``
    (the head cut ``make_dataset`` also cuts its head box at) and
    ``precaudal_pit`` -- which are exactly the spans ``--head-signal`` and
    ``--flank-signal`` drive, so an ablation of a knob is read out on the band
    that knob owns.
    """
    schema = exclusions.load_schema(pattern.DEFAULT_SCHEMA_PATH)
    stations = exclusions.default_stations(schema) if stations is None else stations
    head_s = float(stations["gill_slit_7_dorsal_origin"])
    tail_s = float(stations["precaudal_pit"])
    s_axis, _ = exclusions.chart_axes(resolution)
    S = np.repeat(s_axis[None, :], int(resolution[0]), axis=0)
    return {
        "whole body": np.ones(S.shape, dtype=bool),
        "head (s < %.2f)" % head_s: S < head_s,
        "trunk (%.2f <= s < %.2f)" % (head_s, tail_s): (S >= head_s) & (S < tail_s),
        "tail (s >= %.2f)" % tail_s: S >= tail_s,
    }


# ---------------------------------------------------------------------------
# (b) Similarity
# ---------------------------------------------------------------------------

def chart_ncc(a_val, a_cov, b_val, b_cov, min_cells, band=None):
    """Zero-mean NCC over the JOINTLY covered cells. ``(ncc, n_joint)``.

    ``ncc`` is NaN when the two unwraps share fewer than ``min_cells`` cells:
    an undefined pair is reported as undefined, never as a low score, because
    "these two images never saw the same skin" is not evidence that they are
    different animals.
    """
    joint = a_cov & b_cov
    if band is not None:
        joint = joint & band
    n = int(joint.sum())
    if n < int(min_cells) or n < 2:
        return float("nan"), n
    x = a_val[joint]
    y = b_val[joint]
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float(x.dot(x)) * float(y.dot(y)))
    if denom <= 0.0:
        return float("nan"), n
    return float(x.dot(y) / denom), n


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus(root):
    """``(metadata DataFrame, truth records by image_id, dataset.json or {})``."""
    import pandas as pd

    meta = pd.read_csv(os.path.join(root, "metadata.csv"))
    truth = {}
    truth_path = os.path.join(root, "truth.jsonl")
    if os.path.exists(truth_path):
        with open(truth_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    truth[rec["image_id"]] = rec
    summary = {}
    summary_path = os.path.join(root, "dataset.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
    return meta, truth, summary


def unwrap_corpus(root, meta, truth, params, excl):
    """Unwrap every image that has chart GT. ``(values, coverage, skipped)``.

    ``values`` / ``coverage`` are dicts keyed by ``image_id``; ``skipped``
    lists the images with no GT npz (``--no-gt`` corpora) or an empty
    selection mask, which are excluded from every metric and counted.
    """
    if Image is None:
        raise RuntimeError("Pillow is required to read the corpus images")
    values, coverage, skipped = {}, {}, []
    for image_id, path in zip(meta["image_id"].astype(str), meta["path"].astype(str)):
        rec = truth.get(image_id, {})
        gt_rel = rec.get("chart_gt_path") or os.path.join("gt", image_id + ".npz")
        gt_path = os.path.join(root, gt_rel)
        if not gt_rel or not os.path.exists(gt_path):
            skipped.append({"image_id": image_id, "reason": "no chart GT"})
            continue
        with np.load(gt_path) as gt:
            chart_s = np.asarray(gt["chart_s"], dtype=np.float64)
            chart_phi = np.asarray(gt["chart_phi"], dtype=np.float64)
            if params.use_identity_mask:
                select = np.asarray(gt["identity"], dtype=bool)
            else:
                select = np.asarray(gt["visible_skin"], dtype=bool)
        rgb = np.asarray(Image.open(os.path.join(root, path)).convert("RGB"),
                         dtype=np.float64) / 255.0
        val, cov = unwrap(rgb, chart_s, chart_phi, select,
                          params.chart_resolution, exclusion=excl,
                          highpass_frac=params.highpass_frac)
        if not cov.any():
            skipped.append({"image_id": image_id, "reason": "empty selection mask"})
            continue
        values[image_id] = val.astype(np.float32)
        coverage[image_id] = cov
    return values, coverage, skipped


# ---------------------------------------------------------------------------
# (c) One-shot open-set Rank-1
# ---------------------------------------------------------------------------

def _auroc(pos, neg):
    """Rank AUROC of ``pos`` over ``neg``; NaN if either side is empty."""
    pos = np.asarray([v for v in pos if np.isfinite(v)], dtype=np.float64)
    neg = np.asarray([v for v in neg if np.isfinite(v)], dtype=np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = _rank_average(allv)
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def _rank_average(values):
    """1-based average ranks with tie handling; numpy only."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 3:
        return float("nan")
    a = a[ok] - a[ok].mean()
    b = b[ok] - b[ok].mean()
    denom = math.sqrt(float(a.dot(a)) * float(b.dot(b)))
    return float(a.dot(b) / denom) if denom > 0 else float("nan")


def _spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 3:
        return float("nan")
    return _pearson(_rank_average(a[ok]), _rank_average(b[ok]))


def rank1_open_set(meta, values, coverage, params, min_cells, band=None,
                   split_seed=0, cutoff_fraction=0.5):
    """One-shot open-set Rank-1, side-partitioned, on the chart NCC.

    The split is ``protocol.one_shot_open_set_split`` IMPORTED from
    prototype 01 -- the same gallery, the same known/novel labelling and the
    same same-date exclusion the ablation runs under, so this number and the
    ablation's Rank-1 are commensurate.  Matching is within side: an
    ``(identity, side)`` unit is what is enrolled.

    Returns a dict with ``rank1``, ``chance``, the counts, the mean
    same/different-identity NCC, their separation, the pairwise AUROC and the
    number of queries no gallery entry was DEFINED against.
    """
    if _MELOPS not in sys.path:
        sys.path.insert(0, _MELOPS)
    import protocol  # noqa: E402  (prototype 01, imported not reimplemented)

    usable = meta[meta["image_id"].astype(str).isin(values)].reset_index(drop=True)
    gallery_df, query_df = protocol.one_shot_open_set_split(
        usable, cutoff_fraction=cutoff_fraction, seed=int(split_seed))
    g_ids = [str(v) for v in gallery_df["image_id"]]
    g_ident = [str(v) for v in gallery_df["identity"]]
    g_side = [str(v) for v in gallery_df["side"]]

    known = query_df[query_df["is_known"].astype(bool)].reset_index(drop=True)
    n_hit = 0
    n_scored = 0
    n_unresolved = 0
    chance_terms = []
    same_scores, diff_scores = [], []
    for _, row in known.iterrows():
        qid = str(row["image_id"])
        qident = str(row["identity"])
        qside = str(row["side"])
        best, best_ident, n_cand = None, None, 0
        for gid, gident, gside in zip(g_ids, g_ident, g_side):
            if gside != qside or gid == qid:
                continue
            score, _n = chart_ncc(values[qid], coverage[qid], values[gid],
                                  coverage[gid], min_cells, band=band)
            if not np.isfinite(score):
                continue
            n_cand += 1
            (same_scores if gident == qident else diff_scores).append(score)
            if best is None or score > best:
                best, best_ident = score, gident
        if best is None or n_cand == 0:
            n_unresolved += 1
            continue
        n_scored += 1
        chance_terms.append(1.0 / n_cand)
        if best_ident == qident:
            n_hit += 1
    same = np.asarray(same_scores, dtype=np.float64)
    diff = np.asarray(diff_scores, dtype=np.float64)
    return {
        "rank1": float(n_hit) / n_scored if n_scored else float("nan"),
        "n_hit": int(n_hit),
        "n_scored": int(n_scored),
        "n_known_queries": int(len(known)),
        "n_unresolved": int(n_unresolved),
        "n_gallery": int(len(gallery_df)),
        "n_novel_queries": int((~query_df["is_known"].astype(bool)).sum()),
        "n_same_date_excluded": int(query_df.attrs.get("n_same_date_excluded", 0)),
        "chance": float(np.mean(chance_terms)) if chance_terms else float("nan"),
        "mean_same_identity_ncc": float(same.mean()) if len(same) else float("nan"),
        "mean_diff_identity_ncc": float(diff.mean()) if len(diff) else float("nan"),
        "separation": (float(same.mean() - diff.mean())
                       if len(same) and len(diff) else float("nan")),
        "auroc": _auroc(same, diff),
        "cutoff_fraction": float(cutoff_fraction),
        "split_seed": int(split_seed),
    }


# ---------------------------------------------------------------------------
# (d) The recapture-gap curve
# ---------------------------------------------------------------------------

def _true_similarity_table(summary, identities):
    """``{(identity, date): Individual}`` rebuilt from the run's own seed.

    Returns ``None`` when ``dataset.json`` does not carry the arguments (an
    old or hand-assembled corpus): the true-similarity columns are then simply
    absent rather than guessed.
    """
    args = (summary or {}).get("args")
    if not args or "seed" not in args:
        return None
    context = make_dataset.build_pattern_context(
        head_signal=args.get("head_signal", 1.0),
        flank_signal=args.get("flank_signal", 1.0),
        n_spots=args.get("n_spots", 220),
        n_common=args.get("n_common", 40),
        chart_resolution=tuple(args.get("chart_resolution", (96, 192))))
    states = {}
    for index in range(int(summary.get("n_individuals", 0))):
        identity, _length, timeline = make_dataset.individual_timeline(
            context, args["seed"], index,
            sightings_per_individual=args.get("sightings_per_individual", 6),
            years=args.get("years", 4),
            start_date=args.get("start_date", "2019-03-01"))
        if identity not in identities:
            continue
        for date, _side, ind in timeline:
            states[(identity, str(date))] = ind
    return states or None


def recapture_gap(meta, values, coverage, params, min_cells, band=None,
                  states=None, buckets=GAP_BUCKETS):
    """Same-individual, same-flank pairs bucketed by elapsed days.

    Every ordered-once pair of images of one animal on one flank is scored.
    When ``states`` is available the TRUE chart similarity of the two
    generative states (``drift.similarity``) is scored alongside, which is the
    only way to ask whether the READOUT tracks the drift it was given.
    """
    rows = [(str(r["image_id"]), str(r["identity"]), str(r["side"]), str(r["date"]))
            for _, r in meta.iterrows() if str(r["image_id"]) in values]
    by_unit = {}
    for image_id, identity, side, date in rows:
        by_unit.setdefault((identity, side), []).append((date, image_id))

    pairs = []
    true_cache = {}
    for (identity, _side), items in sorted(by_unit.items()):
        items.sort()
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (d_a, id_a), (d_b, id_b) = items[i], items[j]
                score, n_joint = chart_ncc(values[id_a], coverage[id_a],
                                           values[id_b], coverage[id_b],
                                           min_cells, band=band)
                gap = abs(float(pattern.days_between(d_a, d_b)))
                true_ncc = float("nan")
                if states is not None:
                    key = (identity, d_a, d_b)
                    if key not in true_cache:
                        a = states.get((identity, d_a))
                        b = states.get((identity, d_b))
                        true_cache[key] = (drift.similarity(a, b)
                                           if a is not None and b is not None
                                           else float("nan"))
                    true_ncc = true_cache[key]
                pairs.append({"identity": identity, "image_a": id_a,
                              "image_b": id_b, "gap_days": gap,
                              "measured_ncc": score, "true_ncc": true_ncc,
                              "n_joint_cells": n_joint})

    gaps = np.array([p["gap_days"] for p in pairs], dtype=np.float64)
    meas = np.array([p["measured_ncc"] for p in pairs], dtype=np.float64)
    true = np.array([p["true_ncc"] for p in pairs], dtype=np.float64)
    scored = np.isfinite(meas)

    table = []
    for lo, hi in buckets:
        sel = (gaps >= lo) & (gaps <= hi)
        m = meas[sel & scored]
        t = true[sel & scored]
        table.append({
            "bucket": "%d-%d" % (lo, hi) if hi < 10 ** 9 else "%d+" % lo,
            "n_pairs": int(sel.sum()),
            "n_scored": int(len(m)),
            "mean_measured_ncc": float(m.mean()) if len(m) else float("nan"),
            "mean_true_ncc": (float(np.nanmean(t)) if len(t) and np.isfinite(t).any()
                              else float("nan")),
        })
    return {
        "buckets": table,
        "n_pairs": len(pairs),
        "n_scored_pairs": int(scored.sum()),
        "n_undefined_pairs": int((~scored).sum()),
        "median_joint_cells": (float(np.median([p["n_joint_cells"] for p in pairs]))
                               if pairs else float("nan")),
        "spearman_measured_vs_elapsed": _spearman(meas, gaps),
        "pearson_measured_vs_elapsed": _pearson(meas, gaps),
        "spearman_true_vs_elapsed": _spearman(true, gaps),
        "spearman_measured_vs_true": _spearman(meas, true),
        "pearson_measured_vs_true": _pearson(meas, true),
        "sd_measured_ncc": (float(np.nanstd(meas[scored])) if scored.any()
                            else float("nan")),
        "sd_true_ncc": (float(np.nanstd(true[scored])) if scored.any()
                        and np.isfinite(true[scored]).any() else float("nan")),
        "pairs": pairs,
    }


# ---------------------------------------------------------------------------
# The whole readout
# ---------------------------------------------------------------------------

def run(root, params=None, split_seed=0, cutoff_fraction=0.5, with_truth=True,
        with_bands=True, sensitivity=False, progress=False):
    """Compute the full readout for one corpus directory. Returns the record."""
    params = params or ReadoutParams()
    meta, truth, summary = load_corpus(root)
    excl = exclusion_mask(params.chart_resolution)
    n_valid_cells = int((~excl).sum())
    min_cells = int(math.ceil(params.min_joint_coverage * n_valid_cells))

    values, coverage, skipped = unwrap_corpus(root, meta, truth, params, excl)
    if not values:
        raise RuntimeError("no image in %r could be unwrapped (missing chart GT?)"
                           % (root,))
    cov_frac = [float(c.sum()) / max(n_valid_cells, 1) for c in coverage.values()]

    bands = band_masks(params.chart_resolution) if with_bands else {}
    identity = rank1_open_set(meta, values, coverage, params, min_cells,
                              split_seed=split_seed,
                              cutoff_fraction=cutoff_fraction)
    band_rows = []
    for name, mask in bands.items():
        n_cells = int((mask & ~excl).sum())
        band_cov = float(np.mean([float((c & mask).sum()) / max(n_cells, 1)
                                  for c in coverage.values()]))
        if name == "whole body":
            res = identity
        else:
            band_min = int(math.ceil(params.min_joint_coverage * n_cells))
            res = rank1_open_set(meta, values, coverage, params, band_min,
                                 band=mask, split_seed=split_seed,
                                 cutoff_fraction=cutoff_fraction)
        band_rows.append({"band": name, "n_cells": n_cells,
                          "mean_coverage": band_cov,
                          **{k: res[k] for k in ("rank1", "n_hit", "n_scored",
                                                 "n_unresolved", "chance")}})

    states = None
    if with_truth:
        states = _true_similarity_table(summary,
                                        set(meta["identity"].astype(str)))
    gap = recapture_gap(meta, values, coverage, params, min_cells, states=states)

    record = {
        "corpus": os.path.abspath(root),
        "params": params.as_dict(),
        "n_images_in_metadata": int(len(meta)),
        "n_unwrapped": len(values),
        "n_skipped": len(skipped),
        "skipped": skipped[:20],
        "n_valid_cells": n_valid_cells,
        "min_joint_cells": min_cells,
        "coverage_fraction": {
            "mean": float(np.mean(cov_frac)),
            "min": float(np.min(cov_frac)),
            "max": float(np.max(cov_frac)),
        },
        "identity": identity,
        "bands": band_rows,
        "recapture_gap": {k: v for k, v in gap.items() if k != "pairs"},
        "true_similarity_available": states is not None,
        "corpus_args": (summary or {}).get("args", {}),
    }
    if sensitivity:
        record["sensitivity"] = sensitivity_sweep(
            root, params, split_seed=split_seed, cutoff_fraction=cutoff_fraction,
            progress=progress)
    return record


def sensitivity_grid(params):
    """One-at-a-time variations of every knob the headline moves with.

    Each row changes exactly one setting from ``params``, so a difference in
    the swept table is attributable.  Duplicates (a variation that lands back
    on the default) are dropped, and ``params`` itself is always the first row.
    """
    grid = []
    for rows in (48, 96, 128, 192):
        grid.append(params.replace(chart_resolution=pattern.isotropic_resolution(rows)))
    for cov in (0.02, 0.05, 0.10, 0.20):
        grid.append(params.replace(min_joint_coverage=cov))
    for frac in (0.01, 0.02, 0.04, 0.08):
        grid.append(params.replace(highpass_frac=frac))
    grid.append(params.replace(use_identity_mask=not params.use_identity_mask))
    seen, out = set(), []
    for p in [params] + grid:
        key = (p.chart_resolution, p.min_joint_coverage, p.use_identity_mask,
               p.highpass_frac)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def sensitivity_sweep(root, params, split_seed=0, cutoff_fraction=0.5,
                      progress=False):
    """Rank-1 and separation across the grid. One row per configuration."""
    meta, truth, _summary = load_corpus(root)
    rows = []
    for p in sensitivity_grid(params):
        excl = exclusion_mask(p.chart_resolution)
        n_valid = int((~excl).sum())
        min_cells = int(math.ceil(p.min_joint_coverage * n_valid))
        values, coverage, _skipped = unwrap_corpus(root, meta, truth, p, excl)
        res = rank1_open_set(meta, values, coverage, p, min_cells,
                             split_seed=split_seed,
                             cutoff_fraction=cutoff_fraction)
        rows.append({"params": p.as_dict(), "label": p.label(),
                     "n_valid_cells": n_valid, "min_joint_cells": min_cells,
                     **{k: res[k] for k in ("rank1", "n_hit", "n_scored",
                                            "n_unresolved", "separation",
                                            "auroc", "chance")}})
        if progress:
            print("  %-46s rank1=%.3f" % (p.label(), res["rank1"]),
                  file=sys.stderr)
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(value, digits=3):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return ("%." + str(digits) + "f") % float(value)


def markdown(record):
    """The tables README.md quotes, verbatim."""
    out = []
    p = record["params"]
    out.append("### Chart-space readout -- `%s`" % record["corpus"])
    out.append("")
    out.append("Config: %d x %d cells (H_phi x W_s), min joint coverage %.2f "
               "(%d of %d non-excluded cells), identity mask %s, high-pass "
               "radius %.2f x W_s."
               % (p["chart_resolution"][0], p["chart_resolution"][1],
                  p["min_joint_coverage"], record["min_joint_cells"],
                  record["n_valid_cells"],
                  "on" if p["use_identity_mask"] else "off", p["highpass_frac"]))
    out.append("")
    ident = record["identity"]
    out.append("| quantity | value |")
    out.append("|---|---|")
    out.append("| images unwrapped | %d of %d |"
               % (record["n_unwrapped"], record["n_images_in_metadata"]))
    out.append("| mean chart coverage per image | %s of non-excluded cells |"
               % _fmt(record["coverage_fraction"]["mean"]))
    out.append("| gallery / known / novel | %d / %d / %d |"
               % (ident["n_gallery"], ident["n_known_queries"],
                  ident["n_novel_queries"]))
    out.append("| **one-shot open-set Rank-1 (same flank)** | **%s** (%d/%d) |"
               % (_fmt(ident["rank1"]), ident["n_hit"], ident["n_scored"]))
    out.append("| chance | %s |" % _fmt(ident["chance"]))
    out.append("| queries with no defined gallery pair | %d |"
               % ident["n_unresolved"])
    out.append("| mean same-identity NCC | %s |"
               % _fmt(ident["mean_same_identity_ncc"]))
    out.append("| mean different-identity NCC | %s |"
               % _fmt(ident["mean_diff_identity_ncc"]))
    out.append("| separation | %s |" % _fmt(ident["separation"]))
    out.append("| pairwise AUROC | %s |" % _fmt(ident["auroc"]))
    out.append("")

    if record.get("bands"):
        out.append("| band | cells | mean coverage | Rank-1 | n | undefined "
                   "queries | chance |")
        out.append("|---|---|---|---|---|---|---|")
        for row in record["bands"]:
            out.append("| %s | %d | %s | %s | %d/%d | %d | %s |"
                       % (row["band"], row["n_cells"],
                          _fmt(row["mean_coverage"]), _fmt(row["rank1"]),
                          row["n_hit"], row["n_scored"], row["n_unresolved"],
                          _fmt(row["chance"])))
        out.append("")

    gap = record["recapture_gap"]
    out.append("| gap (days) | pairs | scored | measured NCC | TRUE chart NCC |")
    out.append("|---|---|---|---|---|")
    for row in gap["buckets"]:
        out.append("| %s | %d | %d | %s | %s |"
                   % (row["bucket"], row["n_pairs"], row["n_scored"],
                      _fmt(row["mean_measured_ncc"]),
                      _fmt(row["mean_true_ncc"])))
    out.append("")
    out.append("- same-flank resight pairs: %d (%d scored, %d undefined below "
               "the coverage floor)"
               % (gap["n_pairs"], gap["n_scored_pairs"], gap["n_undefined_pairs"]))
    out.append("- Spearman(measured NCC, elapsed days) = %s   "
               "(Pearson %s)" % (_fmt(gap["spearman_measured_vs_elapsed"]),
                                 _fmt(gap["pearson_measured_vs_elapsed"])))
    out.append("- Spearman(TRUE chart NCC, elapsed days) = %s"
               % _fmt(gap["spearman_true_vs_elapsed"]))
    out.append("- Spearman(measured, TRUE) = %s   (Pearson %s)"
               % (_fmt(gap["spearman_measured_vs_true"]),
                  _fmt(gap["pearson_measured_vs_true"])))
    out.append("- sd of measured NCC %s vs sd of TRUE chart NCC %s"
               % (_fmt(gap["sd_measured_ncc"]), _fmt(gap["sd_true_ncc"])))
    out.append("")

    if record.get("sensitivity"):
        out.append("| chart cells | min joint cov | high-pass | identity mask "
                   "| Rank-1 | n | separation | AUROC |")
        out.append("|---|---|---|---|---|---|---|---|")
        for row in record["sensitivity"]:
            rp = row["params"]
            out.append("| %d x %d | %.2f | %.2f | %s | %s | %d/%d | %s | %s |"
                       % (rp["chart_resolution"][0], rp["chart_resolution"][1],
                          rp["min_joint_coverage"], rp["highpass_frac"],
                          "on" if rp["use_identity_mask"] else "off",
                          _fmt(row["rank1"]), row["n_hit"], row["n_scored"],
                          _fmt(row["separation"]), _fmt(row["auroc"])))
        out.append("")
    return "\n".join(out)


def _resolution(text):
    parts = str(text).lower().replace(",", "x").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("resolution must be H_phixW_s, got %r"
                                         % (text,))
    return (int(parts[0]), int(parts[1]))


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="a make_dataset.py output dir")
    p.add_argument("--chart-resolution", type=_resolution,
                   default=DEFAULT_CHART_RESOLUTION,
                   help="readout chart H_phixW_s (default %dx%d)"
                        % DEFAULT_CHART_RESOLUTION)
    p.add_argument("--min-joint-coverage", type=float,
                   default=DEFAULT_MIN_JOINT_COVERAGE,
                   help="a pair is UNDEFINED below this fraction of the "
                        "non-excluded cells (default %.2f)"
                        % DEFAULT_MIN_JOINT_COVERAGE)
    p.add_argument("--no-identity-mask", action="store_true",
                   help="unwrap every visible skin pixel instead of only the "
                        "render-time identity mask")
    p.add_argument("--highpass-frac", type=float, default=DEFAULT_HIGHPASS_FRAC,
                   help="chart-space high-pass box radius, as a fraction of "
                        "the chart width (default %.2f). Too large leaves "
                        "shading in; too small eats the spots"
                        % DEFAULT_HIGHPASS_FRAC)
    p.add_argument("--split-seed", type=int, default=0,
                   help="tie-break seed handed to protocol.one_shot_open_set_split")
    p.add_argument("--cutoff-fraction", type=float, default=0.5)
    p.add_argument("--no-truth", action="store_true",
                   help="skip the drift.similarity ground truth (faster)")
    p.add_argument("--no-bands", action="store_true",
                   help="skip the head/trunk/tail band table")
    p.add_argument("--sensitivity", action="store_true",
                   help="also sweep chart resolution, coverage floor and the "
                        "identity mask")
    p.add_argument("--out", default=None,
                   help="JSON output path (default <data>/readout.json)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    params = ReadoutParams(chart_resolution=args.chart_resolution,
                           min_joint_coverage=args.min_joint_coverage,
                           use_identity_mask=not args.no_identity_mask,
                           highpass_frac=args.highpass_frac)
    record = run(args.data, params=params, split_seed=args.split_seed,
                 cutoff_fraction=args.cutoff_fraction,
                 with_truth=not args.no_truth, with_bands=not args.no_bands,
                 sensitivity=args.sensitivity, progress=not args.quiet)
    out_path = args.out or os.path.join(args.data, "readout.json")
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    print(markdown(record))
    print("wrote %s" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
