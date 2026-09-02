"""Benchmarks for the non-RGB constellation matcher.

Three modes, all writing ``results/constellation/<mode>_summary.json`` plus a
figure named ``<mode>_*_contact.png`` (the ``.gitignore`` keeps ``*_contact.png``):

  --toy    a controlled generator: N identities of 40-120 spots on the unit
           chart rectangle (s in [0,1] x r in [-1,1]); each sighting gets a
           random visible-extent crop, a smooth monotone s-warp, spot dropout,
           clutter, Gaussian jitter and a random (s, r) flip. Reports Rank-1 /
           Rank-5 / AUROC over a jitter x dropout grid. This is the only mode
           with ground truth, so it is the one the tests assert on.

  --real   detections.jsonl produced from catalog.db by the ingest agent.
           Same-individual vs different-individual score distributions over the
           tagged images. NOTE the leakage ceiling: 12 of the 13 same-individual
           pairs in catalog.db today are also same-encounter pairs, so a high
           same-individual score measures near-duplicate robustness, NOT
           re-identification. The 13th -- the only cross-encounter positive --
           is AOTB_A014 photographed left flank in 2019 and right flank in 2020,
           and an L-vs-R pair shares no spot pattern at all, so it belongs in
           the chance-floor population, not the positives. ``_pair_eval``
           therefore splits it out as ``opposite_side`` and the real arm has
           **zero** usable cross-encounter same-side positives; the reported
           Cohen's d / AUROC are a near-duplicate measurement and nothing more.
           The summary also splits the pairs by rectification frame (chart vs
           pca), because a chart-vs-pca pair is a weaker comparison than a
           same-frame one.

  --synth  truth.jsonl (image_id -> individual/date/side) + detections.jsonl for
           the renders. Identical maths to --real; the difference is that the
           truth is exact and cross-encounter same-side pairs actually exist, so
           this is the only arm that measures re-identification rather than
           near-duplicate robustness.

Every detections arm reports, beside the pair statistics:

  rank_same_side  leave-one-out closed-set identification. The gallery is every
                  other image of the SAME flank from a DIFFERENT encounter, and
                  the rank is that of the best correct entry with ties counted
                  against the query (``_rank_eval``). An L-vs-R pair of one
                  animal shares no spots, so an opposite-side gallery entry is
                  neither a retrievable target nor an honest distractor.
  drift           positive score by elapsed time between the two sightings
                  (0-6 months / 6-12 / 1-2 years / 2+), AUROC per bucket against
                  the whole negative population.
  opposite_side   the L-vs-R same-individual pairs on their own. An AUROC near
                  0.5 there is the CORRECT answer.

``--conf-min`` drops spots below a detector confidence before rectification,
which is where a threshold has to be applied: the frame is fitted to the spots
that remain.

Usage
-----
    P=".venv/bin/python"   # the MAIN checkout venv, py3.9
    $P eval_constellation.py --toy
    $P eval_constellation.py --toy --compare-models
    $P eval_constellation.py --real results/real/detections.jsonl
    $P eval_constellation.py --synth results/synth_calib/detections.jsonl \
        --truth results/synth_calib/truth.jsonl
    $P eval_constellation.py --synth results/synth_calib/detections.jsonl \
        --truth results/synth_calib/truth.jsonl --conf-min 0.40 --prefix synth_c40
    $P eval_constellation.py --synth-from-gt results/synth_calib/gt \
        --truth results/synth_calib/truth.jsonl --prefix synth_calib_gt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constellation import (  # noqa: E402
    DEFAULT_ASPECT, SpotSet, descriptors, extract_spotset, match_score)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "constellation")

JITTERS = (0.005, 0.01, 0.02, 0.04)
DROPOUTS = (0.0, 0.2, 0.4)


# --------------------------------------------------------------------------- #
# toy generator                                                                #
# --------------------------------------------------------------------------- #
def _monotone_warp(rng, amp):
    """A smooth monotone map of [0, 1] onto itself: s + a1 sin(pi s) + a2 sin(2 pi s).

    Coefficients are resampled until the derivative stays positive on a grid,
    so the warp can bend the arc-length parameterisation (the residual of a
    body bend + foreshortening) without ever reordering the spots.
    """
    grid = np.linspace(0.0, 1.0, 201)
    for _ in range(64):
        a1 = rng.uniform(-amp, amp)
        a2 = rng.uniform(-amp / 2, amp / 2)
        d = 1 + a1 * np.pi * np.cos(np.pi * grid) + a2 * 2 * np.pi * np.cos(2 * np.pi * grid)
        if d.min() > 0.05:
            f = grid + a1 * np.sin(np.pi * grid) + a2 * np.sin(2 * np.pi * grid)
            return lambda s, _f=f, _g=grid: np.interp(s, _g, _f)
    return lambda s: s


def make_identity(rng, n_min=40, n_max=120):
    """A base constellation: (n, 2) of (s, r) uniform on [0,1] x [-1,1]."""
    n = int(rng.integers(n_min, n_max + 1))
    return np.column_stack([rng.uniform(0, 1, n), rng.uniform(-1, 1, n)])


def make_sighting(rng, base, jitter=0.02, dropout=0.2, clutter=0.2,
                  warp_amp=0.03, crop=(1.0, 1.0), aspect=DEFAULT_ASPECT,
                  allow_flip=True, jitter_mode="chart", r_gamma_sd=0.0,
                  r_delta_sd=0.0):
    """One noisy view of ``base``. Returns (SpotSet, truth dict).

    Order of operations mirrors the physical chain: what is in frame (crop) ->
    how the body is bent/foreshortened (warp) -> what the detector found
    (dropout, clutter) -> where it put the box (jitter) -> which way round the
    animal happened to be (flip).

    ``jitter`` is a fraction of the chart extent: sd(ds) = jitter (of the body
    LENGTH) and, under the default ``jitter_mode="chart"``, sd(dr) = jitter (of
    the local HALF WIDTH). That is the right convention for this pipeline because
    the dominant per-spot error is *rectification* error -- an error in the
    centerline position or in the estimated local half width displaces r by a
    fraction of the half width, not by a fixed number of pixels. ``jitter_mode=
    "physical"`` gives the pixel-isotropic alternative, sd(dr) = jitter * aspect,
    which is the right model for pure detector box-centre noise; it is ``aspect``
    times harsher in r and is reported as a separate stress table.

    The brief's generator is jitter + dropout + clutter + flip + a mild s-warp;
    that is the default and it is what the headline table and the tests use.
    ``crop`` and ``r_gamma_sd`` / ``r_delta_sd`` are two further nuisances that
    the real photos do have, off by default and switched on together by
    ``--toy-hard``; they cost a lot (Rank-1 0.917 -> 0.733 for the crop alone,
    -> 0.850 for the r frame alone, -> 0.567 for both, at 2% jitter).

    ``crop`` is the fraction of the body length in frame: the sighting keeps
    ``s in [u0, u0+span]`` and renormalises it to [0, 1], so the two sightings of
    one animal differ by ``s' = alpha*s + beta`` with alpha up to 1/0.75.

    ``r_gamma_sd`` / ``r_delta_sd`` perturb the *frame* rather than the points:
    r -> gamma*r + delta with gamma ~ N(1, r_gamma_sd), delta ~ N(0, r_delta_sd).
    This is the residual of a half-width estimated from a hand-occluded mask
    (gamma) and of a laterally displaced centerline (delta). Without it a model
    that holds r rigid is flattered by a toy that never moves r.
    """
    s, r = base[:, 0].copy(), base[:, 1].copy()
    keep_idx = np.arange(len(s))

    span = rng.uniform(crop[0], crop[1])
    u0 = rng.uniform(0.0, 1.0 - span)
    sel = (s >= u0) & (s <= u0 + span)
    s, r, keep_idx = (s[sel] - u0) / span, r[sel], keep_idx[sel]

    s = np.clip(_monotone_warp(rng, warp_amp)(s), 0.0, 1.0)

    if dropout > 0 and len(s):
        keep = rng.random(len(s)) >= dropout
        if keep.sum() < 4:
            keep[rng.choice(len(s), size=min(4, len(s)), replace=False)] = True
        s, r, keep_idx = s[keep], r[keep], keep_idx[keep]

    r_sd = jitter * (aspect if jitter_mode == "physical" else 1.0)
    s = np.clip(s + rng.normal(0, jitter, len(s)), 0.0, 1.0)
    r = r + rng.normal(0, r_sd, len(r))
    gamma = float(np.clip(rng.normal(1.0, r_gamma_sd), 0.80, 1.25))
    delta = float(rng.normal(0.0, r_delta_sd))
    r = np.clip(gamma * r + delta, -1.0, 1.0)

    n_clutter = int(round(clutter * len(s)))
    if n_clutter:
        s = np.concatenate([s, rng.uniform(0, 1, n_clutter)])
        r = np.concatenate([r, rng.uniform(-1, 1, n_clutter)])
        keep_idx = np.concatenate([keep_idx, -np.ones(n_clutter, dtype=int)])

    flip = (bool(rng.integers(2)), bool(rng.integers(2))) if allow_flip else (False, False)
    if flip[0]:
        s = 1.0 - s
    if flip[1]:
        r = -r

    size = np.full(len(s), 0.02)
    conf = np.full(len(s), 0.9)
    truth = {"flip": flip, "crop": (float(u0), float(u0 + span)),
             "r_gamma": gamma, "r_delta": delta, "jitter_mode": jitter_mode,
             "n_true": int((keep_idx >= 0).sum()), "n_clutter": int(n_clutter),
             "base_idx": keep_idx.tolist()}
    return SpotSet(s, r, size, conf, frame="toy", aspect=aspect, meta={"truth": truth}), truth


def build_toy_corpus(n_ids=20, jitter=0.02, dropout=0.2, clutter=0.2, seed=0,
                     aspect=DEFAULT_ASPECT, **kw):
    """N identities x 2 sightings. Returns (gallery, queries, truth_flips)."""
    rng = np.random.default_rng(seed)
    gallery, queries, flips = [], [], []
    for _ in range(n_ids):
        base = make_identity(rng)
        g, tg = make_sighting(rng, base, jitter, dropout, clutter, aspect=aspect, **kw)
        q, tq = make_sighting(rng, base, jitter, dropout, clutter, aspect=aspect, **kw)
        gallery.append(g)
        queries.append(q)
        # match_score(query, gallery) flips the GALLERY set, so the flip that
        # brings it into the query's orientation is the XOR of the two.
        flips.append((tg["flip"][0] != tq["flip"][0], tg["flip"][1] != tq["flip"][1]))
    return gallery, queries, flips


# --------------------------------------------------------------------------- #
# metrics                                                                      #
# --------------------------------------------------------------------------- #
def auroc(pos, neg):
    """Rank-based AUROC with correct tie handling (Mann-Whitney U)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), float)
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    rsum = ranks[:len(pos)].sum()
    u = rsum - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def cohens_d(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    sp = np.sqrt(((len(pos) - 1) * pos.var(ddof=1) + (len(neg) - 1) * neg.var(ddof=1))
                 / (len(pos) + len(neg) - 2))
    return float((pos.mean() - neg.mean()) / max(sp, 1e-12))


def score_matrix(queries, gallery, **kw):
    """(n_q, n_g) score matrix plus the parallel flip / inlier matrices."""
    nq, ng = len(queries), len(gallery)
    sc = np.zeros((nq, ng))
    inl = np.zeros((nq, ng), int)
    flips = np.empty((nq, ng), object)
    for i, q in enumerate(queries):
        for j, g in enumerate(gallery):
            res = match_score(q, g, **kw)
            sc[i, j] = res["score"]
            inl[i, j] = res["n_inliers"]
            flips[i, j] = res["flip"]
    return sc, inl, flips


def rank_metrics(sc, truth_col):
    """Rank-1/Rank-5 given a score matrix and the true gallery column per query."""
    nq, ng = sc.shape
    r1 = r5 = 0
    ranks = []
    for i in range(nq):
        order = np.argsort(-sc[i], kind="mergesort")
        pos = int(np.where(order == truth_col[i])[0][0]) + 1
        ranks.append(pos)
        r1 += pos == 1
        r5 += pos <= 5
    return {"rank1": r1 / nq, "rank5": r5 / nq, "mean_rank": float(np.mean(ranks)),
            "n_query": nq, "n_gallery": ng}


# --------------------------------------------------------------------------- #
# --toy                                                                        #
# --------------------------------------------------------------------------- #
def run_toy(n_ids=20, seed=0, model="axis", clutter=0.2, jitters=JITTERS,
            dropouts=DROPOUTS, out_prefix="toy", gen=None):
    cells = []
    t0 = time.time()
    for dr in dropouts:
        for ji in jitters:
            g, q, tflips = build_toy_corpus(n_ids=n_ids, jitter=ji, dropout=dr,
                                            clutter=clutter, seed=seed, **(gen or {}))
            sc, inl, flips = score_matrix(q, g, model=model, seed=seed)
            truth_col = np.arange(n_ids)
            rm = rank_metrics(sc, truth_col)
            pos = sc[np.arange(n_ids), truth_col]
            neg = sc[~np.eye(n_ids, dtype=bool)]
            flip_ok = float(np.mean([flips[i, i] == tflips[i] for i in range(n_ids)]))
            cells.append(OrderedDict([
                ("jitter", ji), ("dropout", dr), ("clutter", clutter),
                ("rank1", rm["rank1"]), ("rank5", rm["rank5"]),
                ("mean_rank", rm["mean_rank"]),
                ("auroc", auroc(pos, neg)), ("cohens_d", cohens_d(pos, neg)),
                ("same_score_mean", float(pos.mean())),
                ("diff_score_mean", float(neg.mean())),
                ("diff_score_p99", float(np.percentile(neg, 99))),
                ("flip_recovered", flip_ok),
                ("mean_n_spots", float(np.mean([len(x) for x in q + g]))),
                ("median_nn", float(np.mean([descriptors(x)[1] for x in q + g]))),
                ("jitter_over_median_nn",
                 float(ji * np.sqrt(2.0) / np.mean([descriptors(x)[1] for x in q + g]))),
            ]))
            print("  jitter=%5.1f%% dropout=%3.0f%%  R1=%.2f R5=%.2f AUROC=%.3f "
                  "same=%.2f diff=%.3f flip=%.2f"
                  % (ji * 100, dr * 100, rm["rank1"], rm["rank5"], cells[-1]["auroc"],
                     pos.mean(), neg.mean(), flip_ok))
    summary = OrderedDict([
        ("mode", "toy"), ("model", model), ("n_ids", n_ids), ("clutter", clutter),
        ("seed", seed), ("aspect", DEFAULT_ASPECT), ("generator", gen or {}),
        ("seconds", round(time.time() - t0, 1)), ("cells", cells),
    ])
    _write(summary, out_prefix)
    _plot_toy(cells, out_prefix, jitters, dropouts)
    return summary


def compare_models(n_ids=20, seed=0, clutter=0.2, gen=None):
    """Head-to-head of the three geometric models at one operating point."""
    rows = []
    for model in ("axis", "s_affine", "sim"):
        cells = []
        t0 = time.time()
        for ji, dr in ((0.01, 0.2), (0.02, 0.2), (0.04, 0.4)):
            g, q, _ = build_toy_corpus(n_ids=n_ids, jitter=ji, dropout=dr,
                                       clutter=clutter, seed=seed, **(gen or {}))
            sc, _, _ = score_matrix(q, g, model=model, seed=seed)
            tc = np.arange(n_ids)
            rm = rank_metrics(sc, tc)
            pos, neg = sc[tc, tc], sc[~np.eye(n_ids, dtype=bool)]
            cells.append({"jitter": ji, "dropout": dr, "rank1": rm["rank1"],
                          "auroc": auroc(pos, neg),
                          "same": float(pos.mean()), "diff": float(neg.mean())})
        rows.append({"model": model, "seconds": round(time.time() - t0, 1),
                     "cells": cells,
                     "mean_rank1": float(np.mean([c["rank1"] for c in cells])),
                     "mean_auroc": float(np.mean([c["auroc"] for c in cells]))})
        print("  %-9s mean R1=%.3f mean AUROC=%.3f (%.1fs)"
              % (model, rows[-1]["mean_rank1"], rows[-1]["mean_auroc"], rows[-1]["seconds"]))
    out = {"mode": "model_comparison", "n_ids": n_ids, "clutter": clutter,
           "seed": seed, "models": rows}
    _write(out, "model_comparison")
    return out


# --------------------------------------------------------------------------- #
# --ablate                                                                     #
# --------------------------------------------------------------------------- #
#: The knob tables quoted in ``constellation.py``'s comments, as data.  Each row
#: is (group, label, match_score kwargs, build_toy_corpus kwargs).  Everything
#: not named in a row is left at the module default, so a row measures ONE knob.
#: ``HARD`` is the ``--toy-hard`` generator (a visible-extent crop and an
#: r-frame perturbation), the case that separates ``axis`` from ``s_affine``.
HARD = {"crop": (0.75, 1.0), "r_gamma_sd": 0.05, "r_delta_sd": 0.04}
ABLATIONS = (
    ("model", "axis (default)", {"model": "axis"}, {}),
    ("model", "s_affine", {"model": "s_affine"}, {}),
    ("model", "sim", {"model": "sim"}, {}),
    ("model_hard", "axis (default), hard", {"model": "axis"}, HARD),
    ("model_hard", "s_affine, hard", {"model": "s_affine"}, HARD),
    ("desc_gate", "quantile (default)", {"desc_gate": "quantile"}, {}),
    ("desc_gate", "rank_union t=3", {"desc_gate": "rank_union", "rank_t": 3}, {}),
    ("desc_gate", "rank_union t=5", {"desc_gate": "rank_union", "rank_t": 5}, {}),
    ("desc_gate", "none", {"desc_gate": "none"}, {}),
    ("seed_mode", "hungarian (default)", {"seed_mode": "hungarian"}, {}),
    ("seed_mode", "mutual", {"seed_mode": "mutual"}, {}),
    ("seed_mode", "top1", {"seed_mode": "top1"}, {}),
    ("aspect", "1 (default)", {}, {"aspect": 1.0}),
    ("aspect", "2", {}, {"aspect": 2.0}),
    ("aspect", "3", {}, {"aspect": 3.0}),
    ("k_nn", "k=12", {"k": 12}, {}),
    ("k_nn", "k=20 (default)", {"k": 20}, {}),
    ("k_nn", "k=30", {"k": 30}, {}),
    ("n_icp", "n_icp=1", {"n_icp": 1}, {}),
    ("n_icp", "n_icp=2", {"n_icp": 2}, {}),
    ("n_icp", "n_icp=4 (default)", {"n_icp": 4}, {}),
    ("n_icp", "n_icp=6", {"n_icp": 6}, {}),
    ("symmetry", "symmetric=True (default)", {"symmetric": True}, {}),
    ("symmetry", "symmetric=False (directed)", {"symmetric": False}, {}),
)


def run_ablate(n_ids=20, seeds=(0, 1, 2, 3, 4), jitter=0.02, dropout=0.2,
               clutter=0.2, model="axis", out_prefix="ablate"):
    """Regenerate the knob tables quoted in ``constellation.py``'s comments.

    Existence is the point: those tables were once measured under a *different*
    model than the shipped default and drifted silently (the ``_seed_pairs``
    numbers matched ``s_affine`` to four decimals and none of them matched
    ``axis``). Run this after touching any default and paste the output back.

    Protocol: ``len(seeds)`` seeds x ``n_ids`` identities x 2 sightings, scored
    all-against-all, at the stated jitter / dropout / clutter, with every
    parameter other than the one under test left at the module default.
    """
    rows = []
    t0 = time.time()
    print("ablations: %d seeds x %d identities, jitter=%.0f%% dropout=%.0f%% "
          "clutter=%.0f%%, model default=%s"
          % (len(seeds), n_ids, jitter * 100, dropout * 100, clutter * 100, model))
    group = None
    for grp, label, mkw, gkw in ABLATIONS:
        if grp != group:
            group = grp
            print("  [%s]" % grp)
        kw = dict(mkw)
        kw.setdefault("model", model)
        r1s, aurocs, diffs, sames = [], [], [], []
        for sd in seeds:
            g, q, _ = build_toy_corpus(n_ids=n_ids, jitter=jitter, dropout=dropout,
                                       clutter=clutter, seed=sd, **gkw)
            sc, _, _ = score_matrix(q, g, seed=sd, **kw)
            tc = np.arange(n_ids)
            r1s.append(rank_metrics(sc, tc)["rank1"])
            pos, neg = sc[tc, tc], sc[~np.eye(n_ids, dtype=bool)]
            aurocs.append(auroc(pos, neg))
            sames.append(float(pos.mean()))
            diffs.append(float(neg.mean()))
        rows.append(OrderedDict([
            ("group", grp), ("label", label), ("match_kwargs", kw),
            ("corpus_kwargs", gkw),
            ("rank1", float(np.mean(r1s))), ("rank1_per_seed", r1s),
            ("auroc", float(np.mean(aurocs))),
            ("same_score_mean", float(np.mean(sames))),
            ("diff_score_mean", float(np.mean(diffs))),
        ]))
        print("    %-26s R1=%.3f AUROC=%.4f same=%.3f diff=%.3f  per-seed %s"
              % (label, rows[-1]["rank1"], rows[-1]["auroc"],
                 rows[-1]["same_score_mean"], rows[-1]["diff_score_mean"],
                 " ".join("%.2f" % x for x in r1s)))
    out = OrderedDict([
        ("mode", "ablate"), ("model", model), ("n_ids", n_ids),
        ("seeds", list(seeds)), ("jitter", jitter), ("dropout", dropout),
        ("clutter", clutter), ("seconds", round(time.time() - t0, 1)),
        ("rows", rows),
    ])
    _write(out, out_prefix)
    return out


# --------------------------------------------------------------------------- #
# --real / --synth                                                             #
# --------------------------------------------------------------------------- #
def _to_detection(rec):
    """Normalise one jsonl row to the flat contract shape ``extract_spotset`` wants.

    Two layouts are accepted, because the ingest agent writes the second:
      flat    {"body_polygon": ..., "spots": ..., "width": ..., "height": ...}
      nested  {"image_id": ..., "individual_code": ..., "det": {"body_polygon": ...,
               "spots": ..., "image_width": ..., "image_height": ...}, "feats": {...}}
    Everything outside the detection (image_id, individual_code, encounter_id,
    side, filename) is carried through so the pair evaluation can label it.
    """
    det = rec.get("det") if isinstance(rec.get("det"), dict) else rec
    out = {k: v for k, v in rec.items() if k not in ("det", "feats")}
    out["body_polygon"] = det.get("body_polygon")
    out["obstruction_polygons"] = det.get("obstruction_polygons")
    out["spots"] = det.get("spots")
    out["width"] = det.get("width", det.get("image_width", rec.get("width")))
    out["height"] = det.get("height", det.get("image_height", rec.get("height")))
    return out


def detections_from_synth_gt(gt_dir, truth_path, out_path):
    """Build a contract-shaped detections.jsonl from prototype-05 render GROUND TRUTH.

    Each ``<image_id>.npz`` carries ``visible_skin`` (the body mask actually seen
    by the camera) and each ``<image_id>_spots.json`` the projected spot centres
    with ``visible`` and ``radius_px``. The body polygon is the silhouette of the
    visible-skin mask, scanned along its long axis (top edge one way, bottom edge
    back) -- a simple polygon of the kind the body detector emits.

    This is the CEILING for the synthetic arm, not a substitute for running the
    detector: the spot centres are exact, so any loss measured here is
    rectification and matching loss with zero detection error.
    """
    truth = {}
    with open(truth_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                t = json.loads(line)
                truth[str(t["image_id"])] = t
    n = 0
    with open(out_path, "w") as out:
        for image_id in sorted(truth):
            npz = os.path.join(gt_dir, "%s.npz" % image_id)
            spots_json = os.path.join(gt_dir, "%s_spots.json" % image_id)
            if not (os.path.exists(npz) and os.path.exists(spots_json)):
                continue
            with np.load(npz) as z:
                mask = np.asarray(z["visible_skin"], dtype=bool)
            poly = _mask_to_polygon(mask)
            if poly is None:
                continue
            with open(spots_json) as fh:
                spots = json.load(fh)
            keep, ids = [], []
            for sp in spots:
                if not sp.get("visible") or sp.get("cx") is None:
                    continue
                ids.append(int(sp["id"]))
                rad = float(sp.get("radius_px") or 0.0)
                keep.append({"x": round(float(sp["cx"]) - rad, 1),
                             "y": round(float(sp["cy"]) - rad, 1),
                             "w": round(2 * rad, 1), "h": round(2 * rad, 1),
                             "cx": round(float(sp["cx"]), 1),
                             "cy": round(float(sp["cy"]), 1), "conf": 1.0})
            t = truth[image_id]
            out.write(json.dumps({
                "image_id": image_id,
                "individual_code": t.get("identity"),
                "encounter_id": t.get("date"),
                "date": t.get("date"),
                "side": t.get("side"),
                "width": int(mask.shape[1]), "height": int(mask.shape[0]),
                "body_polygon": poly, "obstruction_polygons": None,
                "gt_spot_ids": ids, "spots": keep}) + "\n")
            n += 1
    print("wrote %s (%d records)" % (out_path, n))
    return out_path


def _mask_to_polygon(mask, step=4):
    """Silhouette polygon of a boolean mask, scanned along its longer axis."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 16:
        return None
    transpose = (ys.max() - ys.min()) > (xs.max() - xs.min())
    m = mask.T if transpose else mask
    cols = np.nonzero(m.any(axis=0))[0]
    cols = cols[::max(1, step)]
    top, bot = [], []
    for c in cols:
        rows = np.nonzero(m[:, c])[0]
        top.append((float(c), float(rows[0])))
        bot.append((float(c), float(rows[-1])))
    ring = top + bot[::-1]
    if transpose:
        ring = [(y, x) for x, y in ring]
    return [[round(x, 1), round(y, 1)] for x, y in ring]


def load_detections(path, limit=None, conf_min=None, **kw):
    """Read a detections.jsonl and rectify every record. Returns (sets, records, drops).

    ``conf_min`` drops spots below that detector confidence BEFORE rectification,
    which is the only correct place for it: ``extract_spotset`` normalises r by
    the body half width at each spot's station, so filtering afterwards would
    keep a frame fitted to spots that are no longer in the set.  The v1 spot
    head runs at a 0.25 floor and 46% of its real detections sit below 0.40
    (``results/real/summary.json``), so the threshold is a real knob, not a
    formality.
    """
    sets, records, drops = [], [], []
    n_spots_in = n_spots_kept = 0
    with open(path) as fh:
        for line_no, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            rec = _to_detection(json.loads(line))
            if conf_min is not None:
                spots = rec.get("spots") or []
                n_spots_in += len(spots)
                rec["spots"] = [sp for sp in spots
                                if float(sp.get("conf", 1.0)) >= float(conf_min)]
                n_spots_kept += len(rec["spots"])
            try:
                ss = extract_spotset(rec, **kw)
            except Exception as exc:  # noqa: BLE001
                drops.append({"line": line_no, "image_id": rec.get("image_id"),
                              "reason": "%s: %s" % (type(exc).__name__, str(exc)[:120])})
                continue
            if ss is None:
                drops.append({"line": line_no, "image_id": rec.get("image_id"),
                              "reason": "no body polygon or too few spots"})
                continue
            sets.append(ss)
            records.append(rec)
            if limit and len(sets) >= limit:
                break
    if conf_min is not None:
        # the filter's own bookkeeping, on the records rather than in ``drops``
        # (which counts IMAGES that could not be rectified, not spots)
        for r in records:
            r["conf_min"] = float(conf_min)
        load_detections.last_spot_filter = {
            "conf_min": float(conf_min), "spots_in": n_spots_in,
            "spots_kept": n_spots_kept,
            "kept_frac": (n_spots_kept / float(n_spots_in)) if n_spots_in else None}
    else:
        load_detections.last_spot_filter = None
    return sets, records, drops


def _population(pos, neg):
    """n / means / Cohen's d / AUROC for one (positive, negative) split."""
    return OrderedDict([
        ("n_same", len(pos)), ("n_diff", len(neg)),
        ("same_mean", float(np.mean(pos)) if pos else None),
        ("diff_mean", float(np.mean(neg)) if neg else None),
        ("cohens_d", cohens_d(pos, neg)), ("auroc", auroc(pos, neg)),
    ])


DRIFT_BUCKETS = ((0, 182, "0-6 months"), (183, 365, "6-12 months"),
                 (366, 730, "1-2 years"), (731, 10 ** 6, "2+ years"))


_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _as_date(v):
    """A calendar date from an ISO string or a POSIX timestamp, else None.

    Deliberately strict. ``np.datetime64("21")`` is the YEAR 21, so feeding it a
    catalog encounter id silently produces a two-decade recapture interval --
    which is exactly what happened on the first real run (AOTB_A014's pair came
    out at 6940 days from encounter ids 2 and 21). Only a string that starts
    ``YYYY-MM-DD`` or a number big enough to be a POSIX timestamp is accepted.
    """
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if float(v) < 1e8:            # not a plausible epoch second (1973-03-03)
            return None
        return (np.datetime64("1970-01-01") +
                np.timedelta64(int(round(float(v))), "s")).astype("datetime64[D]")
    if isinstance(v, str) and _ISO_DATE.match(v):
        try:
            return np.datetime64(v[:10])
        except ValueError:
            return None
    return None


def _elapsed_days(a, b):
    """|a - b| in days, or None when either side is not a usable date."""
    da, db = _as_date(a), _as_date(b)
    if da is None or db is None:
        return None
    return int(abs((da - db).astype("timedelta64[D]").astype(int)))


def _rank_eval(score, labels, sides, encs):
    """Closed-set identification, gallery restricted to the SAME flank.

    Leave-one-out over every image: the gallery is every OTHER image of the same
    side that is not from the same encounter.  Both restrictions are forced by
    what the score can mean.  An L photograph and an R photograph of one animal
    share no spot pattern, so an opposite-side gallery entry is neither a
    retrievable target nor an honest distractor; a same-encounter entry is a
    near-duplicate and retrieving it measures nothing (that is exactly the
    leakage that makes the real arm unusable).  A query whose gallery holds no
    same-individual entry is not scorable and is counted in ``n_unscorable``.

    The score is ``n_inliers / min(n_a, n_b)``, which is quantised, so ties at
    the top happen.  The headline rank is PESSIMISTIC -- every wrong gallery
    entry tied with the best correct one is counted ahead of it -- and the
    optimistic rank (ties counted behind) is reported beside it so the gap is
    visible rather than hidden in a sort order.
    """
    n = len(labels)
    ranks_p, ranks_o, n_unscorable, n_tied_top = [], [], 0, 0
    gallery_sizes = []
    for i in range(n):
        g = [j for j in range(n)
             if j != i and sides[i] and sides[j] and sides[i] == sides[j]
             and not (encs[i] is not None and encs[i] == encs[j])]
        correct = [j for j in g if labels[j] == labels[i]]
        if not g or not correct:
            n_unscorable += 1
            continue
        wrong = [j for j in g if labels[j] != labels[i]]
        best = max(score[i, j] for j in correct)
        n_gt = sum(1 for j in wrong if score[i, j] > best)
        n_eq = sum(1 for j in wrong if score[i, j] == best)
        ranks_p.append(1 + n_gt + n_eq)
        ranks_o.append(1 + n_gt)
        n_tied_top += int(n_eq > 0 and n_gt == 0)
        gallery_sizes.append(len(g))
    nq = len(ranks_p)
    if nq == 0:
        return OrderedDict([("n_query", 0), ("n_unscorable", n_unscorable)])
    rp, ro = np.asarray(ranks_p, float), np.asarray(ranks_o, float)
    return OrderedDict([
        ("protocol", "leave-one-out, gallery = same side, different encounter; "
                     "rank of the best correct entry; ties counted AGAINST"),
        ("n_query", int(nq)), ("n_unscorable", int(n_unscorable)),
        ("gallery_size_median", float(np.median(gallery_sizes))),
        ("rank1", float((rp <= 1).mean())), ("rank5", float((rp <= 5).mean())),
        ("rank10", float((rp <= 10).mean())),
        ("mean_rank", float(rp.mean())), ("median_rank", float(np.median(rp))),
        ("mrr", float((1.0 / rp).mean())),
        ("rank1_optimistic", float((ro <= 1).mean())),
        ("rank5_optimistic", float((ro <= 5).mean())),
        ("n_query_tied_at_top", int(n_tied_top)),
    ])


def _drift_eval(pos_pairs, neg):
    """Same-side positive score by elapsed time between the two sightings.

    The pattern drifts (05's ``drift.resight``: spots grow with the animal,
    a few appear, a few fade), so a 2-year recapture is a harder match than a
    2-week one.  AUROC per bucket is against the WHOLE negative population, so
    the buckets are comparable to each other and to the headline number.
    """
    out = OrderedDict()
    for lo, hi, name in DRIFT_BUCKETS:
        sc = [p["score"] for p in pos_pairs
              if p.get("elapsed_days") is not None and lo <= p["elapsed_days"] <= hi]
        out[name] = OrderedDict([
            ("days", [lo, hi if hi < 10 ** 6 else None]), ("n_pairs", len(sc)),
            ("mean_score", float(np.mean(sc)) if sc else None),
            ("median_score", float(np.median(sc)) if sc else None),
            ("auroc_vs_all_negatives", auroc(sc, neg) if sc and neg else None),
        ])
    known = [p["elapsed_days"] for p in pos_pairs if p.get("elapsed_days") is not None]
    out["n_pairs_without_dates"] = len(pos_pairs) - len(known)
    if known:
        out["elapsed_days"] = {"min": int(min(known)), "median": float(np.median(known)),
                               "max": int(max(known))}
    return out


def _pair_eval(sets, records, label_key, mode, model="axis", encounter_key="encounter_id"):
    """All-pairs scoring, split into the populations that are not comparable.

    Three splits are reported rather than one pooled number, because pooling
    them hides two real problems:

    * **frame**. ``extract_spotset`` emits either a prototype-02 ``chart`` frame
      or the ``pca`` fallback. Both now normalise r by a local half width, so
      they *are* comparable, but a cross-frame pair is still the weaker
      comparison (two different rectifications of the same animal) and its
      population is reported separately so that can be checked rather than
      assumed.
    * **flank**. A left-flank photo and a right-flank photo of one animal share
      no spot pattern; such a pair is a chance-floor draw wearing a positive
      label. They are counted in ``opposite_side`` and kept OUT of ``pos``.

    ``pos``/``neg`` (and the headline ``cohens_d``/``auroc``) are the
    same-side same-individual and different-individual pairs.
    """
    labels = [r.get(label_key) for r in records]
    idx = [i for i, lb in enumerate(labels) if lb]
    sets = [sets[i] for i in idx]
    labels = [labels[i] for i in idx]
    encs = [records[i].get(encounter_key) for i in idx]
    sides = [records[i].get("side") for i in idx]
    # NOT encounter_id: on the real corpus that is an integer catalog id, and
    # numpy would read it as a year. The real capture time is ``exif_ts``.
    dates = [records[i].get("date") or records[i].get("exif_ts") for i in idx]
    n = len(sets)
    score = np.full((n, n), np.nan)
    pos, neg, pos_pairs, opp_pairs = [], [], [], []
    same_enc_pos = 0
    cross_enc_same_side = 0
    by_frame = {"same_frame": ([], []), "cross_frame": ([], [])}
    for i in range(n):
        for j in range(i + 1, n):
            res = match_score(sets[i], sets[j], model=model)
            score[i, j] = score[j, i] = res["score"]
            bucket = "same_frame" if sets[i].frame == sets[j].frame else "cross_frame"
            same_ind = labels[i] == labels[j]
            opposite_side = (same_ind and sides[i] and sides[j] and sides[i] != sides[j])
            if same_ind:
                rec_a, rec_b = records[idx[i]], records[idx[j]]
                same_enc = encs[i] is not None and encs[i] == encs[j]
                entry = {"a": rec_a.get("image_id"), "b": rec_b.get("image_id"),
                         "individual": labels[i], "score": res["score"],
                         "n_inliers": res["n_inliers"], "flip": list(res["flip"]),
                         "direction": res.get("direction"),
                         "frames": [sets[i].frame, sets[j].frame],
                         "sides": [rec_a.get("side"), rec_b.get("side")],
                         "opposite_side": bool(opposite_side),
                         "same_encounter": same_enc,
                         "elapsed_days": _elapsed_days(dates[i], dates[j])}
                ga, gb = rec_a.get("gt_spot_ids"), rec_b.get("gt_spot_ids")
                if ga and gb:
                    # exact ceiling: the fraction of the smaller set that the two
                    # sightings actually share, from the render ground truth
                    shared = len(set(ga) & set(gb))
                    entry["gt_shared_spots"] = shared
                    entry["gt_ceiling"] = shared / float(min(len(ga), len(gb)))
                if opposite_side:
                    # an L-vs-R pair cannot be a spot match; it is a labelled
                    # chance-floor draw, so it is reported, not scored as a hit
                    opp_pairs.append(entry)
                    continue
                pos.append(res["score"])
                pos_pairs.append(entry)
                by_frame[bucket][0].append(res["score"])
                same_enc_pos += int(same_enc)
                cross_enc_same_side += int(not same_enc)
            else:
                neg.append(res["score"])
                by_frame[bucket][1].append(res["score"])
    neg_arr = np.asarray(neg, dtype=float)
    for e in opp_pairs:
        e["percentile_of_diff"] = (float((neg_arr < e["score"]).mean() * 100.0)
                                   if neg_arr.size else None)
    out = OrderedDict([
        ("mode", mode), ("model", model), ("n_images_used", n),
        ("n_individuals", len(set(labels))),
        ("n_same_pairs", len(pos)), ("n_diff_pairs", len(neg)),
        ("same_encounter_same_individual_pairs", same_enc_pos),
        ("cross_encounter_same_side_positives", cross_enc_same_side),
        ("opposite_side_same_individual_pairs", len(opp_pairs)),
        ("leakage_note",
         "all same-side same-individual pairs are same-encounter"
         if len(pos) and same_enc_pos == len(pos)
         else "%d/%d same-side same-individual pairs are same-encounter"
              % (same_enc_pos, len(pos))),
        ("usable_note",
         "%d cross-encounter same-side same-individual pair(s): %s"
         % (cross_enc_same_side,
            "the headline d/AUROC measure near-duplicate robustness only"
            if cross_enc_same_side == 0 else "some genuine re-ID evidence")),
        ("same_mean", float(np.mean(pos)) if pos else None),
        ("same_sd", float(np.std(pos, ddof=1)) if len(pos) > 1 else None),
        ("diff_mean", float(np.mean(neg)) if neg else None),
        ("diff_sd", float(np.std(neg, ddof=1)) if len(neg) > 1 else None),
        ("diff_p99", float(np.percentile(neg, 99)) if neg else None),
        ("cohens_d", cohens_d(pos, neg)), ("auroc", auroc(pos, neg)),
        ("by_frame", OrderedDict(
            (k, _population(*by_frame[k])) for k in ("same_frame", "cross_frame"))),
        ("spots_per_image_mean", float(np.mean([len(s) for s in sets])) if sets else 0.0),
        ("frames", {f: sum(1 for s in sets if s.frame == f) for f in ("chart", "pca")}),
        ("aspect_measured_median",
         float(np.median([s.meta.get("aspect_measured", np.nan) for s in sets]))
         if sets else None),
        ("rank_same_side", _rank_eval(score, labels, sides, encs)),
        ("drift", _drift_eval(pos_pairs, neg)),
        ("opposite_side", OrderedDict([
            ("n_pairs", len(opp_pairs)),
            ("mean_score", float(np.mean([e["score"] for e in opp_pairs]))
             if opp_pairs else None),
            ("auroc_vs_negatives",
             auroc([e["score"] for e in opp_pairs], neg) if opp_pairs and neg else None),
            ("note", "L-vs-R pairs of one animal: no shared spot pattern, so an "
                     "AUROC near 0.5 is the CORRECT answer, not a failure"),
        ])),
        ("opposite_side_pairs", sorted(opp_pairs, key=lambda d: -d["score"])[:20]),
        ("top_same_pairs", sorted(pos_pairs, key=lambda d: -d["score"])[:20]),
    ])
    return out, pos, neg


def run_frames(path, limit=None, out_prefix="real_frames", conf_min=None):
    """Rectify every record and tally which frame was used and why.

    No pair scoring, so it is cheap next to the full arm, and it is the source
    for the "the chart frame fails on real OSEA masks" claim: the fallback rate
    and the reasons behind it, over the whole corpus rather than the tagged
    subset the pair evaluation can use.
    """
    from collections import Counter

    sets, records, drops = load_detections(path, limit=limit, conf_min=conf_min)
    frames = Counter(ss.frame for ss in sets)
    reasons = Counter()
    chart_rejected = Counter()
    max_abs_r = []
    for ss in sets:
        seen = set()
        for d in ss.meta.get("drops") or ():
            reason = d.get("reason")
            if reason not in seen:
                seen.add(reason)
                reasons[reason] += 1
            if reason == "chart_frame_rejected":
                for why in (d.get("detail") or ()):
                    # the reasons carry their measured value in parentheses
                    # ("spots_outside_body(0.12)"); tally the kind, not the value
                    chart_rejected[str(why).split("(")[0]] += 1
        max_abs_r.append(float(ss.meta.get("max_abs_r_raw", 0.0)))
    arr = np.asarray(max_abs_r, dtype=float)
    out = OrderedDict([
        ("mode", "frames"), ("path", path),
        ("n_records_read", len(sets) + len(drops)),
        ("n_rectified", len(sets)), ("n_dropped", len(drops)),
        ("frames", dict(frames)),
        ("pca_fallback_frac", (frames["pca"] / float(len(sets))) if sets else None),
        ("images_with_drop_reason", dict(reasons)),
        ("chart_rejection_reasons", dict(chart_rejected)),
        ("max_abs_r_raw", {"min": float(arr.min()) if arr.size else None,
                           "median": float(np.median(arr)) if arr.size else None,
                           "max": float(arr.max()) if arr.size else None,
                           "n_over_1": int((arr > 1.0).sum())}),
        ("spots_per_image_mean", float(np.mean([len(s) for s in sets])) if sets else 0.0),
        ("dropped", drops[:50]),
    ])
    _write(out, out_prefix)
    return out


def run_real(path, model="axis", limit=None, out_prefix="real", conf_min=None):
    sets, records, drops = load_detections(path, limit=limit, conf_min=conf_min)
    out, pos, neg = _pair_eval(sets, records, "individual_code", "real", model=model)
    out["n_records_read"] = len(sets) + len(drops)
    out["dropped"] = drops[:50]
    out["n_dropped"] = len(drops)
    out["spot_filter"] = load_detections.last_spot_filter
    _write(out, out_prefix)
    _plot_hist(pos, neg, out_prefix, "real: same vs different individual")
    return out


def run_synth(path, truth_path, model="axis", limit=None, out_prefix="synth",
              conf_min=None):
    sets, records, drops = load_detections(path, limit=limit, conf_min=conf_min)
    truth = {}
    if truth_path and os.path.exists(truth_path):
        with open(truth_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    t = json.loads(line)
                    truth[str(t.get("image_id"))] = t
    for r in records:
        t = truth.get(str(r.get("image_id")), {})
        if t:
            if not r.get("individual_code"):
                r["individual_code"] = (t.get("individual_code") or t.get("identity")
                                        or t.get("individual"))
            if not r.get("encounter_id"):
                r["encounter_id"] = t.get("encounter_id") or t.get("date")
            if not r.get("date"):
                r["date"] = t.get("date")
            if not r.get("side"):
                r["side"] = t.get("side")
    out, pos, neg = _pair_eval(sets, records, "individual_code", "synth", model=model)
    out["n_records_read"] = len(sets) + len(drops)
    out["dropped"] = drops[:50]
    out["n_dropped"] = len(drops)
    out["truth_rows"] = len(truth)
    out["spot_filter"] = load_detections.last_spot_filter
    _write(out, out_prefix)
    _plot_hist(pos, neg, out_prefix, "synthetic: same vs different individual")
    return out


# --------------------------------------------------------------------------- #
# output                                                                       #
# --------------------------------------------------------------------------- #
def _write(obj, prefix):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "%s_summary.json" % prefix)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=float)
    print("wrote", path)
    return path


def _plot_toy(cells, prefix, jitters, dropouts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(RESULTS, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for dr in dropouts:
        sub = [c for c in cells if c["dropout"] == dr]
        x = [c["jitter"] * 100 for c in sub]
        axes[0].plot(x, [c["rank1"] for c in sub], "o-", label="dropout %.0f%%" % (dr * 100))
        axes[1].plot(x, [c["auroc"] for c in sub], "o-", label="dropout %.0f%%" % (dr * 100))
        axes[2].plot(x, [c["same_score_mean"] for c in sub], "o-",
                     label="same, dropout %.0f%%" % (dr * 100))
        axes[2].plot(x, [c["diff_score_mean"] for c in sub], "s--", alpha=0.6,
                     label="diff, dropout %.0f%%" % (dr * 100))
    for ax, ttl, yl in zip(axes, ("Rank-1", "AUROC", "mean score"),
                           ("rank-1 accuracy", "AUROC", "inliers / min(n_a, n_b)")):
        ax.set_xlabel("jitter sigma (% of body length in s,\n% of local half width in r)")
        ax.set_ylabel(yl)
        ax.set_title(ttl)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].set_ylim(0.45, 1.02)
    fig.suptitle("06-spot-proxy constellation matcher, toy benchmark "
                 "(%d identities, %.0f%% clutter, random flips)"
                 % (cells[0].get("n_ids", 20) if "n_ids" in cells[0] else 20,
                    cells[0]["clutter"] * 100), fontsize=10)
    fig.tight_layout()
    path = os.path.join(RESULTS, "%s_grid_contact.png" % prefix)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def _plot_hist(pos, neg, prefix, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(RESULTS, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, max(0.05, max(list(pos) + list(neg) + [0.05])), 30)
    if neg:
        ax.hist(neg, bins=bins, alpha=0.6, density=True, label="different (n=%d)" % len(neg))
    if pos:
        ax.hist(pos, bins=bins, alpha=0.6, density=True, label="same (n=%d)" % len(pos))
    ax.set_xlabel("score = inliers / min(n_a, n_b)")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(RESULTS, "%s_scores_contact.png" % prefix)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--compare-models", action="store_true")
    ap.add_argument("--ablate", action="store_true",
                    help="regenerate the knob tables quoted in constellation.py "
                         "(descriptor gate, RANSAC seed pool, aspect, symmetry) "
                         "under the SHIPPED defaults")
    ap.add_argument("--real", metavar="DETECTIONS_JSONL")
    ap.add_argument("--frames", metavar="DETECTIONS_JSONL",
                    help="rectify every record and report which frame was used "
                         "and why, with no pair scoring")
    ap.add_argument("--synth", metavar="DETECTIONS_JSONL")
    ap.add_argument("--truth", metavar="TRUTH_JSONL")
    ap.add_argument("--synth-from-gt", metavar="GT_DIR",
                    help="build a detections.jsonl from prototype-05 render ground "
                         "truth (needs --truth) and evaluate it; exact spot centres, "
                         "so this measures the rectification+matching ceiling")
    ap.add_argument("--model", default="axis", choices=("axis", "s_affine", "sim"))
    ap.add_argument("--toy-hard", action="store_true",
                    help="toy plus the two nuisances the brief does not ask for: "
                         "a 75-100%% visible-extent crop and an r-frame perturbation")
    ap.add_argument("--n-ids", type=int, default=20)
    ap.add_argument("--clutter", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--conf-min", type=float, default=None,
                    help="drop spots below this detector confidence BEFORE "
                         "rectification (the v1 spot head's own floor is 0.25)")
    args = ap.parse_args(argv)

    did = False
    if args.toy:
        print("toy benchmark (model=%s, %d identities)" % (args.model, args.n_ids))
        run_toy(n_ids=args.n_ids, seed=args.seed, model=args.model, clutter=args.clutter,
                out_prefix=args.prefix or "toy")
        did = True
    if args.toy_hard:
        print("toy benchmark, HARD generator (crop + r-frame)")
        run_toy(n_ids=args.n_ids, seed=args.seed, model=args.model, clutter=args.clutter,
                out_prefix=args.prefix or "toy_hard",
                gen={"crop": (0.75, 1.0), "r_gamma_sd": 0.05, "r_delta_sd": 0.04})
        did = True
    if args.ablate:
        run_ablate(n_ids=args.n_ids, clutter=args.clutter, model=args.model,
                   out_prefix=args.prefix or "ablate")
        did = True
    if args.compare_models:
        print("model comparison")
        compare_models(n_ids=args.n_ids, seed=args.seed, clutter=args.clutter)
        did = True
    if args.real:
        if not os.path.exists(args.real):
            print("MISSING: %s -- run the ingest agent first" % args.real)
            return 2
        print("real evaluation on", args.real)
        run_real(args.real, model=args.model, limit=args.limit,
                 out_prefix=args.prefix or "real", conf_min=args.conf_min)
        did = True
    if args.frames:
        if not os.path.exists(args.frames):
            print("MISSING: %s" % args.frames)
            return 2
        print("frame report on", args.frames)
        run_frames(args.frames, limit=args.limit,
                   out_prefix=args.prefix or "real_frames", conf_min=args.conf_min)
        did = True
    if args.synth_from_gt:
        if not (args.truth and os.path.exists(args.truth)):
            print("--synth-from-gt needs --truth <truth.jsonl>")
            return 2
        os.makedirs(RESULTS, exist_ok=True)
        built = os.path.join(RESULTS, "%s_detections.jsonl" % (args.prefix or "synth_gt"))
        detections_from_synth_gt(args.synth_from_gt, args.truth, built)
        run_synth(built, args.truth, model=args.model, limit=args.limit,
                  out_prefix=args.prefix or "synth_gt", conf_min=args.conf_min)
        did = True
    if args.synth:
        if not os.path.exists(args.synth):
            print("MISSING: %s -- run the render agent first" % args.synth)
            return 2
        print("synthetic evaluation on", args.synth)
        run_synth(args.synth, args.truth, model=args.model, limit=args.limit,
                  out_prefix=args.prefix or "synth", conf_min=args.conf_min)
        did = True
    if not did:
        ap.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
