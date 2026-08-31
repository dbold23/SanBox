"""Run-2 diagnostics for the Melops ablation (see results/ANALYSIS.md).

Run 1 was INCONCLUSIVE: zero-shot Rank-1 sat at 1-2 points and open-set AUROC
was BELOW 0.5 on every same-side arm. Before any re-run is interpreted, this
CLI answers four questions on one split (same defaults as ``run_ablation.py``:
``one_shot_open_set_split`` with ``cutoff_fraction=0.5``,
``same_date_policy="exclude"``):

a. RECAPTURE-GAP CURVE -- true-mate similarity and Rank-1 vs elapsed days
   between a known query and its unit's gallery image. Doubles as a
   pattern-stability measurement (how fast the individual pattern signal
   decays with time), which is independently valuable.
b. AUROC STRATA BY QUERY YEAR -- tests ANALYSIS.md's temporal-confound
   hypothesis for the below-0.5 pooled AUROC (known queries span all years,
   novel queries are late-years only). Emits an automated one-line reading.
c. SMALL-GALLERY CALIBRATION -- K enrolled + K novel units, 3 seeds, for
   comparability with the NLDL 2023 one-shot 0.35 anchor (arXiv:2301.00596,
   which used a far smaller gallery than our full 10k-unit one).
d. CROP CONTACT SHEET -- a seeded-random PIL grid per requested arm with the
   bbox applied: the human check that head crops actually contain heads.

Usage:
    python diagnose.py --data melops|synthetic --root PATH --backbone NAME \\
        --arm body --out DIR [--emb-cache DIR] [--seed 0]

Outputs ``diagnostics.json`` and ``diagnostics.md`` plus one
``contact_sheet_{arm}.png`` per sheet arm in ``--out``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

import emb_cache
import embedders
import melops_data
import protocol

CROP_ARMS = ("head", "body", "headless")

GAP_BUCKETS = (
    ("0-30", 0, 30),
    ("31-180", 31, 180),
    ("181-365", 181, 365),
    ("366-730", 366, 730),
    ("731+", 731, None),
)

READING_NO_INVERSION = (
    "pooled AUROC >= 0.5 - no below-chance inversion to explain; "
    "temporal-confound test not applicable"
)
READING_CONFOUND = (
    "within-year AUROC >= 0.5 everywhere while pooled < 0.5 "
    "=> temporal confound confirmed"
)
READING_NOT_TEMPORAL = (
    "below 0.5 even within years => confound NOT temporal, investigate further"
)
READING_UNDEFINED = (
    "no year stratum contains both known and novel queries - "
    "the stratified test cannot run"
)

# Contact-sheet cell geometry (fixed so the sheet size is deterministic).
SHEET_CELL_W = 120
SHEET_CELL_H = 80


# ---------------------------------------------------------------------------
# Per-query retrieval details (side-partitioned, same rules as protocol.evaluate)
# ---------------------------------------------------------------------------


def per_query_details(gallery_emb, gallery_df, query_emb, query_df):
    """Per-query max-sim / true-mate-sim / rank under the same-side protocol.

    Mirrors ``protocol.evaluate``'s matching rules (side-partitioned,
    L2-normalized cosine) but returns per-query arrays instead of aggregates:
    ``max_sim`` (-inf for a novel query on an un-enrolled side), ``true_sim``
    (NaN for novel), ``rank`` (0 for novel), ``is_known``. Raises
    ``ProtocolViolation`` on the same misalignments ``evaluate`` would.
    """
    gallery_emb = np.asarray(gallery_emb, dtype=np.float64)
    query_emb = np.asarray(query_emb, dtype=np.float64)
    if gallery_emb.shape[0] != len(gallery_df):
        raise protocol.ProtocolViolation("gallery embeddings misaligned with gallery frame")
    if query_emb.shape[0] != len(query_df):
        raise protocol.ProtocolViolation("query embeddings misaligned with query frame")
    for emb, label in ((gallery_emb, "gallery"), (query_emb, "query")):
        norms = np.linalg.norm(emb, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-6):
            raise protocol.ProtocolViolation("%s embeddings are not L2-normalized" % label)
    if "is_known" not in query_df.columns:
        raise protocol.ProtocolViolation("query frame missing is_known")

    gallery_sides = gallery_df["side"].to_numpy()
    gallery_identities = gallery_df["identity"].to_numpy()
    n_query = len(query_df)
    max_sims = np.full(n_query, -np.inf)
    true_sims = np.full(n_query, np.nan)
    ranks = np.zeros(n_query, dtype=np.int64)
    is_known = query_df["is_known"].to_numpy().astype(bool)

    for qi in range(n_query):
        q_side = query_df["side"].iloc[qi]
        cols = np.flatnonzero(gallery_sides == q_side)
        if len(cols) == 0:
            if is_known[qi]:
                raise protocol.ProtocolViolation(
                    "known query on side %r has no same-side gallery" % q_side
                )
            continue
        sims = query_emb[qi] @ gallery_emb[cols].T
        max_sims[qi] = float(sims.max())
        if is_known[qi]:
            q_identity = query_df["identity"].iloc[qi]
            match = (gallery_identities[cols] == q_identity) & (gallery_sides[cols] == q_side)
            true_pos = np.flatnonzero(match)
            if len(true_pos) != 1:
                raise protocol.ProtocolViolation(
                    "known query %r must have exactly one gallery match, got %d"
                    % (q_identity, len(true_pos))
                )
            true_sims[qi] = float(sims[true_pos[0]])
            ranks[qi] = int((sims > sims[true_pos[0]]).sum()) + 1

    return {"max_sim": max_sims, "true_sim": true_sims, "rank": ranks, "is_known": is_known}


# ---------------------------------------------------------------------------
# a. Recapture-gap curve
# ---------------------------------------------------------------------------


def recapture_gap_section(details, gallery_df, query_df):
    """Bucket known queries by elapsed days since their gallery image."""
    g_dates = pd.to_datetime(gallery_df["date"])
    unit_date = {}
    for i in range(len(gallery_df)):
        unit_date[(gallery_df["identity"].iloc[i], gallery_df["side"].iloc[i])] = g_dates.iloc[i]
    q_dates = pd.to_datetime(query_df["date"])
    is_known = details["is_known"]

    gaps = np.full(len(query_df), np.nan)
    for qi in np.flatnonzero(is_known):
        key = (query_df["identity"].iloc[qi], query_df["side"].iloc[qi])
        gap = int((q_dates.iloc[qi] - unit_date[key]).days)
        if gap < 0:
            raise protocol.ProtocolViolation(
                "known query %r predates its gallery image (gap %d days)" % (key, gap)
            )
        gaps[qi] = gap

    buckets = []
    for label, lo, hi in GAP_BUCKETS:
        mask = is_known & (gaps >= lo)
        if hi is not None:
            mask &= gaps <= hi
        n = int(mask.sum())
        buckets.append(
            {
                "bucket_days": label,
                "n": n,
                "mean_true_mate_sim": float(details["true_sim"][mask].mean()) if n else None,
                "rank1": float((details["rank"][mask] == 1).mean()) if n else None,
            }
        )
    return {"n_known": int(is_known.sum()), "buckets": buckets}


# ---------------------------------------------------------------------------
# b. AUROC strata by query year
# ---------------------------------------------------------------------------


def _finite_mean(values):
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else None


def temporal_reading(pooled_auroc, strata):
    """One-line automated reading of the temporal-confound test.

    Strata with only one class (AUROC undefined) cannot support or refute the
    reading; when any exist, the returned line says how many were excluded so
    "everywhere" is never quietly overstated.
    """
    valid = [s["auroc"] for s in strata if s["auroc"] is not None]
    n_untestable = sum(1 for s in strata if s["auroc"] is None)
    if pooled_auroc is None or not valid:
        return READING_UNDEFINED
    if pooled_auroc >= 0.5:
        reading = READING_NO_INVERSION
    elif all(a >= 0.5 for a in valid):
        reading = READING_CONFOUND
    else:
        reading = READING_NOT_TEMPORAL
    if n_untestable:
        reading += " [%d of %d strata untestable (single-class) and excluded]" % (
            n_untestable, len(strata))
    return reading


def year_strata_section(details, query_df):
    """Per-query-year open-set AUROC on max similarity, plus pooled + reading."""
    years = pd.to_datetime(query_df["date"]).dt.year.to_numpy()
    is_known = details["is_known"]
    max_sim = details["max_sim"]

    strata = []
    for year in sorted(set(years.tolist())):
        in_year = years == year
        known_sims = max_sim[in_year & is_known]
        novel_sims = max_sim[in_year & ~is_known]
        strata.append(
            {
                "year": int(year),
                "n_known": int(len(known_sims)),
                "n_novel": int(len(novel_sims)),
                "auroc": protocol._auroc(known_sims, novel_sims),
                "mean_max_sim_known": _finite_mean(known_sims),
                "mean_max_sim_novel": _finite_mean(novel_sims),
            }
        )
    pooled = protocol._auroc(max_sim[is_known], max_sim[~is_known])
    return {
        "pooled_auroc": pooled,
        "strata": strata,
        "reading": temporal_reading(pooled, strata),
    }


# ---------------------------------------------------------------------------
# c. Small-gallery calibration
# ---------------------------------------------------------------------------


def small_gallery_calibration(gallery_emb, gallery_df, query_emb, query_df,
                              k=500, n_seeds=3, base_seed=0):
    """Subsample K enrolled + K novel units, ``n_seeds`` times; re-evaluate.

    Frames are rebuilt from unit subsets and pushed back through
    ``protocol.evaluate``, which re-runs the no-leakage / one-shot / alignment
    checks -- a subsample that broke the protocol would raise
    ``ProtocolViolation``. Novel queries whose side has no enrolled unit in
    the subsample keep the -inf max-similarity semantics of the full protocol
    (auto-rejected).
    """
    gallery_emb = np.asarray(gallery_emb, dtype=np.float64)
    query_emb = np.asarray(query_emb, dtype=np.float64)
    is_known = query_df["is_known"].to_numpy().astype(bool)

    enrolled_units = sorted(zip(gallery_df["identity"].tolist(), gallery_df["side"].tolist()))
    novel_units = sorted(
        set(
            zip(
                query_df.loc[~is_known, "identity"].tolist(),
                query_df.loc[~is_known, "side"].tolist(),
            )
        )
    )
    if not novel_units:
        raise ValueError("no novel units in the query frame; calibration needs an open set")
    k_eff = min(int(k), len(enrolled_units), len(novel_units))
    if k_eff < 1:
        raise ValueError("cannot subsample %d units" % k_eff)

    g_units = list(zip(gallery_df["identity"].tolist(), gallery_df["side"].tolist()))
    q_units = list(zip(query_df["identity"].tolist(), query_df["side"].tolist()))

    runs = []
    for s in range(int(n_seeds)):
        rng = np.random.default_rng([int(base_seed), 0x5CA1, s])
        keep_enrolled = set(
            enrolled_units[i]
            for i in rng.choice(len(enrolled_units), size=k_eff, replace=False)
        )
        keep_novel = set(
            novel_units[i] for i in rng.choice(len(novel_units), size=k_eff, replace=False)
        )
        g_mask = np.asarray([u in keep_enrolled for u in g_units], dtype=bool)
        q_mask = np.asarray(
            [
                (known and u in keep_enrolled) or ((not known) and u in keep_novel)
                for u, known in zip(q_units, is_known.tolist())
            ],
            dtype=bool,
        )
        sub_gallery = gallery_df.loc[g_mask].reset_index(drop=True)
        sub_query = query_df.loc[q_mask].reset_index(drop=True)
        metrics = protocol.evaluate(
            gallery_emb[g_mask], sub_gallery, query_emb[q_mask], sub_query
        )
        runs.append(
            {
                "subsample_seed": s,
                "n_gallery": metrics["n_gallery"],
                "n_known": metrics["n_known"],
                "n_novel": metrics["n_novel"],
                "rank1": metrics["rank1"],
                "open_set_auroc": metrics["open_set_auroc"],
            }
        )

    def _agg(key):
        vals = [r[key] for r in runs if r[key] is not None]
        if not vals:
            return {"mean": None, "min": None, "max": None}
        return {
            "mean": float(np.mean(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    return {
        "k_requested": int(k),
        "k_effective": int(k_eff),
        "n_seeds": int(n_seeds),
        "base_seed": int(base_seed),
        "runs": runs,
        "rank1": _agg("rank1"),
        "open_set_auroc": _agg("open_set_auroc"),
        "anchor_note": (
            "compare against the NLDL 2023 one-shot 0.35 anchor "
            "(arXiv:2301.00596, population-trained model, small gallery) "
            "[SEARCH-grade, confirm against the PDF before quoting]"
        ),
    }


# ---------------------------------------------------------------------------
# d. Crop contact sheet
# ---------------------------------------------------------------------------


def contact_sheet(root, arm, out_path, grid=6, seed=0):
    """Save a ``grid x grid`` sheet of seeded-random crops for ``arm``.

    The catalogue is loaded with ``bbox=arm`` so every cell shows the crop the
    embedder would actually see. Sampling is deterministic under
    ``(seed, arm)``; if the corpus has fewer images than cells, sampling is
    with replacement so the sheet size stays ``(grid*CELL_W, grid*CELL_H)``.
    """
    if arm not in CROP_ARMS:
        raise ValueError("contact sheet arm must be one of %r, got %r" % (CROP_ARMS, arm))
    df = melops_data.load_melops(root, bbox=arm)
    grid = int(grid)
    n_cells = grid * grid
    arm_sub = sum(ord(c) for c in arm)
    rng = np.random.default_rng([int(seed), 0x5EED, arm_sub])
    replace = len(df) < n_cells
    picks = rng.choice(len(df), size=n_cells, replace=replace)

    sheet = Image.new("RGB", (grid * SHEET_CELL_W, grid * SHEET_CELL_H), (32, 32, 32))
    for cell, idx in enumerate(picks):
        crop = melops_data.load_crop(root, df.iloc[int(idx)])
        crop = crop.resize((SHEET_CELL_W, SHEET_CELL_H), Image.BILINEAR)
        x = (cell % grid) * SHEET_CELL_W
        y = (cell // grid) * SHEET_CELL_H
        sheet.paste(crop, (x, y))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sheet.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _fmt(value):
    return "n/a" if value is None else "%.3f" % value


def write_diagnostics(diag, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    diag = _json_safe(diag)
    json_path = os.path.join(out_dir, "diagnostics.json")
    with open(json_path, "w") as f:
        json.dump(diag, f, indent=2)

    lines = [
        "# Melops run-2 diagnostics",
        "",
        "Backbone: `%s` | analysis arm: `%s` | seed %d | cutoff fraction %.2f" % (
            diag["backbone"], diag["arm"], diag["seed"], diag["cutoff_fraction"]),
        "",
        "Gallery %d units | %d known / %d novel queries | %d same-date near-duplicates excluded."
        % (diag["n_gallery"], diag["n_known"], diag["n_novel"], diag["n_same_date_excluded"]),
        "",
        "The recapture-gap curve below doubles as a PATTERN-STABILITY measurement:",
        "it tracks how the true-mate similarity of the same (identity, side) unit",
        "decays with elapsed time, which is independently valuable beyond debugging",
        "run 1's inconclusive numbers.",
        "",
        "## a. Recapture-gap curve (known queries vs their gallery image)",
        "",
        "| gap (days) | n | mean true-mate cosine sim | Rank-1 |",
        "|---|---|---|---|",
    ]
    for b in diag["recapture_gap"]["buckets"]:
        lines.append("| %s | %d | %s | %s |" % (
            b["bucket_days"], b["n"], _fmt(b["mean_true_mate_sim"]), _fmt(b["rank1"])))
    lines += [
        "",
        "## b. Open-set AUROC strata by query year",
        "",
        "Tests the ANALYSIS.md temporal-confound hypothesis (known queries span",
        "all years; novel queries are late-years only, so acquisition drift can",
        "invert the pooled AUROC even when every within-year comparison is sane).",
        "",
        "| query year | n_known | n_novel | AUROC | mean max-sim (known) | mean max-sim (novel) |",
        "|---|---|---|---|---|---|",
    ]
    for s in diag["auroc_year_strata"]["strata"]:
        lines.append("| %d | %d | %d | %s | %s | %s |" % (
            s["year"], s["n_known"], s["n_novel"], _fmt(s["auroc"]),
            _fmt(s["mean_max_sim_known"]), _fmt(s["mean_max_sim_novel"])))
    lines += [
        "",
        "Pooled AUROC: %s" % _fmt(diag["auroc_year_strata"]["pooled_auroc"]),
        "",
        "**Automated reading: %s**" % diag["auroc_year_strata"]["reading"],
        "",
        "## c. Small-gallery calibration (NLDL one-shot 0.35 anchor comparability)",
        "",
    ]
    cal = diag["small_gallery_calibration"]
    lines += [
        "K = %d enrolled + %d novel units (requested %d), %d subsample seeds."
        % (cal["k_effective"], cal["k_effective"], cal["k_requested"], cal["n_seeds"]),
        "",
        "| metric | mean | min | max |",
        "|---|---|---|---|",
        "| Rank-1 | %s | %s | %s |" % (
            _fmt(cal["rank1"]["mean"]), _fmt(cal["rank1"]["min"]), _fmt(cal["rank1"]["max"])),
        "| open-set AUROC | %s | %s | %s |" % (
            _fmt(cal["open_set_auroc"]["mean"]), _fmt(cal["open_set_auroc"]["min"]),
            _fmt(cal["open_set_auroc"]["max"])),
        "",
        cal["anchor_note"],
        "",
        "## d. Crop contact sheets",
        "",
    ]
    for arm, path in diag["contact_sheets"].items():
        lines.append("* `%s`: `%s` -- eyeball that %s crops contain what they claim." % (
            arm, os.path.basename(path), arm))
    lines.append("")
    md_path = os.path.join(out_dir, "diagnostics.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    return json_path, md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_diagnostics(root, backbone="hist", arm="body", seed=0, cutoff_fraction=0.5,
                    emb_cache_dir=None, calibration_k=500):
    """Compute diagnostic sections a-c; returns the diagnostics dict.

    Contact sheets (section d) need an output path, so the caller renders
    them via ``contact_sheet`` and records the paths in
    ``diag["contact_sheets"]`` before ``write_diagnostics``."""
    if arm not in CROP_ARMS:
        raise ValueError("analysis arm must be one of %r, got %r" % (CROP_ARMS, arm))
    df = melops_data.load_melops(root, bbox=arm)
    gallery_df, query_df = protocol.one_shot_open_set_split(
        df, cutoff_fraction=cutoff_fraction, seed=seed
    )
    embedder = embedders.get_embedder(backbone, seed=seed)
    cache_backbone = backbone if backbone != "random" else "random-seed%d" % int(seed)
    gallery_emb = emb_cache.embed_frame(
        embedder, root, gallery_df, cache_backbone, arm, cache_dir=emb_cache_dir
    )
    query_emb = emb_cache.embed_frame(
        embedder, root, query_df, cache_backbone, arm, cache_dir=emb_cache_dir
    )
    details = per_query_details(gallery_emb, gallery_df, query_emb, query_df)

    diag = {
        "backbone": backbone,
        "arm": arm,
        "seed": int(seed),
        "cutoff_fraction": float(cutoff_fraction),
        "root": os.path.abspath(root),
        "n_gallery": int(len(gallery_df)),
        "n_known": int(details["is_known"].sum()),
        "n_novel": int((~details["is_known"]).sum()),
        "n_same_date_excluded": int(query_df.attrs.get("n_same_date_excluded", 0)),
        "recapture_gap": recapture_gap_section(details, gallery_df, query_df),
        "auroc_year_strata": year_strata_section(details, query_df),
        "small_gallery_calibration": small_gallery_calibration(
            gallery_emb, gallery_df, query_emb, query_df,
            k=calibration_k, n_seeds=3, base_seed=seed,
        ),
        "contact_sheets": {},
    }
    return diag


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", choices=("synthetic", "melops"), required=True)
    parser.add_argument("--root", required=True, help="corpus root directory")
    parser.add_argument("--backbone", default="hist",
                        choices=("hist", "random", "megadescriptor", "dinov2", "miewid"))
    parser.add_argument("--arm", default="body", choices=CROP_ARMS,
                        help="crop arm for the a/b/c analyses (default body)")
    parser.add_argument("--out", default="diagnostics")
    parser.add_argument("--emb-cache", default=None, metavar="DIR",
                        help="optional embedding-cache directory shared with "
                             "run_ablation.py --emb-cache (one per corpus root)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cutoff-fraction", type=float, default=0.5)
    parser.add_argument("--calibration-k", type=int, default=500,
                        help="enrolled/novel units per calibration subsample")
    parser.add_argument("--grid", type=int, default=6, help="contact-sheet grid side")
    parser.add_argument("--sheet-arms", default="head,body,headless",
                        help="comma-separated crop arms to render contact sheets for")
    parser.add_argument("--n-individuals", type=int, default=40,
                        help="synthetic only; used when --root has no metadata.csv yet")
    args = parser.parse_args(argv)

    if args.data == "synthetic" and not os.path.exists(os.path.join(args.root, "metadata.csv")):
        melops_data.make_synthetic(args.root, n_individuals=args.n_individuals, seed=args.seed)

    sheet_arms = tuple(a.strip() for a in args.sheet_arms.split(",") if a.strip())
    diag = run_diagnostics(
        args.root, backbone=args.backbone, arm=args.arm, seed=args.seed,
        cutoff_fraction=args.cutoff_fraction, emb_cache_dir=args.emb_cache,
        calibration_k=args.calibration_k,
    )
    os.makedirs(args.out, exist_ok=True)
    for sheet_arm in sheet_arms:
        path = os.path.join(args.out, "contact_sheet_%s.png" % sheet_arm)
        contact_sheet(args.root, sheet_arm, path, grid=args.grid, seed=args.seed)
        # basename only: the sheet sits beside diagnostics.json, and an
        # absolute path made byte-identical reruns impossible across --out dirs
        diag["contact_sheets"][sheet_arm] = os.path.basename(path)
    json_path, md_path = write_diagnostics(diag, args.out)

    print("known=%d novel=%d gallery=%d same_date_excluded=%d"
          % (diag["n_known"], diag["n_novel"], diag["n_gallery"], diag["n_same_date_excluded"]))
    for b in diag["recapture_gap"]["buckets"]:
        print("gap %-8s n=%-5d true-mate-sim=%s rank1=%s"
              % (b["bucket_days"], b["n"], _fmt(b["mean_true_mate_sim"]), _fmt(b["rank1"])))
    strata = diag["auroc_year_strata"]
    print("pooled AUROC=%s" % _fmt(strata["pooled_auroc"]))
    for s in strata["strata"]:
        print("year %d known=%-5d novel=%-5d AUROC=%s"
              % (s["year"], s["n_known"], s["n_novel"], _fmt(s["auroc"])))
    print("READING: %s" % strata["reading"])
    cal = diag["small_gallery_calibration"]
    print("calibration K=%d rank1 mean/min/max = %s/%s/%s AUROC mean = %s"
          % (cal["k_effective"], _fmt(cal["rank1"]["mean"]), _fmt(cal["rank1"]["min"]),
             _fmt(cal["rank1"]["max"]), _fmt(cal["open_set_auroc"]["mean"])))
    print("Wrote %s and %s (+%d contact sheets)" % (json_path, md_path, len(sheet_arms)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
