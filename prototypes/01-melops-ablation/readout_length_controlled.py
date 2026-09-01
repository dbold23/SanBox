"""Length-controlled supplementary readout (run 2, protocol-owner signed off).

SCOPE: supplementary readout ONLY. No changes to ``split_identities``, to
``one_shot_open_set_split``, or to the verdict rules. All length logic is
applied as masks on the similarity matrix AFTER the split, so no leakage path
exists. Rows with missing ``length`` are excluded from the stratified
readouts only (and counted); headline metrics elsewhere are untouched.

Three readouts, same-side crop arms only:

1. SIZE-ASSORTATIVITY INDEX
   ``1 - mean|len_q - len_argmax| / mean|len_q - len_random|`` over all
   queries (known + novel) with lengths on both ends. The random baseline
   draws same-side gallery entries under a fixed seed. This is the
   fine-tune's nuisance-suppression success metric: report before vs after.

2. LENGTH-STRATIFIED RANK-1
   Known queries bucketed by ``|len_q - len_truemate|`` terciles; standard
   same-side Rank-1 per tercile. Collapse in the top tercile means growth
   destroys the match.

3. LENGTH-MATCHED IMPOSTOR AUROC
   Open-set AUROC (known vs novel on max cosine similarity, known positive)
   with each query's gallery restricted to same-side entries within +/-band
   (default 10%) of the query's length -- impostors AND mate alike. Queries
   with no in-band gallery are excluded and counted.

Embeddings come from the npz cache written by ``run_ablation.py --emb-cache``
/ ``diagnose.py --emb-cache``; a cache miss is a hard error (this tool never
embeds, so it can run CPU-only while training owns the GPU).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import emb_cache
import melops_data
import protocol
from run_ablation import _exclude_train_identities


def _same_side_sims(gallery_df, query_df, gallery_emb, query_emb):
    S = query_emb @ gallery_emb.T
    mask = query_df["side"].to_numpy()[:, None] == gallery_df["side"].to_numpy()[None, :]
    return np.where(mask, S, -np.inf)


def compute_assortativity(S, qlen, glen, rng):
    """Readout 1. Returns dict with index, means and the usable-row count."""
    am = S.argmax(axis=1)
    ok = ~np.isnan(qlen) & ~np.isnan(glen[am])
    d_argmax = np.abs(qlen[ok] - glen[am][ok])
    rand_idx = rng.integers(0, S.shape[1], size=int(ok.sum()))
    ok_rand = ~np.isnan(glen[rand_idx])
    d_random = np.abs(qlen[ok][ok_rand] - glen[rand_idx][ok_rand])
    mean_am, mean_rand = float(d_argmax.mean()), float(d_random.mean())
    return {
        "index": 1.0 - mean_am / mean_rand,
        "mean_absdiff_argmax_mm": mean_am,
        "mean_absdiff_random_mm": mean_rand,
        "n_queries_used": int(ok.sum()),
        "n_queries_missing_length": int(len(qlen) - ok.sum()),
    }


def compute_stratified_rank1(S, q_units, g_units, known, qlen, mate_len):
    """Readout 2. Terciles of |len_q - len_truemate| over known queries."""
    am = S.argmax(axis=1)
    correct = np.array([g_units[a] == qu for a, qu in zip(am, q_units)])
    d_mate = np.abs(qlen - mate_len)
    ok = known & ~np.isnan(d_mate)
    d = d_mate[ok]
    edges = np.quantile(d, [1.0 / 3.0, 2.0 / 3.0])
    buckets = []
    lo = -np.inf
    for j, hi in enumerate(list(edges) + [np.inf]):
        sel = ok & (d_mate > lo) & (d_mate <= hi)
        buckets.append(
            {
                "tercile": j + 1,
                "absdiff_range_mm": [None if not np.isfinite(lo) else float(lo),
                                     None if not np.isfinite(hi) else float(hi)],
                "n": int(sel.sum()),
                "rank1": float(correct[sel].mean()) if sel.any() else None,
                "mean_absdiff_mm": float(d_mate[sel].mean()) if sel.any() else None,
            }
        )
        lo = hi
    return {
        "terciles": buckets,
        "n_known_used": int(ok.sum()),
        "n_known_missing_length": int(known.sum() - ok.sum()),
    }


def compute_band_auroc(S, known, qlen, glen, band):
    """Readout 3. AUROC with per-query same-side gallery banded to +/-band."""
    glen_row = glen[None, :]
    with np.errstate(invalid="ignore"):
        in_band = (
            ~np.isnan(glen_row)
            & (glen_row >= qlen[:, None] * (1.0 - band))
            & (glen_row <= qlen[:, None] * (1.0 + band))
        )
    S_band = np.where(in_band, S, -np.inf)
    mx = S_band.max(axis=1)
    usable = np.isfinite(mx) & ~np.isnan(qlen)
    pos = mx[usable & known]
    neg = mx[usable & ~known]
    return {
        "auroc": None if (len(pos) == 0 or len(neg) == 0)
        else float(protocol._auroc(pos, neg)),
        "band_fraction": float(band),
        "n_known_used": int(len(pos)),
        "n_novel_used": int(len(neg)),
        "n_excluded_no_inband_gallery_or_no_length": int(len(mx) - usable.sum()),
    }


def run_readout(root, backbone, arm="body", seed=0, cutoff_fraction=0.5,
                emb_cache_dir="emb_cache", band=0.10, baseline_seed=17,
                dense_min_images=None):
    from run_ablation import _filter_dense_units

    df = melops_data.load_melops(root, bbox=arm)
    if dense_min_images is not None:
        df, _, _ = _filter_dense_units(df, dense_min_images)
    df, n_train_excluded, _ = _exclude_train_identities(df, backbone)
    gallery_df, query_df = protocol.one_shot_open_set_split(
        df, cutoff_fraction=cutoff_fraction, seed=seed
    )
    cache_backbone = backbone if backbone != "random" else "random-seed%d" % int(seed)
    if dense_min_images is not None:
        # match run_ablation's dense cache namespace
        cache_backbone = "%s-dense%d" % (cache_backbone, int(dense_min_images))
    ge = emb_cache.load(emb_cache_dir, cache_backbone, arm,
                        emb_cache._ids_list(gallery_df["image_id"].tolist()))
    qe = emb_cache.load(emb_cache_dir, cache_backbone, arm,
                        emb_cache._ids_list(query_df["image_id"].tolist()))
    if ge is None or qe is None:
        raise RuntimeError(
            "embedding cache miss for backbone=%r arm=%r in %r -- run "
            "run_ablation.py/diagnose.py with --emb-cache first; this tool "
            "never embeds" % (backbone, arm, emb_cache_dir)
        )

    meta = pd.read_csv(os.path.join(root, "Melops_metadata.txt"),
                       sep=None, engine="python")
    meta["image_id"] = meta["filename_year"].astype(str)
    lengths = meta.set_index("image_id")["length"]
    qlen = query_df["image_id"].map(lengths).to_numpy(dtype=np.float64)
    glen = gallery_df["image_id"].map(lengths).to_numpy(dtype=np.float64)

    S = _same_side_sims(gallery_df, query_df, ge, qe)
    known = query_df["is_known"].to_numpy(dtype=bool)
    g_units = list(zip(gallery_df["identity"].astype(str),
                       gallery_df["side"].astype(str)))
    q_units = list(zip(query_df["identity"].astype(str),
                       query_df["side"].astype(str)))
    unit_to_len = {u: glen[i] for i, u in enumerate(g_units)}
    mate_len = np.array([unit_to_len.get(u, np.nan) for u in q_units],
                        dtype=np.float64)

    rng = np.random.default_rng(int(baseline_seed))
    out = {
        "backbone": backbone,
        "arm": arm,
        "seed": int(seed),
        "cutoff_fraction": float(cutoff_fraction),
        "baseline_seed": int(baseline_seed),
        "n_gallery": int(len(gallery_df)),
        "n_known": int(known.sum()),
        "n_novel": int((~known).sum()),
        "n_train_identities_excluded": int(n_train_excluded),
        "assortativity": compute_assortativity(S, qlen, glen, rng),
        "length_stratified_rank1": compute_stratified_rank1(
            S, q_units, g_units, known, qlen, mate_len
        ),
        "length_matched_auroc": compute_band_auroc(S, known, qlen, glen, band),
    }
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--backbone", required=True,
                        help="megadescriptor | finetuned:CHECKPOINT_PATH | ...")
    parser.add_argument("--arm", default="body", choices=("head", "body", "headless"))
    parser.add_argument("--emb-cache", required=True)
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cutoff-fraction", type=float, default=0.5)
    parser.add_argument("--band", type=float, default=0.10)
    parser.add_argument("--baseline-seed", type=int, default=17)
    parser.add_argument("--dense-min-images", type=int, default=None,
                        help="apply the run-3 Leg A dense-subset catalogue filter "
                             "before the split (matches run_ablation's cache keys)")
    args = parser.parse_args(argv)

    out = run_readout(args.root, args.backbone, arm=args.arm, seed=args.seed,
                      cutoff_fraction=args.cutoff_fraction,
                      emb_cache_dir=args.emb_cache, band=args.band,
                      baseline_seed=args.baseline_seed,
                      dense_min_images=args.dense_min_images)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    a = out["assortativity"]
    print("ASSORTATIVITY index=%.3f (argmax %.2fmm vs random %.2fmm, n=%d)"
          % (a["index"], a["mean_absdiff_argmax_mm"], a["mean_absdiff_random_mm"],
             a["n_queries_used"]))
    for b in out["length_stratified_rank1"]["terciles"]:
        print("STRATIFIED tercile %d n=%-5d mean|dlen|=%.1fmm rank1=%s"
              % (b["tercile"], b["n"], b["mean_absdiff_mm"] or float("nan"),
                 "n/a" if b["rank1"] is None else "%.4f" % b["rank1"]))
    m = out["length_matched_auroc"]
    print("BAND-AUROC(+/-%d%%)=%s (known n=%d, novel n=%d, excluded=%d)"
          % (round(m["band_fraction"] * 100),
             "n/a" if m["auroc"] is None else "%.4f" % m["auroc"],
             m["n_known_used"], m["n_novel_used"],
             m["n_excluded_no_inband_gallery_or_no_length"]))
    print("Wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
