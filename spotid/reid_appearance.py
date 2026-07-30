"""Appearance-based individual re-identifier (Stage A of the frontier design).

Matches flank crops with DISK deep local features + mutual-NN + RANSAC
verification; the number of geometrically-verified inliers is the match
score. Evaluated leave-one-individual-out (each image queried against a
gallery of all others) with a real ranking metric, plus an open-set
"new individual" threshold analysis.

Usage:
    python -m spotid.reid_appearance --root realdata/realworldspots.yolov8
"""

import argparse
import itertools
import time

import cv2
import numpy as np

from .probe_matchers import DiskMatcher, _ransac_inliers, flank_crop, sift_match
from .realdata import group_by_individual, load_dataset


def score_from_features(ka, da, kb, db, min_pts=8):
    """RANSAC-verified inlier count between two DISK feature sets."""
    if len(ka) < min_pts or len(kb) < min_pts:
        return 0
    da = da / (np.linalg.norm(da, axis=1, keepdims=True) + 1e-9)
    db = db / (np.linalg.norm(db, axis=1, keepdims=True) + 1e-9)
    sim = da @ db.T
    ab = sim.argmax(1)
    ba = sim.argmax(0)
    mut = [(i, ab[i]) for i in range(len(ka)) if ba[ab[i]] == i]
    if len(mut) < min_pts:
        return 0
    pa = np.float32([ka[i] for i, _ in mut])
    pb = np.float32([kb[j] for _, j in mut])
    return _ransac_inliers(pa, pb)


def build_feature_bank(shots, matcher):
    bank = {}
    for i, s in enumerate(shots):
        bank[s.filename] = matcher._feat(flank_crop(s))
        print(f"  features {i+1}/{len(shots)} {s.individual}", flush=True)
    return bank


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="realdata/realworldspots.yolov8")
    ap.add_argument("--min-spots", type=int, default=40)
    ap.add_argument("--also-sift", action="store_true")
    args = ap.parse_args()

    shots = [s for s in load_dataset(args.root) if s.n_spots >= args.min_spots]
    by = group_by_individual(shots)
    N = len(shots)
    print(f"{N} flank photos, {len(by)} individuals, "
          f"{sum(1 for v in by.values() if len(v) > 1)} re-sighted")

    t0 = time.time()
    disk = DiskMatcher()
    bank = build_feature_bank(shots, disk)
    print(f"feature bank in {time.time()-t0:.0f}s")

    # full symmetric score matrix
    S = np.zeros((N, N))
    t0 = time.time()
    for i, j in itertools.combinations(range(N), 2):
        ka, da = bank[shots[i].filename]
        kb, db = bank[shots[j].filename]
        S[i, j] = S[j, i] = score_from_features(ka, da, kb, db)
    print(f"pairwise matching in {time.time()-t0:.0f}s")

    lbl = [s.individual for s in shots]
    has_twin = [sum(1 for k in range(N) if k != i and lbl[k] == lbl[i]) > 0
                for i in range(N)]

    # --- leave-one-out ranking: query each re-sighted image vs all others ---
    ranks, top1 = [], 0
    nq = 0
    print("\n=== leave-one-individual-out ranking (DISK) ===")
    for i in range(N):
        if not has_twin[i]:
            continue
        order = np.argsort(-S[i])
        order = [j for j in order if j != i]
        # rank of the first gallery image sharing the query's identity
        rank = next(r for r, j in enumerate(order, 1) if lbl[j] == lbl[i])
        ranks.append(rank)
        top1 += rank == 1
        nq += 1
        best = order[0]
        print(f"  {lbl[i]:11s} -> top match {lbl[best]:11s} "
              f"score {S[i,best]:.0f} (2nd {S[i,order[1]]:.0f})  "
              f"true-rank {rank}  {'OK' if rank==1 else 'MISS'}")
    mrr = np.mean([1.0 / r for r in ranks]) if ranks else 0.0
    print(f"\nDISK top-1: {top1}/{nq}   MRR: {mrr:.3f}   "
          f"(centroid-matcher baseline was 2/8)")

    # --- open-set: can a threshold tell 're-sighted' from 'new individual'? ---
    print("\n=== open-set separation (best-match score per query) ===")
    best_twin = [max((S[i, j] for j in range(N) if j != i and lbl[j] == lbl[i]),
                     default=0) for i in range(N) if has_twin[i]]
    best_none = []
    for i in range(N):
        if has_twin[i]:
            continue
        best_none.append(max((S[i, j] for j in range(N) if j != i), default=0))
    best_twin, best_none = np.array(best_twin), np.array(best_none)
    if len(best_twin):
        print(f"  genuine (has true match): min {best_twin.min():.0f} "
              f"mean {best_twin.mean():.0f}")
    else:
        print("  genuine (has true match): none in this dataset")
    if len(best_none):
        print(f"  new individual (no match): max {best_none.max():.0f} "
              f"mean {best_none.mean():.0f}")
    else:
        print("  new individual (no match): none — every individual has a twin")
    if len(best_twin) and len(best_none):
        gap = best_twin.min() - best_none.max()
        print(f"  separation margin: {gap:+.0f}  "
              f"({'cleanly separable' if gap > 0 else 'overlap'})")

    if args.also_sift:
        print("\n=== SIFT cross-check (true vs impostor best scores) ===")
        # quick sanity on a few
        for ind, g in list(by.items())[:6]:
            if len(g) > 1:
                _, si = sift_match(flank_crop(g[0]), flank_crop(g[1]))
                print(f"  {ind}: SIFT inliers {si}")


if __name__ == "__main__":
    main()
