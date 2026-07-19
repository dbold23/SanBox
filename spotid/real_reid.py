"""Real individual re-identification on the sevengill dataset.

Gallery = one photo per well-annotated individual. Query = a *different*
photo of a re-sighted individual. We ask: does the constellation matcher
rank the query's true individual first, purely from the 2-D arrangement of
its spots? This is the real-world analogue of the synthetic surface test —
and the honest check of whether a (curved) shark flank is planar enough for
the homography-based matcher.

Usage:
    python -m spotid.real_reid --root realdata/realworldspots.yolov8
"""

import argparse

import numpy as np

from .realdata import group_by_individual, load_dataset, resighting_pairs
from .surface_matcher import (SurfaceMatcher, _local_signatures,
                              _whiten_points)


class _CentroidSurface:
    """Minimal Surface adapter: constellation geometry from raw centroids
    (no synthetic contours), so SurfaceMatcher can enroll real photos."""

    def __init__(self, surface_id, centroids):
        self.surface_id = surface_id
        self.positions = np.asarray(centroids, float)
        self.spots = [None] * len(self.positions)

    def spot_contour(self, i):  # never called when use_shape_descriptors=False
        raise NotImplementedError


def build_matcher(gallery_shots):
    """Enroll one _CentroidSurface per gallery photo, keyed by individual."""
    matcher = SurfaceMatcher(use_shape_descriptors=False)
    ids = []
    for shot in gallery_shots:
        matcher.enroll_surface(_CentroidSurface(shot.individual, shot.centroids))
        ids.append(shot.individual)
    return matcher, ids


def identify_centroids(matcher, cents, top_k=5):
    """Run the matcher's geometric pipeline directly on query centroids
    (we already have ground-truth spot positions, so skip segmentation)."""
    results = matcher._identify_global(cents, None, top_k)
    n = results[0].n_matched if results else 0
    if n < max(20, 0.3 * len(cents)):
        partial = matcher._identify_partial(cents, top_k)
        if partial and (not results or partial[0].n_matched > n):
            return partial
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="realdata/realworldspots.yolov8")
    ap.add_argument("--min-spots", type=int, default=40)
    args = ap.parse_args()

    shots = load_dataset(args.root)
    well = [s for s in shots if s.n_spots >= args.min_spots]
    pairs = resighting_pairs(shots, args.min_spots)
    print(f"loaded {len(shots)} photos, {len(well)} well-annotated "
          f"(>={args.min_spots} spots), {len(pairs)} re-sighted individuals\n")

    # Leave-one-out re-ID: for each re-sighted individual, hold out one
    # photo as the query; the gallery is one photo of every OTHER
    # well-annotated individual plus the remaining photo(s) of this one.
    by_ind = group_by_individual(well)
    trials = 0
    correct = 0
    reciprocal_ranks = []
    for ind, group in pairs:
        for qi in range(len(group)):
            query = group[qi]
            gallery = []
            seen = set()
            # the held-in photo of the SAME individual (its enrolled twin)
            for gj, g in enumerate(group):
                if gj != qi:
                    gallery.append(g)
                    break
            # one photo of every other well-annotated individual
            for other, og in by_ind.items():
                if other == ind:
                    continue
                if other not in seen:
                    gallery.append(og[0])
                    seen.add(other)
            matcher, gids = build_matcher(gallery)
            res = identify_centroids(matcher, query.centroids, top_k=len(gallery))
            ranked = [r.surface_id for r in res]
            rank = ranked.index(ind) + 1 if ind in ranked else None
            hit = bool(res) and res[0].surface_id == ind
            trials += 1
            correct += hit
            reciprocal_ranks.append(1.0 / rank if rank else 0.0)
            top = res[0] if res else None
            print(f"{ind:12s} q={query.n_spots:3d} spots  gallery={len(gallery):2d}"
                  f"  -> pred {top.surface_id if top else '-':12s}"
                  f" ({top.mode if top else '-':7s}, matched {top.n_matched if top else 0:3d})"
                  f"  {'OK' if hit else 'MISS'}  rank={rank}")

    print(f"\n=== real individual re-ID ===")
    print(f"trials: {trials}   top-1: {correct}/{trials} "
          f"({correct / max(trials,1):.3f})   MRR: {np.mean(reciprocal_ranks):.3f}")


if __name__ == "__main__":
    main()
