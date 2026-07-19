"""Curvature-aware local re-identification for real spot patterns.

The global-homography matcher fails on real sharks because a flank is a
curved 3-D surface (two views are not one homography) and because dense
spot clouds let RANSAC find spurious consensus sets. This matcher fixes
both:

  1. Each spot gets a rotation/scale-invariant *local shape-context*
     descriptor of its neighborhood.
  2. Candidate spot matches come from mutual nearest neighbors in
     descriptor space.
  3. Matches are verified by *neighborhood preservation* — a match a_i->b_j
     is kept only if several of a_i's spatial neighbors are themselves
     matched to spatial neighbors of b_j. This is a NON-RIGID consistency
     check: it survives curvature (which preserves local neighborhoods)
     but rejects the coincidental descriptor collisions that inflate false
     matches on dense clouds.
  4. The pair score is the count of verified matches, each weighted by how
     *distinctive* its descriptor is across the database (common
     neighborhoods count for little). Optional LNBNN normalization ranks a
     query against a whole gallery.
"""

import numpy as np
from scipy.spatial import cKDTree

# Descriptor geometry.
DESC_K = 12               # neighbors used to build a spot's descriptor
RAD_BINS = 5             # log-radial bins
ANG_BINS = 12            # angular bins
RAD_MIN, RAD_MAX = 0.35, 3.0   # radial range in units of median NN distance

# Verification.
VERIFY_K = 8            # spatial neighbors checked for preservation
MIN_SUPPORT = 3        # neighbors that must be co-matched-and-still-near


def local_descriptors(pts: np.ndarray, k: int = DESC_K) -> np.ndarray:
    """Rotation- and scale-invariant shape-context descriptor per point.

    Log-polar histogram of the k nearest neighbors (radius normalized by
    the median neighbor distance), made rotation-invariant by taking the
    magnitude of the angular FFT within each radial ring.
    """
    n = len(pts)
    dim = RAD_BINS * (ANG_BINS // 2 + 1)
    if n < k + 1:
        return np.zeros((n, dim))
    tree = cKDTree(pts)
    dist, idx = tree.query(pts, k=k + 1)
    out = np.zeros((n, dim))
    log_edges = np.linspace(np.log(RAD_MIN), np.log(RAD_MAX), RAD_BINS + 1)
    for i in range(n):
        rel = pts[idx[i, 1:]] - pts[i]
        r = np.linalg.norm(rel, axis=1)
        med = np.median(r) if len(r) else 1.0
        rn = r / max(med, 1e-9)
        ang = np.arctan2(rel[:, 1], rel[:, 0]) % (2 * np.pi)
        hist = np.zeros((RAD_BINS, ANG_BINS))
        rb = np.digitize(np.log(np.clip(rn, 1e-6, None)), log_edges) - 1
        ab = np.minimum((ang / (2 * np.pi) * ANG_BINS).astype(int), ANG_BINS - 1)
        for rr, aa in zip(rb, ab):
            if 0 <= rr < RAD_BINS:
                hist[rr, aa] += 1.0
        # rotation invariance: angular-FFT magnitude per ring
        mag = np.abs(np.fft.rfft(hist, axis=1))
        out[i] = mag.ravel()
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norm, 1e-9)


def _mutual_nn(dA: np.ndarray, dB: np.ndarray):
    """Mutual nearest neighbors in descriptor space, with distances."""
    if len(dA) == 0 or len(dB) == 0:
        return np.empty((0, 2), int), np.empty(0)
    tA, tB = cKDTree(dA), cKDTree(dB)
    dab, jab = tB.query(dA)
    _, jba = tA.query(dB)
    ia = np.arange(len(dA))
    keep = jba[jab] == ia
    pairs = np.stack([ia[keep], jab[keep]], axis=1)
    return pairs, dab[keep]


def verify_matches(ptsA, ptsB, pairs, k=VERIFY_K, min_support=MIN_SUPPORT):
    """Keep matches whose local neighborhood is preserved: >= min_support of
    a_i's k spatial neighbors are matched to points among b_j's k spatial
    neighbors. Non-rigid — no global transform assumed."""
    if len(pairs) == 0:
        return pairs, np.zeros(0, int)
    a2b = {int(a): int(b) for a, b in pairs}
    tA = cKDTree(ptsA)
    tB = cKDTree(ptsB)
    ka = min(k, len(ptsA) - 1)
    kb = min(k, len(ptsB) - 1)
    nbrA = tA.query(ptsA, k=ka + 1)[1]
    nbrB = tB.query(ptsB, k=kb + 1)[1]
    nbrB_sets = [set(row[1:]) for row in nbrB]
    verified = []
    support = []
    for a, b in pairs:
        sup = 0
        for na in nbrA[a, 1:]:
            nb = a2b.get(int(na))
            if nb is not None and nb in nbrB_sets[b]:
                sup += 1
        if sup >= min_support:
            verified.append((a, b))
            support.append(sup)
    return np.array(verified) if verified else np.empty((0, 2), int), \
        np.array(support, int)


def match_pair(ptsA, ptsB, descA=None, descB=None,
               idf: dict | None = None):
    """Verified correspondences between two spot patterns and a score.

    ``idf`` (optional): per-descriptor-cluster inverse document frequency
    for distinctiveness weighting; if None, all matches weigh 1.
    Returns (verified_pairs, support, score)."""
    if descA is None:
        descA = local_descriptors(ptsA)
    if descB is None:
        descB = local_descriptors(ptsB)
    pairs, _ = _mutual_nn(descA, descB)
    vpairs, support = verify_matches(ptsA, ptsB, pairs)
    if idf is None:
        score = float(len(vpairs))
    else:
        w = np.array([idf.get(int(a), 1.0) for a, _ in vpairs])
        score = float((w * support).sum())
    return vpairs, support, score
