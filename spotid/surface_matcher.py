"""Identify a surface — and every individual spot on it — from one image.

Pipeline:
  1. Segment all splotches in the query image; take centroids + shape
     descriptors.
  2. Whiten the query centroid cloud and each enrolled surface's canonical
     spot cloud (inverse-sqrt covariance). An affine view of the same
     constellation then differs only by rotation.
  3. Recover that rotation per candidate surface with FFT cross-correlation
     of radius-weighted angular histograms, keep the best peaks.
  4. Mutual nearest-neighbor matching + iterative affine refinement (ICP);
     surfaces are ranked by inlier count (+ shape-descriptor agreement).
  5. For the winning surface, fit a full RANSAC homography for exact
     per-spot assignment: every matched query blob is labeled with the
     spot index it corresponds to on the identified surface.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .features import describe_contour, segment_all_spots
from .surface import Surface

ANGLE_BINS = 256
ROTATION_CANDIDATES = 4
ICP_ROUNDS = 3

# Local constellation signatures (partial-view mode).
SIG_NEIGHBORS = 6
SIG_CANDIDATES = 3      # gallery hits considered per query spot
# Global mode is accepted when it tight-matches at least this fraction of
# query blobs AND this many blobs absolutely (clutter inflates the blob
# count, so the absolute count matters more than the fraction).
GLOBAL_ACCEPT_FRACTION = 0.35
GLOBAL_ACCEPT_MIN = 40
# ICP convergence uses a loose radius (fraction of spot spacing); scoring
# uses a tight one — genuine alignments land within a few percent of the
# spacing, chance alignments do not.
ICP_LOOSE = 0.5
ICP_TIGHT = 0.15
# Final per-spot assignment: max residual as a fraction of projected spot
# spacing (with a small absolute floor in pixels).
ASSIGN_FRACTION = 0.3
ASSIGN_FLOOR_PX = 3.0


@dataclass
class SurfaceMatch:
    surface_id: int
    score: float                 # inliers + descriptor agreement bonus
    n_matched: int               # matched spot count
    n_query_spots: int           # segmented blobs in the query image
    n_surface_spots: int         # spots enrolled for this surface
    mode: str = "global"         # "global" (whole surface visible) or "partial"
    homography: np.ndarray | None = None   # surface coords -> query pixels
    # rows of (query_blob_index, surface_spot_index, pixel_residual)
    assignments: list = field(default_factory=list)


def _whiten_points(pts: np.ndarray):
    """Return (whitened points, mean, inv-sqrt-covariance)."""
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    w, v = np.linalg.eigh(cov)
    w = np.clip(w, 1e-12, None)
    t = v @ np.diag(1.0 / np.sqrt(w)) @ v.T
    return (pts - mean) @ t.T, mean, t


def _angular_histogram(pts: np.ndarray) -> np.ndarray:
    """Radius-weighted circular histogram of point directions."""
    ang = np.arctan2(pts[:, 1], pts[:, 0]) % (2.0 * np.pi)
    r = np.linalg.norm(pts, axis=1)
    hist, _ = np.histogram(ang, bins=ANGLE_BINS, range=(0.0, 2.0 * np.pi),
                           weights=r)
    # Circular smoothing for robustness to bin-edge jitter.
    kernel = np.array([0.25, 0.5, 1.0, 0.5, 0.25])
    kernel /= kernel.sum()
    padded = np.concatenate([hist[-2:], hist, hist[:2]])
    return np.convolve(padded, kernel, mode="same")[2:-2]


def _rotation_candidates(hq: np.ndarray, hg: np.ndarray, k: int) -> np.ndarray:
    """Angles (radians) that best rotate the query histogram onto the
    gallery histogram, from FFT circular cross-correlation peaks."""
    corr = np.real(np.fft.ifft(np.fft.fft(hg) * np.conj(np.fft.fft(hq))))
    order = np.argsort(corr)[::-1]
    picked = []
    for idx in order:
        if all(min(abs(idx - p), ANGLE_BINS - abs(idx - p)) > 4 for p in picked):
            picked.append(int(idx))
        if len(picked) == k:
            break
    return np.array(picked) * 2.0 * np.pi / ANGLE_BINS


def _signature_of(center, nb, ii, jj, trips):
    """Sorted triangle-area-ratio signature of one point + its neighbors.

    Uses |areas| of all (p, ni, nj) and all (ni, nj, nl) triangles,
    normalized to unit sum. Affine maps scale every area by the same
    factor, so the ratios survive; sorting removes neighbor labeling."""
    rel = nb - center
    a1 = 0.5 * np.abs(rel[ii, 0] * rel[jj, 1] - rel[ii, 1] * rel[jj, 0])
    v1 = nb[trips[:, 1]] - nb[trips[:, 0]]
    v2 = nb[trips[:, 2]] - nb[trips[:, 0]]
    a2 = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    sig = np.concatenate([np.sort(a1), np.sort(a2)])
    return sig / max(sig.sum(), 1e-12)


def _local_signatures(pts: np.ndarray, k: int = SIG_NEIGHBORS,
                      loo: bool = True):
    """Affine-invariant local constellation signatures for every point.

    With ``loo`` (leave-one-out), each point takes its k+1 nearest
    neighbors and emits k+1 signatures, each omitting one neighbor. A
    point then still matches its gallery twin when one neighbor is
    missing (worn/faded/glare-hidden) or one clutter blob intrudes into
    the neighborhood. Returns (signatures, owner_point_index_per_row).
    """
    from itertools import combinations

    n = len(pts)
    ii, jj = np.triu_indices(k, 1)
    trips = np.array(list(combinations(range(k), 3)))
    dim = len(ii) + len(trips)
    n_nbrs = k + 1 if loo else k
    if n < n_nbrs + 1:
        return np.zeros((0, dim)), np.zeros(0, int)
    tree = cKDTree(pts)
    _, nbrs = tree.query(pts, k=n_nbrs + 1)
    sigs, owners = [], []
    for p in range(n):
        nb_all = pts[nbrs[p, 1:]]
        if loo:
            for drop in range(n_nbrs):
                nb = np.delete(nb_all, drop, axis=0)
                sigs.append(_signature_of(pts[p], nb, ii, jj, trips))
                owners.append(p)
        else:
            sigs.append(_signature_of(pts[p], nb_all, ii, jj, trips))
            owners.append(p)
    return np.array(sigs), np.array(owners, int)


def _mutual_matches(a: np.ndarray, b: np.ndarray, max_dist: float):
    """Mutual nearest-neighbor pairs (ia, ib) with distance < max_dist."""
    if len(a) == 0 or len(b) == 0:
        return np.empty((0, 2), int)
    ta, tb = cKDTree(a), cKDTree(b)
    d_ab, j_ab = tb.query(a)          # for each a: nearest b
    _, j_ba = ta.query(b)             # for each b: nearest a
    ia = np.arange(len(a))
    mutual = (j_ba[j_ab] == ia) & (d_ab < max_dist)
    return np.stack([ia[mutual], j_ab[mutual]], axis=1)


class SurfaceMatcher:
    def __init__(self, use_shape_descriptors: bool = True,
                 min_blob_area: float = 12.0):
        self.use_shape_descriptors = use_shape_descriptors
        self.min_blob_area = min_blob_area
        self._surfaces: list[dict] = []
        self._sig_index: np.ndarray | None = None
        self._sig_tree: cKDTree | None = None
        self._sig_owner: list[tuple] = []
        # Query blob centroids from the most recent identify() call; the
        # qi indices in SurfaceMatch.assignments index into this array.
        # Exposed so callers can reuse them instead of re-segmenting.
        self.last_query_centroids: np.ndarray | None = None

    def enroll_surface(self, surface: Surface) -> None:
        pts = surface.positions
        white, _, _ = _whiten_points(pts)
        tree = cKDTree(white)
        entry = {
            "id": surface.surface_id,
            "positions": pts,
            "white": white,
            "hist": _angular_histogram(white),
            "tree": tree,
            "nn_spacing": float(np.median(tree.query(white, k=2)[0][:, 1])),
            "signatures": _local_signatures(pts),
            "descs": None,
        }
        if self.use_shape_descriptors:
            descs = []
            for i in range(len(surface.spots)):
                d = describe_contour(surface.spot_contour(i))
                descs.append(d if d is not None else np.zeros(1))
            dim = max(len(d) for d in descs)
            mat = np.zeros((len(descs), dim))
            for i, d in enumerate(descs):
                if len(d) == dim:
                    n = np.linalg.norm(d)
                    mat[i] = d / max(n, 1e-9)
            entry["descs"] = mat
        self._surfaces.append(entry)
        self._sig_index = None  # invalidate partial-view index

    def _prepare_signatures(self) -> None:
        blocks, owners = [], []
        for si, entry in enumerate(self._surfaces):
            sig, own = entry["signatures"]
            blocks.append(sig)
            owners.extend((si, int(spot)) for spot in own)
        self._sig_index = np.vstack(blocks)
        self._sig_owner = owners
        self._sig_tree = cKDTree(self._sig_index)

    # ------------------------------------------------------------------

    def _score_surface(self, entry, qwhite, qdescs):
        """Align query cloud to one enrolled surface; return
        (score, matches, affine) where matches are (query_idx, spot_idx)."""
        loose = ICP_LOOSE * entry["nn_spacing"]
        tight = ICP_TIGHT * entry["nn_spacing"]
        hq = _angular_histogram(qwhite)
        best = (0.0, np.empty((0, 2), int), None)
        for ang in _rotation_candidates(hq, entry["hist"], ROTATION_CANDIDATES):
            c, s = np.cos(ang), np.sin(ang)
            cur = qwhite @ np.array([[c, -s], [s, c]]).T
            aff = None
            matches = _mutual_matches(cur, entry["white"], loose)
            for _ in range(ICP_ROUNDS):
                if len(matches) < 4:
                    break
                # Affine LSQ: cur -> gallery-white on current matches.
                src = np.hstack([qwhite[matches[:, 0]],
                                 np.ones((len(matches), 1))])
                dst = entry["white"][matches[:, 1]]
                aff, *_ = np.linalg.lstsq(src, dst, rcond=None)
                cur = np.hstack([qwhite, np.ones((len(qwhite), 1))]) @ aff
                matches = _mutual_matches(cur, entry["white"], loose)
            # Score with the tight radius only: chance alignments rarely
            # land points this precisely.
            matches = _mutual_matches(cur, entry["white"], tight)
            score = float(len(matches))
            if self.use_shape_descriptors and len(matches) and qdescs is not None:
                sims = np.einsum("ij,ij->i", qdescs[matches[:, 0]],
                                 entry["descs"][matches[:, 1]])
                score += 2.0 * float(np.clip(sims, 0.0, None).mean())
            if score > best[0]:
                best = (score, matches, aff)
        return best

    @staticmethod
    def _ransac_threshold(cents: np.ndarray) -> float:
        """Reprojection tolerance scaled to the observed blob spacing."""
        spacing = float(np.median(cKDTree(cents).query(cents, k=2)[0][:, 1]))
        return float(np.clip(0.25 * spacing, 3.0, 12.0))

    def _finalize(self, res: SurfaceMatch, entry, cents, matches) -> SurfaceMatch:
        """Fit a RANSAC homography on the matched pairs and re-assign every
        query blob to its surface spot under that homography."""
        if len(matches) < 6:
            return res
        src = entry["positions"][matches[:, 1]].astype(np.float64)
        dst = cents[matches[:, 0]].astype(np.float64)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC,
                                  self._ransac_threshold(cents))
        if H is None:
            return res
        res.homography = H
        proj = cv2.perspectiveTransform(
            entry["positions"].reshape(-1, 1, 2), H).reshape(-1, 2)
        final = _mutual_matches(cents, proj, np.inf)
        if not len(final):
            return res
        resid = np.linalg.norm(cents[final[:, 0]] - proj[final[:, 1]], axis=1)
        # Absolute gate: a correct homography places spots within a small
        # fraction of the projected spot spacing.
        spacing = float(np.median(cKDTree(proj).query(proj, k=2)[0][:, 1]))
        keep = resid < max(ASSIGN_FRACTION * spacing, ASSIGN_FLOOR_PX)
        res.assignments = [(int(qi), int(si), float(r))
                           for (qi, si), r in zip(final[keep], resid[keep])]
        res.n_matched = len(res.assignments)
        return res

    def _identify_global(self, cents, qdescs, top_k):
        """Whole-surface alignment: whiten + rotation search + ICP."""
        qwhite, _, _ = _whiten_points(cents)
        scored = []
        for entry in self._surfaces:
            score, matches, _ = self._score_surface(entry, qwhite, qdescs)
            scored.append((score, matches, entry))
        scored.sort(key=lambda t: t[0], reverse=True)
        out = []
        for score, matches, entry in scored[:top_k]:
            res = SurfaceMatch(
                surface_id=entry["id"], score=score, n_matched=len(matches),
                n_query_spots=len(cents),
                n_surface_spots=len(entry["positions"]), mode="global",
            )
            out.append(self._finalize(res, entry, cents, matches))
        return out

    def _identify_partial(self, cents, top_k):
        """Partial-view fallback: local constellation signatures vote for
        surfaces; candidate correspondences are verified with RANSAC."""
        if self._sig_index is None:
            self._prepare_signatures()
        qsig, qowner = _local_signatures(cents)
        if not len(qsig) or qsig.shape[1] != self._sig_index.shape[1]:
            return []
        d, hits = self._sig_tree.query(qsig, k=SIG_CANDIDATES)
        d = np.atleast_2d(d)
        hits = np.atleast_2d(hits)
        # Best hit per (query blob, surface): a blob emits several
        # leave-one-out rows, but may vote once per surface.
        best_pair: dict[tuple, tuple] = {}
        for row in range(len(qsig)):
            qi = int(qowner[row])
            for c in range(SIG_CANDIDATES):
                si, spot = self._sig_owner[hits[row, c]]
                w = 1.0 / (1.0 + 50.0 * d[row, c])
                key = (qi, si)
                if key not in best_pair or w > best_pair[key][0]:
                    best_pair[key] = (w, spot)
        votes = np.zeros(len(self._surfaces))
        for (qi, si), (w, _) in best_pair.items():
            votes[si] += w
        order = np.argsort(votes)[::-1][:max(top_k, 2)]
        out = []
        for si in order:
            entry = self._surfaces[si]
            pairs = [(qi, spot) for (qi, s2), (_, spot) in best_pair.items()
                     if s2 == si]
            res = SurfaceMatch(
                surface_id=entry["id"], score=0.0, n_matched=0,
                n_query_spots=len(cents),
                n_surface_spots=len(entry["positions"]), mode="partial",
            )
            if len(pairs) >= 6:
                pairs = np.array(pairs)
                src = entry["positions"][pairs[:, 1]].astype(np.float64)
                dst = cents[pairs[:, 0]].astype(np.float64)
                H, inliers = cv2.findHomography(
                    src, dst, cv2.RANSAC,
                    max(self._ransac_threshold(cents), 6.0), maxIters=5000)
                if H is not None and inliers is not None and inliers.sum() >= 6:
                    matches = pairs[inliers.ravel().astype(bool)]
                    # De-duplicate (a query blob may appear in several pairs).
                    seen_q, seen_s, uniq = set(), set(), []
                    for qi, sp in matches:
                        if qi not in seen_q and sp not in seen_s:
                            uniq.append((qi, sp))
                            seen_q.add(qi); seen_s.add(sp)
                    res = self._finalize(res, entry, cents, np.array(uniq))
                    # Partial views: keep only assignments near observed blobs
                    # (the homography also projects spots outside the frame).
                    res.score = float(res.n_matched)
            out.append(res)
        out.sort(key=lambda r: r.score, reverse=True)
        return out[:top_k]

    def _query_features(self, img):
        contours = segment_all_spots(img, min_area=self.min_blob_area)
        if len(contours) < 8:
            return None, None
        cents = []
        descs = []
        for c in contours:
            m = cv2.moments(c.astype(np.float32))
            if abs(m["m00"]) < 1e-9:
                cents.append(c.mean(axis=0))
            else:
                cents.append([m["m10"] / m["m00"], m["m01"] / m["m00"]])
            if self.use_shape_descriptors:
                descs.append(describe_contour(c))
        cents = np.array(cents)
        qdescs = None
        if self.use_shape_descriptors:
            dim = max((len(d) for d in descs if d is not None), default=0)
            qdescs = np.zeros((len(descs), dim))
            for i, d in enumerate(descs):
                if d is not None and len(d) == dim:
                    qdescs[i] = d / max(np.linalg.norm(d), 1e-9)
        return cents, qdescs

    def identify(self, img: np.ndarray, top_k: int = 1,
                 mode: str = "auto") -> list[SurfaceMatch]:
        """Identify the surface in ``img`` and label every matched spot.

        mode: "global" (whole surface visible), "partial" (cropped /
        partial view), or "auto" — try global, fall back to partial when
        too few blobs matched.
        """
        if not self._surfaces:
            return []
        cents, qdescs = self._query_features(img)
        self.last_query_centroids = cents
        if cents is None:
            return []
        if mode in ("global", "auto"):
            results = self._identify_global(cents, qdescs, top_k)
            n = results[0].n_matched if results else 0
            if mode == "global" or n >= max(
                    GLOBAL_ACCEPT_MIN, GLOBAL_ACCEPT_FRACTION * len(cents)):
                return results
        return self._identify_partial(cents, top_k)
