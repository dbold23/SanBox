"""Surface-level evaluation: identify which surface (of many) an image
shows, and label every individual spot on it, from full and partial views
at random tilt angles.

Usage:
    python -m spotid.evaluate_surface --surfaces 12 --spots 600 --views 6
"""

import argparse
import time

import numpy as np
from scipy.spatial import cKDTree

from .surface import SurfaceViewConfig, generate_surface, render_surface_view
from .surface_matcher import SurfaceMatcher

# Ground-truth association gate: a query blob centroid must lie within this
# many pixels of a true projected spot center to be scored.
GT_GATE_PX = 6.0


def score_view(matcher, surface, img, info, offset=(0.0, 0.0)):
    """Return (predicted_id, mode, n_good, n_bad, n_visible_gt).

    A spot counts as visible when its center is in frame, it was actually
    drawn (not worn away), and it is not hidden under glare."""
    res = matcher.identify(img, mode="auto")
    gt = info["spot_centroids_px"] - np.asarray(offset)
    h, w = img.shape
    in_frame = ((gt[:, 0] >= 0) & (gt[:, 0] < w)
                & (gt[:, 1] >= 0) & (gt[:, 1] < h))
    present = in_frame
    n = len(gt)
    present = present & info.get("drawn", np.ones(n, bool))
    present = present & ~info.get("obscured", np.zeros(n, bool))
    visible = int(np.sum(present))
    if not res:
        return None, "none", 0, 0, visible
    r = res[0]
    good = bad = 0
    if r.surface_id == surface.surface_id and r.assignments:
        # identify() already segmented the image; its assignment indices
        # refer to exactly these centroids.
        cents = matcher.last_query_centroids
        tree = cKDTree(gt)
        for qi, si, _ in r.assignments:
            d, j = tree.query(cents[qi])
            # Score only blobs that land on a countably-visible spot, so
            # coverage stays comparable to the visible denominator.
            if d < GT_GATE_PX and present[j]:
                if j == si:
                    good += 1
                else:
                    bad += 1
    return r.surface_id, r.mode, good, bad, visible


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--surfaces", type=int, default=12)
    ap.add_argument("--spots", type=int, default=600)
    ap.add_argument("--views", type=int, default=6,
                    help="views per surface (half full, half cropped)")
    ap.add_argument("--crop", type=float, default=0.55,
                    help="cropped views keep this fraction of each side")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    t0 = time.time()
    surfaces = [generate_surface(i, n_spots=args.spots)
                for i in range(args.surfaces)]
    matcher = SurfaceMatcher()
    for s in surfaces:
        matcher.enroll_surface(s)
    print(f"enrolled {args.surfaces} surfaces x {args.spots} spots "
          f"in {time.time() - t0:.0f}s", flush=True)

    rng = np.random.default_rng(args.seed)
    cfg = SurfaceViewConfig()
    stats = {"full": [0, 0, 0, 0, 0], "crop": [0, 0, 0, 0, 0]}
    # per kind: [surface_ok, n_queries, assign_good, assign_bad, visible]
    t0 = time.time()
    n_q = 0
    for s in surfaces:
        for v in range(args.views):
            img, info = render_surface_view(s, rng, cfg)
            kind = "full" if v % 2 == 0 else "crop"
            offset = (0.0, 0.0)
            if kind == "crop":
                h, w = img.shape
                ch, cw = int(h * args.crop), int(w * args.crop)
                y0 = int(rng.integers(0, h - ch))
                x0 = int(rng.integers(0, w - cw))
                img = img[y0:y0 + ch, x0:x0 + cw]
                offset = (x0, y0)
            pred, mode, good, bad, visible = score_view(
                matcher, s, img, info, offset)
            st = stats[kind]
            st[0] += pred == s.surface_id
            st[1] += 1
            st[2] += good
            st[3] += bad
            st[4] += visible
            n_q += 1
            print(f"  s{s.surface_id:02d} {kind:4s} tilt "
                  f"{info['view']['tilt_deg']:4.0f} -> pred "
                  f"{pred if pred is not None else '-'} ({mode}) "
                  f"spots {good}/{good + bad} of {visible} visible",
                  flush=True)

    print(f"\n=== surface-level results "
          f"({(time.time() - t0) / n_q:.1f}s/query) ===")
    for kind, st in stats.items():
        ok, n, good, bad, vis = st
        if not n:
            continue
        prec = good / max(good + bad, 1)
        cov = good / max(vis, 1)
        print(f"{kind:4s} views: surface top-1 {ok}/{n} ({ok / n:.4f}) | "
              f"spot assignment precision {prec:.4f} ({good}/{good + bad}) | "
              f"coverage {cov:.4f} of visible spots")


if __name__ == "__main__":
    main()
