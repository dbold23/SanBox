"""Spot-level evaluation: identify single splotches across thousands of
permutations (rotation, scale, perspective tilt, lighting, noise).

Usage:
    python -m spotid.evaluate --identities 150 --views 70 --seed 7
"""

import argparse
import time

import numpy as np

from .features import describe_image
from .matcher import SpotMatcher
from .render import ViewConfig, render_view
from .shapes import generate_identity

# Enrollment poses: uniform tilt coverage so the matcher's within-identity
# noise estimate reflects the query distribution.
ENROLL_TILTS = (0, 12, 24, 34, 42, 50, 56)
ENROLL_PER_TILT = 2


def enroll_identity(matcher: SpotMatcher, seed: int,
                    cfg: ViewConfig = ViewConfig()) -> None:
    """Enroll spot identity ``seed`` from stratified-tilt sample views."""
    ident = generate_identity(seed)
    rng = np.random.default_rng(np.random.SeedSequence([913_001, seed]))
    descs = []
    for tilt in ENROLL_TILTS:
        for _ in range(ENROLL_PER_TILT):
            img, _ = render_view(ident, rng, cfg, tilt_deg=tilt)
            descs.append(describe_image(img))
    matcher.enroll(seed, descs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--identities", type=int, default=150)
    ap.add_argument("--views", type=int, default=70,
                    help="query permutations per identity")
    ap.add_argument("--tilt-max", type=float, default=55.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    qcfg = ViewConfig(tilt_max_deg=args.tilt_max)
    t0 = time.time()
    matcher = SpotMatcher()
    for s in range(args.identities):
        enroll_identity(matcher, s)
    print(f"enrolled {len(matcher)} identities "
          f"({len(ENROLL_TILTS) * ENROLL_PER_TILT} views each) "
          f"in {time.time() - t0:.0f}s", flush=True)

    rng = np.random.default_rng(args.seed)
    buckets: dict[int, list[int]] = {}
    seg_fail = 0
    n_total = args.identities * args.views
    t0 = time.time()
    done = 0
    for s in range(args.identities):
        ident = generate_identity(s)
        for _ in range(args.views):
            img, info = render_view(ident, rng, qcfg)
            res = matcher.identify(img)
            bucket = int(info["tilt_deg"] // 15) * 15
            hit = bool(res) and res[0][0] == s
            if not res:
                seg_fail += 1
            b = buckets.setdefault(bucket, [0, 0])
            b[0] += hit
            b[1] += 1
            done += 1
        if (s + 1) % 25 == 0:
            print(f"  {done}/{n_total} queries...", flush=True)

    correct = sum(b[0] for b in buckets.values())
    total = sum(b[1] for b in buckets.values())
    print(f"\n=== spot-level results ===")
    print(f"gallery: {args.identities} identities | "
          f"queries: {total} permutations | "
          f"query rate {total / (time.time() - t0):.1f}/s")
    print(f"top-1 accuracy: {correct / total:.4f} ({correct}/{total})"
          + (f", segmentation failures: {seg_fail}" if seg_fail else ""))
    print("by tilt:")
    for k in sorted(buckets):
        c, n = buckets[k]
        print(f"  {k:2d}-{k + 15:2d} deg: {c / n:.4f} ({c}/{n})")


if __name__ == "__main__":
    main()
