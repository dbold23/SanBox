"""Head-to-head: learned embedding vs handcrafted descriptor.

Both are evaluated on the *same* rendered query images with the same
enrollment protocol and the same Fisher-weighted matcher, so the only
variable is the descriptor.

Usage:
    python -m spotid.ml.evaluate_ml --checkpoint spotid/ml/checkpoints/encoder.pt \
        --identities 100 --views 30
"""

import argparse
import time

import numpy as np

from ..evaluate import ENROLL_TILTS, ENROLL_PER_TILT
from ..features import describe_image
from ..matcher import SpotMatcher
from ..render import ViewConfig, render_view
from ..shapes import generate_identity
from .infer import MLSpotDescriptor


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="spotid/ml/checkpoints/encoder.pt")
    ap.add_argument("--identities", type=int, default=100)
    ap.add_argument("--views", type=int, default=30)
    ap.add_argument("--tilt-max", type=float, default=55.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ml = MLSpotDescriptor(args.checkpoint, args.device)
    describers = {"classical": describe_image, "learned": ml.describe_image}
    matchers = {name: SpotMatcher() for name in describers}

    t0 = time.time()
    ecfg = ViewConfig()
    for s in range(args.identities):
        ident = generate_identity(s)
        descs = {name: [] for name in describers}
        rng = np.random.default_rng(np.random.SeedSequence([913_001, s]))
        for tilt in ENROLL_TILTS:
            for _ in range(ENROLL_PER_TILT):
                img, _ = render_view(ident, rng, ecfg, tilt_deg=tilt)
                for name, fn in describers.items():
                    descs[name].append(fn(img))
        for name in describers:
            matchers[name].enroll(s, descs[name])
    print(f"enrolled {args.identities} identities for both descriptors "
          f"in {time.time() - t0:.0f}s", flush=True)

    qcfg = ViewConfig(tilt_max_deg=args.tilt_max)
    rng = np.random.default_rng(args.seed)
    buckets = {name: {} for name in describers}
    t0 = time.time()
    n_q = 0
    for s in range(args.identities):
        ident = generate_identity(s)
        for _ in range(args.views):
            img, info = render_view(ident, rng, qcfg)
            n_q += 1
            b = int(info["tilt_deg"] // 15) * 15
            for name, fn in describers.items():
                res = matchers[name].identify_descriptor(fn(img))
                hit = bool(res) and res[0][0] == s
                bk = buckets[name].setdefault(b, [0, 0])
                bk[0] += hit
                bk[1] += 1

    print(f"\n=== learned vs classical ({n_q} queries, "
          f"{n_q / (time.time() - t0):.1f}/s) ===")
    for name in describers:
        bk = buckets[name]
        c = sum(v[0] for v in bk.values())
        n = sum(v[1] for v in bk.values())
        by_tilt = "  ".join(f"{k}-{k + 15}: {v[0] / v[1]:.3f}"
                            for k, v in sorted(bk.items()))
        print(f"{name:10s} top-1 {c / n:.4f} ({c}/{n})   {by_tilt}")


if __name__ == "__main__":
    main()
