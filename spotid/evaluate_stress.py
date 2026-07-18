"""Real-world stress matrix for surface identification.

Each condition isolates one degradation (plus a combined worst case):
harsh lighting, glossy sheen, missing + clutter spots, faded spots, far
away / low resolution, close-up partial crops, cracks — on surfaces whose
spots vary in size (specks to blobs) and elongation (round to streaks).

Usage:
    python -m spotid.evaluate_stress --surfaces 10 --spots 600 --views 4
"""

import argparse
import time

import numpy as np

from .evaluate_surface import score_view
from .surface import (SurfaceViewConfig, generate_surface,
                      harsh_view_config, render_surface_view)
from .surface_matcher import SurfaceMatcher


def conditions() -> list[tuple]:
    return [
        ("baseline", SurfaceViewConfig(), False),
        ("harsh lighting", SurfaceViewConfig(
            gamma_range=(0.6, 1.6), gradient_strength=(0.2, 0.45),
            vignette_strength=(0.2, 0.45), contrast_range=(0.22, 0.45)), False),
        ("glossy sheen", SurfaceViewConfig(
            gloss_range=(3, 7), gloss_strength=(0.55, 0.95)), False),
        ("missing + clutter", SurfaceViewConfig(
            dropout_range=(0.1, 0.25), clutter_range=(15, 40)), False),
        ("heavy fade", SurfaceViewConfig(
            contrast_range=(0.2, 0.4)), False),
        ("far away", SurfaceViewConfig(
            fill_range=(0.35, 0.5), resolution_range=(0.45, 0.7),
            blur_sigma_range=(0.4, 1.2)), False),
        ("close-up crop", SurfaceViewConfig(), True),
        ("cracks + texture", SurfaceViewConfig(
            crack_range=(3, 6), noise_sigma_range=(0.015, 0.035)), False),
        ("everything at once", harsh_view_config(), False),
        ("everything + crop", harsh_view_config(fill_range=(0.8, 0.95)), True),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--surfaces", type=int, default=10)
    ap.add_argument("--spots", type=int, default=600)
    ap.add_argument("--views", type=int, default=4, help="views per surface per condition")
    ap.add_argument("--crop", type=float, default=0.55)
    ap.add_argument("--seed", type=int, default=9)
    args = ap.parse_args()

    t0 = time.time()
    surfaces = [generate_surface(i, n_spots=args.spots)
                for i in range(args.surfaces)]
    matcher = SurfaceMatcher()
    for s in surfaces:
        matcher.enroll_surface(s)
    print(f"enrolled {args.surfaces} surfaces x {args.spots} spots "
          f"in {time.time() - t0:.0f}s\n", flush=True)

    rows = []
    for name, cfg, do_crop in conditions():
        rng = np.random.default_rng(args.seed)
        ok = n = good = bad = vis = 0
        modes = {"global": 0, "partial": 0, "none": 0}
        t0 = time.time()
        for s in surfaces:
            for _ in range(args.views):
                img, info = render_surface_view(s, rng, cfg)
                offset = (0.0, 0.0)
                if do_crop:
                    h, w = img.shape
                    ch, cw = int(h * args.crop), int(w * args.crop)
                    y0 = int(rng.integers(0, h - ch))
                    x0 = int(rng.integers(0, w - cw))
                    img = img[y0:y0 + ch, x0:x0 + cw]
                    offset = (x0, y0)
                pred, mode, g, b, v = score_view(matcher, s, img, info, offset)
                ok += pred == s.surface_id
                n += 1
                good += g
                bad += b
                vis += v
                modes[mode if mode in modes else "none"] += 1
        prec = good / max(good + bad, 1)
        cov = good / max(vis, 1)
        dt = (time.time() - t0) / n
        rows.append((name, ok, n, prec, good, good + bad, cov, modes, dt))
        print(f"{name:22s} surface {ok:3d}/{n:3d}  spot precision "
              f"{prec:.4f} ({good}/{good + bad})  coverage {cov:.3f}  "
              f"modes {modes}  {dt:.1f}s/query", flush=True)

    print("\n=== stress matrix summary ===")
    print(f"{'condition':22s} {'surface':>10s} {'precision':>10s} {'coverage':>9s}")
    for name, ok, n, prec, g, gb, cov, modes, dt in rows:
        print(f"{name:22s} {ok:>5d}/{n:<4d} {prec:>10.4f} {cov:>9.3f}")


if __name__ == "__main__":
    main()
