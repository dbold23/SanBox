"""Generate demo images: permutations of one spot, cross-angle matches,
and an annotated surface identification.

Usage:
    python -m spotid.demo --out-dir demo_out
"""

import argparse
import os

import cv2
import numpy as np

from .evaluate import enroll_identity
from .matcher import SpotMatcher
from .render import ViewConfig, render_view
from .shapes import generate_identity
from .surface import generate_surface, render_surface_view
from .surface_matcher import SurfaceMatcher


def permutation_grid(seed: int, rows: int = 3, cols: int = 6) -> np.ndarray:
    """One identity rendered under rows*cols random permutations."""
    ident = generate_identity(seed)
    rng = np.random.default_rng(101)
    tiles = []
    for r in range(rows):
        row = []
        for c in range(cols):
            img, info = render_view(ident, rng)
            tile = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.putText(tile, f"tilt {info['tilt_deg']:.0f}", (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 90, 220), 1,
                        cv2.LINE_AA)
            row.append(tile)
        tiles.append(np.hstack(row))
    return np.vstack(tiles)


def match_grid(n_ids: int = 4, views: int = 5) -> np.ndarray:
    """Different identities from different angles, with predictions."""
    matcher = SpotMatcher()
    for s in range(n_ids):
        enroll_identity(matcher, s)
    rng = np.random.default_rng(55)
    rows = []
    for s in range(n_ids):
        ident = generate_identity(s)
        row = []
        for _ in range(views):
            img, info = render_view(ident, rng)
            res = matcher.identify(img)
            pred = res[0][0] if res else "?"
            ok = pred == s
            tile = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            color = (40, 160, 40) if ok else (30, 30, 220)
            cv2.putText(tile, f"true {s} pred {pred}", (6, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            cv2.putText(tile, f"tilt {info['tilt_deg']:.0f}", (6, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1,
                        cv2.LINE_AA)
            row.append(tile)
        rows.append(np.hstack(row))
    return np.vstack(rows)


def surface_demo(n_spots: int = 600) -> np.ndarray:
    """Annotated identification of a tilted surface with n_spots spots."""
    surfaces = [generate_surface(i, n_spots=n_spots) for i in range(3)]
    matcher = SurfaceMatcher()
    for s in surfaces:
        matcher.enroll_surface(s)
    rng = np.random.default_rng(21)
    img, info = render_surface_view(surfaces[1], rng, tilt_deg=40.0)
    res = matcher.identify(img)[0]
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if res.homography is not None:
        proj = cv2.perspectiveTransform(
            matcher._surfaces[1]["positions"].reshape(-1, 1, 2),
            res.homography).reshape(-1, 2)
        for _, si, _ in res.assignments:
            x, y = proj[si]
            cv2.putText(out, str(si), (int(x) - 8, int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (40, 150, 40), 1,
                        cv2.LINE_AA)
    banner = (f"pred surface {res.surface_id} ({res.mode}) | "
              f"{res.n_matched}/{res.n_query_spots} spots identified | "
              f"tilt {info['view']['tilt_deg']:.0f} deg")
    cv2.putText(out, banner, (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (30, 90, 220), 2, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="demo_out")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(args.out_dir, "permutations.png"),
                permutation_grid(seed=3))
    print("wrote permutations.png")
    cv2.imwrite(os.path.join(args.out_dir, "matches.png"), match_grid())
    print("wrote matches.png")
    cv2.imwrite(os.path.join(args.out_dir, "surface.png"), surface_demo())
    print("wrote surface.png")


if __name__ == "__main__":
    main()
