"""Full pairwise match matrix across real photos, to reveal how many
distinct individuals are actually present and expose false matches.

Score = fraction of the smaller spot set matched under one homography with
tight residuals (a scale-free measure that a large true overlap should
push high and coincidental alignments should not)."""

import re

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .realdata import load_dataset
from .real_reid import _CentroidSurface, identify_centroids
from .surface_matcher import SurfaceMatcher

OUT = "/tmp/claude-0/-home-user-SanBox/b501c8e0-afbc-56a3-aade-8375de946ebf/scratchpad/viz"


def coarse(name):
    m = re.match(r'([A-Za-z]+_[A-Za-z])', name)
    return m.group(1) if m else name


def match_fraction(a, b):
    m = SurfaceMatcher(use_shape_descriptors=False)
    m.enroll_surface(_CentroidSurface(a.individual, a.centroids))
    res = identify_centroids(m, b.centroids, top_k=1)
    if not res or not res[0].assignments:
        return 0.0, 0
    n = res[0].n_matched
    return n / max(min(len(a.centroids), len(b.centroids)), 1), n


def main():
    shots = [s for s in load_dataset("realdata/realworldspots.yolov8")
             if s.n_spots >= 40]
    shots.sort(key=lambda s: (coarse(s.individual), s.individual))
    N = len(shots)
    frac = np.zeros((N, N))
    cnt = np.zeros((N, N), int)
    for i in range(N):
        for j in range(N):
            if i == j:
                frac[i, j] = 1.0
                continue
            frac[i, j], cnt[i, j] = match_fraction(shots[i], shots[j])
        print(f"row {i+1}/{N} ({shots[i].individual}) done", flush=True)

    labels = [s.individual for s in shots]
    groups = [coarse(s.individual) for s in shots]
    np.savez(f"{OUT}/match_matrix.npz", frac=frac, cnt=cnt,
             labels=labels, groups=groups)

    # heatmap
    sym = np.maximum(frac, frac.T)
    vis = (np.clip(sym, 0, 0.5) / 0.5 * 255).astype(np.uint8)
    hm = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
    scale = 18
    hm = cv2.resize(hm, (N * scale, N * scale), interpolation=cv2.INTER_NEAREST)
    # group divider lines
    prev = None
    for k, g in enumerate(groups):
        if g != prev:
            cv2.line(hm, (k * scale, 0), (k * scale, N * scale), (80, 255, 80), 1)
            cv2.line(hm, (0, k * scale), (N * scale, k * scale), (80, 255, 80), 1)
            cv2.putText(hm, g, (k * scale + 2, 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (80, 255, 80), 1, cv2.LINE_AA)
            prev = g
    cv2.imwrite(f"{OUT}/match_matrix.png", hm)

    # summary: within-group vs cross-group fraction
    groups = np.array(groups)
    within, cross = [], []
    for i in range(N):
        for j in range(i + 1, N):
            (within if groups[i] == groups[j] else cross).append(sym[i, j])
    print(f"\nwithin-group fraction: mean {np.mean(within):.3f} "
          f"p90 {np.percentile(within,90):.3f}")
    print(f"cross-group  fraction: mean {np.mean(cross):.3f} "
          f"p90 {np.percentile(cross,90):.3f}")
    print(f"separation (within.mean - cross.p90): "
          f"{np.mean(within) - np.percentile(cross,90):+.3f}")
    print("wrote match_matrix.png")


if __name__ == "__main__":
    main()
