"""Visualize real sevengill spot matching: re-sighting matches, homography
alignments, and per-individual constellation fingerprints."""

import os

import cv2
import numpy as np

from .realdata import group_by_individual, load_dataset, resighting_pairs
from .real_reid import _CentroidSurface, build_matcher, identify_centroids
from .surface_matcher import SurfaceMatcher

OUT = os.environ.get("SPOTID_VIZ_OUT", "viz_out")


def _fit(img, H):
    h, w = img.shape[:2]
    s = H / h
    return cv2.resize(img, (int(w * s), H)), s


def _label(img, text, org, color=(40, 200, 60), scale=0.8):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 2, cv2.LINE_AA)


def match_pair_panel(ind, a, b, res, H=520):
    """Side-by-side photos with matched spots color-linked."""
    imgA = cv2.imread(a.image_path)
    imgB = cv2.imread(b.image_path)
    imgA, sA = _fit(imgA, H)
    imgB, sB = _fit(imgB, H)
    gap = 30
    panel = np.full((H, imgA.shape[1] + gap + imgB.shape[1], 3), 30, np.uint8)
    panel[:, :imgA.shape[1]] = imgA
    xoff = imgA.shape[1] + gap
    panel[:, xoff:xoff + imgB.shape[1]] = imgB
    for c in a.centroids * sA:
        cv2.circle(panel, (int(c[0]), int(c[1])), 2, (110, 110, 110), 1)
    for c in b.centroids * sB:
        cv2.circle(panel, (int(c[0] + xoff), int(c[1])), 2, (110, 110, 110), 1)
    rng = np.random.default_rng(1)
    ok = bool(res) and res[0].surface_id == ind
    for k, (qi, si, _) in enumerate(res[0].assignments if res else []):
        col = tuple(int(v) for v in rng.integers(60, 245, 3))
        pa = a.centroids[si] * sA
        pb = b.centroids[qi] * sB
        cv2.circle(panel, (int(pa[0]), int(pa[1])), 4, col, 2)
        cv2.circle(panel, (int(pb[0] + xoff), int(pb[1])), 4, col, 2)
        if k % 2 == 0:
            cv2.line(panel, (int(pa[0]), int(pa[1])),
                     (int(pb[0] + xoff), int(pb[1])), col, 1, cv2.LINE_AA)
    verdict = "MATCH" if ok else f"MISS -> {res[0].surface_id if res else '-'}"
    vcol = (60, 200, 60) if ok else (60, 60, 230)
    _label(panel, f"{ind}: {res[0].n_matched if res else 0} spots linked  [{verdict}]",
           (14, 30), vcol)
    _label(panel, "photo 1", (14, H - 14), (200, 200, 60), 0.6)
    _label(panel, "photo 2", (xoff + 14, H - 14), (200, 200, 60), 0.6)
    return panel


def alignment_overlay(ind, a, b, res, H=560):
    """Warp photo 1 into photo 2's frame via the recovered homography and
    blend, so the two sharks' spots visibly coincide."""
    if res is None or res[0].homography is None:
        return None
    imgA = cv2.imread(a.image_path)
    imgB = cv2.imread(b.image_path)
    hB, wB = imgB.shape[:2]
    warpedA = cv2.warpPerspective(imgA, res[0].homography, (wB, hB))
    blend = cv2.addWeighted(imgB, 0.5, warpedA, 0.5, 0)
    # mark B spots (green) and A spots projected through H (magenta)
    projA = cv2.perspectiveTransform(
        a.centroids.reshape(-1, 1, 2).astype(np.float64),
        res[0].homography).reshape(-1, 2)
    for c in b.centroids:
        cv2.circle(blend, (int(c[0]), int(c[1])), 6, (60, 230, 60), 2)
    for c in projA:
        cv2.circle(blend, (int(c[0]), int(c[1])), 3, (230, 60, 230), -1)
    blend, _ = _fit(blend, H)
    _label(blend, f"{ind}: photo1 warped onto photo2 (green=photo2 spots, "
                  f"magenta=photo1 aligned)", (14, 30), (255, 255, 255), 0.62)
    return blend


def constellation_sheet(shots, cols=6, cell=230, min_spots=40):
    """Contact sheet of individual constellation 'fingerprints' (spots as
    dots, normalized), one tile per well-annotated individual."""
    by_ind = group_by_individual([s for s in shots if s.n_spots >= min_spots])
    inds = sorted(by_ind)
    rows = (len(inds) + cols - 1) // cols
    sheet = np.full((rows * cell, cols * cell, 3), 255, np.uint8)
    for i, ind in enumerate(inds):
        s = by_ind[ind][0]
        pts = s.centroids.copy()
        mn, mx = pts.min(0), pts.max(0)
        span = (mx - mn).max() + 1e-6
        pts = (pts - mn) / span * (cell - 40) + 20
        r, c = divmod(i, cols)
        tile = sheet[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell]
        for p in pts:
            cv2.circle(tile, (int(p[0]), int(p[1])), 2, (40, 40, 40), -1)
        cv2.rectangle(tile, (2, 2), (cell - 3, cell - 3), (210, 210, 210), 1)
        _label(tile, f"{ind} ({s.n_spots})", (8, cell - 10), (150, 60, 40), 0.42)
    return sheet


def main():
    os.makedirs(OUT, exist_ok=True)
    shots = load_dataset("realdata/realworldspots.yolov8")
    pairs = resighting_pairs(shots)
    well = [s for s in shots if s.n_spots >= 40]

    panels = []
    for ind, group in pairs:
        a, b = group[0], group[1]
        m = SurfaceMatcher(use_shape_descriptors=False)
        m.enroll_surface(_CentroidSurface(ind, a.centroids))
        # gallery for honest verdict: include impostors
        gal = [a] + [g[0] for k, g in group_by_individual(well).items() if k != ind]
        gm, _ = build_matcher(gal)
        res = identify_centroids(gm, b.centroids, top_k=len(gal))
        panels.append(match_pair_panel(ind, a, b, res))
        ov = alignment_overlay(ind, a, b, res)
        if ov is not None:
            cv2.imwrite(f"{OUT}/align_{ind}.png", ov)

    if not panels:
        raise SystemExit(
            "no re-sighting pairs found — check that the dataset is unpacked "
            "at realdata/realworldspots.yolov8 (see spotid/README.md)")

    w = max(p.shape[1] for p in panels)
    stacked = np.vstack([cv2.copyMakeBorder(p, 6, 6, 0, w - p.shape[1],
                         cv2.BORDER_CONSTANT, value=(30, 30, 30)) for p in panels])
    cv2.imwrite(f"{OUT}/resighting_matches.png", stacked)
    cv2.imwrite(f"{OUT}/constellations.png", constellation_sheet(shots))
    print("wrote:", os.listdir(OUT))


if __name__ == "__main__":
    main()
