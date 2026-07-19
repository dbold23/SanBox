"""Measure a spot descriptor on REAL cross-view spot pairs.

The constellation matcher, run on re-sighted individuals, yields
geometry-verified correspondences: spot i in photo A is the same physical
spot as spot j in photo B. Those are ground-truth positive pairs for an
*appearance* descriptor (an independent signal from the geometry that
produced them). We embed both spots of each pair and ask: are matched
spots more similar than mismatched spots? Reported as verification AUC and
the mean same/different similarity gap — for the classical descriptor, the
synthetic encoder, and the real-fine-tuned encoder.

Usage:
    python -m spotid.ml.eval_real_embed --root realdata/realworldspots.yolov8 \
        --checkpoints spotid/ml/checkpoints/encoder_gpu.pt \
                      spotid/ml/checkpoints/encoder_real.pt
"""

import argparse

import cv2
import numpy as np

from ..features import describe_image as classical_describe
from ..realdata import load_dataset, resighting_pairs
from ..real_reid import _CentroidSurface, identify_centroids
from ..surface_matcher import SurfaceMatcher
from .infer import MLSpotDescriptor


def _crop(img, px, py, r, pad=0.6):
    h, w = img.shape
    x0, x1 = int(max(0, px - r * (1 + pad))), int(min(w, px + r * (1 + pad)))
    y0, y1 = int(max(0, py - r * (1 + pad))), int(min(h, py + r * (1 + pad)))
    c = img[y0:y1, x0:x1]
    return c if c.shape[0] >= 12 and c.shape[1] >= 12 else None


def gather_correspondences(root):
    """Return list of (cropA, cropB) for geometry-verified same-spot pairs
    across re-sighting photos."""
    shots = load_dataset(root)
    pairs = resighting_pairs(shots)
    corr = []
    for ind, group in pairs:
        a, b = group[0], group[1]
        m = SurfaceMatcher(use_shape_descriptors=False)
        m.enroll_surface(_CentroidSurface(ind, a.centroids))
        res = identify_centroids(m, b.centroids, top_k=1)
        if not res or res[0].surface_id != ind:
            continue
        imgA = cv2.imread(a.image_path, cv2.IMREAD_GRAYSCALE)
        imgB = cv2.imread(b.image_path, cv2.IMREAD_GRAYSCALE)
        for qi, si, _ in res[0].assignments:
            ra = 0.5 * max(a.boxes[si][2], a.boxes[si][3])
            rb = 0.5 * max(b.boxes[qi][2], b.boxes[qi][3])
            ca = _crop(imgA, a.centroids[si][0], a.centroids[si][1], ra)
            cb = _crop(imgB, b.centroids[qi][0], b.centroids[qi][1], rb)
            if ca is not None and cb is not None:
                corr.append((ca, cb))
    return corr


def _embeddings(describe, crops):
    out = []
    for c in crops:
        e = describe(c)
        out.append(None if e is None else e / (np.linalg.norm(e) + 1e-9))
    return out


def _auc(pos, neg):
    """AUC that a positive similarity exceeds a negative (Mann-Whitney)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, len(allv) + 1)
    rp = ranks[:len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def evaluate(name, embA, embB):
    valid = [(a, b) for a, b in zip(embA, embB) if a is not None and b is not None]
    if len(valid) < 5:
        print(f"{name:22s}: too few valid pairs ({len(valid)})")
        return
    A = np.stack([a for a, _ in valid])
    B = np.stack([b for _, b in valid])
    pos = np.sum(A * B, axis=1)
    # negatives: each A vs a shuffled (non-matching) B
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(B))
    perm[perm == np.arange(len(B))] = (perm[perm == np.arange(len(B))] + 1) % len(B)
    neg = np.sum(A * B[perm], axis=1)
    print(f"{name:22s}: {len(valid):4d} real pairs | same-sim {pos.mean():.3f} "
          f"diff-sim {neg.mean():.3f} gap {pos.mean()-neg.mean():+.3f} | "
          f"verification AUC {_auc(pos, neg):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="realdata/realworldspots.yolov8")
    ap.add_argument("--checkpoints", nargs="*",
                    default=["spotid/ml/checkpoints/encoder_gpu.pt",
                             "spotid/ml/checkpoints/encoder_real.pt"])
    args = ap.parse_args()

    corr = gather_correspondences(args.root)
    cropsA = [a for a, _ in corr]
    cropsB = [b for _, b in corr]
    print(f"gathered {len(corr)} geometry-verified real same-spot pairs\n")

    evaluate("classical", _embeddings(classical_describe, cropsA),
             _embeddings(classical_describe, cropsB))
    for ck in args.checkpoints:
        try:
            d = MLSpotDescriptor(ck)
        except Exception as e:
            print(f"{ck}: load failed ({e})")
            continue
        tag = "learned:" + ck.split("/")[-1].replace(".pt", "")
        evaluate(tag, _embeddings(d.describe_image, cropsA),
                 _embeddings(d.describe_image, cropsB))


if __name__ == "__main__":
    main()
