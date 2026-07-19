"""Sevengill individual identifier — a usable catalog tool.

Wraps the validated appearance matcher (DISK deep local features + RANSAC
verification) into a real workflow:

    # build a catalog from a folder of photos (filenames encode the
    # individual id, or pass --flat to treat each photo as its own id)
    python -m spotid.identify enroll  --images catalog/ --out catalog.npz

    # identify a new photo against the catalog
    python -m spotid.identify query   --catalog catalog.npz --image new.jpg

    # batch: identify every photo in a folder, leave-one-out if they are
    # already in the catalog
    python -m spotid.identify batch   --catalog catalog.npz --images test/

It matches the flank *pixels*, so it does NOT need spot annotations — only
a crop that contains the flank. If given a full photo it uses the whole
frame; pass a bbox or pre-crop for best results.

Scores: the raw score is the count of RANSAC-verified inlier
correspondences; ``score_norm`` normalizes by the smaller keypoint count so
the open-set threshold transfers across image resolutions. A query whose
best match falls below the threshold is flagged NEW INDIVIDUAL.
"""

import argparse
import glob
import os
import re

import cv2
import numpy as np

from .probe_matchers import DiskMatcher, _ransac_inliers

# Default open-set threshold on the normalized score. Calibrated loosely
# from the sevengill leave-one-out run (genuine >= 553 raw inliers /
# normalized ~0.28+, impostors <= 43 / ~0.02); recalibrate with `calibrate`.
DEFAULT_NORM_THRESHOLD = 0.10
MAXSIDE = 1024


def _individual_from_name(fn: str) -> str:
    base = re.sub(r'\.rf\.[A-Za-z0-9]+\.(jpg|jpeg|png)$', '', fn, flags=re.I)
    base = re.sub(r'\.(jpg|jpeg|png)$', '', base, flags=re.I)
    base = re.sub(r'_(jpg|JPG|jpeg|png)$', '', base)
    return re.sub(r'\s*\(\d+\)\s*', ' ', base).strip()


def _load_gray_bgr(path, bbox=None):
    img = cv2.imread(path)
    if img is None:
        return None
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        img = img[y0:y1, x0:x1]
    s = MAXSIDE / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def score_pair(ka, da, kb, db, min_pts=8):
    """(raw inliers, normalized) between two DISK feature sets."""
    if len(ka) < min_pts or len(kb) < min_pts:
        return 0, 0.0
    da = da / (np.linalg.norm(da, axis=1, keepdims=True) + 1e-9)
    db = db / (np.linalg.norm(db, axis=1, keepdims=True) + 1e-9)
    sim = da @ db.T
    ab = sim.argmax(1)
    ba = sim.argmax(0)
    mut = [(i, ab[i]) for i in range(len(ka)) if ba[ab[i]] == i]
    if len(mut) < min_pts:
        return len(mut), 0.0
    pa = np.float32([ka[i] for i, _ in mut])
    pb = np.float32([kb[j] for _, j in mut])
    inl = _ransac_inliers(pa, pb)
    norm = inl / max(min(len(ka), len(kb)), 1)
    return inl, float(norm)


class Catalog:
    def __init__(self):
        self.names = []          # per-entry photo id (filename stem)
        self.individuals = []    # per-entry individual id
        self.kpts = []           # per-entry (M,2)
        self.descs = []          # per-entry (M,D)
        self._matcher = None

    @property
    def matcher(self):
        if self._matcher is None:
            self._matcher = DiskMatcher()
        return self._matcher

    def add(self, path, individual, bbox=None):
        img = _load_gray_bgr(path, bbox)
        if img is None:
            return False
        k, d = self.matcher._feat(img)
        self.names.append(os.path.basename(path))
        self.individuals.append(individual)
        self.kpts.append(k.astype(np.float32))
        self.descs.append(d.astype(np.float32))
        return True

    def save(self, path):
        np.savez(path,
                 names=np.array(self.names, object),
                 individuals=np.array(self.individuals, object),
                 kpts=np.array(self.kpts, object),
                 descs=np.array(self.descs, object))

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True)
        c = cls()
        c.names = list(d["names"])
        c.individuals = list(d["individuals"])
        c.kpts = [np.asarray(x, np.float32) for x in d["kpts"]]
        c.descs = [np.asarray(x, np.float32) for x in d["descs"]]
        return c

    def query_features(self, k, d, exclude_name=None, top_k=5):
        rows = []
        for i in range(len(self.names)):
            if exclude_name is not None and self.names[i] == exclude_name:
                continue
            raw, norm = score_pair(k, d, self.kpts[i], self.descs[i])
            rows.append((self.individuals[i], self.names[i], raw, norm))
        rows.sort(key=lambda r: r[2], reverse=True)
        return rows[:top_k]

    def query_image(self, path, bbox=None, top_k=5,
                    threshold=DEFAULT_NORM_THRESHOLD):
        img = _load_gray_bgr(path, bbox)
        if img is None:
            return None
        k, d = self.matcher._feat(img)
        results = self.query_features(k, d, exclude_name=os.path.basename(path),
                                      top_k=top_k)
        is_new = (not results) or results[0][3] < threshold
        return {"results": results, "new_individual": is_new,
                "threshold": threshold}


def _iter_images(folder):
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        yield from glob.glob(os.path.join(folder, ext))


def cmd_enroll(args):
    cat = Catalog()
    paths = sorted(_iter_images(args.images))
    for i, p in enumerate(paths):
        ind = os.path.splitext(os.path.basename(p))[0] if args.flat \
            else _individual_from_name(os.path.basename(p))
        ok = cat.add(p, ind)
        print(f"  [{i+1}/{len(paths)}] {'+' if ok else 'x'} {ind}", flush=True)
    cat.save(args.out)
    n_ind = len(set(cat.individuals))
    print(f"enrolled {len(cat.names)} photos / {n_ind} individuals -> {args.out}")


def cmd_query(args):
    cat = Catalog.load(args.catalog)
    out = cat.query_image(args.image, top_k=args.top_k,
                          threshold=args.threshold)
    if out is None:
        print("could not read image")
        return
    print(f"\nQuery: {os.path.basename(args.image)}")
    if out["new_individual"]:
        print(f"  ==> NEW INDIVIDUAL (best normalized score below "
              f"{out['threshold']:.2f})")
    else:
        print(f"  ==> MATCH: {out['results'][0][0]}")
    print("  ranked candidates (individual | photo | inliers | norm):")
    for ind, name, raw, norm in out["results"]:
        print(f"    {ind:16s} {name:28s} {raw:4d}  {norm:.3f}")


def cmd_batch(args):
    cat = Catalog.load(args.catalog)
    paths = sorted(_iter_images(args.images))
    correct = total = 0
    for p in paths:
        out = cat.query_image(p, top_k=args.top_k, threshold=args.threshold)
        if out is None:
            continue
        truth = _individual_from_name(os.path.basename(p))
        pred = "NEW" if out["new_individual"] else out["results"][0][0]
        # only score images whose individual has another photo in the catalog
        others = sum(1 for j, ind in enumerate(cat.individuals)
                     if ind == truth and cat.names[j] != os.path.basename(p))
        mark = ""
        if others > 0:
            total += 1
            hit = pred == truth
            correct += hit
            mark = "OK" if hit else "MISS"
        print(f"  {truth:16s} -> {pred:16s} "
              f"(norm {out['results'][0][3]:.3f}) {mark}")
    if total:
        print(f"\nre-sighted-image accuracy: {correct}/{total} "
              f"({correct/total:.3f})")


def cmd_calibrate(args):
    """Print genuine vs impostor score distributions to pick a threshold."""
    cat = Catalog.load(args.catalog)
    gen, imp = [], []
    for i in range(len(cat.names)):
        best_same = best_diff = 0.0
        for j in range(len(cat.names)):
            if i == j:
                continue
            _, norm = score_pair(cat.kpts[i], cat.descs[i],
                                 cat.kpts[j], cat.descs[j])
            if cat.individuals[j] == cat.individuals[i]:
                best_same = max(best_same, norm)
            else:
                best_diff = max(best_diff, norm)
        if any(cat.individuals[j] == cat.individuals[i] and j != i
               for j in range(len(cat.names))):
            gen.append(best_same)
        imp.append(best_diff)
    gen, imp = np.array(gen), np.array(imp)
    print(f"genuine best-match norm: n={len(gen)} min {gen.min():.3f} "
          f"mean {gen.mean():.3f}")
    print(f"impostor best-match norm: n={len(imp)} max {imp.max():.3f} "
          f"mean {imp.mean():.3f}")
    if len(gen):
        print(f"suggested threshold: {(gen.min()+imp.max())/2:.3f} "
              f"(midpoint of genuine-min and impostor-max)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("enroll", help="build a catalog from a folder")
    e.add_argument("--images", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--flat", action="store_true",
                   help="treat each photo as its own individual id")
    e.set_defaults(func=cmd_enroll)

    q = sub.add_parser("query", help="identify one image")
    q.add_argument("--catalog", required=True)
    q.add_argument("--image", required=True)
    q.add_argument("--top-k", type=int, default=5)
    q.add_argument("--threshold", type=float, default=DEFAULT_NORM_THRESHOLD)
    q.set_defaults(func=cmd_query)

    b = sub.add_parser("batch", help="identify every image in a folder")
    b.add_argument("--catalog", required=True)
    b.add_argument("--images", required=True)
    b.add_argument("--top-k", type=int, default=5)
    b.add_argument("--threshold", type=float, default=DEFAULT_NORM_THRESHOLD)
    b.set_defaults(func=cmd_batch)

    c = sub.add_parser("calibrate", help="show score distributions + threshold")
    c.add_argument("--catalog", required=True)
    c.set_defaults(func=cmd_calibrate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
