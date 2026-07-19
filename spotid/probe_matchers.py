"""Foundation/local-feature matcher probe (frontier design experiment B).

Decisive question: do generic local-feature matchers align DIFFERENT sharks
as readily as true re-sightings? If yes, they key on body outline / gill
slits / fins (not the discriminative spots), and "zero-shot deep matcher as
the accuracy driver" is falsified for this problem.

Runs DISK (learned deep local features) and SIFT (classical) on flank crops,
matches + RANSAC-verifies (fundamental matrix — a curved/perspective scene),
and compares verified-inlier counts for TRUE pairs vs DIFFERENT-individual
pairs.
"""

import itertools

import cv2
import numpy as np
import torch

from .realdata import group_by_individual, load_dataset

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MAXSIDE = 1024


def flank_crop(shot, pad=0.06):
    """Grayscale + color crop around the spot bounding box (the flank)."""
    img = cv2.imread(shot.image_path)
    c = shot.centroids
    x0, y0 = c.min(0)
    x1, y1 = c.max(0)
    w, h = x1 - x0, y1 - y0
    X0 = int(max(0, x0 - pad * w)); X1 = int(min(img.shape[1], x1 + pad * w))
    Y0 = int(max(0, y0 - pad * h)); Y1 = int(min(img.shape[0], y1 + pad * h))
    crop = img[Y0:Y1, X0:X1]
    s = MAXSIDE / max(crop.shape[:2])
    if s < 1:
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return crop


def _ransac_inliers(pa, pb, min_pts=8):
    if len(pa) < min_pts:
        return 0
    F, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 3.0, 0.99)
    return int(mask.sum()) if mask is not None else 0


def sift_match(imgA, imgB):
    g = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=4000)
    ka, da = sift.detectAndCompute(g(imgA), None)
    kb, db = sift.detectAndCompute(g(imgB), None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0, 0
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(da, db, k=2)
    good = [m for m, n in knn if m.distance < 0.8 * n.distance]
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    return len(good), _ransac_inliers(pa, pb)


class DiskMatcher:
    def __init__(self):
        from kornia.feature import DISK
        self.disk = DISK.from_pretrained("depth").to(DEV).eval()

    @torch.no_grad()
    def _feat(self, img):
        t = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).float() / 255.0
        t = t.permute(2, 0, 1)[None].to(DEV)
        # DISK needs dims divisible by 16
        _, _, H, W = t.shape
        H2, W2 = (H // 16) * 16, (W // 16) * 16
        t = t[:, :, :H2, :W2]
        f = self.disk(t, n=2048, window_size=5, score_threshold=0.0)[0]
        return f.keypoints.cpu().numpy(), f.descriptors.cpu().numpy()

    def match(self, imgA, imgB):
        ka, da = self._feat(imgA)
        kb, db = self._feat(imgB)
        if len(ka) < 8 or len(kb) < 8:
            return 0, 0
        # mutual nearest neighbor on L2-normalized descriptors
        da /= np.linalg.norm(da, axis=1, keepdims=True) + 1e-9
        db /= np.linalg.norm(db, axis=1, keepdims=True) + 1e-9
        sim = da @ db.T
        ab = sim.argmax(1)
        ba = sim.argmax(0)
        mut = [(i, ab[i]) for i in range(len(ka)) if ba[ab[i]] == i]
        if len(mut) < 8:
            return len(mut), 0
        pa = np.float32([ka[i] for i, _ in mut])
        pb = np.float32([kb[j] for _, j in mut])
        return len(mut), _ransac_inliers(pa, pb)


def main():
    shots = [s for s in load_dataset("realdata/realworldspots.yolov8")
             if s.n_spots >= 40]
    by = group_by_individual(shots)
    crops = {s.filename: flank_crop(s) for s in shots}
    print(f"{len(shots)} flank crops ready ({DEV})")

    # TRUE re-sighting pairs
    true_pairs = []
    for ind, g in by.items():
        for a, b in itertools.combinations(g, 2):
            true_pairs.append((ind, a, b))
    # DIFFERENT-individual pairs (sample)
    rng = np.random.default_rng(0)
    diff = []
    keys = [s for s in shots]
    for _ in range(400):
        a, b = rng.choice(len(keys), 2, replace=False)
        if keys[a].individual != keys[b].individual:
            diff.append((keys[a], keys[b]))
        if len(diff) >= 24:
            break

    disk = DiskMatcher()

    def run(pairs, label):
        rows = []
        for item in pairs:
            a, b = item[-2], item[-1]
            sg, si = sift_match(crops[a.filename], crops[b.filename])
            dg, di = disk.match(crops[a.filename], crops[b.filename])
            rows.append((si, di))
            tag = item[0] if len(item) == 3 else f"{a.individual}|{b.individual}"
            print(f"  {label} {tag:24s} SIFT inliers={si:3d}  DISK inliers={di:3d}")
        return np.array(rows)

    print("\n=== TRUE re-sightings (same shark) ===")
    T = run(true_pairs, "TRUE")
    print("\n=== DIFFERENT individuals ===")
    D = run(diff, "DIFF")

    print("\n=== VERDICT (RANSAC-verified inliers) ===")
    for k, col in [("SIFT", 0), ("DISK", 1)]:
        t, d = T[:, col], D[:, col]
        print(f"{k}: TRUE mean {t.mean():.1f} (vals {sorted(t.tolist())})  |  "
              f"DIFF mean {d.mean():.1f} p90 {np.percentile(d,90):.0f} max {d.max()}")
        # can inlier count separate true from different?
        thr = np.percentile(d, 90)
        print(f"      TRUE pairs above DIFF-p90 ({thr:.0f}): "
              f"{int((t > thr).sum())}/{len(t)}")


if __name__ == "__main__":
    main()
