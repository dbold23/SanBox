"""Domain-adapt the spot encoder on real sevengill spot crops.

The synthetic encoder has only ever seen bold synthetic blobs, but real
spots are faint (median contrast ~0.055, 67% below 0.08) and textured.
This fine-tunes it, self-supervised, on real crops: each real spot yields
two augmented, canonicalized views that a supervised-contrastive loss pulls
together — shifting the feature distribution toward real appearance while
the moment-canonicalization keeps geometry handling intact.

Usage:
    python -m spotid.ml.finetune_real \
        --init spotid/ml/checkpoints/encoder_gpu.pt \
        --root realdata/realworldspots.yolov8 \
        --out spotid/ml/checkpoints/encoder_real.pt
"""

import argparse

import cv2
import numpy as np
import torch

from ..features import segment_spot
from ..realdata import load_dataset
from .dataset import _augment, canonical_patch
from .model import SpotEncoder, supcon_loss


def collect_real_patches(root: str, pad: float = 0.6, min_radius: float = 4.0):
    """Return [(crop_image_float, contour), ...] for every real spot whose
    crop segments cleanly — the inputs to the canonicalization path."""
    shots = load_dataset(root)
    out = []
    for s in shots:
        if s.n_spots == 0:
            continue
        img = cv2.imread(s.image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        for (px, py, bw, bh) in s.boxes:
            r = 0.5 * max(bw, bh)
            if r < min_radius:
                continue
            x0, x1 = int(max(0, px - r * (1 + pad))), int(min(w, px + r * (1 + pad)))
            y0, y1 = int(max(0, py - r * (1 + pad))), int(min(h, py + r * (1 + pad)))
            crop = img[y0:y1, x0:x1]
            if crop.shape[0] < 12 or crop.shape[1] < 12:
                continue
            contour = segment_spot(crop)
            if contour is None or len(contour) < 8:
                continue
            out.append((crop, contour))
    return out


def _two_views(crop, contour, rng):
    """Two canonicalized + photometrically-augmented views of one spot."""
    views = []
    for _ in range(2):
        p = canonical_patch(crop, contour, jitter_rng=rng)
        if rng.uniform() < 0.5:
            p = p[:, ::-1].copy()  # canonicalization leaves a reflection d.o.f.
        views.append(_augment(p, rng))
    return views


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", default="spotid/ml/checkpoints/encoder_gpu.pt")
    ap.add_argument("--root", default="realdata/realworldspots.yolov8")
    ap.add_argument("--out", default="spotid/ml/checkpoints/encoder_real.pt")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=64, help="spots per step (x2 views)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temperature", type=float, default=0.15)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpt = torch.load(args.init, map_location=args.device, weights_only=True)
    model = SpotEncoder(ckpt["embed_dim"], ckpt["width"]).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.train()
    print(f"loaded encoder (embed {ckpt['embed_dim']}, width {ckpt['width']}) "
          f"from {args.init}", flush=True)

    patches = collect_real_patches(args.root)
    print(f"collected {len(patches)} cleanly-segmented real spot crops", flush=True)
    if len(patches) < args.batch:
        raise SystemExit("too few usable real crops")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    idx_all = np.arange(len(patches))
    for step in range(1, args.steps + 1):
        pick = rng.choice(idx_all, args.batch, replace=False)
        views, labels = [], []
        for lab, i in enumerate(pick):
            crop, contour = patches[i]
            for v in _two_views(crop, contour, rng):
                views.append(v)
                labels.append(lab)
        x = torch.from_numpy(np.stack(views)[:, None].astype(np.float32)).to(args.device)
        y = torch.tensor(labels, device=args.device)
        emb = model(x)
        loss = supcon_loss(emb, y, args.temperature)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                sim = emb @ emb.T
                sim.fill_diagonal_(-1e9)
                acc = (y[sim.argmax(1)] == y).float().mean().item()
            print(f"step {step:4d}/{args.steps}  loss {loss.item():.4f}  "
                  f"batch-NN acc {acc:.3f}", flush=True)

    torch.save({"model": model.state_dict(), "embed_dim": ckpt["embed_dim"],
                "width": ckpt["width"], "step": args.steps,
                "finetuned_on": "sevengill-real"}, args.out)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
