"""
Benchmark the seafloor segmentation against hand-labeled ground truth.

For every frame in benchmark/frames/ that has a matching mask in benchmark/gt/,
this runs the pipeline (and ablations) and scores each against the ground truth
with IoU, Dice, precision and recall. Precision is the one to watch here: it
punishes exactly the failures we fought (smoke / water counted as ground),
while recall measures missed floor.

Ground-truth masks: same filename as the frame, white (>127) = seafloor,
black = everything else. See benchmark/README.md for how to make them.

    python benchmark.py                 # score all variants over all labeled frames
    python benchmark.py --variant current
"""

import argparse
import csv
import glob
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM

import segment_ground as sg

# Ablation grid: each entry is (sensitivity, texture) passed to predict_ground.
VARIANTS = {
    "current (auto, s=0.4)":  (0.4, "auto"),
    "no texture gate":        (0.4, "0"),
    "fixed texture 1.5":      (0.4, "1.5"),
    "strict (s=0.0)":         (0.0, "auto"),
    "aggressive (s=0.8)":     (0.8, "auto"),
}


def scores(pred, gt):
    pred = pred > 0
    gt = gt > 127
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = inter / union if union else 1.0
    dice = 2 * inter / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) else 1.0
    prec = inter / pred.sum() if pred.sum() else (1.0 if gt.sum() == 0 else 0.0)
    rec = inter / gt.sum() if gt.sum() else 1.0
    return iou, dice, prec, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="benchmark/frames")
    ap.add_argument("--gt", default="benchmark/gt")
    ap.add_argument("--model", default="sam2.1_t.pt")
    ap.add_argument("--variant", default=None, help="run just one named variant")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    gt_dir = Path(args.gt)
    pairs = []
    for fp in sorted(glob.glob(f"{args.frames}/*.png")):
        gp = gt_dir / Path(fp).name
        if gp.exists():
            pairs.append((fp, str(gp)))
    if not pairs:
        print(f"No labeled frames found. Put ground-truth masks in {args.gt}/ "
              f"(same names as benchmark/frames/, white=seafloor).")
        return
    print(f"Scoring {len(pairs)} labeled frames.\n")

    variants = VARIANTS if not args.variant else {args.variant: VARIANTS[args.variant]}
    model = SAM(args.model)

    rows = []
    per_frame = []
    for name, (sens, tex) in variants.items():
        acc = np.zeros(4)
        for fp, gp in pairs:
            frame = cv2.imread(fp)
            gt = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
            if gt.shape != frame.shape[:2]:
                gt = cv2.resize(gt, (frame.shape[1], frame.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            pred = sg.predict_ground(model, frame, args.imgsz, sens, tex)
            s = scores(pred, gt)
            acc += s
            per_frame.append((name, Path(fp).name, *[round(x, 4) for x in s]))
        acc /= len(pairs)
        rows.append((name, *acc))

    print(f"{'variant':24s} {'IoU':>7} {'Dice':>7} {'Prec':>7} {'Recall':>7}")
    print("-" * 58)
    for name, iou, dice, prec, rec in rows:
        print(f"{name:24s} {iou:7.3f} {dice:7.3f} {prec:7.3f} {rec:7.3f}")

    with open("benchmark/results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "mean_iou", "mean_dice", "mean_precision", "mean_recall"])
        for r in rows:
            w.writerow([r[0], *[round(x, 4) for x in r[1:]]])
    with open("benchmark/results_per_frame.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "frame", "iou", "dice", "precision", "recall"])
        w.writerows(per_frame)
    print("\nWrote benchmark/results.csv and benchmark/results_per_frame.csv")


if __name__ == "__main__":
    main()
