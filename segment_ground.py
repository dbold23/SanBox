"""
Segment the seafloor ("ground") in MBARI ROV video.

Pipeline
--------
1. Read an ROV video frame by frame (optionally subsampling for speed on CPU).
2. For each processed frame, prompt SAM (Segment Anything) with a set of
   positive points seeded across the lower part of the frame -- where the
   seafloor sits in forward/down-looking ROV footage -- plus a few negative
   points high in the frame to push the water column out of the mask.
3. Keep the parts of the predicted mask that are connected to the bottom of
   the frame, so we end up with one coherent "ground" region rather than
   scattered blobs.
4. Temporally smooth the mask a little to reduce flicker.
5. Write three things to the output directory:
     - overlay.mp4        original footage with the ground tinted green
     - masks/000123.png   per-frame binary ground masks
     - coverage.csv       fraction of each frame covered by ground

The default model is SAM 2.1 tiny (``sam2.1_t.pt``), whose weights are pulled
automatically from the ultralytics GitHub release on first run. Everything
runs on CPU.

Usage
-----
    python segment_ground.py --video data/V4361_20211006T162656Z_h265_1sec.mp4
    python segment_ground.py --video data/clip.mp4 --model sam2.1_t.pt --stride 2
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM


def seed_points(width, height):
    """Positive points on the (lower) seafloor, negative points up in the water.

    Returns (points, labels) where labels are 1 for foreground (ground) and
    0 for background (water column / open field above the bottom).
    """
    # Positive points: a small grid across the lower 45% of the frame.
    pos = []
    for fy in (0.62, 0.78, 0.92):
        for fx in (0.2, 0.5, 0.8):
            pos.append((fx * width, fy * height))
    # Negative points: top strip, almost always open water in ROV footage.
    neg = [(0.5 * width, 0.08 * height),
           (0.2 * width, 0.12 * height),
           (0.8 * width, 0.12 * height)]

    points = pos + neg
    labels = [1] * len(pos) + [0] * len(neg)
    return points, labels


def illumination_gate(frame):
    """Mask of pixels lit enough to actually be visible seafloor.

    ROV lights illuminate the bottom; the open water column above recedes into
    near-black. Forward-looking footage therefore has a dark upper region that
    is *not* ground we can see, just water. We separate lit surface from dark
    water with an Otsu-derived threshold, clamped to a low band so that a
    uniformly bright down-looking frame keeps all of its ground while a mostly
    dark forward-looking frame still drops the black water column.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = int(np.clip(otsu * 0.4, 8, 40))
    return (gray > thr).astype(np.uint8)


def keep_bottom_connected(mask):
    """Keep only mask components that touch the bottom edge of the frame.

    The seafloor is continuous with the bottom of the image, so this drops
    floating midwater detections (fish, marine snow, lit particles) while
    keeping the ground itself.
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    if mask_u8.sum() == 0:
        return mask_u8

    # Drop tiny isolated specks (marine snow) before grouping components.
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, open_k)
    if mask_u8.sum() == 0:
        return mask_u8

    num, labels = cv2.connectedComponents(mask_u8, connectivity=8)
    bottom_row = labels[-1, :]
    keep_ids = set(np.unique(bottom_row)) - {0}
    if not keep_ids:
        # Nothing reaches the bottom edge; fall back to the largest component.
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        keep_ids = {int(counts.argmax())}

    out = np.isin(labels, list(keep_ids)).astype(np.uint8)
    # Close small gaps so the mask reads as one surface.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    return out


def predict_ground(model, frame, imgsz):
    """Run SAM on one frame and return a binary ground mask (H x W uint8)."""
    h, w = frame.shape[:2]
    points, labels = seed_points(w, h)

    results = model(
        frame,
        points=points,
        labels=labels,
        imgsz=imgsz,
        verbose=False,
    )

    combined = np.zeros((h, w), dtype=np.uint8)
    r = results[0]
    if r.masks is not None:
        for m in r.masks.data.cpu().numpy():
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            combined |= (m > 0).astype(np.uint8)

    # Restrict to lit pixels so the mask cannot bleed into the dark water column.
    combined &= illumination_gate(frame)

    return keep_bottom_connected(combined)


def tint(frame, mask, color=(0, 255, 0), alpha=0.45):
    """Overlay a translucent colored mask on a BGR frame."""
    overlay = frame.copy()
    overlay[mask > 0] = color
    out = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    # Draw the mask outline for a crisp boundary.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


def main():
    ap = argparse.ArgumentParser(description="Segment the seafloor in ROV video.")
    ap.add_argument("--video", required=True, help="input video path")
    ap.add_argument("--model", default="sam2.1_t.pt", help="SAM model weights")
    ap.add_argument("--out", default="outputs", help="output directory")
    ap.add_argument("--stride", type=int, default=1,
                    help="process every Nth frame (>=1) to save CPU time")
    ap.add_argument("--imgsz", type=int, default=640, help="inference image size")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="stop after this many processed frames (0 = no limit)")
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    out_dir = Path(args.out)
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {args.model} (downloads on first run)...")
    model = SAM(args.model)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = max(1.0, fps / max(1, args.stride))

    writer = cv2.VideoWriter(
        str(out_dir / "overlay.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (width, height),
    )

    csv_rows = []
    ema = None  # floating-point mask accumulator for temporal smoothing
    smooth_alpha = 0.6  # weight of the current frame
    frame_idx = 0
    processed = 0

    print(f"Video: {width}x{height} @ {fps:.1f} fps | stride={args.stride}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.stride != 0:
            frame_idx += 1
            continue

        mask = predict_ground(model, frame, args.imgsz)

        # Temporal smoothing via an exponential moving average of the float
        # mask. This fills single-frame dropouts and releases stale pixels
        # within a frame or two -- unlike averaging two binary masks, it does
        # not behave as a union that pins coverage high across scene changes.
        m_float = mask.astype(np.float32)
        ema = m_float if ema is None else smooth_alpha * m_float + (1 - smooth_alpha) * ema
        mask = (ema >= 0.5).astype(np.uint8)

        coverage = float(mask.sum()) / float(mask.size)
        csv_rows.append((processed, frame_idx, round(coverage, 4)))

        cv2.imwrite(str(masks_dir / f"{processed:06d}.png"), mask * 255)
        writer.write(tint(frame, mask))

        processed += 1
        if processed % 10 == 0:
            print(f"  processed {processed} frames (coverage {coverage:.1%})")
        if args.max_frames and processed >= args.max_frames:
            break
        frame_idx += 1

    cap.release()
    writer.release()

    with open(out_dir / "coverage.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["processed_index", "source_frame", "ground_coverage_fraction"])
        w.writerows(csv_rows)

    if csv_rows:
        mean_cov = sum(r[2] for r in csv_rows) / len(csv_rows)
        print(f"\nDone. {processed} frames processed, mean ground coverage {mean_cov:.1%}.")
    print(f"Outputs in: {out_dir.resolve()}")
    print("  overlay.mp4, masks/, coverage.csv")


if __name__ == "__main__":
    main()
