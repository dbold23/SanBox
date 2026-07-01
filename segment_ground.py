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
4. Temporally median-filter each mask against its neighbouring frames, so a
   single-frame "flash" (SAM briefly grabbing the dark water column) is
   outvoted and erased while stable seafloor is kept.
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


def seed_points(frame, lit, grid=(7, 5)):
    """Content-adaptive SAM prompts driven by the lit-surface (colour) gate.

    A blind fixed grid under-prompts: it never reaches the frame edges or the
    upper seafloor, so SAM is never told about ground sitting at the left/right
    margins and under-covers it. Instead we tile the frame into a grid of cells
    and, using the colour gate (which reliably marks lit seafloor), drop a
    POSITIVE point on the lit pixels of every cell that is mostly ground -- edges
    included -- and a NEGATIVE point in every cell that is essentially open
    water. This spreads foreground prompts across the true extent of the ground
    and anchors the water column as background.

    Returns (points, labels): labels are 1 for foreground, 0 for background.
    """
    h, w = lit.shape
    gx, gy = grid
    y0 = int(0.10 * h)                 # ignore the very top for positives
    cw = w / gx
    ch = (h - y0) / gy

    pos, neg = [], []
    for j in range(gy):
        for i in range(gx):
            x1, x2 = int(i * cw), int((i + 1) * cw)
            ya, yb = int(y0 + j * ch), int(y0 + (j + 1) * ch)
            cell = lit[ya:yb, x1:x2]
            frac = float(cell.mean())
            if frac > 0.35:
                # Put the point on the cell's lit pixels (its ground centroid).
                ys, xs = np.nonzero(cell)
                pos.append((x1 + xs.mean(), ya + ys.mean()))
            elif frac < 0.05:
                neg.append((x1 + cw / 2.0, ya + ch / 2.0))

    # Always anchor the top strip (open water) as background.
    neg += [(0.5 * w, 0.06 * h), (0.2 * w, 0.10 * h), (0.8 * w, 0.10 * h)]

    # Fallback: if the gate found almost nothing lit, fall back to a lower grid
    # so SAM still gets a reasonable seafloor prompt.
    if not pos:
        for fy in (0.7, 0.85):
            for fx in (0.2, 0.5, 0.8):
                pos.append((fx * w, fy * h))

    points = pos + neg
    labels = [1] * len(pos) + [0] * len(neg)
    return points, labels


def illumination_gate(frame, sensitivity=0.0):
    """Mask of pixels that are actually lit seafloor, not open water column.

    ROV lights illuminate the bottom; the open water column recedes into a dark,
    strongly *blue* haze. The key discriminator is colour, not just brightness:
    red light attenuates fastest underwater, so only a near, lit surface returns
    appreciable red/green, while the water column is blue-dominant with almost no
    red. A plain grayscale threshold fails because the blue haze inflates
    luminance enough to survive (that is what caused whole-frame "flashes").

    We therefore gate on a green+red "surface illumination" channel (ignoring the
    blue the water inflates), Otsu-thresholded and clamped, and additionally
    reject pixels that are dark *and* strongly blue -- i.e. the water column.

    ``sensitivity`` in [0, 1] trades coverage against water bleed: 0.0 is strict
    (only clearly-lit ground); higher values lower the brightness threshold and
    soften the blue-water rejection so dimmer seafloor is included too.
    """
    s = float(np.clip(sensitivity, 0.0, 1.0))
    b, g, r = cv2.split(frame)
    surf = cv2.addWeighted(g, 0.5, r, 0.5, 0.0)      # lit-surface proxy, no blue
    surf = cv2.GaussianBlur(surf, (5, 5), 0)
    otsu, _ = cv2.threshold(surf, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = int(np.clip(otsu * (0.5 - 0.35 * s), 12 - 8 * s, 60))
    lit = surf > thr

    # Explicit water-column rejection: dark red channel + blue-dominant.
    # As sensitivity rises, only more extreme blue/red-starved pixels are cut.
    ri = r.astype(np.int16)
    bi = b.astype(np.int16)
    water = (ri < 18 - 10 * s) & ((bi - ri) > 12 + 10 * s)

    return (lit & ~water).astype(np.uint8)


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
    return fill_interior_holes(out)


def fill_interior_holes(mask, max_hole_frac=0.02):
    """Fill small enclosed holes (dark crevices) inside the ground region.

    A "hole" is a background component fully surrounded by ground. Filling these
    keeps the seafloor reading as one clean surface. We only fill holes below a
    fraction of the frame so that genuine open-water gaps -- which are large and
    reach the frame border -- are never filled in.
    """
    inv = (mask == 0).astype(np.uint8)
    n, lab = cv2.connectedComponents(inv, connectivity=8)
    if n <= 1:
        return mask
    border = set(np.unique(np.concatenate(
        [lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])).tolist())
    counts = np.bincount(lab.ravel())
    limit = max_hole_frac * mask.size
    fill_ids = [i for i in range(1, n)
                if i not in border and counts[i] <= limit]
    if fill_ids:
        mask = mask.copy()
        mask[np.isin(lab, fill_ids)] = 1
    return mask


def predict_ground(model, frame, imgsz, sensitivity=0.0):
    """Run SAM on one frame and return a binary ground mask (H x W uint8)."""
    h, w = frame.shape[:2]
    lit = illumination_gate(frame, sensitivity)   # lit-surface / colour gate
    points, labels = seed_points(frame, lit)

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
    combined &= lit

    return keep_bottom_connected(combined)


class OverlayWriter:
    """Write an H.264 mp4 that plays in browsers, falling back to mp4v.

    Browsers won't decode OpenCV's default ``mp4v`` (MPEG-4 Part 2) stream, so
    we prefer H.264 via the ffmpeg binary bundled with ``imageio-ffmpeg``. If
    that package isn't installed we fall back to OpenCV's ``mp4v`` writer and
    say so, so the pipeline still produces a file.
    """

    def __init__(self, path, fps, size):
        self.path = str(path)
        self._imageio = None
        self._cv2 = None
        try:
            import imageio  # noqa: F401  (imageio-ffmpeg provides the encoder)
            import imageio.v2 as iio
            self._imageio = iio.get_writer(
                self.path, fps=fps, codec="libx264",
                format="ffmpeg", pixelformat="yuv420p",
                macro_block_size=1, output_params=["-movflags", "+faststart"],
            )
        except Exception:
            self._cv2 = cv2.VideoWriter(
                self.path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            print("  (imageio-ffmpeg not available; writing mp4v -- may not "
                  "play in browsers. `pip install imageio-ffmpeg` for H.264.)")

    @property
    def codec(self):
        return "H.264" if self._imageio is not None else "mp4v"

    def write(self, bgr_frame):
        if self._imageio is not None:
            self._imageio.append_data(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
        else:
            self._cv2.write(bgr_frame)

    def release(self):
        if self._imageio is not None:
            self._imageio.close()
        else:
            self._cv2.release()


def temporal_vote(idx, total, half, loader):
    """Majority-vote a frame's ground mask against its temporal neighbours.

    A pixel is kept only if it is ground in a strict majority of the frames in
    the centered window [idx-half, idx+half] (clamped at the ends). Because a
    single-frame "flash" -- where SAM briefly balloons the mask over the dark
    water column -- appears in only one frame of the window, it is outvoted by
    its neighbours and erased, while genuine seafloor (present in nearly every
    frame) survives untouched. Window is clamped at the clip boundaries.
    """
    lo = max(0, idx - half)
    hi = min(total - 1, idx + half)
    acc = None
    n = 0
    for j in range(lo, hi + 1):
        m = loader(j)
        acc = m.astype(np.uint16) if acc is None else acc + m
        n += 1
    # Strict majority: pixel on in more than half the window.
    return (acc * 2 > n).astype(np.uint8)


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
    ap.add_argument("--smooth-window", type=int, default=5,
                    help="temporal median window in frames (odd; 1 disables). "
                         "Larger = fewer background flashes, but laggier masks.")
    ap.add_argument("--sensitivity", type=float, default=0.4,
                    help="ground coverage vs water bleed, 0..1. 0=strict (only "
                         "clearly-lit ground); higher includes dimmer seafloor. "
                         "0.4 is the balanced default; ~0.8 starts to bleed.")
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

    writer = OverlayWriter(out_dir / "overlay.mp4", out_fps, (width, height))

    half = max(0, args.smooth_window // 2)
    raw_dir = out_dir / "_raw"          # temp per-frame masks, removed at the end
    raw_dir.mkdir(parents=True, exist_ok=True)

    # ---- Pass 1: run SAM once per frame and save the raw ground mask. ----
    print(f"Video: {width}x{height} @ {fps:.1f} fps | stride={args.stride} "
          f"| overlay codec={writer.codec} | smooth_window={args.smooth_window}")
    print("Pass 1/2: segmenting frames...")
    src_frames = []      # source frame index for each processed frame
    frame_idx = 0
    processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.stride != 0:
            frame_idx += 1
            continue

        mask = predict_ground(model, frame, args.imgsz, args.sensitivity)
        cv2.imwrite(str(raw_dir / f"{processed:06d}.png"), mask * 255)
        src_frames.append(frame_idx)

        processed += 1
        if processed % 10 == 0:
            cov = float(mask.sum()) / float(mask.size)
            print(f"  segmented {processed} frames (raw coverage {cov:.1%})")
        if args.max_frames and processed >= args.max_frames:
            break
        frame_idx += 1
    cap.release()

    total = processed

    def load_raw(j):
        m = cv2.imread(str(raw_dir / f"{j:06d}.png"), cv2.IMREAD_GRAYSCALE)
        return (m > 127).astype(np.uint8)

    # ---- Pass 2: temporally vote each mask, then composite the overlay. ----
    print(f"Pass 2/2: temporal median (window={args.smooth_window}) + overlay...")
    cap = cv2.VideoCapture(str(video_path))
    csv_rows = []
    frame_idx = 0
    p = 0
    while p < total:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx != src_frames[p]:
            frame_idx += 1
            continue

        if args.smooth_window <= 1:
            mask = load_raw(p)
        else:
            mask = temporal_vote(p, total, half, load_raw)

        coverage = float(mask.sum()) / float(mask.size)
        csv_rows.append((p, frame_idx, round(coverage, 4)))
        cv2.imwrite(str(masks_dir / f"{p:06d}.png"), mask * 255)
        writer.write(tint(frame, mask))

        p += 1
        if p % 50 == 0:
            print(f"  composited {p}/{total} frames")
        frame_idx += 1

    cap.release()
    writer.release()

    # Remove the temporary raw masks now that voted masks are written.
    for f in raw_dir.glob("*.png"):
        f.unlink()
    raw_dir.rmdir()

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
