"""Load the real sevengill-shark spot dataset (YOLOv8 export) and run the
constellation matcher on it.

Each image is one photo of one shark's flank; the ``target`` boxes are its
spots. The individual's identity is encoded in the filename
(e.g. ``AOTB_A002``), and some individuals were photographed more than once
(``AOTB_A002 (2)``), which gives ground-truth re-sighting pairs — the data
needed to test genuine individual re-identification.
"""

import glob
import os
import re
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

# YOLO class ids in this export.
CLASS_TARGET = 0        # an individual spot
CLASS_ROI = 2           # flank region-of-interest box


def individual_id(filename: str) -> str:
    """Recover the individual-shark id from a Roboflow export filename.

    ``AOTB_A002 (2)_jpg.rf.<hash>.jpg`` -> ``AOTB_A002``. The ``(2)``
    re-sighting marker and the roboflow hash suffix are stripped.
    """
    base = re.sub(r'\.rf\.[A-Za-z0-9]+\.(jpg|jpeg|png)$', '', filename, flags=re.I)
    base = re.sub(r'_(jpg|JPG|jpeg|png)$', '', base)
    return re.sub(r'\s*\(\d+\)\s*', ' ', base).strip()


@dataclass
class ShotData:
    """One photograph and its annotated spots."""
    individual: str
    filename: str
    split: str
    image_path: str
    label_path: str
    width: int
    height: int
    centroids: np.ndarray   # (N, 2) spot centers in pixels
    boxes: np.ndarray       # (N, 4) x, y, w, h in pixels
    roi: np.ndarray | None  # (4,) x0,y0,x1,y1 in pixels, or None

    @property
    def n_spots(self) -> int:
        return len(self.centroids)


def _read_label(path: str, w: int, h: int):
    cents, boxes, roi = [], [], None
    if not os.path.exists(path):
        return np.zeros((0, 2)), np.zeros((0, 4)), None
    for line in open(path):
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, bw, bh = (float(x) for x in parts[1:5])
        px, py = cx * w, cy * h
        if cls == CLASS_TARGET:
            cents.append([px, py])
            boxes.append([px, py, bw * w, bh * h])
        elif cls == CLASS_ROI:
            roi = np.array([px - bw * w / 2, py - bh * h / 2,
                            px + bw * w / 2, py + bh * h / 2])
    cents_a = np.array(cents) if cents else np.zeros((0, 2))
    boxes_a = np.array(boxes) if boxes else np.zeros((0, 4))
    return cents_a, boxes_a, roi


def load_dataset(root: str) -> list[ShotData]:
    """Load every annotated photo under a YOLOv8 export ``root``."""
    shots = []
    for split in ("train", "valid", "test"):
        for img_path in sorted(glob.glob(os.path.join(root, split, "images", "*"))):
            fn = os.path.basename(img_path)
            lbl = os.path.join(root, split, "labels",
                               re.sub(r'\.(jpg|jpeg|png)$', '.txt', fn, flags=re.I))
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            h, w = img.shape
            cents, boxes, roi = _read_label(lbl, w, h)
            shots.append(ShotData(
                individual=individual_id(fn), filename=fn, split=split,
                image_path=img_path, label_path=lbl, width=w, height=h,
                centroids=cents, boxes=boxes, roi=roi))
    return shots


def group_by_individual(shots: list[ShotData]) -> dict[str, list[ShotData]]:
    groups: dict[str, list[ShotData]] = defaultdict(list)
    for s in shots:
        groups[s.individual].append(s)
    return dict(groups)


def resighting_pairs(shots: list[ShotData], min_spots: int = 40):
    """Return [(individual, [well-annotated ShotData, ...]), ...] for
    individuals photographed more than once with enough spots each."""
    out = []
    for ind, group in group_by_individual(shots).items():
        good = [s for s in group if s.n_spots >= min_spots]
        if len(good) >= 2:
            out.append((ind, good))
    return out
