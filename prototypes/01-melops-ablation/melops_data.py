"""Data adapters for the Melops rigid-part vs deformable-body ablation.

Two sources, one contract:

* ``load_melops(root, bbox)`` returns a catalogue DataFrame with the columns
  the protocol layer depends on: ``image_id``, ``identity``, ``path`` (relative
  to ``root``), ``date`` (``YYYY-MM-DD`` string), ``side``, ``orientation``
  (== ``side``, mirroring the real loader's ``df["orientation"] = df["side"]``),
  ``bbox`` (the crop for the requested part) and ``bbox_body`` / ``bbox_head``
  / ``bbox_headless``. All bboxes are ``[left, top, width, height]`` float
  arrays in the pixel space of the stored image, matching the semantics the
  real ``wildlife_datasets`` Melops loader produces for its ``bbox_{part}``
  columns (LTWH in body-crop pixels).
* ``make_synthetic(...)`` renders a deterministic miniature corpus with the
  same column layout, so the whole pipeline is exercisable with zero downloads
  and zero optional dependencies.

``bbox`` must be one of ``("body", "head", "headless")`` -- exactly the
``bbox_parts`` tuple of the real loader.
"""

from __future__ import annotations

import csv
import os

import numpy as np
import pandas as pd
from PIL import Image

BBOX_PARTS = ("body", "head", "headless")

_PLAIN_METADATA = "metadata.csv"

_REQUIRED_PLAIN_COLUMNS = (
    "image_id",
    "identity",
    "path",
    "date",
    "side",
    "bbox_body",
    "bbox_head",
    "bbox_headless",
)


def _parse_bbox(text):
    parts = [float(v) for v in str(text).split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be 'left,top,width,height', got %r" % (text,))
    return np.asarray(parts, dtype=np.float64)


def _format_bbox(bbox):
    return ",".join("%.2f" % float(v) for v in bbox)


def load_melops(root, bbox="body"):
    """Load a Melops-layout catalogue from ``root``.

    If ``root`` contains a plain ``metadata.csv`` (the synthetic / offline
    layout) it is used directly. Otherwise the real ``wildlife_datasets``
    Melops class is used (guarded import; a clear error names the pip line if
    it is missing).
    """
    if bbox not in BBOX_PARTS:
        raise ValueError("bbox must be one of %r, got %r" % (BBOX_PARTS, bbox))
    plain = os.path.join(root, _PLAIN_METADATA)
    if os.path.exists(plain):
        df = _load_plain(root, plain)
    else:
        df = _load_wildlife_datasets(root, bbox)
    df = _normalize(df, bbox)
    _check_catalogue(df)
    return df


def _load_plain(root, metadata_path):
    df = pd.read_csv(metadata_path)
    missing = [c for c in _REQUIRED_PLAIN_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("metadata.csv missing required columns: %r" % (missing,))
    for part in BBOX_PARTS:
        col = "bbox_%s" % part
        df[col] = df[col].apply(_parse_bbox)
    return df


def _load_wildlife_datasets(root, bbox):
    try:
        from wildlife_datasets import datasets  # noqa: guarded optional dep
    except ImportError:
        raise RuntimeError(
            "No plain metadata.csv found in %r and wildlife-datasets is not "
            "installed. On the lab machine run: pip install wildlife-datasets "
            "then datasets.Melops.get_data(root). NOTE the real Melops "
            "get_data downloads only the body-crop archive; head/headless "
            "crops come from the bbox_head/bbox_headless columns applied to "
            "the body crops (bbox='head'/'headless'), matching melops.py." % root
        )
    ds = datasets.Melops(root, bbox=bbox)
    return ds.df.copy()


def _normalize(df, bbox):
    df = df.copy()
    if "orientation" not in df.columns:
        df["orientation"] = df["side"]
    df["identity"] = df["identity"].astype(str)
    df["image_id"] = df["image_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["bbox"] = df["bbox_%s" % bbox]
    keep = [
        "image_id",
        "identity",
        "path",
        "date",
        "side",
        "orientation",
        "bbox",
        "bbox_body",
        "bbox_head",
        "bbox_headless",
    ]
    return df[keep].reset_index(drop=True)


def _check_catalogue(df):
    if df["image_id"].duplicated().any():
        dup = df.loc[df["image_id"].duplicated(), "image_id"].iloc[0]
        raise ValueError("duplicate image_id in catalogue: %r" % dup)
    if not (df["orientation"] == df["side"]).all():
        raise ValueError("orientation column must equal side (Melops semantics)")
    sides = set(df["side"].unique().tolist())
    if not sides.issubset({"L", "R"}):
        raise ValueError("side must be 'L' or 'R', got %r" % sorted(sides))


def load_crop(root, row):
    """Return the PIL RGB crop for one catalogue row (its ``bbox`` column)."""
    img = Image.open(os.path.join(root, row["path"])).convert("RGB")
    left, top, w, h = [float(v) for v in row["bbox"]]
    right = min(left + w, img.size[0])
    bottom = min(top + h, img.size[1])
    left = max(left, 0.0)
    top = max(top, 0.0)
    if right - left < 1 or bottom - top < 1:
        raise ValueError("degenerate bbox %r for %r" % (row["bbox"], row["path"]))
    return img.crop((int(round(left)), int(round(top)), int(round(right)), int(round(bottom))))


# ---------------------------------------------------------------------------
# Synthetic corpus
# ---------------------------------------------------------------------------

_W, _H = 160, 64
_HEAD_FRAC = 1.0 / 3.0
_N_HEAD_SPOTS = 6
_N_BODY_SPOTS = 14
_N_COMMON_SPOTS = 8


def _spot_layer(rng, n, x_lo, x_hi):
    """Sample spot (x, y, radius) triples inside [x_lo, x_hi) x [4, H-4)."""
    xs = rng.uniform(x_lo, x_hi, size=n)
    ys = rng.uniform(4, _H - 4, size=n)
    rs = rng.uniform(2.5, 4.5, size=n)
    return np.column_stack([xs, ys, rs])


def _render_base(spots_amp):
    """Render the canonical left-facing fish as a float array.

    ``spots_amp`` is a list of (spots, amplitude) layers; amplitude is the
    darkening applied at each spot (identity signal strength lives here).
    """
    yy, xx = np.mgrid[0:_H, 0:_W]
    shade = 150.0 + 30.0 * np.sin(np.pi * yy / _H)
    img = np.stack([shade * 0.9, shade, shade * 0.85], axis=-1)
    for spots, amp in spots_amp:
        if amp <= 0:
            continue
        for x, y, r in spots:
            mask = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
            img[mask] -= amp
    return img


def make_synthetic(
    out_dir,
    n_individuals=30,
    images_per_individual_dist=(1, 1, 2, 3, 4),
    seed=0,
    head_signal=1.0,
    body_signal=1.0,
):
    """Render a deterministic miniature Melops-layout corpus into ``out_dir``.

    Contract:
    * Each individual carries a fixed spot constellation: ``_N_HEAD_SPOTS``
      spots on the head third and ``_N_BODY_SPOTS`` on the rest. The darkening
      amplitude of the head layer is ``90 * head_signal`` and of the body
      layer ``90 * body_signal``; amplitude 0 removes the identity signal from
      that region entirely (a shared low-amplitude confounder texture remains
      everywhere, so a zero-signal region is textured but uninformative).
      This is the knob that lets tests build a head-concentrated corpus
      (head_signal=1, body_signal=0) and a distributed one (1, 1).
    * Per-sighting nuisance: small affine jitter (rotation, translation),
      brightness jitter, Gaussian pixel noise.
    * ``images_per_individual_dist`` is a sequence of counts sampled uniformly
      per individual; include 1s to get singletons (Melops averages ~2.5
      images/individual, so the protocol must survive singletons).
    * Dates span ~3 synthetic years (2018-01-01 .. 2020-12-31) so a
      time-separated split is exercised. Sides mix L and R; the R rendering is
      the horizontal mirror of the canonical left-facing fish, with the head
      bbox on the mirrored side.
    * Everything derives from ``seed``; identical calls are byte-identical in
      metadata (images identical up to PNG encoding, which is deterministic).

    Returns the catalogue DataFrame (same layout as ``load_melops``).
    """
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "body")
    os.makedirs(img_dir, exist_ok=True)
    rng = np.random.default_rng([int(seed), 0xC0FFEE])
    common_spots = _spot_layer(np.random.default_rng([int(seed), 7]), _N_COMMON_SPOTS, 0, _W)

    head_x_hi = _W * _HEAD_FRAC
    rows = []
    base_date = np.datetime64("2018-01-01")
    for i in range(int(n_individuals)):
        rng_i = np.random.default_rng([int(seed), 1, i])
        head_spots = _spot_layer(rng_i, _N_HEAD_SPOTS, 4, head_x_hi - 4)
        body_spots = _spot_layer(rng_i, _N_BODY_SPOTS, head_x_hi + 4, _W - 4)
        canonical = _render_base(
            [
                (common_spots, 25.0),
                (head_spots, 90.0 * float(head_signal)),
                (body_spots, 90.0 * float(body_signal)),
            ]
        )
        n_images = int(rng.choice(np.asarray(images_per_individual_dist)))
        days = np.sort(rng.integers(0, 1095, size=n_images))
        for j in range(n_images):
            side = "L" if rng.random() < 0.5 else "R"
            arr = canonical if side == "L" else canonical[:, ::-1, :]
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            angle = float(rng.uniform(-2.0, 2.0))
            tx, ty = int(rng.integers(-2, 3)), int(rng.integers(-2, 3))
            img = img.rotate(angle, resample=Image.BILINEAR, translate=(tx, ty), fillcolor=(135, 150, 128))
            out = np.asarray(img, dtype=np.float64)
            out = out * float(rng.uniform(0.85, 1.15))
            out = out + rng.normal(0.0, 5.0, size=out.shape)
            out = np.clip(out, 0, 255).astype(np.uint8)

            image_id = "ind%04d_img%02d" % (i, j)
            rel_path = os.path.join("body", image_id + ".png")
            Image.fromarray(out).save(os.path.join(out_dir, rel_path))

            head_w = head_x_hi
            if side == "L":
                bbox_head = (0.0, 0.0, head_w, float(_H))
                bbox_headless = (head_w, 0.0, _W - head_w, float(_H))
            else:
                bbox_head = (_W - head_w, 0.0, head_w, float(_H))
                bbox_headless = (0.0, 0.0, _W - head_w, float(_H))
            date = str((base_date + np.timedelta64(int(days[j]), "D")))
            rows.append(
                {
                    "image_id": image_id,
                    "identity": "ind%04d" % i,
                    "path": rel_path,
                    "date": date,
                    "side": side,
                    "bbox_body": _format_bbox((0.0, 0.0, float(_W), float(_H))),
                    "bbox_head": _format_bbox(bbox_head),
                    "bbox_headless": _format_bbox(bbox_headless),
                }
            )

    metadata_path = os.path.join(out_dir, _PLAIN_METADATA)
    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(_REQUIRED_PLAIN_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return load_melops(out_dir)
