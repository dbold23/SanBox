# Prototype 06 "spot-proxy" — the OSEA contract and the real-photo pass

The matcher in this prototype never sees RGB. Real photos and synthetic renders
of the scanned body are both reduced to the *same* non-RGB representation — a
YOLO body polygon plus YOLO spot boxes — by the *same* extractor, and identity
is matched on the spot constellation alone. This file documents the door both
domains go through (`osea_contract.py`) and the real-photo distribution the
synthetic side has to be pushed onto (`real_features.py`).

## Files

| path | what it is |
|---|---|
| `osea_contract.py` | model loading, `detect()` in catalog.db column shapes, `features()` |
| `real_features.py` | CLI: run the contract over catalog.db, write the feature distribution |
| `tests/test_contract.py` | PCA-frame invariance, hand-checked `features()`, the `max_det` truncation report, live `detect()` |
| `tests/test_real_features.py` | body-conditioned scalar table, the slim record |
| `results/real/` | output of the full-catalog run (see below) |

## Public interface (what the rest of prototype 06 calls)

```python
import osea_contract as oc

oc.main_root() -> pathlib.Path          # $SEVENGILL_MAIN_ROOT, else walk up to spot_detector/
oc.weight_paths() -> {"body_obstr", "body_only", "head", "spots"}   # values are Path or None
oc.load_models(device="cpu") -> (body_model, spot_model)            # cached per device

oc.detect(img_rgb_uint8, models=None, device="cpu",
          body_conf=0.40, head_conf=0.40, spot_conf=0.25) -> dict
oc.features(det, image_size=None) -> dict
oc.flat_scalars(feats) -> {str: float or None}
oc.to_db_row(det) -> {the ten catalog.db columns}
oc.pca_frame(polygon, spot_centres_px=None) -> dict or None
oc.raster_moments(polygon, long_side=1024) -> (area_px2, centroid_xy, cov_2x2) or None
oc.polygon_area(polygon) -> float       # shoelace; see the self-intersection note
```

### `detect()` output — exactly the DB column shapes

```
body_polygon          [[x, y], ...] floats at 1 dp, original EXIF-transposed px   | None
body_bbox             {"x","y","w","h"} ints (polygon extent padded 5%, clipped)  | None
body_conf             float                                                       | None
obstruction_polygons  [ [[x,y], ...], ... ]        (empty list when none)
obstruction_count     int
head_polygon / head_bbox / head_conf   always None — runs/head/v1 does not exist
spots                 [ {"x","y","w","h","cx","cy","conf"}, ... ]  1 dp / conf 3 dp
spot_count            int
image_width, image_height   ints   (extra, non-DB; used for bbox_width_frac)
spots_raw_count       int | None — boxes NMS returned BEFORE run_image filtered
                      them; None when the spot model never ran (no body)
spots_max_det         int  — the ultralytics detection cap in force (300)
spots_truncated       bool — spots_raw_count >= spots_max_det
```

`to_db_row()` JSON-serialises these into the ten columns
`pipeline_worker.stage_detect` writes, with the same "empty means NULL" rule.
The three `spots_*` diagnostics are not DB columns.

**The spot count is capped, and the cap is invisible in `spot_count`.**
`run_image` calls `predict` with no `max_det`, so ultralytics' default of 300
applies and NMS silently returns its 300 highest-confidence boxes. The
centre-inside-body / not-in-obstruction filter then runs *after* the cap, so a
truncated image can report any `spot_count` at or below 300: on this catalog
**24 images are truncated and only 6 of them show `spot_count == 300`** (image
619 stores 239 with 300 raw boxes; at `max_det=5000` it keeps 300, and image
979 stores 260 against a true 417). We keep the cap — raising it would break
fidelity with what the tagger stores — and report it. `n_spots` and `density`
on a truncated image are lower bounds; `summary.json` lists the ids under
`counts.spots_truncated_image_ids` so they can be dropped before the real
distribution is used as a synthetic target.

### `features()` output

```
ok         bool  — False when there is no usable body polygon
frame      {origin[2], e_major[2], e_minor[2], L_major, D_minor, area_px2,
            aspect, theta_deg, sign_rule, degenerate_contour, area_shoelace_px2}
spots_uv   [[u, v, size, conf], ...]   u along e_major, v across, both / D_minor;
                                       size = sqrt(w*h) / D_minor
spots_raw  the untouched detector spot dicts (original px)
scalars    n_spots, density, area_px2, area_norm, L_major, D_minor, aspect,
           size{q05,q25,q50,q75,q95,mean}, nn{...}, nn_median, conf{...},
           bbox_width_frac, body_conf, obstruction_count,
           obstruction_area_frac, obstruction_area_ratio, degenerate_contour
```

`density = n_spots / (area_px2 / D_minor**2)`; `area_norm = area_px2 / D_minor**2`.
`obstruction_area_frac` is the *rasterised* fraction of the body covered by
obstruction polygons (they routinely extend past the animal, so summing their
areas overstates it); `obstruction_area_ratio` is that raw uncapped sum divided
by the body area. **Both areas in that ratio are measured by raster fill**, the
same way `area_px2` is: dividing by the *shoelace* body area inflated the ratio
by exactly the factor `raster_moments` exists to remove — 19 obstruction rows
were affected and the corpus maximum read 74.90 where the truth is 1.67 (image
799: an obstruction covering 26% of the body, over a shoelace area that had
cancelled to 1/70th of the filled area).

## The wrapper was checked against real stored rows

Two catalog images (20 `AOTB_A014.jpg`, 160 `IMG_20200811_132759.jpg`) already
carry detections written by the OSEA pipeline itself. Re-running
`to_db_row(detect(img))` on them reproduces the stored columns:

| column | result |
|---|---|
| `body_polygon_json` | identical string (7801 / 14761 chars) |
| `body_bbox_json`, `body_conf` | identical |
| `head_*` | identical (all NULL) |
| `spots_json` | **same spot set** — 277 / 18 spots, zero differing entries once sorted by centre, zero centres unique to either side; only the list *order* differs (YOLO box ordering) |
| `spot_count` | identical (277 / 18) |
| `obstruction_polygon_json` | identical (NULL) |
| `obstruction_count` | ours `0`, stored `NULL` — those legacy rows predate the column being populated; `stage_detect` writes `0` today, so ours matches current behaviour |

`detect()` is deterministic: two consecutive calls on the same array return
equal spot lists.

## Three things a caller must know

**1. Channel order.** OSEA hands ultralytics an **RGB** array straight from
`PIL.ImageOps.exif_transpose` even though ultralytics documents numpy input as
BGR. That quirk is part of the deployed contract, so `detect()` replicates it.
Synthetic renders must be fed the same way — RGB, uint8, `H x W x 3` — or they
will not land in the same detector input distribution.

**2. The frame is defined only up to sign.** An eigenvector has no canonical
sign, so the body frame is defined up to `{u -> ±u} x {v -> ±v}`.

* Every scalar in `feats["scalars"]` is built from counts, areas, extents,
  pairwise distances or per-spot magnitudes, so the whole scalar block is
  identical under both flips, under rotation and translation, and under
  mirroring. `tests/test_contract.py` monkeypatches `pca_frame` to negate each
  axis and asserts bit-equality.
* `spots_uv` is **not** sign-canonical. Its sign is pinned by a documented
  tie-break (third central moment of the spot cloud, falling back to the
  polygon's, then to a fixed sign), recorded in `frame["sign_rule"]`. It is
  deterministic and rotation-equivariant but carries no biological meaning, so
  **downstream matchers must be flip-invariant** — try all four sign
  combinations, or use flip-invariant descriptors.
* `bbox_width_frac` is the one deliberate exception to rotation invariance: the
  body bbox is axis-aligned in *image* space, so it is a framing descriptor
  (how much of the frame the animal fills), not a shape descriptor.

**3. The body contour self-intersects on ~11% of real photos.** `mask.xy` can
hand back a single contour that snakes across the frame and doubles back. The
shoelace area then partially cancels and the analytic moments are meaningless —
on catalog image 799 that drove `density` to 23159 against a corpus median of
99, and put a spot at `v = 13.1` (impossible for a point inside the body).
`raster_moments()` fixes this by measuring the *filled* region on a 1024-px
canvas instead, and `degenerate_contour` flags the disagreement so callers can
drop or down-weight those images. Known cost: `cv2.fillPoly` paints boundary
pixels, so areas run ~0.3% high — a systematic bias of the shared extractor,
identical on both domains, so it cancels in every real-vs-synthetic comparison.

## Running the real pass

```bash
MAIN=/Volumes/External\ Dive\ 2TB/projects/marine-cv/7Gill
"$MAIN/.venv/bin/python" real_features.py --overlay-n 12 --overlay-every 88
"$MAIN/.venv/bin/python" -m pytest tests -q
```

Outputs land in `results/real/`: `detections.jsonl` (one record per processed
image), `detections_slim.jsonl`, `skipped.jsonl` (one per skip, with a reason),
`summary.json` (counts, per-scalar quantiles, pooled per-spot histograms with
explicit bin edges), `features_contact.png` and `detections_contact.png`.
Full-resolution single overlays go to `results/real/overlays/` and are
gitignored.

`detections.jsonl` is 32.4 MB on the full catalog (11.2 MB of body polygons,
10.1 MB of spot boxes, 10.2 MB of feats) and is **gitignored**. The tracked
variant is `detections_slim.jsonl` (15.3 MB), written by the same run: the body
polygon is dropped and each spot becomes `[cx, cy, w, h, conf]`, while
`body_bbox` and the whole `feats` block are kept, so every number in
`summary.json` is recomputable from it. The constellation matcher rectifies
against the body outline, so it needs the full file — re-run this script.

**The per-image scalar quantiles in `summary.json` are conditioned on a usable
body polygon.** An image with no body has no spot field to measure, so its
structural `n_spots = 0` goes in as missing, not as a zero; each scalar carries
`n` and `n_missing`. Pooling the 61 no-body images in moved `n_spots` q05 from
39 to 0, q25 from 69 to 64 and q50 from 112 to 107 — a body-detector failure
rate masquerading as a spot-density fact, in exactly the distribution the
synthetic corpus is pushed onto.

`--min-width 800` (the default) drops 6 catalog images — and **all six are
tagged to an individual**, i.e. 10% of the 61 tagged photos. The detector
handles them fine (5 of 6 yield a body polygon at 480–640 px). For the
identity-matching arm, run with `--min-width 0`.
