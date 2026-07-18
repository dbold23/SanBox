# spotid — spot generation & viewpoint-invariant recognition

Generate unique splotch identities, render them in unlimited permutations
(rotation, scale, **out-of-plane viewing angle**, lighting, noise), and
recognize which spot is which from any of those views. Scales up to whole
*surfaces* carrying hundreds of spots: identify the surface from one photo
— full or partial, at an angle — and label **every individual spot** on it.

Built for real-world imagery: spots range from tiny specks to long
streaks (4:1 elongation), some faded to near-invisibility; views suffer
uneven lighting, vignettes, gamma shifts, glossy wet sheen, cracks and
scratches, clutter blobs that belong to no surface, missing/worn-away
spots, and photos taken from far away or close up. See the stress matrix
below.

## Quick start

```bash
pip install numpy opencv-python-headless scipy

# 10,500-permutation single-spot benchmark (150 identities x 70 views)
python -m spotid.evaluate --identities 150 --views 70

# multi-surface benchmark (12 surfaces x 600 spots, full + cropped views)
python -m spotid.evaluate_surface --surfaces 12 --spots 600 --views 6

# real-world stress matrix (glare, fade, clutter, distance, ... combined)
python -m spotid.evaluate_stress --surfaces 10 --spots 600 --views 4

# demo images (permutation grid, cross-angle matches, annotated surface)
python -m spotid.demo --out-dir demo_out

python -m pytest tests/
```

## Library usage

```python
import numpy as np
from spotid import generate_identity, render_view, SpotMatcher
from spotid.evaluate import enroll_identity

# a seed IS a spot identity — same seed, same splotch, forever
contour = generate_identity(seed=42)

# render a random permutation of it
img, info = render_view(contour, np.random.default_rng())

# enroll identities, then identify any permutation
matcher = SpotMatcher()
for seed in range(100):
    enroll_identity(matcher, seed)
[(who, dist)] = matcher.identify(img)     # -> 42
```

Surfaces:

```python
from spotid.surface import generate_surface, render_surface_view
from spotid.surface_matcher import SurfaceMatcher

surfaces = [generate_surface(i, n_spots=600) for i in range(12)]
m = SurfaceMatcher()
for s in surfaces:
    m.enroll_surface(s)

img, info = render_surface_view(surfaces[5], np.random.default_rng())
res = m.identify(img)[0]
res.surface_id      # -> 5
res.assignments     # -> [(query_blob_i, spot_index_on_surface, residual_px), ...]
res.mode            # "global" (whole surface seen) or "partial" (cropped view)
```

## How it works

**Spot shapes.** An identity seed drives a radial-harmonic blob plus smooth
random warps — deterministic, organic, unique. The renderer applies exact
perspective projection of the contour (tilt up to ~55°), then rasterizes
into a scene with lighting gradients, texture, blur and noise.

**Viewpoint invariance.** A tilted view of a planar spot ≈ an affine
transform of its shape. The descriptor cancels it:

1. *Whiten* the segmented contour by the inverse square root of its
   second-moment matrix — any affine distortion collapses, leaving only an
   unknown rotation.
2. Describe the whitened contour with rotation-invariant signatures:
   Fourier-descriptor magnitudes (normalized by |Z₁|, log-compressed),
   a radial-distance histogram, and Flusser affine moment invariants.
3. The matcher enrolls several tilted views per identity and learns
   per-dimension within-identity noise vs. between-identity signal
   (a diagonal Fisher/LDA transform), then matches by cosine distance.

**Surfaces (constellations).** A surface = ~600 spots placed deterministically
on a plane; the layout is its fingerprint (like whale-shark photo-ID):

- *Global mode* (whole surface visible): whiten the centroid cloud (cancels
  the affine view), recover rotation by FFT correlation of angular
  histograms, refine with mutual-nearest-neighbor ICP, score with a tight
  inlier radius. Winner gets a RANSAC homography fit that assigns every
  blob to its spot.
- *Partial mode* (cropped view — automatic fallback): each spot carries an
  affine-invariant local signature (sorted triangle-area ratios over its 6
  nearest neighbors); signatures vote for candidate surfaces and seed a
  RANSAC homography that verifies and labels the visible spots.
- Per-spot shape descriptors add agreement evidence to the surface score.

## Measured results

Spot level — 150 identities, 10,500 rendered permutations, tilts 0–55°:

| tilt      | top-1 accuracy |
|-----------|----------------|
| 0–15°     | 99.9 %         |
| 15–30°    | 99.7 %         |
| 30–45°    | 95.2 %         |
| 45–55°    | 86.1 %         |
| **all**   | **96.1 %**     |

Surface level — 600 spots/surface, gallery of 12 surfaces (7,200 enrolled
spots), 72 query views at tilts 0–50°:

| view kind          | surface top-1 | spot assignment precision | coverage of visible spots |
|--------------------|---------------|---------------------------|---------------------------|
| full surface       | 36/36 (100 %) | 21,510/21,510 (100 %)     | 99.6 %                    |
| ~30 %-area crop    | 36/36 (100 %) | 11,871/11,871 (100 %)     | 99.8 %                    |

Isolated single-spot ID degrades gracefully with tilt; once spots live on a
surface, constellation geometry makes identification nearly exact — the
600-spot layout is far more distinctive than any single shape.

## Learned embeddings (spotid/ml)

The synthetic generator doubles as an infinite labeled dataset, so the
handcrafted descriptor can be replaced by a *trained* one:

```bash
pip install torch

# train the encoder (CPU proof of concept, ~25 min)
python -m spotid.ml.train --steps 1200 --out spotid/ml/checkpoints/encoder.pt

# head-to-head vs the classical descriptor on identical images
python -m spotid.ml.evaluate_ml --checkpoint spotid/ml/checkpoints/encoder.pt
```

- `dataset.py` renders fresh views on the fly (no stored dataset, no
  labeling); training identities use seeds ≥ 1,000,000 so evaluation
  identities are never seen in training.
- `model.py` is a compact CNN (~1.2M params) mapping a 96×96 spot patch
  to an L2-normalized embedding, trained with supervised-contrastive
  loss (views of the same spot attract, different spots repel).
- `infer.py` wraps the trained encoder as a drop-in `describe_image`,
  so `SpotMatcher` works unchanged with learned embeddings.

Scaling on a GPU is the same script:
`python -m spotid.ml.train --device cuda --width 64 --embed-dim 256
--steps 20000 --ids-per-batch 32 --id-pool 20000`.

## Files

| file                 | what it does                                    |
|----------------------|-------------------------------------------------|
| `shapes.py`          | seed → deterministic splotch contour             |
| `render.py`          | single-spot permutation renderer                 |
| `features.py`        | segmentation + affine-invariant descriptors      |
| `matcher.py`         | spot gallery, Fisher-weighted nearest neighbor   |
| `surface.py`         | surface generation + angled surface renderer     |
| `surface_matcher.py` | constellation matching, full & partial views     |
| `evaluate.py`        | spot-level benchmark CLI                         |
| `evaluate_surface.py`| surface-level benchmark CLI                      |
| `demo.py`            | demo image generation                            |
