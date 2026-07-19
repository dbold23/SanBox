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

Real-world stress matrix — 10 surfaces × 600 spots, 40 views per
condition (spot sizes speck-to-blob, elongation to 4:1, 15 % faded):

| condition            | surface top-1 | assignment precision | coverage |
|----------------------|---------------|----------------------|----------|
| baseline             | 40/40         | 0.999                | 0.86     |
| harsh lighting       | 40/40         | 0.999                | 0.87     |
| glossy sheen         | 40/40         | 0.998                | 0.87     |
| missing + clutter    | 40/40         | 0.991                | 0.76     |
| heavy fade           | 40/40         | 0.999                | 0.87     |
| far away             | 36/40         | 0.998                | 0.61     |
| close-up crop        | 40/40         | 1.000                | 0.87     |
| cracks + texture     | 40/40         | 0.992                | 0.80     |
| everything at once   | 40/40         | 1.000                | 0.84     |
| everything + crop    | 39/40         | 1.000                | 0.87     |

Coverage = correctly labeled fraction of spots that are actually visible
(drawn, in frame, not under glare). When a spot *is* assigned a label,
that label is right ≥ 99 % of the time in every condition; what harsh
conditions cost is how many spots remain readable at all.

## Learned embeddings (spotid/ml)

The synthetic generator doubles as an infinite labeled dataset, so the
handcrafted descriptor can be replaced by a *trained* one:

```bash
pip install torch

# train the encoder (CPU proof of concept, ~20 min)
python -m spotid.ml.train --steps 2500 --out spotid/ml/checkpoints/encoder.pt

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

Measured after 2,500 CPU steps (~20 min), 100 unseen identities × 30
views, tilts to 55°, identical images and matcher for all rows:

| descriptor              | top-1      | at 45–55° tilt |
|-------------------------|------------|----------------|
| classical (handcrafted) | 96.0 %     | 87.1 %         |
| learned (CNN)           | 97.6 %     | 93.2 %         |
| **ensemble (both)**     | **98.7 %** | **96.3 %**     |

Twenty minutes of CPU training beats the handcrafted descriptor,
especially at extreme viewing angles, and the concatenated ensemble
(`spotid.ml.infer.EnsembleDescriptor`) beats both — the two make
different mistakes. Scaling on a GPU is the same script:
`python -m spotid.ml.train --device cuda --width 64 --embed-dim 256
--steps 20000 --ids-per-batch 32 --id-pool 20000`.

### learned (GPU)

Running that GPU recipe end to end (`spotid/ml/gpu_train.sh`: 20,000
steps, width 64, embed 256, ~56 min on an RTX 5090) and benchmarking on
150 unseen identities × 40 views with tilts to 60° (6,000 queries),
identical images and matcher for all rows:

| descriptor              | top-1       | 30–45° tilt | 45–60° tilt |
|-------------------------|-------------|-------------|-------------|
| classical (handcrafted) | 95.6 %      | 95.2 %      | 83.7 %      |
| **learned (GPU)**       | **100.0 %** | **100.0 %** | **100.0 %** |
| ensemble (both)         | 100.0 %     | 100.0 %     | 100.0 %     |

The scaled-up encoder saturates the benchmark — perfect top-1 in every
tilt bucket, including the 45–60° range where the classical descriptor
drops to 83.7 %. Checkpoint: `spotid/ml/checkpoints/encoder_gpu.pt`
(trained on an RTX 5090, torch 2.13.0+cu130, sm_120).

Independently reproduced on CPU with a separate protocol (100 unseen
identities × 30 views, seed 7): 3,000/3,000 top-1, all tilt buckets
1.000, vs classical 0.960 on identical images.

## Real-world validation (sevengill sharks) — negative result

The synthetic pipeline was tested on a real dataset: 58 photos of
sevengill shark flanks, 6,618 hand-annotated spots (`spotid/realdata.py`,
`spotid/real_reid.py`, `spotid/match_matrix.py`). Some individuals were
photographed more than once, giving true re-sighting pairs.

**The synthetic result did NOT transfer. The constellation matcher does
not re-identify real sevengill individuals as implemented.** Scoring each
photo pair by the fraction of the smaller constellation explained under a
single homography:

| pairs                                   | mean fraction |
|-----------------------------------------|---------------|
| true re-sightings (same individual)     | 0.33          |
| all impostor pairs                      | 0.18 (but p90 0.37, **max 0.65**) |

The true signal is real but weak and drowned by false matches: **11 % of
impostor pairs beat the median true pair**, and using each re-sighting
image as a query, the correct individual is the top match only **2 of 8**
times. The full 41×41 match matrix shows no per-individual block
structure — instead, dense-spot images match *everything* (bright stripes).

Why it fails on real data (all absent from the synthetic model):

- **Curved, non-planar flanks.** A shark's side is a 3-D curved surface;
  two views are not related by a single homography, so true matches are
  suppressed while dense clouds still yield coincidental partial
  alignments.
- **High spot density + RANSAC.** 150–400 spots give many chances for a
  spurious consensus set (impostors reach 0.65 explained).
- **Partial overlap.** Re-sighting photos often show different extents of
  the flank, so genuine overlap is limited.

**A curvature-aware matcher was then built** (`spotid/reid_local.py`):
rotation/scale-invariant local shape-context descriptors + mutual-NN
candidates + *neighborhood-preservation* verification (a match is kept
only if several of a spot's spatial neighbors are matched to the other
spot's neighbors — a non-rigid check that survives curvature but rejects
coincidental collisions).

This **fixed the false-match problem** (impostor pairs now score ≈ 0,
where the global homography reached 0.65) but revealed the deeper limit:
it re-identifies only the mild-viewpoint pair (`AOTB_A002`: 5 verified
matches, cleanly rank 1, zero for all 37 impostors) and scores 0 on the
larger-viewpoint true pairs. Diagnosis: candidate correspondences are
found for every pair, but the local neighborhood is only *preserved* when
the two photos are taken from similar angles. Across big viewpoint changes
on a curved flank, centroid geometry alone is not preserved — so both the
global and local matchers land at **2/8 top-1**.

**Honest conclusions:**

- The spot pattern *is* individually distinctive, and matching works
  cleanly and confidently **when the two photos share a similar
  viewpoint** (A002). A standardized photographing angle/distance would
  likely make centroid matching usable in practice.
- Across large viewpoint changes it does not work from centroids alone.
  The algorithmic path is appearance-based local features (HotSpotter /
  Wild-ID: image descriptors at each spot + distinctiveness scoring), and
  the learned path needs many labeled individuals with re-sightings — far
  more than this 58-image set provides.

**Update — real re-ID does work, with the right features.** The negative
result above is specific to matching *spot centroids*. A follow-up probe
(`spotid/probe_matchers.py`) matched flank crops with off-the-shelf local
image features (DISK deep features, SIFT) + RANSAC:

| matcher | TRUE re-sightings (inliers) | DIFFERENT individuals |
|---------|-----------------------------|-----------------------|
| SIFT    | 57–156                      | 0–13                  |
| DISK    | 553–821                     | 14–23                 |

All 4 true re-sighting pairs separate cleanly from all 24 impostor pairs
(zero overlap), including the pairs the centroid matcher missed. The
discriminative signal is in the spot *appearance and local skin texture*,
not just centroid geometry — reducing each spot to a point threw it away.
See `spotid/FRONTIER.md` for the full frontier design and roadmap.

The synthetic benchmarks below remain valid *for the synthetic setting*
(planar, controlled); they should not be read as real-world performance.

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
| `evaluate_stress.py` | real-world stress-matrix CLI                     |
| `demo.py`            | demo image generation                            |
| `ml/`                | learned embeddings: dataset, model, train, infer |
| `realdata.py`        | load real YOLO dataset, parse individuals        |
| `real_reid.py`       | real individual re-ID via constellation matching |
| `ml/finetune_real.py`| domain-adapt encoder on real crops               |
| `ml/eval_real_embed.py` | appearance eval on real same-spot pairs       |
