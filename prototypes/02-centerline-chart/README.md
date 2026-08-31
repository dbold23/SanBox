# Prototype 02 — Centerline chart + strain measurement harness

The bend-invariant **(arc-length `s` × circumferential offset)** parameterization core from
**Approach 2** of `docs/sevengill-canonical-reid/03-candidate-approaches.md`, built 2D-first in pure
numpy/scipy/PIL and verified end-to-end on synthetic ground truth. No torch, no timm, no pretrained
weights, no real data — everything here runs and is measured in this repo as-is.

**This module is simultaneously the Phase 1B midline rectifier from Approach 1.** That synergy is why
it is worth building now: `mask → extract_centerline → rectify` *is* the "fit a spline midline, resample
the flank normal to that curve into a fixed-size rectangle" step of Approach 1B, and it is the offline
2D restriction of Approach 2's (s × φ) tube chart (in 2D the circumferential angle collapses to the
signed normal offset `r`). One codebase serves both experiments, so the Approach 1B kill test
("rectification must beat raw crops by ≥3 Rank-1 points") and the Approach 2 chart both depend on the
code that is verified here.

## Design decisions carried over from the approach spec

- `s` is computed **on the centerline** (medial-weighted shortest path over the distance-transform
  ridge), never integrated along the skin — the neutral-axis argument (spec decision 1).
- Head→tail orientation is the deterministic **widest-end-first rule** (mean distance-transform
  half-width over the first vs last 10% of stations; wider end = head).
- 3D frames are **rotation-minimizing** (double-reflection method, Wang et al., ACM TOG 2008), not
  Frenet — no sign flip at curvature inflections, zero twist on planar bends (spec decision 2). The 2D
  chart uses the left normal; the RMF implementation is in place and tested for the future mesh chart.
- The chart **inverts** (`chart_to_image` / `image_to_chart`), so findings measured in chart space can
  be annotated back onto the frame (spec failure-mode 4: invertibility designed in from day one).

## Modules

| file | contract |
|---|---|
| `centerline.py` | `extract_centerline(mask, n_stations)` → (n, 2) polyline, uniform arc length, head-first, deterministic, robust to ragged masks (largest component + hole fill + medial-weighted path). `arc_length`, `resample_polyline`. |
| `frames.py` | 2D unit tangent + left normal (central differences); 3D rotation-minimizing frames (double reflection), tested orthonormal, zero-twist on planar curves, and against the analytic RMF of a helix. |
| `chart.py` | `rectify(image, centerline, half_width, n_s, n_r, mask)` → (n_s × n_r) strip, bilinear, off-body samples NaN. `chart_to_image` / `image_to_chart` inverse pair. |
| `strain_demo.py` | The measurement (below). `python strain_demo.py` → `results/metrics.json` + `results/panel_eps_*.png`. |
| `tests/` | pytest; 22 tests, ~3 s, zero optional deps. |

## The measurement (`strain_demo.py`)

A synthetic elongate fish with a seeded spot texture painted in body-frame (s, r) is rendered straight
and bent through a known constant-curvature centerline (total turn 1.2 rad), with a multiplicative
arc-length stretch ramping linearly across the body: strain `ε·r/W` — `+ε` at the convex outer fibre,
`−ε` at the concave outer fibre, zero at the midline (the bending-beam profile). The **real pipeline**
(mask → centerline → rectify) is run on both renders; spot centroids are detected in both charts
(threshold + connected components), matched nearest-neighbour, and per-spot chart displacement is
reported split by convex/concave side.

**ε defaults to 0.05** — the midpoint of the ±3.9–6.6% longitudinal skin-strain bracket measured by
sonomicrometry in a swimming leopard shark at ~1 BL/s (Donley & Shadwick, *J. Exp. Biol.* 206(7), 2003;
[SEARCH]-grade per the evidence rules in `docs/sevengill-canonical-reid/README.md` — do not promote).

### Measured numbers (seed 0, 28 spots, body length 560 px)

| | ε = 0 (bend only) | ε = 0.05 (bend + strain) |
|---|---|---|
| matched spots | 28 / 28 | 28 / 28 |
| mean \|Δs\| | **0.40 px (0.072% BL)** | **3.89 px (0.69% BL)** |
| max \|Δs\| | 1.58 px (0.28% BL) | 8.06 px (1.44% BL) |
| mean \|Δr\| | 0.14 px | 0.90 px |
| convex side, mean signed Δs | −0.31 px | **+3.55 px** (tail-ward) |
| concave side, mean signed Δs | −0.09 px | **−4.32 px** (head-ward) |
| measured-vs-predicted slope (Δs = ε·(r/W)·s) | n/a | **1.019** (intercept −0.19 px) |

Two claims, both demonstrated in a controlled setting:

1. **Bend invariance (ε = 0):** rectifying a strongly bent body recovers spot positions to a mean of
   0.07% of body length — the residual is pipeline error (centerline extraction + resampling), not
   chart instability.
2. **Strain is the irreducible error (ε = 0.05):** with literature-bracket skin strain the residual is
   ~0.7% BL mean / 1.4% BL max, antisymmetric between flanks, and matches the beam-model prediction
   with slope ≈ 1. **No centerline chart can remove this** — it is a property of the skin, not of the
   parameterization (spec: "flatten then nonrigidly register, never rigidly compare"). This is finding
   02 of the report, quantified. Note it scales with the *imposed* ε: at a hard-turn outer-fibre
   estimate (~10–12%) the displacement doubles-plus.

## What this is NOT yet (next milestones)

- **No fins-as-separate-charts** — trunk-only strip; fin insertion curves as landmarks and per-fin
  patches (BFF-style cone singularities) are milestone 2.
- **No 3D mesh chart** — the (s × φ) cylindrical chart on a template mesh; the RMF machinery in
  `frames.py` is already built and tested for it.
- **No LBS carry** — computing the chart once on a rest-pose template and carrying it through skinning
  (spec decision 4) needs the serial-spine skeleton fix and a sevengill template (blocked on P0-b).
- **No anatomical anchoring** — `s=0` sits at the extracted snout tip (a gauge the tests measure as a
  ~3 px offset); gill-slit landmark anchoring (spec decision 3) replaces it on real data.
- **Fused self-touching bends are undetectable** — a mask whose flanks fuse (gap = 0) yields a
  confidently wrong shortcut centerline that passes the tubularity check, because the fused
  region widens exactly as the path shortcuts. Keep bends resolvable in the segmentation; the
  extractor warns on blob/disc-like masks (ratio < 0.8) but cannot warn on this one.
- **No φ periodicity / ventral seam** — 2D has no circumference; seam handling arrives with the mesh
  chart.

## Run

```
python -m pytest tests -q     # 22 tests, ~3 s
python strain_demo.py         # writes results/metrics.json, results/panel_eps_*.png
```

Python ≥3.9 (the lab Mac), numpy/scipy/Pillow only.
