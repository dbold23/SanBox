# Prototype 06 — spot-proxy

**Can a sevengill be re-identified from the spot pattern the OSEA tagger
already stores, and can that be measured before the catalogue holds a single
usable recapture?**

Prototype 06 answers both with one idea: put a synthetic domain and the real
domain through the *same detector*, and match on what comes out the other side.

**The answer, measured.** On a 160-frame calibrated corpus (40 animals, 4
sightings each over three years, prototype 05 drift) run through the deployed
OSEA detector, the constellation matcher retrieves the right animal at
**Rank-1 0.279, Rank-5 0.404, Rank-10 0.500, AUROC 0.670**, over 136 same-side
leave-one-out queries against a median gallery of 78. Replacing the detector
with the renderer's exact spot centres and changing nothing else lifts that to
**Rank-1 0.571, AUROC 0.830** — so roughly half the loss is detection and half
is rectification and matching. Opposite-flank pairs of the same animal, which
share no pattern, score at chance (AUROC 0.473), which is the control that says
the signal is real. Over the same three years there is no measurable drift
decay. On the real catalogue none of this is measurable at all: it holds zero
cross-encounter same-flank pairs, and the flank is recorded on 2 of its 1091
photographs.

**What is calibrated, and how well.** The synthetic detection distribution sits
at KS objective **0.126** against the real one, on a measured sampling floor of
0.035 at this corpus size. Whole tables below; every number names the file under
`results/` it came from.

## The sim-to-real argument

The matcher never sees RGB. Its entire input is one detection dict in the shape
`tagger/data/catalog.db` already stores per photograph — a YOLO body polygon,
zero or more obstruction polygons, and a list of YOLO spot boxes:

```python
{"width": int, "height": int,
 "body_polygon": [[x, y], ...],
 "obstruction_polygons": [[[x, y], ...], ...] | None,
 "spots": [{"x","y","w","h","cx","cy","conf"}, ...]}
```

Three consequences, and they are the whole prototype:

1. **A render and a photograph are interchangeable inputs** as long as the same
   extractor produced both. `osea_contract.py` is that extractor, verified
   byte-identical to what the OSEA pipeline wrote for the catalog rows that had
   detections (`README_contract.md`). It runs unchanged on
   `results/synth_calib/body/*.jpg` and on `tagger/data/images_raw/*.jpg`.
2. **Photorealism is not the target.** The renderer only has to be right about
   what survives the detector: spot size, spacing, count and position on the
   body; the body silhouette and pose; and enough scene junk (hands, deck, blur,
   JPEG) that the detector behaves as it does on a real frame. A synthetic frame
   that looks wrong to a person but produces the same detection distribution is
   a *success* here. This is the shark-pose-3d principle — train and measure on
   a representation the simulator can actually get right.
3. **Calibration is therefore a distribution-matching problem, not an art
   problem.** `compare_features.py` runs two-sample KS tests between the real
   and synthetic feature distributions of exactly the quantities the ingest
   stores, and `calibrate.py` searches the render config against that objective.
   Every calibration decision in this prototype was made by that number, not by
   eye — the eye is used only to explain a number and to catch defects the
   objective cannot see.

The payoff is the thing the real catalogue cannot supply. `catalog.db` today has
**zero cross-encounter same-side same-individual pairs**: every same-individual
pair is either a same-encounter near-duplicate or the one L-vs-R pair of
`AOTB_A014`. There is no re-identification signal in it to measure. The
calibrated synthetic corpus has 40 identities re-sighted four times over three
years with prototype 05's pattern drift, at least twice on the same flank, so it
has the recaptures the catalogue is missing — and it was built to reproduce the
catalogue's own detection statistics, which is the only argument that a number
measured on it says anything about the real one.

## Pipeline

```
          REAL                                       SYNTHETIC
 tagger/data/catalog.db                    prototype 04 scan (results/real_v11)
  1091 photographs                                     |
          |                              real_body.py: chart (s, phi), decimate
          |                                 to 1.5 mm, pose (planar bend + yaw)
          |                                            |
          |                              prototype 05 pattern.Individual
          |                              + drift.resight() per sighting date
          |                                            |
          |                              synth_render.py: albedo chart on the
          |                              de-lit real skin, OSEA-like camera,
          |                              sun + ambient + Blinn-Phong, deck
          |                              background, hand occluders, blur, JPEG
          |                                            |
          |                                  body/*.jpg  +  gt/*.npz
          |                                            |
          +-------------------+     +------------------+
                              |     |
                   osea_contract.detect()   <-- THE SAME EXTRACTOR, both domains
                 body polygon + obstruction polygons + spot boxes
                              |
              +---------------+------------------+
              |                                  |
   osea_contract.features()          constellation.extract_spotset()
   scalars: n_spots, density,        prototype 02 medial-axis chart (s, r),
   nn_median, size/conf quantiles,   PCA frame as the fallback
   aspect, area_norm, body_conf                   |
              |                       shape contexts -> Hungarian
   real_features.py / synth_features.py  -> RANSAC on an 'axis' model
              |                                  |
   compare_features.py: KS tests        match_score in [0, 1]
              |                                  |
   calibrate.py: config search        eval_constellation.py:
   -> results/calibration/best.json    Rank-1/5, AUROC, drift buckets
```

## Files

| path | what it is |
|---|---|
| `osea_contract.py` | the door both domains go through: `load_models`, `detect`, `features`. See **`README_contract.md`** for the column shapes, the three caveats and the verification against stored rows. |
| `real_features.py` | CLI: run the contract over `catalog.db`, write the real feature distribution |
| `real_body.py` | loads prototype 04's `results/real_v11` bind pose, charts it, decimates by vertex clustering, poses it |
| `synth_render.py` | the frame generator, the config, the corpus plan, the contact sheets, the benchmark |
| `synth_features.py` | run the contract over a rendered corpus and score the detections against the render's own spot ground truth |
| `compare_features.py` | two-sample KS between the two domains; `objective()` is the calibration loss |
| `calibrate.py` | render -> detect -> compare driver and the config search |
| `constellation.py` | the non-RGB matcher: rectify, describe, assign, RANSAC |
| `eval_constellation.py` | the benchmarks: toy grid, real arm, synthetic arm, GT ceiling, ablations |
| `results/` | every number in this file; each table names its own source |

`results/synth_smoke` and everything measured off it (`results/compare/synth_smoke`,
`results/constellation/bridge_smoke_*`, `synth_gt_*`) predate the four render
defect fixes and describe a renderer that no longer exists. They are kept only so
the before/after comparison stays auditable; nothing in this file cites them.

Tests: **160 passed in 50 s** (`python -W ignore -m pytest tests -q`).

| file | tests | what it pins |
|---|---:|---|
| `tests/test_contract.py` | 38 | PCA-frame invariance, hand-checked `features()`, the `max_det` truncation report, live `detect()` |
| `tests/test_render.py` | 25 | the chart, the decimation, the pose identity, the spot ground truth |
| `tests/test_calibrate.py` | 27 | config round-trip, the objective is exactly 0 on identical summaries, one regression test per fixed render defect |
| `tests/test_corpus_eval.py` | 28 | the benchmark corpus plan, the same-side rank protocol, the drift buckets, the `--conf-min` filter, the frame report |
| `tests/test_constellation.py` | 21 | rectification, descriptor invariance, score symmetry, chart rejection |
| `tests/test_bridge.py` | 15 | the detector-vs-GT matcher and the record shape shared by both domains |
| `tests/test_real_features.py` | 6 | the body-conditioned scalar table and the slim record |

## Runbook

One interpreter throughout — the main checkout's venv, Python 3.9.6 — and
always with `-W ignore` (see the environment note in the appendix). Timings are
wall clock on a 10-core M-series Mac, CPU only.

```zsh
P="/Volumes/External Dive 2TB/projects/marine-cv/7Gill/.venv/bin/python"
cd .../prototypes/06-spot-proxy
```

| # | command | writes | time |
|---|---|---|---|
| 1 | `"$P" -W ignore real_body.py --cells 1.5,2,2.5,4,6` | `assets/real_body_*mm.npz`, `results/decimation.json` | chart 4.6 s, then 0.6 s per decimation level |
| 2 | `"$P" -W ignore real_features.py --overlay-n 12 --overlay-every 90` | `results/real/{detections,detections_slim}.jsonl`, `summary.json`, `skipped.jsonl`, two contact sheets | 352 s (0.212 s/image detect, 3.10 img/s over 1097 candidate rows) |
| 3 | `"$P" -W ignore synth_render.py --bench` | `results/bench.json` | ~60 s (3 frames at each of 5 cell sizes) |
| 4 | `"$P" -W ignore calibrate.py --before --stage1 --stage1b --stage2 --stage3 --stage4 --final --workers 6` | `results/calibration/{grid.jsonl,best.json,best_record.json,report.md,calibration_contact.png}` | ~25 min at 5-7 workers (57 configs, 15,426 s of config time) |
| 5 | `"$P" -W ignore synth_render.py --out results/synth_calib --identities 40 --sightings 4 --seed 0 --config results/calibration/best.json --min-same-side 2` | `results/synth_calib/{body,gt}/*`, `truth.jsonl`, `config.json`, `summary.json` | **1238 s** = 7.74 s/frame for 160 frames |
| 6 | `"$P" -W ignore synth_features.py --corpus results/synth_calib --contact-n 4` | `results/synth_calib/{detections.jsonl,detector_summary.json,detector_contact.png}` | 45 s (0.258 s/image on cpu) |
| 7 | `"$P" -W ignore compare_features.py --real results/real/detections.jsonl --synth results/synth_calib/detections.jsonl --out results/compare/synth_calib` | `results/compare/synth_calib/{summary.json,summary.md,compare_contact.png}` | 4.4 s |
| 8 | `"$P" -W ignore eval_constellation.py --toy` | `results/constellation/toy_summary.json`, `toy_grid_contact.png` | 114 s |
| 9 | `"$P" -W ignore eval_constellation.py --real results/real/detections.jsonl` | `results/constellation/real_summary.json`, `real_scores_contact.png` | 428 s (7.1 min; dominated by rectifying all 1030 body-bearing records) |
| 10 | `"$P" -W ignore eval_constellation.py --synth results/synth_calib/detections.jsonl --truth results/synth_calib/truth.jsonl --prefix synth_calib` | `results/constellation/synth_calib_summary.json` | 760 s |
| 11 | same, `--conf-min 0.40 --prefix synth_calib_c40` and `--conf-min 0.50 --prefix synth_calib_c50` | `..._c40_summary.json`, `..._c50_summary.json` | 326 s / 213 s |
| 12 | `"$P" -W ignore eval_constellation.py --synth-from-gt results/synth_calib/gt --truth results/synth_calib/truth.jsonl --prefix synth_calib_gt` | `results/constellation/synth_calib_gt_{detections.jsonl,summary.json}` | 2016 s (34 min; 249 GT spots per image over 12,720 pairs) |
| 13 | `"$P" -W ignore eval_constellation.py --frames results/real/detections.jsonl` and `--frames results/synth_calib/detections.jsonl --prefix synth_calib_frames` | `results/constellation/{real_frames,synth_calib_frames}_summary.json` | 306 s / 49 s |
| 14 | `"$P" -W ignore eval_constellation.py --ablate` | `results/constellation/ablate_summary.json` | 998 s |
| 15 | `"$P" -W ignore -m pytest tests -q` | — | **160 passed** in 50 s |

Steps 10-12 are independent and were run in parallel.

`--min-same-side 2` on step 5 is what makes the corpus a benchmark rather than a
simulated field catalogue. Prototype 05's `make_dataset.plan_sightings` draws
deliberate singletons, varies the sighting count around the target, and flips
the flank per sighting: on the smoke corpus that produced three identities whose
*both* same-individual pairs were L-vs-R, so they shared no spots by
construction and the resulting AUROC measured nothing.
`synth_render.plan_same_side_sightings` keeps 05's log-uniform recapture gaps and
its distinct-date rule but fixes the count and guarantees at least two sightings
on one flank. Everything else — the identity, its length, its drifted pattern —
still comes from 05: `identity_timeline` uses the same generator seeds and the
same `drift.resight` walk, so it is the same population of animals.
`corpus.min_same_side` defaults to `None`, which leaves 05's plan in place.

## The real distribution

`real_features.py` runs the contract over every ingested photograph in
`tagger/data/catalog.db` and writes the distribution the synthetic side has to
be pushed onto. Source: `results/real/summary.json`; overlays in
`results/real/detections_contact.png` and `features_contact.png`.

1097 candidate rows, 6 skipped as under 800 px wide, **1091 processed** in 352 s
(0.212 s/image in `detect()`, 3.10 images/s). Body detected on **1030**; 61
frames produced no body polygon at the OSEA `body_conf >= 0.40` floor and are
excluded from every spot statistic, because there is no frame to normalise by.
**128,145 spots** in total, of which 24 images are censored at `max_det = 300`
and 117 have a self-intersecting body contour.

Per-image scalars. `density` is spots per `D_minor^2` of body area, `nn_median`
and `size_q50` are normalised by `D_minor` (the body's minor axis in pixels), so
all three are scale free.

| scalar | n | q05 | q25 | q50 | q75 | q95 |
|---|---:|---:|---:|---:|---:|---:|
| `n_spots` | 1030 | 39 | 69 | 112 | 165 | 249.5 |
| `density` | 1030 | 30.14 | 62.53 | 98.53 | 146.2 | 249.1 |
| `nn_median` | 1029 | 0.02835 | 0.03762 | 0.04512 | 0.05516 | 0.07608 |
| `size_q50` | 1030 | 0.01743 | 0.02568 | 0.0313 | 0.0384 | 0.04862 |
| `conf_q50` | 1030 | 0.3415 | 0.389 | 0.4223 | 0.459 | 0.5068 |
| `aspect` | 1030 | 1.371 | 1.745 | 1.947 | 2.173 | 2.813 |
| `area_norm` | 1030 | 0.5853 | 0.9565 | 1.175 | 1.423 | 1.833 |
| `bbox_width_frac` | 1030 | 0.5751 | 0.8403 | 0.9134 | 0.9798 | 1 |
| `body_conf` | 1030 | 0.5432 | 0.7198 | 0.8019 | 0.8573 | 0.9079 |
| `D_minor` | 1030 | 971.5 | 1564 | 1854 | 2105 | 2644 |
| `obstruction_count` | 1030 | 0 | 0 | 0 | 0 | 1 |
| `obstruction_area_frac` | 1030 | 0 | 0 | 0 | 0 | 0.02924 |

Pooled over every detected spot:

| per-spot | n | q05 | q25 | q50 | q75 | q95 |
|---|---:|---:|---:|---:|---:|---:|
| `size` | 128145 | 0.01511 | 0.02177 | 0.02911 | 0.03924 | 0.05875 |
| `nn` | 128144 | 0.009114 | 0.02942 | 0.04097 | 0.05678 | 0.09556 |
| `conf` | 128145 | 0.264 | 0.324 | 0.414 | 0.533 | 0.698 |

Three things in this table drive everything downstream. `conf` q05 = 0.264 with
q50 = 0.414 says the v1 spot head is running wide open and 46% of what it emits
is below 0.40 — hence three confidence floors everywhere. `bbox_width_frac`
q50 = 0.913 says a real photograph usually has the animal running nearly the
full width but not clipped, which is a framing constraint the renderer got wrong
until the calibration search. And `body_conf` q50 = 0.802 is the bar a synthetic
frame has to clear: below the OSEA 0.40 body floor the frame produces no body
polygon and therefore no spots at all.

## Calibration: matching the detection distribution

`compare_features.py` runs a two-sample KS test per feature between the real and
synthetic sides, at three spot-confidence floors. The loss the search minimises
is

```
objective = 0.5 * mean KS D over per-image {density, size_q50, nn_median, conf_q50}
          + 0.5 * mean KS D over pooled   {size, nn, conf}          (spot conf >= 0.25)
```

with a separate `geometry_objective` over per-image `{aspect, area_norm,
bbox_width_frac}`. Both are in `[0, 1]`; 0 means identical distributions.

**The objective has a sampling floor, and it was measured rather than assumed.**
Draw *n* images and *m* spots at random *from the real corpus* and score them
against the whole of it: whatever D that gives is what a perfectly matched
synthetic corpus of that size would score. Over 300 bootstrap draws:

| corpus size | per-image half | pooled half | objective floor (q05/q50/q95) |
|---|---:|---:|---|
| 37 images, 4,540 spots (the search) | 0.134 | 0.013 | 0.059 / **0.073** / 0.093 |
| 152 images, 19,292 spots (the corpus) | 0.064 | 0.006 | 0.028 / **0.035** / 0.043 |

So at the size the search ran at, a per-image D under ~0.13 was already
indistinguishable from a perfect match, which is why configurations were ranked
on the pooled half — the half that still discriminates there.

### Before and after the calibration search

Same 38 draws, seed 0, real side = all 1091 ingested photographs (1030 with a
body). BEFORE is the pre-fix renderer replayed frame for frame through
`calibrate.BEFORE_OVERRIDE`, so the two columns differ only by the four defect
fixes and the config search.
Source: `results/calibration/compare/{before_big,b3_r37_elev}/summary.json`,
also tabulated across all 57 configs in `results/calibration/report.md`.

| feature | kind | D before | D after | real q25/q50/q75 | before q25/q50/q75 | after q25/q50/q75 |
|---|---|---:|---:|---|---|---|
| `area_norm` | img | 0.186 | 0.297 | 0.9565/1.175/1.423 | 0.8712/1.246/1.616 | 0.7592/0.9352/1.524 |
| `aspect` | img | 0.247 | 0.151 | 1.745/1.947/2.173 | 1.79/2.026/2.478 | 1.734/1.907/2.296 |
| `bbox_width_frac` | img | 0.744 | 0.193 | 0.8403/0.9134/0.9798 | 1/1/1 | 0.8821/0.9157/0.9763 |
| `body_conf` | img | 0.386 | 0.219 | 0.7198/0.8019/0.8573 | 0.5872/0.6719/0.7823 | 0.7461/0.7893/0.8421 |
| `conf_q50` | img | 0.125 | 0.318 | 0.389/0.4223/0.459 | 0.4022/0.4217/0.4453 | 0.3978/0.4073/0.4235 |
| `density` | img | 0.408 | 0.222 | 62.53/98.53/146.2 | 36.08/59.73/81.23 | 77.38/128.9/181.7 |
| `n_spots` | img | 0.416 | 0.286 | 69/112/165 | 42.25/65.5/95.5 | 112.8/130.5/150.2 |
| `nn_median` | img | 0.226 | 0.094 | 0.03762/0.04512/0.05516 | 0.04032/0.05212/0.0609 | 0.03928/0.04481/0.05891 |
| `size_q50` | img | 0.268 | 0.131 | 0.02568/0.0313/0.0384 | 0.02269/0.0264/0.03121 | 0.02697/0.03103/0.03707 |
| `conf` | spot | 0.031 | 0.017 | 0.324/0.414/0.533 | 0.333/0.424/0.538 | 0.324/0.41/0.528 |
| `nn` | spot | 0.183 | 0.105 | 0.02942/0.04097/0.05678 | 0.03672/0.05077/0.0695 | 0.03414/0.0455/0.06186 |
| `size` | spot | 0.070 | 0.092 | 0.02177/0.02911/0.03924 | 0.02076/0.02738/0.03621 | 0.02429/0.03185/0.04177 |
| `u` | spot | 0.142 | 0.087 | -0.4233/-0.1006/0.2559 | -0.6092/-0.261/0.281 | -0.54/-0.1895/0.2426 |
| `v` | spot | 0.146 | 0.051 | -0.1729/-0.0508/0.08561 | -0.235/-0.1186/0.003291 | -0.1953/-0.07286/0.06553 |

| | objective @0.25 | @0.40 | @0.50 | geometry | body_conf median | bodies kept |
|---|---:|---:|---:|---:|---:|---:|
| before | 0.1757 | 0.1810 | 0.2050 | 0.3924 | 0.672 | 36/38 |
| after | **0.1313** | **0.1042** | **0.1200** | **0.2137** | **0.789** | 36/38 |
| real corpus | - | - | - | - | 0.802 | 1030/1091 |

11 of the 14 features improved at the 0.25 floor, 12 of 14 at 0.40 and 13 of 14
at 0.50; the only feature that regresses at every threshold is `area_norm`. The
biggest single move is `bbox_width_frac`, 0.744 -> 0.193: the pre-fix camera
clipped the animal at the frame edge on 38 of 38 frames.

The winning override (`results/calibration/best.json`) is

```json
{"camera": {"elevation_deg": [0.0, 32.0]},
 "pattern": {"n_spots": 1300, "min_sep": 0.0080, "radius_median": 0.0037}}
```

on top of a `DEFAULT_CONFIG` that also carries the four defect fixes. `best.json`
alone will not reproduce this against the old defaults.

The three that regress at the 0.25 floor are honest costs of the winning trade.
`conf_q50` went 0.125 -> 0.318: the *pooled* confidence distribution matches
almost exactly (D 0.017), but every synthetic frame is the same mesh, skin and
sensor, so the per-image median spans 0.398-0.424 against a real 0.34-0.51. It
is a spread problem, and no knob in the config space moves spread. `area_norm`
went 0.186 -> 0.297, which is the price of capping elevation at 32 degrees to
keep `body_conf` (0.219 against 0.386) — the two trade against each other and
0-32 degrees is the measured optimum. Pooled `size` went 0.070 -> 0.092 because
the spot radius was raised to fix the much larger per-image `size_q50` gap
(0.268 -> 0.131); it is back to 0.046 at the 0.40 floor.

### The 160-frame corpus against the real corpus

The corpus this prototype actually evaluates on (described in full in the next
section) is 40 identities x 4 sightings = 160 frames, four times the size the
search ran at, so the sampling floor is lower here. Source:
`results/compare/synth_calib/summary.json` (and `summary.md` for the full
three-threshold table).

| feature | kind | D @0.25 | D @0.40 | D @0.50 | real q25/q50/q75 @0.25 | synthetic q25/q50/q75 @0.25 |
|---|---|---:|---:|---:|---|---|
| `area_norm` | per-image | 0.230 | 0.230 | 0.230 | 0.9565 / 1.175 / 1.423 | 0.7309 / 0.9934 / 1.6 |
| `aspect` | per-image | 0.155 | 0.155 | 0.155 | 1.745 / 1.947 / 2.173 | 1.631 / 1.91 / 2.42 |
| `bbox_width_frac` | per-image | 0.151 | 0.151 | 0.151 | 0.8403 / 0.9134 / 0.9798 | 0.8574 / 0.9182 / 0.9893 |
| `body_conf` | per-image | 0.160 | 0.160 | 0.160 | 0.7198 / 0.8019 / 0.8573 | 0.7101 / 0.7825 / 0.8263 |
| `conf_q50` | per-image | 0.323 | 0.165 | 0.097 | 0.389 / 0.4223 / 0.459 | 0.3929 / 0.407 / 0.4233 |
| `density` | per-image | 0.192 | 0.176 | 0.133 | 62.53 / 98.53 / 146.2 | 71.44 / 129.2 / 180.5 |
| `n_spots` | per-image | 0.255 | 0.207 | 0.166 | 69 / 112 / 165 | 108 / 129.5 / 148 |
| `nn_median` | per-image | 0.124 | 0.140 | 0.071 | 0.03762 / 0.04512 / 0.05516 | 0.03837 / 0.04371 / 0.06215 |
| `size_q50` | per-image | 0.117 | 0.189 | 0.225 | 0.02568 / 0.0313 / 0.0384 | 0.02677 / 0.0303 / 0.04117 |
| `conf` | per-spot | 0.021 | 0.021 | 0.012 | 0.324 / 0.414 / 0.533 | 0.3227 / 0.408 / 0.525 |
| `nn` | per-spot | 0.090 | 0.057 | 0.073 | 0.02942 / 0.04097 / 0.05678 | 0.03353 / 0.04521 / 0.06227 |
| `size` | per-spot | 0.079 | 0.046 | 0.112 | 0.02177 / 0.02911 / 0.03924 | 0.02389 / 0.03152 / 0.04199 |
| `u` | per-spot | 0.074 | 0.092 | 0.099 | -0.4233 / -0.1006 / 0.2559 | -0.5212 / -0.1772 / 0.264 |
| `v` | per-spot | 0.029 | 0.043 | 0.067 | -0.1729 / -0.0508 / 0.08561 | -0.1877 / -0.05923 / 0.07939 |

160 frames, 152 with a body polygon (8 fell under the OSEA 0.40 body floor,
5.0% attrition against the real corpus's 5.6%). The four geometry rows are
properties of the body polygon alone, so they do not move with the spot
threshold.

| | objective @0.25 | @0.40 | @0.50 | geometry | per-image half | pooled half |
|---|---:|---:|---:|---:|---:|---:|
| calibrated corpus, 160 frames | **0.1262** | 0.1045 | 0.0987 | 0.1788 | 0.1890 | 0.0635 |
| sampling floor at this size | 0.0346 | - | - | - | 0.0637 | 0.0056 |
| ratio to floor | 3.6x | - | - | - | 3.0x | 11.4x |

The floor row is measured, not assumed: 300 bootstrap draws of 152 images and
19,292 spots *from the real corpus*, scored against the whole of it
(q05/q95 of the objective floor 0.0275/0.0430). So the corpus is 3.6x its own
noise floor, and the gap is concentrated in the per-image half — a spread
problem, not a location problem, exactly as the 38-frame search predicted.
`conf_q50` alone (D 0.323 at the 0.25 floor, but 0.097 at 0.50) contributes
almost half of the per-image mean, and the *pooled* confidence distribution
matches to D 0.021.

## The calibrated corpus (`results/synth_calib`)

The thing the whole prototype was built to produce: a corpus with the recaptures
`catalog.db` does not have, drawn to reproduce `catalog.db`'s own detection
statistics. Source: `results/synth_calib/{summary.json,truth.jsonl,config.json}`.

| | |
|---|---|
| frames | **160** = 40 identities x 4 sightings, seed 0 |
| config | `results/calibration/best.json` on the shipped `DEFAULT_CONFIG`, `corpus.min_same_side = 2` |
| plan | `plan_same_side_sightings(min_same_side=2)` — every identity has exactly 4 distinct dates and at least 2 on one flank |
| dates | 2019-03-03 to 2021-12-10; per-identity span 14 / 418 / 849 days (min/median/max) |
| pair intervals | 1 / 104 / 849 days over the 240 within-identity pairs |
| sides | 87 L, 73 R; per identity the majority flank appears 2x on 15 animals, 3x on 13, 4x on 12 |
| pairs | **141 same-side** and 99 opposite-side same-individual pairs (126 and 91 survive the OSEA body floor) |
| lengths | 141 to 284 cm, from 05's `LENGTH_CM_BRACKET` |
| rendered spots | 1300 per identity; **164 / 248.5 / 390** visible per frame (min/median/max) |
| GT spot size | `radius_px` p5/median/p95 = 2.76 / 10.88 / 26.13 |
| cost | 1238 s, 7.74 s/frame; 1.2 GB on disk with the per-frame `.npz` ground truth |

Contact sheets: `results/synth_calib/corpus_contact.png` (all 160 frames),
`detector_contact.png` (detections drawn over the GT on four frames) and
`zoomed_contact.png` (two synthetic flank crops beside two real ones at matched
scale — the source of the visual verdict under **Known limitations**).

## The detector on synthetic frames

`synth_features.py` runs `osea_contract.detect` over the corpus and scores every
detection against the render's own spot ground truth: a detection is a true
positive when its centre lands within `max(GT radius_px, 6 px)` of a *visible*
GT centre, one-to-one, greedy by descending detector confidence.
Source: `results/synth_calib/detector_summary.json`.

| spot conf floor | detections | TP | FP | FN | precision | recall |
|---|---:|---:|---:|---:|---:|---:|
| >= 0.25 | 19,292 | 18,855 | 437 | 20,961 | **0.977** | **0.474** |
| >= 0.40 | 10,097 | 10,036 | 61 | 29,780 | 0.994 | 0.252 |
| >= 0.50 | 5,675 | 5,651 | 24 | 34,165 | 0.996 | 0.142 |
| >= 0.60 | 2,698 | 2,688 | 10 | 37,128 | 0.996 | 0.068 |

The denominator is 39,816 *visible* rendered spots over 160 frames. 152 of the
160 kept a body at the OSEA 0.40 body floor (`body_conf` q05/q25/q50/q75/q95 =
0.515/0.710/0.782/0.826/0.883 against a real 0.543/0.720/0.802/0.857/0.908), and
a frame with no body polygon yields no spots at all, because the spot head only
runs inside the body crop. Per image, recall q05/q50/q95 is 0.233/0.498/0.598 and
precision 0.949/0.979/1.000. Detection cost 41.2 s wall, 36.3 s inside
`detect()`, 0.258 s/image on cpu.

True-positive confidence q05/q50/q95 is 0.264/0.412/0.698 (n = 18,855) against
0.254/0.302/0.511 for the 437 false positives, so the false positives really are
concentrated at the bottom of the range — which is why raising the floor to 0.40
removes 86% of them but also 47% of the true positives. The matcher section
shows which of those two matters more.

The eight frames that lost their body (`body_ok: false` in
`detector_summary.json`) draw camera elevations from 3.4 to 25.7 degrees over
three different backgrounds, so this is not the near-dorsal attrition the
38-frame search attributed it to; it has no single cause yet. One of them is the
bottom-right tile of `results/synth_calib/detector_contact.png` — a clearly
visible animal with 201 visible GT spots and zero detections, because the spot
head only runs inside a body crop. In the same sheet the missed spots (blue) sit
overwhelmingly on the dorsal rim, where the projection is edge-on, and among the
smallest spots; the detections (green) cover the flank.

**This precision is not comparable to any real-photo precision, and none is
reported anywhere in this prototype, because there is no per-spot ground truth
on a real photograph.** What the table measures is only whether the renderer
drew spots the deployed v1 detector can see. The real detector is visibly *less*
precise than this: 46% of its real detections sit below confidence 0.40 and the
overlays in `results/real/detections_contact.png` show boxes on plain skin. The
synthetic skin does not generate that clutter, which is the honest reading of
the near-perfect synthetic precision — the synthetic domain is *cleaner* than
the real one, not the detector *better* on it.

Recall is against the renderer's own `visible` flag, which includes spots at
grazing angles whose projected footprint is a two-pixel sliver. It is a floor,
not a failure.

| GT radius_px | visible GT spots | recall |
|---|---:|---:|
| [0, 2) | 1,215 | 0.000 |
| [2, 4) | 2,410 | 0.024 |
| [4, 6) | 3,745 | 0.199 |
| [6, 8) | 4,917 | 0.487 |
| [8, 12) | 10,365 | **0.664** |
| [12, 20) | 12,022 | 0.590 |
| [20, 40) | 4,726 | 0.346 |
| [40, inf) | 416 | 0.103 |

Detected spots have `radius_px` q05/q50/q95 = 6.18/11.59/22.24; missed ones
1.78/9.44/30.27. Recall peaks at 0.66 for spots of 8-12 px radius and falls away
at both ends: below 4 px nothing is detectable (3,625 spots, 9.1% of the
denominator), and above 20 px the spots are the near-silhouette ones the
projection stretches into slivers.

## The matcher

`constellation.py` takes one detection dict and returns a score in `[0, 1]`:

1. **Rectify.** `build_body_mask` -> prototype 02's medial-axis chart, giving
   each spot `s` in `[0, 1]` (arc length) and `r` in `[-1, 1]` (signed offset
   normalised by the *local* half width). A PCA frame is the fallback when the
   chart is unusable, and which one was used is recorded per set.
2. **Describe.** A shape context per spot over its K = 20 nearest neighbours,
   orientation fixed by the body axis (the chart has already removed rotation),
   scale normalised by the set's median nearest-neighbour distance.
3. **Assign and fit.** Chi-squared descriptor cost -> Hungarian assignment ->
   RANSAC on an `axis` model (translation and scale, separately in `s` and in
   `r`) -> inliers inside a gate of 0.6 x the median NN distance.
   `score = inliers / min(n_a, n_b)`, scored in both directions and the better
   kept, so the score is exactly symmetric.
4. **Orientation.** There is no head detector in the shipped weights, so all
   four `s -> 1-s` x `r -> -r` flips of the gallery set are tried.

Knob-by-knob ablations are in `results/constellation/ablate_summary.json`
(5 seeds x 20 identities at 2% jitter / 20% dropout / 20% clutter). The
headline: the descriptor gate is worth Rank-1 0.98 against 0.81 with no gate,
`axis` beats `s_affine` badly once a crop and an r-frame perturbation are added
(Rank-1 0.64 vs 0.45, AUROC 0.914 vs 0.760), and symmetrising costs nothing
measurable in accuracy (AUROC 0.9932 vs 0.9940 directed) at 2x the compute.

### Toy benchmark — the controlled ceiling

N identities of 40-120 spots on the unit chart rectangle, each sighting given a
random visible-extent crop, a smooth monotone `s`-warp, dropout, clutter,
Gaussian jitter and a random flip. The only mode with exact ground truth, and
the one the tests assert on. Source: `results/constellation/toy_summary.json`,
20 identities, 20% clutter, seed 0.

| dropout | jitter | Rank-1 | Rank-5 | mean rank | AUROC | d | same mean | diff mean | jitter / median NN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 0.5% | 1.00 | 1.00 | 1.00 | 1.0000 | 11.38 | 0.825 | 0.220 | 0.10 |
| 0% | 1.0% | 1.00 | 1.00 | 1.00 | 1.0000 | 10.67 | 0.784 | 0.222 | 0.21 |
| 0% | 2.0% | 1.00 | 1.00 | 1.00 | 1.0000 | 6.51 | 0.565 | 0.223 | 0.41 |
| 0% | 4.0% | 0.45 | 0.75 | 3.55 | 0.8346 | 1.83 | 0.328 | 0.228 | 0.80 |
| 20% | 0.5% | 1.00 | 1.00 | 1.00 | 1.0000 | 6.91 | 0.682 | 0.261 | 0.09 |
| 20% | 1.0% | 1.00 | 1.00 | 1.00 | 1.0000 | 6.19 | 0.643 | 0.259 | 0.18 |
| 20% | 2.0% | 0.95 | 1.00 | 1.05 | 0.9936 | 3.77 | 0.502 | 0.258 | 0.35 |
| 20% | 4.0% | 0.20 | 0.50 | 5.40 | 0.7176 | 0.99 | 0.322 | 0.261 | 0.70 |
| 40% | 0.5% | 0.85 | 0.95 | 1.35 | 0.9882 | 4.26 | 0.567 | 0.305 | 0.07 |
| 40% | 1.0% | 0.85 | 0.95 | 1.35 | 0.9849 | 3.79 | 0.559 | 0.306 | 0.15 |
| 40% | 2.0% | 0.65 | 0.95 | 2.10 | 0.9221 | 2.69 | 0.494 | 0.305 | 0.29 |
| 40% | 4.0% | 0.10 | 0.50 | 6.80 | 0.6872 | 0.59 | 0.338 | 0.301 | 0.60 |

20 identities, 20% clutter, seed 0, 114 s.

Jitter is in chart units; the ratio to the median nearest-neighbour spacing is
what actually matters, and the wall is at roughly 0.6 of it.

### The evaluation protocol

Two statistics per arm, and they answer different questions.

*Pairwise.* Every pair of images is scored. Positives are same-individual pairs
on the **same flank**; negatives are all different-individual pairs. Cohen's d
and AUROC are over those two populations. Opposite-flank pairs of one animal
carry a positive label but share no pattern, so they are pulled out into their
own population rather than counted as positives the matcher failed to find.

*Rank (`_rank_eval`).* Leave-one-out closed-set identification. Each image in
turn is the query and the gallery is every other image **of the same flank from
a different encounter**; the rank is that of the best correct entry, with every
wrong entry tied with it counted ahead of it. A query whose gallery holds no
same-individual entry is unscorable and is reported, not scored. The
same-encounter exclusion is what stops a near-duplicate from being retrieved
instead of a recapture — which is exactly the leakage that makes the real
catalogue unusable.

Both are reported at three spot-confidence floors, applied *before*
rectification (`--conf-min`), because the frame is fitted to the spots that
survive the filter.

### The three arms on real and synthetic detections

| arm | images | ids | same-side pos | diff pairs | opp-side pairs | same mean | diff mean | d | AUROC | Rank-1 | Rank-5 | queries | gallery | spots/img | frames |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| real, conf>=0.25 | 60 | 43 | 21 | 1748 | 1 | 0.343 | 0.256 | 0.84 | 0.720 | - | - | 0 | - | 128.8 | {"chart": 13, "pca": 47} |
| synthetic detector, conf>=0.25 | 152 | 40 | 126 | 11259 | 91 | 0.264 | 0.208 | 1.16 | 0.670 | 0.279 | 0.404 | 136 | 78 | 126.9 | {"chart": 29, "pca": 123} |
| synthetic detector, conf>=0.40 | 152 | 40 | 126 | 11259 | 91 | 0.300 | 0.262 | 0.54 | 0.624 | 0.110 | 0.272 | 136 | 78 | 66.4 | {"chart": 31, "pca": 121} |
| synthetic detector, conf>=0.50 | 152 | 40 | 126 | 11259 | 91 | 0.324 | 0.317 | 0.07 | 0.512 | 0.037 | 0.184 | 136 | 78 | 37.3 | {"chart": 32, "pca": 120} |
| synthetic GT centres (ceiling) | 160 | 40 | 141 | 12480 | 99 | 0.250 | 0.146 | 3.04 | 0.830 | 0.571 | 0.707 | 147 | 85 | 245.4 | {"chart": 89, "pca": 71} |

Sources: `results/constellation/{real,synth_calib,synth_calib_c40,synth_calib_c50,synth_calib_gt}_summary.json`.
`d` and `AUROC` are over same-side same-individual pairs against all
different-individual pairs. Rank-1/Rank-5 are the leave-one-out same-side
protocol described above; `queries` is how many of the images had a retrievable
same-side entry and `gallery` the median gallery size. The real arm has **no
scorable query at all** — every positive it has is a same-encounter
near-duplicate, and `side` is null on 58 of its 60 images, so a same-side gallery
cannot even be formed. Its `d` and `AUROC` are a near-duplicate measurement and
its rank columns are empty by construction, not by failure. Its "same-side
positives" are strictly flank-unrecorded positives (see the limitations).

Rank detail (same protocol, same summaries):

| arm | mean rank | median rank | MRR | Rank-10 | Rank-1 optimistic | queries tied at top | unscorable |
|---|---:|---:|---:|---:|---:|---:|---:|
| real, conf>=0.25 | - | - | - | - | - | - | 60 |
| synthetic detector, conf>=0.25 | 19.4 | 10 | 0.352 | 0.500 | 0.287 | 1 | 16 |
| synthetic detector, conf>=0.40 | 22.1 | 16 | 0.208 | 0.382 | 0.118 | 1 | 16 |
| synthetic detector, conf>=0.50 | 31.1 | 28 | 0.124 | 0.265 | 0.051 | 2 | 16 |
| synthetic GT centres (ceiling) | 9.7 | 1 | 0.641 | 0.803 | 0.578 | 1 | 13 |

The pessimistic and optimistic Rank-1 differ by at most 0.008 anywhere, so tie
handling is not doing the work.

Split by rectification frame (`by_frame` in each summary):

| arm | same-frame n+/n- | same-frame d | same-frame AUROC | cross-frame n+/n- | cross-frame d | cross-frame AUROC |
|---|---:|---:|---:|---:|---:|---:|
| real, conf>=0.25 | 17/1141 | 0.96 | 0.751 | 4/607 | 0.05 | 0.576 |
| synthetic detector, conf>=0.25 | 86/7759 | 1.26 | 0.662 | 40/3500 | 0.94 | 0.687 |
| synthetic detector, conf>=0.40 | 83/7581 | 0.47 | 0.592 | 43/3678 | 0.67 | 0.684 |
| synthetic detector, conf>=0.50 | 82/7493 | 0.09 | 0.518 | 44/3766 | 0.05 | 0.501 |
| synthetic GT centres (ceiling) | 69/6280 | 3.84 | 0.870 | 72/6200 | 2.24 | 0.807 |

The opposite-flank pairs, kept out of the positives and reported on their own:

| arm | L-vs-R pairs | mean score | AUROC vs negatives |
|---|---:|---:|---:|
| real, conf>=0.25 | 1 | 0.389 | 0.896 |
| synthetic detector, conf>=0.25 | 91 | 0.203 | 0.473 |
| synthetic detector, conf>=0.40 | 91 | 0.246 | 0.429 |
| synthetic detector, conf>=0.50 | 91 | 0.289 | 0.440 |
| synthetic GT centres (ceiling) | 99 | 0.141 | 0.435 |

91 opposite-flank pairs at AUROC 0.473 against the negatives is the control this
prototype needs and the real corpus cannot supply: pairs that carry a positive
label and share no pattern score at chance, so the positive signal elsewhere is
not an artefact of the labelling. The real corpus's single such pair sits at
0.896, which is one draw and means nothing on its own.

**What the three thresholds say.** More spots is unambiguously better: the
detector's own 0.25 floor gives 127 spots per image and AUROC 0.670 / Rank-1
0.279; raising the floor to 0.40 halves the spots and drops it to 0.624 / 0.110;
0.50 leaves 37 spots and the matcher is at chance (0.512 / 0.037). The
low-confidence detections are individually unreliable — 437 of the 19,292 at the
0.25 floor are false — but the constellation needs the density more than it
needs the precision. **Do not raise the ingest's spot threshold.**

**What the ceiling says.** Replacing the detector with the renderer's exact spot
centres, and changing nothing else, takes Rank-1 from 0.279 to 0.571 and AUROC
from 0.670 to 0.830. So roughly half the identification loss is detection loss —
the 53% of visible spots the detector misses at its own floor — and the rest is
rectification and matching loss, which the ceiling still carries. The ceiling
also uses the chart frame far more often (89 of 160 frames, against 29 of 152 on
the detector output), because its body polygon is the silhouette of the clean
visible-skin mask rather than a YOLO wedge.

Two caveats on that comparison. The ceiling keeps all 160 frames where the
detector arm keeps 152 — the eight frames lost to the OSEA body floor are
present in the ceiling, so it scores 141 positives against 126 — and it changes
the rectification input as well as the spot list, so "detection loss" here means
everything downstream of the render, not the spot head alone.

### What each arm does and does not measure

**The real arm measures near-duplicate robustness, not re-identification.**
Every one of its same-individual pairs is a same-encounter pair, so
`cross_encounter_same_side_positives` is 0; and because `side` is recorded on
only 2 of the 1091 catalog rows, the same-side rank protocol can form no gallery
and reports **zero scorable queries**. The one cross-encounter positive in `catalog.db` is
`AOTB_A014`, photographed left flank in 2019 and right flank in 2020 — an L-vs-R
pair, which shares no spot pattern at all and belongs in the chance-floor
population. `_pair_eval` reports it as `opposite_side_pairs` and keeps it out of
the positives:

| field | value |
|---|---|
| individual | `AOTB_A014` |
| images | 20 (left flank, 2019-10-24) and 160 (right flank, 2020-08-11) |
| elapsed | 292 days |
| sides | L vs R — no shared spot pattern |
| frames | pca / pca |
| score | 0.389 (7 inliers), against a different-individual mean of 0.256 and p99 0.531 |
| percentile of the negative distribution | 89.0 |

Source: `results/constellation/real_summary.json`, `opposite_side_pairs`. A score
at the 89th percentile of the chance distribution is what an L-vs-R pair should
look like: high enough to notice, well inside the negative population, and not
evidence of anything. It is reported precisely so that it does not get read as a
re-identification.

**The synthetic-by-detector arm is the one that measures re-identification.**
Its positives are cross-encounter, same-flank, and separated by real drift
intervals. It is the number to quote for "does the spot constellation carry
identity through the OSEA ingest", subject to the domain gap the KS table above
quantifies.

**The GT-centre ceiling** replaces the detector with the renderer's exact
projected spot centres (`--synth-from-gt`), keeping everything else identical.
The gap between it and the by-detector arm is detection loss; whatever the
ceiling itself falls short of 1.0 is rectification and matching loss.

### Drift: score against recapture interval

Same-side positives bucketed by the interval between the two sightings, with
each bucket's AUROC against the whole negative population.
Source: the `drift` block of each summary.

| arm | 0-6 months n / mean / AUROC | 6-12 months n / mean / AUROC | 1-2 years n / mean / AUROC | 2+ years n / mean / AUROC |
|---|---|---|---|---|
| real, conf>=0.25 | 21 / 0.343 / 0.720 | 0 / - / - | 0 / - / - | 0 / - / - |
| synthetic detector, conf>=0.25 | 85 / 0.270 / 0.675 | 19 / 0.247 / 0.625 | 20 / 0.247 / 0.674 | 2 / 0.305 / 0.815 |
| synthetic detector, conf>=0.40 | 85 / 0.292 / 0.601 | 19 / 0.318 / 0.642 | 20 / 0.313 / 0.676 | 2 / 0.353 / 0.898 |
| synthetic detector, conf>=0.50 | 85 / 0.328 / 0.520 | 19 / 0.310 / 0.489 | 20 / 0.322 / 0.504 | 2 / 0.300 / 0.456 |
| synthetic GT centres (ceiling) | 95 / 0.259 / 0.839 | 22 / 0.231 / 0.822 | 22 / 0.235 / 0.788 | 2 / 0.230 / 0.967 |

Recapture intervals run from 1 to 787 days, median 97.5
(`drift.elapsed_days`). Within that range there is **no measurable decay**: at
the 0.25 floor the AUROC is 0.675 / 0.625 / 0.674 / 0.815 across the four
buckets, and on the GT ceiling 0.839 / 0.822 / 0.788 / 0.967. The 2+ year bucket
holds two pairs, so its numbers are noise. The honest reading is that prototype
05's drift over three years is small next to the loss the detector and the
rectification already impose — not that the pattern is provably stable, which
this corpus cannot show.

The real arm's drift row is degenerate for the reason above: all 21 of its
positives are same-encounter, so every interval is 0 days.

## Known limitations

**The chart frame fails on real OSEA masks.** `extract_spotset` tries prototype
02's medial-axis chart first and falls back to a PCA frame when the chart is
unusable. Source: `results/constellation/real_frames_summary.json` and
`synth_calib_frames_summary.json`, written by `eval_constellation.py --frames`.

| corpus | rectified | chart frame | PCA fallback | fallback rate | max abs r, median / max |
|---|---:|---:|---:|---:|---:|
| real photographs | 1030 | 135 | 895 | 87% | 0.932 / 1.661 |
| calibrated corpus | 152 | 29 | 123 | 81% | 0.958 / 0.993 |

| chart rejection reason | real | calibrated corpus |
|---|---:|---:|
| `centerline_too_short` | 219 | 16 |
| `centerline_warning` | 864 | 119 |
| `spot_far_outside_body` | 684 | 83 |
| `spots_outside_body` | 519 | 72 |


The two domains agree, which is itself a sim-to-real result: the calibrated
renders are cropped into the same fat-wedge silhouette the real photographs are,
so the synthetic arm exercises the same rectification the real one will. The
rejection reasons are the same too — `centerline_warning` (prototype 02 itself
flagging the mask as non-tubular) on 864 of 1030 real masks and 119 of 152
synthetic ones. A mask can trip several reasons at once, so the columns do not
sum to the fallback count.

The OSEA body masks are fat wedges of a head and forebody rather than tubes, so
prototype 02's medial-weighted longest path curls inside the widest region
instead of spanning the body. Both frames now normalise `r` by a *local* half
width, so they are comparable — but the PCA fallback is a straight axis through
a curved animal, and cross-frame pairs are measurably weaker than same-frame
ones on both arms (see the matcher tables). This is the largest single lever
left on the real arm.

**Side is unrecorded on almost every real photograph.** `catalog.db` has a
`side` column, but it is filled on **2 of 1091** rows — the `AOTB_A014` pair, and
nothing else (counted over `results/real/detections_slim.jsonl`). That has two
consequences the real arm's numbers depend on. Its 21 "same-side" positives are
really *flank-unrecorded* positives: any of them could be an L-vs-R pair that
nothing in the pipeline can identify as one. And the same-side rank protocol
needs a known flank on both images, so with `side` null it finds no gallery at
all — the real arm's zero scorable queries are caused by the missing labels as
well as by the same-encounter leakage.

Nor can the side be inferred: there is no head detector in the shipped weights,
and `extract_centerline`'s widest-end-first rule can pick the wrong end of a
head-and-forebody crop. The matcher compensates by trying all four
`s -> 1-s` x `r -> -r` flips of the gallery set and keeping the best, which costs
4x and inflates the chance floor.

**Spread pectorals.** The rig cannot fold the fins; `real_body.pose` bends the
body but does not drive the fin joints. Above ~35 degrees of elevation the
splayed pectorals turn the silhouette into a fat cross and `body_conf` falls, so
the camera elevation is capped at 32 degrees. The cost is `area_norm`, the one
geometry feature the search could not close.

**Hand occluders are ellipsoids.** They occlude and cast shadow like hands; at
a 2.5x zoom they are brown eggs. The real backgrounds are hands, water surface,
tub rims, gunwales and people abutting the animal; the synthetic background is a
flat pastel wash with a soft gradient and per-pixel noise.

**The v1 spot detector is low precision on real frames.** It runs at a 0.25
confidence floor and 46% of its real detections sit below 0.40
(`results/real/summary.json`, `pooled_histograms.spot_conf`); the overlays in
`results/real/detections_contact.png` show boxes on plain skin. Nothing in this
prototype measures real spot precision, because there is no per-spot ground
truth on a real photograph — see the caveat under the detector table. Every
matcher number is reported at three confidence floors for that reason.

**`spot_count` is censored at 300.** `ultralytics` `max_det` defaults to 300 and
the OSEA pipeline keeps it, so 24 of the 1091 real images are truncated
(`results/real/summary.json`, `counts.spots_truncated_image_ids`). They are
flagged, not corrected: raising the cap would break fidelity with what the
tagger actually stores. `n_spots` and `density` on those 24 are lower bounds.

**Self-intersecting body contours.** 117 of the 1030 real contours self-
intersect, so the shoelace area is meaningless on them; `features()` uses a
raster moment frame instead and flags the contour
(`results/real/summary.json`, `counts.degenerate_contour_image_ids`).

**One animal, one skin, one sensor.** Every synthetic frame is the same mesh,
the same skin chart and the same 2016x1512 output, where the real corpus is 1030
photographs of different animals, cameras, distances and water. That is why the
remaining calibration gap is almost all per-image *spread* rather than a wrong
median (see the objective's two halves against their sampling floors).

**What still looks wrong at 2.5x.** From
`results/synth_calib/zoomed_contact.png`, which puts two synthetic flank crops
above two real ones at matched scale, in the order that should matter to a
detector:

1. *Skin grain.* The real skin carries a fine bright denticle speckle edge to
   edge, visible inside the spots as well as on plain skin, plus a strong wet
   specular sheen that goes pink and purple over the head. The synthetic skin
   has no grain at that scale at all — only a broad, soft mottle and a smooth
   sheen. The synthetic *background* is grainier than the synthetic skin, which
   is the wrong way round. This is a resolution limit, not a parameter:
   `skin.mottle_px` is measured in chart pixels on a 512x2048 chart, about 1.8
   render pixels, so denticle-scale speckle cannot be represented.
2. *Spot shape.* Synthetic spots are clean ellipses with soft edges; real spots
   are irregular blotches with ragged outlines that fuse into lobed clusters of
   two to four. `pattern._stamp_ellipse` draws an ellipse with a linear coverage
   ramp, so no config value makes one ragged.
3. *Pectoral banding.* The synthetic blade shows straight longitudinal stripes —
   the 1.5 mm cluster grid across a blade thinner than a cell. A real pectoral
   base is smooth, and much less spotted than the flank, which the chart does
   not encode (`pattern` has head/flank/tail region signals in `s`, but no
   fin-versus-body signal).
4. *Colour and gills.* The synthetic skin is a fairly neutral mauve-grey where
   the real is brown-purple over olive; the synthetic gill slits are faint
   creases where the real ones are pale flaps with hard edges.

## Next steps

1. **A head/eye anchor for `s`.** Every frame problem above — the chart
   fallback, the unknown side, the four flips — is one missing anchor. The eye
   is already located on this mesh (prototype 04's
   `results/real_v11/eye/eye_patch.json`; `synth_render.eye_chart_mask` draws
   it), and the synthetic corpus carries its chart position for free. A small
   eye/snout detector trained on the calibrated corpus would pin `s = 0` and the
   direction of travel, which removes the flip search, removes the chance-floor
   inflation it causes, and makes the PCA fallback orientable.
2. **Side inference — and, more cheaply, side *recording*.** `catalog.db`'s
   `side` column is filled on 2 of 1091 rows, so most of what the real arm
   cannot do is a data-entry gap, not a modelling one; the OSEA sheet could
   carry the flank at no field cost. Beyond that: with `s` anchored, the sign of
   `r` is a two-class problem and the synthetic corpus has an exact label on
   every frame, so an inferred flank would turn the L-vs-R pairs from unusable
   labels into a second gallery.
3. **Ingest the 69 OSEA sevengills.** The OSEA sheet holds 69 sevengill
   sightings from Dec 2025 to Aug 2026 that are not in `catalog.db`, alongside
   the three known recaptures (A011, A012, A014). Ingesting them is the only
   route to a real cross-encounter same-side pair, which is the one number this
   prototype cannot currently produce. Nothing else on this list changes that.
4. **A learned proxy-tensor embedding.** The constellation matcher is
   hand-built: shape contexts, Hungarian, RANSAC. The calibrated corpus is a
   labelled training set of the exact representation the ingest stores — spot
   constellations in a body-normalised frame, with identity, drift interval and
   flank known. Training an embedding on it (and evaluating with the same
   `eval_constellation` protocol) is the natural next model, and the corpus is
   the reason it is possible without a labelled real catalogue.

## Appendix — the render

### The chart, and why the skin lines up for free

Prototype 04's `results/real_v11` GLB has the de-bent scan as its bind pose:
1,013,814 vertices, 1,961,876 faces, extents 0.682 x 0.230 x 0.108 m.
`report/centerline.json` holds the straight centerline (64 stations, chord
0.4803 m, snout at `+X`) and `mesh3d.canonical_frames` are its frames
(T = -X, N = +Z dorsal, B = +Y = the animal's left), which is what
`mesh3d.tube_frames(straight_centerline, up=(0,0,1))` returns. So

```python
coords = mesh3d.tube_coords(mesh, centerline, mesh3d.canonical_frames(64))
vertex_s, vertex_phi = texture_identity.chart_coords(coords, normalize="extent")
```

reproduces the chart `texture_identity.straighten` builds. Measured on this
mesh, `s` runs from -0.0267 m to +0.6553 m and `normalize="extent"` maps that
onto `[0, 1]`. `assets/chart_skin_x4.png` was made from the same normalisation
(it box-downsamples 4x to 04's `chart_skin.png` with max abs difference 0.0), so
skin chart and vertex chart agree cell for cell with no fitting step. Sanity
check against 05's anatomy schema: `detect_fins` puts the right pectoral at
`s` 0.208-0.334 and the schema's station table puts `pectoral_origin` at 0.245
and `pectoral_insertion` at 0.300.

### Decimation (`results/decimation.json`)

Vertex clustering on a regular grid of `cell`: every occupied cell becomes one
vertex carrying the cell's mean `s` and `r`, its *circular* mean `phi`, and
`is_fin = any`. Rest positions are reconstructed from the cluster's
`(s, r, phi)` through `mesh3d.tube_to_points`, so `pose(amp=0)` is an exact
identity.

| cell | vertices | faces | vs source | degenerate faces dropped | duplicate | cluster drift mean / max |
|---:|---:|---:|---:|---:|---:|---|
| 1.5 mm | 75,731 | 152,467 | 7.47% / 7.77% | 1,804,176 | 5,233 | 0.002 / 0.122 mm |
| 2.0 mm | 43,304 | 87,366 | 4.27% / 4.45% | 1,870,916 | 3,594 | 0.004 / 0.167 mm |
| 2.5 mm | 28,007 | 56,551 | 2.76% / 2.88% | 1,902,688 | 2,637 | 0.007 / 0.197 mm |
| 4.0 mm | 10,933 | 22,212 | 1.08% / 1.13% | 1,938,368 | 1,296 | 0.017 / 0.329 mm |
| 6.0 mm | 4,875 | 9,966 | 0.48% / 0.51% | 1,951,242 | 668 | 0.037 / 0.637 mm |

### Rasterisation cost (`results/bench.json`)

Median of 3 frames at 2016x1512, real `draw_scene` draws, subject plus 0-2
occluders, single-threaded numpy rasteriser.

| cell | raster + shadow map | of which shadow map | whole frame |
|---:|---:|---:|---:|
| 1.5 mm | **5.10 s** | 2.37 s | 6.18 s |
| 2.0 mm | 3.43 s | 1.51 s | 4.51 s |
| 2.5 mm | 2.67 s | 1.11 s | 3.62 s |
| 4.0 mm | 1.62 s | 0.45 s | 2.73 s |
| 6.0 mm | 1.44 s | 0.44 s | 2.44 s |

The default is 1.5 mm. What bites is not time but the pectoral rim: the blade is
thinner than a cell over its last few millimetres, so clustering quantises the
rim to the grid and renders a sawtooth — ~21 px at 2.5 mm on a 2016-wide frame,
~13 px at 1.5 mm, where it reads as a soft edge.

### Pose

`real_body.pose(body, amp, wave, phase, yaw_deg)` uses prototype 05's planar
bend, `kappa(u) = amp*cos(2*pi*wave*u + phase)`, swept with
`mesh3d.tube_to_points` on a 512-station centreline built from the midpoint
heading, so arc length is preserved and `(s, phi)` — and every chart ground-truth
map — is unchanged by the pose. `pose(amp=0)` reproduces `body.vertices` to
2.8e-16 m. Fins ride the chart rather than being carried rigidly:
`real_body.fin_stretch` reports 1.095 at `amp = 0.35` and `r_max = 0.130 m`,
i.e. 9.5% on a pectoral tip that is out of frame or edge-on in almost every
frame.

### Ground truth per frame

* `body/<image_id>.jpg` — 2016x1512, JPEG quality U(75, 95), no chroma
  subsampling. A real frame is 4032x3024, but the OSEA spot model resizes the
  body crop to 1280, so 2016 keeps the effective spot resolution.
* `gt/<image_id>.npz` — `chart_s`, `chart_phi` (float32, NaN off-body),
  `visible_skin`, `shadow`, `occlusion` (bool), pixel-aligned with the JPEG
  before the blur.
* `gt/<image_id>_spots.json` — one row per rendered spot: `id, s, phi, radius,
  rendered_darkness, visible, cx, cy, radius_px, n_pixels`. `radius_px` is the
  area-equivalent radius `sqrt(n/pi)` of the visible pixels, which is honest
  under foreshortening. It is deliberately larger than what a blob detector
  returns on the same spot, because `pattern._stamp_ellipse` only reaches full
  amplitude inside 75% of the radius.
* `truth.jsonl` — `image_id, identity, sighting, date, side, length_cm, pose,
  camera, light, specular, background, occluders, degrade, n_spots,
  n_visible_spots, geometry, timings, paths`.

### Four render defects the bridge run exposed, all fixed before the search

1. **The camera was below the animal.** `frame_camera` negated the Rodrigues
   angle, so a positive `elevation_deg` put the eye under the shark and framed
   the countershaded, near-bare ventrum. That one sign produced the bridge run's
   "spots only in a dorsal band", its 14.9% spot recall and much of its low
   `body_conf`. `calibrate.BEFORE_OVERRIDE` replays the old behaviour frame for
   frame with a reversed elevation range, which is what makes the before/after
   table below a controlled comparison.
2. **Spots stopped at the flank.** `cs_phi_onset` 1.30 with `dorsal_exponent`
   0.80 left the placement rate at `|phi| = 2.3` at 24% of the dorsal rate,
   where a real sevengill is spotted to the ventral transition. Now 2.30 / 0.30,
   which puts it at 64%.
3. **The snout was clipped on every frame.** The framed set is `s <=
   s_frame_max` spanning `width_frac` of the image and centred on `s_target`, so
   the snout is inside only when `s_target < 0.5 * s_frame_max / width_frac`.
   `s_target = 0.25` against `s_frame_max` 0.26-0.38 failed that for every draw.
4. **A white band along the pectoral.** `tone_ventral = 1.18` is a multiplier
   above 1, and on the blade's rounded leading edge it clipped to white.

Two more, found by looking at `zoomed_contact.png` rather than by a test: a bare
band across the gill region (05's `gill_slits` exclusion is a *scoring* mask, not
a claim that the animal has no spots there — dropped for rendering only), and a
rectangular smear for an eye (the schema's generic eye rectangles are the wrong
instrument for "where is the eye on *this* animal").

### Environment notes

* `render` is a pure-numpy rasteriser with a per-face Python loop; it is the
  whole cost of a frame.
* numpy 2.0.2 on Accelerate raises spurious `divide by zero` / `overflow` /
  `invalid value` RuntimeWarnings from `matmul` on finite input. This module
  wraps its own calls in `np.errstate`; the ones raised inside
  `05-synthetic-identities/render.py` cannot be suppressed without editing that
  file, so run everything with `-W ignore`.
* `calibrate.run_many` uses the *spawn* start method. A caller driving it from a
  script file without an `if __name__ == "__main__":` guard fork-bombs.
