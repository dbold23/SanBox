# Prototype 05 — Synthetic Identity Engine

A generator for synthetic sevengill (*Notorynchus cepedianus*) re-identification
corpora. It does the four things the owner asked the dot generator to do:

1. **RANDOMIZE** an individual's speckle pattern,
2. **COPY** a real individual's actual pattern onto the 3D model,
3. **SLIGHTLY CHANGE** it to simulate a **resighting** months or years later,
4. render it with **occlusions, shadows** and turbid water, **excluding the eye
   and the mouth** from anything that counts as identity.

The output is readable **unchanged** by `prototypes/01-melops-ablation`, so
`run_ablation.py`, `diagnose.py` and `readout_length_controlled.py` run on
synthetic sevengills with zero edits.

```
pattern.py    RANDOMIZE   a speckle field in CHART SPACE
unbake.py     COPY        a real pattern out of a photograph into that chart
drift.py      RESIGHT     the same animal later (growth + drift + scars)
bake.py       BAKE        the chart onto the model's UV texture
render.py     RENDER      a numpy z-buffer frame + pixel-aligned chart GT
nuisance.py   DEGRADE     kelp, occluders, turbidity, caustics, blur, jitter
exclusions.py             the anatomical exclusion mask (eye, mouth, nares, gills)
make_dataset.py           wires them into one corpus (randomize path)
chart_readout.py          READ BACK a corpus in chart space and score it
```

Every measured number below is printed by a script in this directory, and the
command that prints it is quoted next to the number.

---

## The design decision: patterns live in chart space, not the UV atlas

Every pattern is generated in **canonical chart coordinates `(s, phi)`**:

- `s ∈ [0,1]` — arc-length fraction along the body centreline, `0` = snout tip,
  `1` = caudal terminus.
- `phi ∈ [-π, π)` — circumferential angle, `0` = dorsal midline, `+π/2` = the
  animal's **left** flank, `-π/2` = right, `±π` = ventral seam.

Chart arrays are `(H_phi, W_s)` in `pattern.py` / `drift.py` / `exclusions.py`
and `(n_s, n_phi)` in `bake.py` / `unbake.py`. `bake.from_pattern_chart` /
`to_pattern_chart` bridge the two, and a test asserts `bake.chart_axes` agrees
cell-for-cell with `exclusions.chart_axes` so the two layouts cannot drift apart
in meaning. Values are *darkness* on one side and *albedo multiplier* on the
other; `bake` warns when a chart's mean says it was handed the other convention,
which is the inversion that would otherwise reach the dataset in silence.

**Why not the mesh's UV atlas.** A UV atlas is a property of one particular
mesh's unwrapping — its seams, its island layout, its texel density. A pattern
authored in UV space is welded to that mesh and means nothing on another. A
pattern authored in `(s, phi)`:

- is **mesh-agnostic** — the same identity transfers to any tube-like body with
  a centreline;
- **maps to any pose through the rig**, because `s` and `phi` are body-intrinsic
  and a bend does not change them (`pose_vertices` re-sweeps the same `(s, phi)`
  along a bent centreline, so it is arc-length preserving by construction and
  the chart GT is *identical* across poses — that invariance is exactly what
  lets one identity be measured across poses);
- gives **every rendered pixel exact ground-truth chart coordinates**
  (`out["chart_s"]`, `out["chart_phi"]`), which is what makes the corpus a test
  instrument rather than just pretty pictures.

Baking to a UV texture is a separate, later step that needs per-vertex
`(s, phi)`. Here that comes from `fixtures.make_uv_tube`; for a real mesh it
comes from prototype 04 (see *Swap-in points*).

**One subtlety that matters.** `s` and `phi` are *normalised*, so purely
isotropic growth moves nothing in the chart — it only multiplies spot spacing in
centimetres. Chart motion comes only from **allometry**. Spot *count* is
invariant by construction; `Individual.spot_spacing('cm')` scales with the
length ratio, `spot_spacing('chart')` does not. This encodes the derived fact
that spots are a fixed cell population that spreads as the animal grows — growth
scales spot **spacing**, not spot **count** *(derived, see `drift.py`)*.

**The schema is read, not assumed.** All landmark stations come from
`phase1b/p0-sevengill-schema/keypoints_sevengill_v1.yaml`. That yaml carries the
seven midline axis fractions, the ordered A–P sequence and its explicit refusal
to order the pelvic/dorsal/cloaca trio — but it carries **no arc-length fraction
for eye, naris, rictus, gills or fins**, and its own `open_questions` says fin
stations are provisional and were never retrieved. So `exclusions.py` reads
everything the yaml *does* contain, keeps the missing numbers in
`DEFAULT_STATIONS` tagged `[UNVERIFIED]` by `station_grades()`, and
`validate_stations()` asserts any table against exactly what the yaml asserts
and nothing more. **Pass a measured `stations` dict and every region moves with
it** — nothing downstream is hard-coded.

---

## The three modes

### 1. RANDOMIZE — invent an individual

```python
import pattern, exclusions
schema   = exclusions.load_schema(pattern.DEFAULT_SCHEMA_PATH)
stations = exclusions.default_stations(schema)
regions  = exclusions.exclusion_regions(schema, stations=stations)

ind = pattern.randomize(seed=3, identity="demo", date="2020-03-01",
                        length_cm=230.0, regions=regions)
# -> 240 spots, spacing 0.0332 chart units / 7.64 cm
```

Spots are rejection-sampled with a minimum separation, radii log-normal,
eccentricity and orientation drawn per spot, darkness attenuated ventrally by
the countershading prior. `PatternParams.head_signal` / `flank_signal` /
`tail_signal` set the identity amplitude **per region** — amplitude `0` strips
identity from a region while `n_common` keeps a shared confounder layer there,
so the region stays *textured but uninformative*. That is the direct
generalisation of `melops_data.make_synthetic`'s `head_signal` / `body_signal`,
and it is what makes a synthetic head-vs-flank ablation possible.

### 2. COPY — lift a real individual's pattern off a photograph

```python
import unbake
individual, result = unbake.copy_from_photo(photo_rgb, body_mask, side="L",
                                            identity="LJ-014", date="2019-08-02")
```

`photo_to_chart` rectifies the silhouette against its medial axis, converts
across-body position to `phi = ±arccos(r/R_local)`, and returns a chart plus a
per-cell **confidence**; `pattern.copy_from_chart` then segments marks out of it
and returns a real `Individual` that the rest of the engine treats exactly like
a randomized one. Measured on the closed synthetic loop (`randomize` →
`render_lateral_tube` → `copy_from_photo`): **48 spots recovered from a source
of 240**, mean confidence 0.269.

**What the confidence does and does not cover.** It is the foreshortening
factor `c = sin|phi| = sqrt(1 − (r/R_local)²)` — the cosine between the surface
normal and the view direction — so it correctly collapses to 0 at the
silhouette, where one pixel covers an unbounded sweep of girth. It says nothing
about **obliquity**: a bent animal is handled (the centerline follows the bend)
but an animal angled *toward* the camera has a shortened silhouette, so every
`s` is wrong and the confidence map still reads high. `unbake` cannot detect
that (`photo_to_chart`'s own LIMITS list says so); only the 3D fit path can.
Treat `confidence` as "was this cell resolved across the girth", never as "is
this cell's `s` correct".

That ~20 % is not a bug and it is the honest headline for this mode: **one
photograph sees one half of one girth.** Half the spots are on the far side and
are never observed (they stay `NaN` and must *not* be filled by mirroring —
Schema S1 measured cross-flank Rank-1 at 0.70 % zero-shot). Of the near half,
the grazing bands near `phi = ±π/2` are unresolvable and the exclusion mask
removes the rest. Copy mode needs multiple views, or prototype 04's
analysis-by-synthesis, to reconstruct a whole animal.

### 3. RESIGHT — the same animal, later

```python
import drift
later = drift.resight(ind, "2020-03-01", "2022-03-01",
                      growth_model=drift.VonBertalanffyGrowth())
```

Measured chart-NCC decay for one individual, verbatim from shipped code:

```console
$ python -c "import pattern, drift; print(drift.similarity_curve(pattern.randomize(3), seed=3))"
[(0, 1.0), (30, 0.9843570459017099), (180, 0.9274954521337114),
 (365, 0.8670025910346812), (730, 0.7889129295560317)]
```

| elapsed | 0 d | 30 d | 180 d | 365 d | 730 d |
|---|---|---|---|---|---|
| chart NCC | 1.000 | 0.984 | 0.927 | 0.867 | 0.789 |

`DEFAULT_JITTER_RATE = 0.0012828125` s-units/√year (0.32 cm/√yr at 250 cm TL) was
**calibrated, not fitted**: `drift.calibrate_jitter_rate()` bisects until
NCC(730 d)/NCC(30 d) lands within `tol = 0.004` of the Melops ratio
`0.474 / 0.605 = 0.7835` (`prototypes/01-melops-ablation/results/CAMPAIGN.md`).
Every rng in that bisection is seeded, so the constant is exactly what a re-run
returns — `tests/test_drift.py` asserts that equality, and the full bisection log
is quoted above the constant in `drift.py`:

```console
$ python -c "import drift; print(drift.calibrate_jitter_rate())"
(0.0012828125, 0.7796889708078755)
```

So the calibration objective — the ratio averaged over the 6 calibration animals
— is **0.780**; the single animal in the table above realises 0.789/0.984 =
**0.801**, which is the spread between individuals, not a second measurement.
**Only the ratio is claimed** — the Melops number is an ArcFace cosine
between two photographs whose absolute level is set by photographic nuisance
this module does not model, whereas a chart NCC at zero elapsed time is 1.0 by
construction. Re-run `calibrate_jitter_rate()` after any change to default spot
size, density or rendering.

Also modelled, each constant graded in-file: scar accumulation and healing
(τ = 98.5 d, bracket [60, 130], solved from a reef-manta wound-length curve and
sanity-checked against a blacktip "undetectable within 179 days" report), a
persistent-residue fraction (0.25 persist, 0.15 residue, from manta scars stable
beyond 3 y), and optional melanism patches (area rate 0.535 /yr, derived from
the white-shark −33 %-in-9-months islet).

---

## Generating a corpus

```bash
python make_dataset.py --out DIR \
    --n-individuals 40 --sightings-per-individual 6 --years 4 \
    [--head-signal 1.0 --flank-signal 1.0] \
    [--occlusion 0.3 --shadow 0.5 --turbidity 0.4] [--seed 0] \
    [--length-noise 0.07]
```

Then, with **no changes to prototype 01**:

```bash
cd ../01-melops-ablation
python run_ablation.py --data melops --root DIR --backbone hist --out results
python diagnose.py     --data melops --root DIR --backbone hist --arm body --out diag
```

And to read the corpus back in **chart space** — the ground-truth readout every
chart-space number below is quoted from — write `DIR/readout.json` with:

```bash
python chart_readout.py --data DIR [--sensitivity]
```

Outputs in `DIR`:

| file | what |
|---|---|
| `metadata.csv` | the `melops_data` contract: `image_id, identity, path, date, side, bbox_body, bbox_head, bbox_headless` |
| `Melops_metadata.txt` | `filename_year,length` — what `readout_length_controlled.py` reads. The **recorded** length, i.e. an estimate with measurement error; the true one stays in `truth.jsonl` |
| `body/<image_id>.png` | the crop referenced by `path` |
| `masks/<image_id>_identity.png` | the render-time identity mask, same crop |
| `gt/<image_id>.npz` | `chart_s`, `chart_phi` and every mask, same crop |
| `truth.jsonl` | per image: pose, camera, light, nuisance draws, growth ratio, elapsed days, spot counts, pixel budgets, and both the **true** (`length_cm` / `length_mm`) and **recorded** (`measured_length_mm`) length |
| `dataset.json` | arguments, constants with evidence grades, summary counts |

Boxes are `[left, top, width, height]` **floats in crop pixels**, with head and
headless expressed *inside* the body crop — which is what `melops_data.load_crop`
applies them to. The head/headless split is cut in **arc length through the
chart GT** at the schema's last gill slit (`gill_slit_7_dorsal_origin`,
`s = 0.22`, `[UNVERIFIED]`), never guessed from the silhouette, so it means the
same anatomical thing at every pose and view angle.

Sightings are planned with **log-uniform gaps**, so `diagnose.py`'s
recapture-gap buckets are populated by construction rather than by luck, and
**singletons are drawn deliberately** (Melops averages ~2.5 images/individual, so
the open-set split must survive them).

Sides `L`/`R` are rendered by **moving the camera**, never by mirroring.

### The recorded length is an estimate, not a name

Each animal draws one initial length and then only grows, so an *exact* recorded
length is very nearly a unique identity code — and every length-stratified
readout downstream (`readout_length_controlled.py`'s size-assortativity index,
its length-stratified Rank-1, its length-matched impostor AUROC) would then be
measuring that code rather than a body. Real field length estimates carry error,
so `--length-noise` (default `0.07` relative sd, drawn per sighting from a
generator seeded on `(seed, 4, individual, sighting)`) is applied to the value
written into `Melops_metadata.txt`. `truth.jsonl` keeps the true length for the
size-assortativity readouts that need it.

That bracket is a **placeholder**: `EVIDENCE["LENGTH_MEASUREMENT_RSD"]` grades it
`[UNVERIFIED]`, bracket `[0.03, 0.15]`, standing in for a photogrammetric
total-length error study that was **not** retrieved for this species. Replace it
before making any length-stratified claim.

Measured on a 40-animal set (`tests/test_dataset.py::test_length_alone_is_not_an_identity_oracle`,
seed 0, 217 sightings, chance = 1/40 = 0.025):

| 1-NN identity from length alone | value | × chance |
|---|---|---|
| true length (`--length-noise 0`) | **0.567** | 22.7× |
| recorded length (default `0.07`) | **0.106** | 4.2× |

Between-individual / within-individual sd of length falls from **21.3×** to
**3.2×** with the noise on. The test asserts the recorded figure stays under
5× chance; widening `LENGTH_CM_BRACKET` from `[150, 275]` to `[140, 285]` came
with it, so the sampled population spans the species range rather than a narrow
slice of it.

---

## What "excluding the eye, the mouth and shadows" means — at three levels

The phrase means three different, separately-enforced things, and the engine
enforces all three.

### Level 1 — generation: the sampling mask

`exclusions.exclusion_regions(schema, stations)` derives regions from the schema
landmarks. On the default station table:

| region | `s` | `phi` |
|---|---|---|
| `eye_left` / `eye_right` | 0.040–0.080 | ±1.20 ± 0.35 |
| `naris_left` / `naris_right` | 0.020–0.050 | ±2.10 ± 0.30 |
| `mouth_jaw` | 0.000–0.110 | π ± 1.20 (ventral) |
| `gill_slits` | 0.130–0.230 | all but a dorsal band of ±0.75 |

No spot is ever **placed** inside one. A copied or drifted pattern cannot leak
identity into an excluded region either, because the mask is enforced at
sampling *and* at render time.

*(The gill slits are excluded on purpose, and this was arbitrated between two
modules that disagreed. A gill slit is a dark linear aperture every individual
has, whose appearance varies with respiration and view angle: a textbook re-ID
shortcut. Excluding it costs the arc-length anchor nothing — re-anchoring runs
on landmarks, not on the identity image — and the La Jolla freckle patch lies
anterior to the band (`chart.head_patch_bounds` in the yaml).
`EXCLUSION_MASK_INCLUDE_GILL_SLITS` is the single flag that reverses it.)*

### Level 2 — texture: de-lighting

When a texture comes from photogrammetry it has **capture lighting baked into
it**, and that lighting is not identity. `bake.bake_chart_to_texture(delight=True)`
removes it by fitting a low-order **Legendre × Fourier** basis in log space
rather than blurring.

This replaced a Gaussian blur deliberately, and the reason is measurable: the
dominant lighting term on a horizontal animal is a dorsal-to-ventral gradient,
i.e. `cos(phi)` — the *lowest* non-constant frequency around the girth. A
Gaussian with `sigma_phi = 0.8 rad` attenuates it only to `exp(-0.32) ≈ 0.73`,
so ~27 % survives. Correlation to the clean pattern: **0.261 (off) → 0.675
(blur) → 0.9999 (basis)**. A spot is not representable at `n_harmonics = 3` and
survives (relative spot contrast 0.489 vs source 0.689 after a full
bake + read-back).

Its documented limits — both from `bake.bake_chart_to_texture`'s own LIMITS
list, and both load-bearing before this is pointed at a real photogrammetry
texture:

- **Low-frequency albedo goes with the light.** It cannot separate low-frequency
  *shading* from low-frequency *albedo*, so countershading is flattened too (a
  test asserts dorso-ventral spread 0.175 → 0.0002). That is the deliberate
  three-layer separation — **capture lighting / species tone / identity**.
- **Hard shadow *edges* are not low frequency and survive.** The estimator is a
  low-order Legendre × Fourier fit sized by `DELIGHT_SIGMA_S = 0.10`
  body-length fractions and `DELIGHT_SIGMA_PHI = 0.80 rad`; only shading
  *softer* than that is removed. A fin's cast-shadow boundary, a kelp-blade
  stripe or a strobe terminator stays baked into the texture — and then becomes
  a per-capture identity shortcut for exactly the individual photographed under
  it, which is the failure de-lighting exists to prevent. Note what kind of
  claim this is: `tests/test_bake.py` exercises de-lighting only against smooth
  `0.55 + 0.27·cos(phi) + 0.25·s` shading, so the edge case is **documented,
  not measured**. De-lighting is not a substitute for rejecting a
  hard-shadowed capture.

For this generator's synthetic albedo there is no light to remove, so
`delight=False`; switching it on here would flatten the countershading, which
is albedo.

### Level 3 — render: the per-pixel identity mask

```
identity = visible_skin ∧ isfinite(chart_s) ∧ ¬exclusion ∧ ¬shadow ∧ ¬occlusion
```

- **`exclusion`** is the chart mask pulled *through* the per-pixel chart GT, so
  eyes, nares, mouth and gill slits are excluded at any pose with **no
  image-space detector**.
- **`shadow`** = attached (`N·L ≤ 0`) ∨ cast (lit-facing but blocked in the
  shadow map). Both are excluded — a pattern you cannot see is not evidence.
- **`occlusion`** = a subject surface is here but no subject is front-most.

**A discretization hole that had to be closed.** `mask_from_regions` marks a
chart *cell* when its *centre* is inside a region, and `render.sample_chart_mask`
looks the mask up by **nearest neighbour** (a boolean has no interpolant).
Composing the two leaves a half-cell sliver along every region border where a
pixel is genuinely inside the anatomical region but its nearest cell centre is
outside — so the pixel was **not** excluded. Measured: **62 of 3935 eye pixels
(1.6 %, ≤2 per frame) reached the identity mask.** Small, but "the eye is
excluded" is a guarantee, and a guarantee with a 1.6 % hole is not one.

`render.dilate_chart_mask` grows the **scoring** mask by one cell first, and
`render.resolve_exclusion_chart("auto")` — the default path — applies it, so the
guarantee holds for any caller of `render.render`, not only for `make_dataset`
(which builds the same mask explicitly because it pins its own chart
resolution). A mask passed in explicitly is used verbatim: spot **placement**
in `pattern.py` needs the exact cells and must not be pushed out of skin it is
entitled to use.

The element must be the **full 3×3 (8-connected)** one, not the 4-connected cross:
nearest-neighbour lookup misplaces a point by half a cell in *each axis
independently*, so a point just inside a region *corner* lands on the diagonal
neighbour. That was not hypothetical — with the cross, one gill-slit pixel still
leaked, sitting 0.17 cells past the `s` border and 0.17 cells past the `phi`
border simultaneously. `tests/test_dataset.py` asserts **zero** leaked
pixels across all six regions on every image of the corpus, and separately
asserts the check is not vacuous (each region must actually be visible
somewhere); `tests/test_render.py` asserts the same guarantee at *module* level
on the default `exclusion="auto"` path — 0 leaks of 918 in-region pixels, where
the undilated mask leaks 16 — plus a ring test that identity pixels survive one
pixel outside every region (17 of 149), so the collar is a collar and not a
blackout.

### Shadows are a first-class, *separable* nuisance

`--shadow` drives a **canopy caster**: kelp blades placed up-light and above the
animal, outside the camera frame, so they cast **without occluding**. In-frame
kelp is what occludes. Keeping the two apart is what lets shadow and occlusion
be ablated independently, which is most of what a synthetic corpus is for.
Verified on the demo corpus: across **104 canopy frames**, canopy-only frames
have **exactly 0 occlusion pixels** while dappling ~10 % of the body.

The key light is also placed **relative to the camera**, and this is the one
scene choice here that is a convention rather than a physical model. A purely
overhead sun on a laterally-viewed cylinder puts the terminator *on the
camera-facing flank* — `N·L ≈ 0` right where the pattern is — so half of every
animal would be attached shadow for a reason that has nothing to do with the
animal. A diver keeps the sun or strobe behind their own shoulder, so the
light's lateral component is drawn on the camera's side for `1 − BACKLIT_PROB`
of frames. Backlit frames are kept on purpose (0.22): a rim-lit animal is a real
and hard encounter and a corpus that never contains one trains a model that has
never seen one.

Measured effect of the convention: identity/body pixels **0.393 → 0.472** on a
10×6 corpus, recorded in `make_dataset.EVIDENCE["LIGHT_FRONTAL_LATERAL_BRACKET"]`
together with the front-lit/backlit split it decomposes into. The counterfactual
half of that pair (a purely overhead key light) is **not** reproducible from any
shipped command — the convention is not a CLI knob — so treat 0.393 as a
recorded observation, not as a number this repository can re-derive. The
convention-on figure for the 40-animal demo below, **0.471**, is
`dataset.json`'s `identity_pixel_fraction_mean` and is re-derived by every run.

---

## Measured results from the demo run

`python make_dataset.py --out demo --n-individuals 40 --sightings-per-individual 6 --years 4 --seed 0`
— 217 images of 40 individuals, 0 dropped, **120 s** on one CPU core.

| | |
|---|---|
| images / individual | min 1, mean 5.4, max 8 (6 singletons) |
| sides | L 109 / R 108 |
| dates | 2019-03-01 … 2023-02-17 |
| mean visibility | 8.1 m |
| identity px / body px | **0.471** (front-lit 0.537, backlit 0.185) |
| exclusion px / body px | 0.187 |
| occlusion px / body px | 0.070 (70 frames) |
| cast-shadow px / body px | 0.043 (110 frames, 104 with canopy) |
| recorded length | mean 2264 mm, sd 461 (true: mean 2267, sd 427) |

Every row is a field of `demo/dataset.json`, written by the command above.

### Prototype 01 runs on it unchanged

`run_ablation.py --data melops --backbone hist` (gallery 44, known 113, novel 60):

| arm | Rank-1 | Rank-5 | mAP | AUROC |
|---|---|---|---|---|
| head | 0.053 | 0.248 | 0.185 | 0.491 |
| body | 0.088 | 0.327 | 0.234 | 0.471 |
| headless | 0.097 | 0.354 | 0.241 | 0.466 |
| cross_orientation | 0.034 | 0.203 | 0.159 | 0.405 |

**Verdict: INCONCLUSIVE** — "every crop arm is under 15 Rank-1 points, so the
≥15-point kill criterion is arithmetically inexpressible". That is the correct
answer and it is a **statement about the backbone, not about the corpus**. A
colour histogram cannot read speckle through turbid water; the stronger
backbones (MegaDescriptor, MiewID, DINOv2) need model downloads unavailable in
this environment. `diagnose.py` populates all five recapture-gap buckets
(n = 39 / 31 / 14 / 17 / 12).

### The corpus does carry identity — the ground-truth chart test

Every chart-space number in this section comes from **`chart_readout.py`**, and
the command that produced it is on the line above the table. Nothing here is
hand-computed.

```bash
# the two corpora, from the same seed
python make_dataset.py --out demo --n-individuals 40 \
    --sightings-per-individual 6 --years 4 --seed 0
python make_dataset.py --out demo_clean --n-individuals 40 \
    --sightings-per-individual 6 --years 4 --seed 0 \
    --occlusion 0 --shadow 0 --turbidity 0
# the readout
python chart_readout.py --data demo --sensitivity
python chart_readout.py --data demo_clean --sensitivity
```

(The corpora are ~20 MB each and are *not* checked in; every command in this
section regenerates its own input, which is the point of the rule that a README
number must come from a shipped script.)

The readout unwraps every render back into the canonical chart **through its own
per-pixel `(s, phi)` GT** (a scatter, not a resampling: each identity-mask pixel
is dropped into the cell its own chart coordinates name and cells are averaged),
high-passes in chart space to drop shading, veil and countershading, and matches
by NCC over the cells the two unwraps **jointly** cover and that are not
anatomically excluded. The split is
`prototypes/01-melops-ablation/protocol.one_shot_open_set_split`, **imported**,
side-partitioned.

Default configuration: `128 × 240` cells (`H_phi × W_s`, isotropic), minimum
joint coverage `0.05` (**1318 of 26342** non-excluded cells), identity mask on,
high-pass radius `0.02 × W_s`. All 217 images unwrap; mean coverage per image is
**0.154** of the non-excluded chart — one flank, once.

| readout | Rank-1 | chance |
|---|---|---|
| raw crop + colour histogram (`run_ablation.py`, body arm) | 0.088 | — |
| **chart-space NCC, same flank** | **0.875** (84/96) | 0.055 |
| same, nuisance off (`demo_clean`) | **0.941** (96/102) | 0.062 |

Same-individual mean NCC **0.768** vs different-individual **0.112**
(separation **0.656**, pairwise AUROC **0.999**); nuisance off, 0.806 / 0.102,
separation 0.704, AUROC 1.000.

`chance` is the readout's own: the mean of `1 / (number of same-side gallery
entries this query was actually scored against)`, which is ~0.055 rather than
`1/40 = 0.025` because matching is within side and under-covered gallery entries
are not scored. **17 of the 113 known queries have no gallery entry they share
enough chart with** and are excluded from the Rank-1 rather than counted as
misses — "these two images never saw the same skin" is not evidence that they
are different animals. That count is reported, not hidden.

**Read this as a ceiling, not a result.** It uses the *oracle* chart GT — the
renderer's exact per-pixel `(s, phi)` — not an estimated one. So 0.875 is what a
*perfect* pose-and-chart estimator would reach on this corpus, and the gap from
0.088 to 0.875 is what rectification is worth **if you can estimate the chart**.
Closing that gap is prototype 04's job.

**And read it as a family of numbers, not one number.** The headline spans
**0.511 – 0.938** across choices that have nothing to do with the corpus, which
is exactly why `chart_readout.py` exposes them as flags and `--sensitivity`
sweeps them. One row per one-at-a-time variation from the default, on `demo`:

| chart cells | min joint cov | high-pass | identity mask | Rank-1 | n | separation | AUROC |
|---|---|---|---|---|---|---|---|
| **128 × 240** | **0.05** | **0.02** | **on** | **0.875** | 84/96 | 0.656 | 0.999 |
| 48 × 90 | 0.05 | 0.02 | on | 0.867 | 98/113 | 0.535 | 0.977 |
| 96 × 180 | 0.05 | 0.02 | on | 0.890 | 97/109 | 0.613 | 0.996 |
| 192 × 360 | 0.05 | 0.02 | on | 0.511 | 23/45 | 0.709 | 1.000 |
| 128 × 240 | 0.02 | 0.02 | on | 0.902 | 101/112 | 0.622 | 0.997 |
| 128 × 240 | 0.10 | 0.02 | on | 0.706 | 60/85 | 0.680 | 1.000 |
| 128 × 240 | 0.20 | 0.02 | on | n/a | 0/0 | n/a | n/a |
| 128 × 240 | 0.05 | 0.01 | on | 0.875 | 84/96 | 0.646 | 0.998 |
| 128 × 240 | 0.05 | 0.04 | on | 0.740 | 71/96 | 0.393 | 0.906 |
| 128 × 240 | 0.05 | 0.08 | on | 0.594 | 57/96 | 0.155 | 0.812 |
| 128 × 240 | 0.05 | 0.02 | off | 0.938 | 106/113 | 0.441 | 0.986 |

Three things to take from it, and the `n` column is load-bearing in all of them:

- **The coverage floor trades Rank-1 against how much of the query set is
  answerable at all.** At `0.02` more pairs clear the bar (112 of 113 queries
  scored) and Rank-1 is 0.902; at `0.10` only 85 are answerable and Rank-1 falls
  to 0.706; at `0.20` *no* pair on this corpus clears it and the readout
  correctly reports nothing rather than a number. Any single Rank-1 quoted
  without its floor and its `n` is uninterpretable.
- **Aliasing costs separation before it costs Rank-1.** At `48 × 90` the default
  spot diameter (`2 × PatternParams.radius_median` = 0.011 s-units) spans ~1.0
  cell, at the Nyquist limit, and separation falls **0.656 → 0.535** with AUROC
  0.999 → 0.977 — but Rank-1 barely moves (0.875 → 0.867) because a coarser
  chart also lets more pairs clear the coverage floor. The two effects partly
  cancel, which is precisely why a single-number headline hid this. Going the
  other way, `192 × 360` collapses Rank-1 to 0.511 not through aliasing but
  through coverage: only 45 of 113 queries stay answerable.
- **The identity mask costs Rank-1 and buys honesty.** Turning it off (unwrap
  every visible-skin pixel, attached shadow and all) *raises* Rank-1 to 0.938
  while *lowering* separation to 0.441 — shading structure is itself a
  per-capture signature, and pooling it inflates the match while making the
  score mean less. The default keeps the mask on.

For the record, the earlier version of this README reported `0.807` here from a
computation that shipped with nothing. An independent re-implementation got a
very different answer, and re-deriving it inside a script — which is what the
project rule demands — showed why: the number was one cell of the table above,
not a property of the corpus.

### The ablation instrument recovers an answer it was given

The whole point of `--head-signal` / `--flank-signal` is to build a corpus whose
*correct* head-vs-flank answer is known, and check that the measurement finds it.

```bash
python make_dataset.py --out demo_noflank --n-individuals 40 \
    --sightings-per-individual 6 --years 4 --seed 0 --flank-signal 0
python chart_readout.py --data demo_noflank
```

Band Rank-1 from `chart_readout.py`'s band table. Bands are cut at the schema's
own stations — the same `gill_slit_7_dorsal_origin` the head box is cut at, and
`precaudal_pit` — so each band is the span its knob owns:

| band | `--flank-signal 1` | `--flank-signal 0` | chance | mean coverage |
|---|---|---|---|---|
| head (`s < 0.22`) | 0.866 (97/112) | 0.866 (97/112) | 0.056 | 0.215 |
| trunk (`0.22 ≤ s < 0.75`) | 0.901 (91/101) | **0.079** (8/101) ← at chance | 0.065 | 0.179 |
| tail (`s ≥ 0.75`) | 0.067 (1/15) | 0.067 (1/15) | 0.767 | 0.079 |
| **whole body** | **0.875** (84/96) | **0.594** (57/96) | 0.055 | 0.154 |

The trunk drops to chance exactly as commanded (0.079 against a measured chance
of 0.065), while the head is **bit-for-bit unchanged** — same 97 of 112 — and so
is the tail, governed by the separate `tail_signal` left at 1.0. The three region
knobs are independent. The trunk is not *blank*: the `n_common` shared-confounder
layer still textures it, so it is **textured but uninformative**, which is the
honest version of an ablation (a blank region would be trivially detectable as
blank).

**The tail row is not usable and the table says so.** Mean coverage there is
0.079 — the caudal region is thin, often at grazing incidence, and frequently out
of frame — so 98 of the 113 known queries have no defined tail pair, the 15 that
survive face ~1.3 candidates each (hence the 0.767 chance column), and the
resulting 0.067 measures the coverage failure, not the pattern. It is printed
because suppressing it would hide the failure; it must not be quoted as a tail
result. Recovering the tail needs either a lower floor with its own caveat, or
framings that put the caudal peduncle in shot.

The whole-body row is the useful one: **pooling a signal-free region actively
hurts** (0.875 → 0.594) even though no information was removed from the head.
That is the quantitative argument for region-restricted matching, measured
rather than asserted — and it is the kind of claim `run_ablation.py` exists to
settle on real data, dry-run here first.

### Dry-running P0-d: what the protocol can and cannot see

P0-d (`docs/sevengill-canonical-reid/03-candidate-approaches.md`) proposes
measuring pattern drift on confirmed multi-year resights. Here it can be run
against a corpus where **the drift is known exactly**, so the protocol itself
can be tested rather than trusted. `chart_readout.py` scores every same-flank
resight pair two ways: the MEASURED chart NCC between the two unwrapped renders,
and the TRUE chart similarity between the two generative states of the animal
(`drift.similarity`, on `Individual`s rebuilt deterministically from the run's
own seed through `make_dataset.individual_timeline` — the same code path that
rendered them).

```bash
python chart_readout.py --data demo          # prints the table below
```

| gap (days) | pairs | scored | MEASURED chart NCC | TRUE chart NCC |
|---|---|---|---|---|
| 0–30 | 95 | 64 | 0.817 | 0.992 |
| 31–180 | 110 | 73 | 0.783 | 0.951 |
| 181–365 | 51 | 34 | 0.736 | 0.865 |
| 366–730 | 82 | 59 | 0.675 | 0.765 |
| 731+ | 32 | 22 | 0.615 | 0.669 |

- Spearman(measured NCC, elapsed days) = **−0.515** (Pearson −0.422); the true
  chart NCC gives −0.975.
- Spearman(measured, TRUE) = **+0.527** (Pearson +0.498).
- sd of measured NCC **0.149** vs sd of true chart NCC **0.125**.
- **118 of the 370 same-flank resight pairs are undefined** — below the 0.05
  joint-coverage floor — and are excluded rather than scored.

Nuisance off (`demo_clean`, `--occlusion 0 --shadow 0 --turbidity 0`) improves
every term: Spearman(measured, elapsed) **−0.591**, Spearman(measured, TRUE)
**+0.612**, measured sd **0.124**, and only 83 of 370 pairs undefined.

**This reverses the conclusion this README used to draw here, and the reversal
is the finding.** The earlier text reported corr(measured, elapsed) = +0.014 and
corr(measured, true) = −0.017 and concluded that "a drift measurement built on
single-image similarity between field photographs will report the viewing
geometry, not the pigment". That was an artefact of the earlier, **unshipped**
computation — most likely of scoring pairs that shared almost no chart, and of a
high-pass wide enough to leave shading in. Computed over *jointly covered*
cells, with the coverage floor stated, chart-space NCC **does** track both
elapsed time and the true pattern change, monotonically across all five buckets.
The claim that the readout "reports the viewing geometry, not the pigment" is
withdrawn.

**What survives of the old caution, and it is not nothing:**

- **A third of the resight pairs cannot be scored at all** (118 of 370 at the
  default floor; 83 of 370 with nuisance off). On this corpus that is visible,
  because the coverage is computed from an oracle chart. On real photographs it
  is *not* visible — you cannot tell a low-similarity pair from a
  never-saw-the-same-skin pair — so a real P0-d must report a defined-pair
  count or it is silently conditioning on framing.
- **The coupling is real but loose.** Spearman 0.53 between measured and true,
  and a measured sd (0.149) *larger* than the drift signal it is estimating
  (0.125). A single pair is a weak estimate of that pair's drift; the curve
  works because it averages tens of pairs per bucket. Do not fit a drift model
  to individual pairwise similarities.
- **Nuisance is part of it after all.** Turning the water off moves
  corr(measured, true) from 0.53 to 0.61 and removes 35 undefined pairs, so
  turbidity, occlusion and cast shadow *do* cost drift sensitivity — the earlier
  "this is not the water" claim does not survive either.
- **This is chart space on oracle geometry.** The raw-pixel readout that a real
  P0-d would start from is the 0.088-Rank-1 histogram arm, not this. Nothing
  here says drift is measurable from field photographs; it says drift is
  measurable *once the chart is*.

This does **not** contradict the identity result above: between-individual
separation (**0.656**) is far larger than the two-year drift effect (true sd
0.125), so *who this is* is an easier question than *how much it changed* —
which is the ordering a re-ID pipeline wants.

**The warning for the real P0-d run**, restated from the numbers that survive:
report the defined-pair count, aggregate over pairs rather than trusting one,
and do not spend the La Jolla archive on a drift readout until the chart
estimate it depends on has been validated. This corpus is the cheap place to
design that readout, because here the right answer is known.

---

## Limitations

State these before citing any number above.

- **Procedural patterns are not real sevengill statistics.** Spot count,
  size distribution, minimum separation and contrast are plausible parameters,
  not measurements; **no measured sevengill speckle morphometry was retrieved.**
  The corpus becomes species-realistic only when `copy_from_photo` is fed real
  La Jolla photographs. Until then, treat it as a test instrument for
  *geometry and protocol*, not as a substitute for data.
- **The tube pose is not the rig.** `pose_vertices` bends a tube along
  `κ(s) = amp·cos(2π·wave·s + phase)`. It has no vertebral limits, no fin
  articulation, no volume preservation. `POSE_AMP_BRACKET` has evidence grade
  *none — placeholder*.
- **The renderer is not photoreal.** It is an opaque numpy z-buffer: no
  refraction, no subsurface scattering, no forward scatter, no strobe
  backscatter, no spectral veil; texture alpha is dropped; pinhole triangles
  crossing the near plane are *dropped, not clipped*. Caustic amplitude has
  evidence grade *none — visual placeholder* (and at 3–6 m, scattering has
  largely washed caustics out anyway).
- **A third of same-flank resight pairs cannot be scored at all.** 118 of 370
  fall below `chart_readout.py`'s 0.05 joint-coverage floor and are reported
  UNDEFINED. The drift signal *does* survive on the pairs that clear it
  (Spearman 0.53 with the true chart similarity, monotone across all five gap
  buckets — see the P0-d section), but the measured sd (0.149) exceeds the drift
  signal it estimates (0.125), so **a single pairwise similarity is not a drift
  measurement**; only the bucketed curve is. An earlier version of this README
  claimed the readout was blind to drift entirely; that claim came from an
  unshipped computation and is withdrawn.
- **Every chart-space number here is a family, not a value.** Rank-1 spans
  0.51–0.94 across readout chart resolution, joint-coverage floor, high-pass
  radius and identity mask (and at a 0.20 floor nothing on this corpus is
  scoreable at all). `chart_readout.py --sensitivity` prints the sweep;
  quote a cell only with its configuration and its `n`.
- **Attached shadow is treated as non-identity**, which is conservative: an
  attached-shadow pixel still shows the pattern at ambient level. It costs ~0.39
  of the body mask. That threshold lives in `render.py`, not here.
- **`unbake`'s `s` is arc length along the silhouette's medial axis**, which
  retracts from the snout and caudal tips, compressing both ends. A
  foreshortened animal is undetected and every `s` is wrong. The whole 2D path
  is **superseded** — not merely complemented — by a prototype 04 rig fit plus
  analysis-by-synthesis for oblique views.
- **The visibility bracket is dive-operator copy**, `[SECONDARY]`, and that
  source itself says to get real figures from SCCOOS.
- **`copy_from_chart`'s fitted radius runs 10–15 % small** because thresholding
  cuts inside a soft-edged mark. Documented as a known bias with a `radius_gain`
  escape hatch rather than papered over with a constant that holds at one
  threshold.

---

## Swap-in points

### Prototype 04 — the sevengill rig (never imported here, only assumed)

Two edits, both inside `make_dataset.py`; nothing else changes.

**1. Geometry + chart coordinates.** Replace `build_model`:

```python
tc         = mesh3d.tube_coords(mesh, centerline)
vertex_s   = tc.s / tc.total_length
vertex_phi = tc.phi          # convention already matches TubeCoords exactly
```

Everything downstream — the bake, the render, the chart GT, the three boxes — is
written against `(mesh, vertex_s, vertex_phi)` and needs no other change. The
phi convention was aligned deliberately: `0 = +Z` dorsal, `+π/2 = +Y` left.

**2. Pose.** Replace `pose_vertices` with the rig's own posing. `PoseParams` is
the record the rig should fill in instead. The current bend is arc-length
preserving so the chart GT is pose-invariant; **any replacement must preserve
that property** or the ground truth stops being ground truth.

Pass a *measured* `stations` dict into `exclusion_regions` /
`build_exclusion_mask` / `Individual.generate` and every exclusion region moves
with it — that is the third, already-open, swap-in point.

### The owner's Blender pipeline — `/home/user/shark-pose-3d`

**Do not modify these files. Extend them.**

| file | what to add |
|---|---|
| `shark_pose/synthetic/blender/underwater_shader.py` → `setup_shark_skin_material` | The identity hand-off. It currently builds the pattern from a procedural `ShaderNodeTexVoronoi` / `TexWave` / `TexGradient` chosen by `skin_params["pattern_type"]` — decorative, and **carrying no identity**. Replace that branch with a `ShaderNodeTexImage` pointing at the per-individual UV texture written by `bake.bake_chart_to_texture`. Keep the existing countershading `ColorRamp` **or** the baked tone, not both. |
| `shark_pose/synthetic/blender/render_pipeline.py` → `_build_annotation` | The JSON hand-off. The annotation dict is a *pose* record (`theta`, `beta`, `keypoints_2d/3d`, `camera_matrix`) with **no `identity` and no `date`** — it cannot express a re-ID corpus. Add `identity`, `date`, `side`, and the texture path, and the same pipeline emits a re-ID dataset. |
| `shark_pose/synthetic/blender/domain_randomization.py` | Its `skin_params` is where per-individual texture selection should be drawn, alongside the existing lighting/water randomization. |
| `shark_pose/synthetic/blender/batch_render.py` | Unchanged — it already checkpoints and parallelises. |

The hand-off is therefore exactly two artefacts: **a UV texture per (individual,
date)** and **a JSON row carrying `identity` / `date` / `side`**. Prototype 05
produces both today; Cycles then replaces this repository's numpy rasteriser for
photorealism while the *identity* keeps coming from the chart.

---

## What this enables

- **Synthetic pretraining before Sharkbook data exists.** A model can be
  pretrained on tens of thousands of labelled sevengill sightings with exact
  identity, pose and chart supervision, before a single real La Jolla image is
  licensed — and the head-vs-flank signal knobs let the pretraining corpus be
  built to match whatever the ablation says.
- **A ground-truth test of the canonical chart.** The 0.088 → 0.875 Rank-1 gap
  above is measured by a shipped script (`chart_readout.py`), not argued: it is
  the value of rectification with a perfect chart estimate, on a corpus where
  the true chart is known per pixel. No real dataset can give that number,
  because no real dataset knows its own chart. It is also a *configured* number
  — see the sensitivity table.
- **Dry-running the P0-d drift protocol** before spending it on the real
  archive. On data whose drift is known exactly, chart-space NCC recovers the
  drift curve (Spearman 0.53 with the true chart similarity, −0.52 with elapsed
  days, monotone across five gap buckets) — but only over jointly covered cells,
  and only in aggregate: a third of resight pairs are undefined, and a single
  pair's similarity is noisier than the drift it estimates. Both the recovery
  and the caveat are cheap, early and specific, obtained without touching the
  La Jolla archive.
- **Validating the ablation instrument itself** before pointing it at real
  sharks: the head/flank/tail signal knobs build a corpus whose correct answer
  is known, and the measured answer matches it (table above).
- **Ablating nuisance factors independently** — occlusion, cast shadow, turbidity
  and pattern-region signal are separate knobs with separate, verified masks;
  the canopy caster exists specifically so shadow moves without occlusion
  moving.
- **Exercising the whole downstream stack** end to end with zero real data and
  zero optional dependencies.

---

## Tests

```bash
python -m pytest tests/ -q                       # 223 tests, ~103 s
python -m pytest tests/test_dataset.py -q        # 17 tests, ~34 s
python -m pytest tests/test_chart_readout.py -q  # 19 tests, ~27 s
```

`tests/test_dataset.py` deliberately does **not** reimplement the downstream
contract: it imports the real `melops_data` and runs the real `run_ablation.py`
and `diagnose.py` as **subprocesses**, the way a user would. A test that mimicked
the loader would pass while the loader failed.

`tests/test_chart_readout.py` pins what makes the chart-space numbers *mean*
something rather than what they happen to be: that a pixel lands in the cell its
own `(s, phi)` GT names and nowhere else; that an under-covered pair is reported
UNDEFINED rather than as a low score; that excluded cells never enter a score;
that the split is prototype 01's own (counts checked against
`protocol.one_shot_open_set_split` directly); that the rebuilt `Individual`
states used for the true-drift column are the ones that were actually rendered;
and that each sensitivity flag really does move the answer — which is the whole
reason it is a flag.
