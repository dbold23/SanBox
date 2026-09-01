# Prototype 04 — Sevengill de-bend, rig and swim

A **mesh-agnostic** pipeline that takes a photogrammetric GLB of an elongate animal, recovers its 3D
centerline, **straightens it to a canonical rest pose**, rigs it with the phase1b sevengill skeleton,
and writes one Khronos-clean skinned GLB carrying real swimming animations.

```
load_mesh → extract_centerline_3d → tube_frames → tube_coords → debend
          → detect_fins → build_skeleton → compute_weights → make_clip → write_skinned_glb
```

Everything runs on numpy/scipy/trimesh/pygltflib/PIL. No Blender, no GPU, no OpenGL — previews are
orthographic painter's-algorithm renders drawn with PIL. Every GLB emitted is checked with the
Khronos `gltf-validator` and has **zero errors and zero warnings**.

---

## 1. The problem: the scan's rest pose is a C

The real input is a textured Meshy-AI GLB generated from photographs of a live sevengill
(*Notorynchus cepedianus*). The animal was **mid-turn** when it was photographed, so the mesh's rest
pose is a strong lateral C-curve, not a straight canonical pose.

That is fatal if you rig it naively. A skinned mesh's *bind pose* is the pose the inverse-bind
matrices encode; every animation is a deformation applied **on top of it**. Bind a spine to a C-shaped
mesh and:

- the "straight" pose of the rig is a C, so a cruising animation is a wave riding on a permanent turn;
- joint arc-length fractions are measured along a curve that is 22% shorter end-to-end than the
  animal, so the fin stations land in the wrong places;
- the two sides of the body have different skin area in the chart, so left/right symmetry is broken;
- a re-pose that *does* straighten the animal has to undo a bend the rig cannot see.

## 2. The fix: tube coordinates — Approach 2's canonical chart, lifted to 3D

Prototype 02 built a 2D chart that rectified a silhouette onto **(arc length `s` × signed normal
offset `r`)**. In 3D the same idea gains one coordinate and becomes a full body-frame parameterisation
of a tube:

| coordinate | meaning |
|---|---|
| `s` | arc length along the centerline, head → tail |
| `r` | perpendicular distance from the centerline |
| `phi` | circumferential angle, measured from the dorsal normal `+Z` toward the left binormal `+Y` |

The frame field at each station is a **rotation-minimising frame** (double reflection, Wang et al.
2008) — prototype 02's `frames.rotation_minimizing_frames`, *imported*, not vendored. RMFs have no
sign flip at curvature inflections and zero twist on a planar bend, which is precisely what a
straightening map must not invent.

**De-bending is then one line of intent: keep `(r, phi)`, replace the centerline.** Vertex positions
move; `faces`, `visual.uv`, the texture image and the material are the *same objects* before and
after. A textured Meshy GLB survives the operation unchanged except for where its vertices sit.

Two details make it exact rather than approximate:

- `tube_coords` returns the **station index** alongside `(s, r, phi)`. Recovering the segment from `s`
  alone is ambiguous at a corner of a bent centerline; with the index, `tube_to_points` inverts to
  `< 1e-12`.
- Material beyond either end of the chart — the snout tip, both caudal lobes — gets `s < 0` or
  `s > S` and is **carried, not clipped**: transported rigidly by the terminal frame. This is
  deliberate. The centerline extractor's thick-core threshold stops the medial path at the
  **peduncle** so it cannot wander down a fin, and charting *through* the caudal would straighten the
  heterocercal upsweep and hand the rig an anatomically wrong rest pose.

## 3. What the rig is

- **Spine**: the 13 serial joints of `phase1b/p0-sevengill-schema/skeleton_sevengill.py`, by name and
  parent order, placed at arc-length fractions of the *straight* centerline. The schema module is
  imported through a path shim (`$SEVENGILL_SCHEMA_DIR` overrides), never copied.
- **Fins**: two joints each — `<fin>_fin_root` at the detected insertion centroid, parented to the
  nearest spine joint, and `<fin>_fin_tip` at the island's distal extremity. The heterocercal caudal
  gets its own `caudal_upper` and `caudal_lower` pairs.
- **Weights**: body vertices bind to their two bracketing spine joints by arc length; fin vertices
  bind to their own fin's root and tip by normalised distance along the fin axis, and still follow
  body bends because the fin root is a child of a spine joint. Rows sum to exactly `1.0f` after the
  float32 cast (the validator's `ACCESSOR_WEIGHTS_NON_NORMALIZED` check does not round).
- **The fin base is blended, not seamed.** Those two rules meet at the edge of a fin island with
  nothing in between — the island is 100% fin root, the body vertices one ring outside it are 100%
  spine, and the fin root carries the fin's own drive rotation on top of the spine's. That is a step
  discontinuity straight across a mesh edge, and it measured **2.13% BL of edge-length change** on
  the demo mesh's fin-base edges under the cruise clip: a visible tear. `compute_weights(faces=...)`
  therefore ramps the fin-root weight linearly out into the body — 1.0 at the island boundary, 0 at
  ring `R` (`--fin-blend-rings`, default **3**), spine weights scaled by the complement so rows
  still sum to 1 and stay inside four influences. The same edges then move **0.34% BL**.
- **Motion**: an anguilliform/subcarangiform travelling body wave with amplitude growing toward the
  tail, expressed as curvature and integrated to per-joint yaw with an arc-length-exact integrator.
  Modes keep `pose_sampler.MODE_CONFIG`'s names: `cruise`, `turn`, `escape` (C-start), `rest`,
  `glide`; `breach` and `strike` exist and raise `NotImplementedError` with their reason.
- **Fins move as hard as the body does.** `DEFAULT_FIN_DRIVES` amplitudes are absolute degrees
  written for a cruise, so every wave-driven channel is multiplied by
  `motion.fin_amplitude_scale(mode, params)` = the mode's tail-tip amplitude over cruise's default,
  clamped to `[0, 2]`. Cruise is the reference and is unchanged (×1.00); `rest` — "near-zero
  articulation" by its own description — gets ×0.091 instead of flapping its pectorals at full
  cruise amplitude; `escape` scales up and hits the clamp at ×2. Curvature-driven channels (the
  passive dorsal) are excluded because `gain × kappa(s, t)` is already proportional to the mode.

## 4. `as_scanned` — the audit clip

`--keep-bent` adds a clip called **`as_scanned`** that bends the straight rig **back onto the scanned
centerline**. Its construction is the inverse of the de-bend, expressed in joint space:

1. each spine joint's arc-length fraction is looked up on the *scanned* centerline, giving a target
   position and a target RMF frame;
2. the frame's tangent column is replaced by the **chord** to the next joint (a bone must point at the
   next joint, not along the tangent at its own end — that is a forward-Euler lag of half a segment)
   and re-orthonormalised;
3. the cumulative world rotation is the rotation carrying the canonical straight frame onto that
   frame, and the local rotation is `M_parent^T · M_j` — exactly what `rig.forward_kinematics`
   composes;
4. a **root translation channel** carries the rig from the rest origin to where the scan actually sits,
   so the clip lands on the scan in world space.

The clip eases from the rest pose at `t = 0` to the scan pose over 1 s and holds it, so a viewer that
autoplays shows *both* poses. Fin joints stay at identity: the chart measures the body, and a fin's
pose in the scan is whatever the body carries it to.

**This is the audit.** If `as_scanned` does not visually reproduce the scan, the chart is wrong. On the
synthetic demo the spine joints land on the scanned centerline to **0.15 voxel** and the skinned
surface matches the original scan to **0.55% BL RMS** — the residual is linear-blend-skinning blending
two rigid transforms per vertex where the chart transports a continuous frame, not chart error.

---

## 5. Running it

### On the owner's Meshy GLB

```bash
cd prototypes/04-sevengill-rig
python rig_sevengill.py \
    --glb /path/to/sevengill_meshy.glb \
    --out sevengill_rigged.glb \
    --motion cruise,turn,escape,rest,glide \
    --fps 30 --seconds 4 \
    --keep-bent \
    --report out/report
```

> **The GLB has to be on the machine that runs this.** This session had no access to the external
> drive the scan lives on, so the pipeline was built and measured entirely on the procedural mesh
> below. Copy the GLB into the repo (or give an absolute path on your machine) and the command above
> is the whole run.

Two flags carry real risk on a real mesh and are worth a first pass with `--report`:

- **`--up X Y Z`** — the mesh's **dorsal** direction (default `+Z`). Everything downstream hangs on
  this seed: `phi = 0` dorsal, the left/right fin split, and the de-bend being a *lateral*
  straightening rather than a sagittal one. Meshy output is often Y-up; if your scan is, pass
  `--up 0 1 0`.
- **`--core-radius-frac`** (default `0.17`) — the thick-core threshold that keeps the medial path out
  of the fins. It must sit between fin half-thickness and peduncle radius. With it off, the path
  escapes down the caudal lobe and the centerline is wrong by `0.27 BL`; with it on the same test is
  accurate to `< 1` voxel. Raise it for fleshier fins.

Sanity-check the head/tail call the CLI prints: orientation is decided by "the wider end by
distance-transform comes first", which holds for sharks but would fail for a body that thickens
posteriorly. `--voxel-pitch AUTO` is `max(extent)/128`; error plateaus below about `0.004 BL`, so
finer is not automatically better.

Other flags: `-n/--n-stations` (chart resolution, default 64), `--sigma` (Gaussian skin-weight falloff
in world units instead of two-joint binding — smoother bends), `--fin-blend-rings` (width in mesh edge
rings of the fin-base weight ramp, default 3; `1` restores the hard seam),
`--precaudal-fraction`, `--seed`, `--no-validate`, `-q`.

### The synthetic demo

```bash
python demo.py            # ~2.5 s, writes demo/ and demo/report/
```

It builds a straight, UV-textured procedural sevengill (seven gill slits, dorsal set far posterior
over the pelvics, pectorals, pelvics, anal, heterocercal caudal with a long upper lobe), bends it onto
a **known** 120° C-curve, exports that as the *input* GLB, then runs the identical CLI code path and
compares every recovered quantity against the ground truth it started from.

### Opening the output

Any glTF 2.0 viewer: [gltf-viewer.donmccurdy.com](https://gltf-viewer.donmccurdy.com), Windows 3D
Viewer, Babylon Sandbox, macOS Quick Look. Pick the clip from the animation dropdown.

**Blender**: `File ▸ Import ▸ glTF 2.0`. The armature comes in with all 29 bones and each clip as an
Action (open the Dope Sheet ▸ Action Editor to switch). Bone *names* are the schema's
(`spine_00_cranium` … `spine_12_caudal_axis_2`, `<fin>_fin_root`/`_tip`). Bone *head/tail* geometry is
glTF's node hierarchy, not Blender's, so the imported bones point parent→child along `-X`, matching
the `create_shark_armature.py` convention (snout `+X`, tail `-X`, head at the joint, tail toward the
first child).

---

## 6. Measured results (synthetic demo, this repo, `python demo.py`)

Procedural sevengill: **6290 verts, 12202 faces, 8 fin islands**, UV-textured, bent into a 120° C
(sagitta 22.1% of chart length). Chart: 64 stations, voxel pitch `0.00625` = `0.782% BL`.
`BL = 0.8000` world units (the tube centerline; the mesh including the caudal lobe is `1.021` long).

**How every "% BL" and "px" in this section is computed.** `% BL` is normalised by
**`BL = 0.8000`**, the *ground-truth* tube centerline of the straight synthetic mesh — one length,
used by the de-bend table, the `as_scanned` numbers and the joint errors alike, so the three are
directly comparable. (Two other lengths exist and are *not* used here: the extracted chart is
`0.7269` long because extraction trims a few percent off each end, and the mesh including the caudal
lobe is `1.021`. `report/skeleton.json` carries its joint error in world units with an explicit
`spine_joint_error_units` key and a `spine_joint_error_bl` copy normalised by the *chart* length,
which is the number the CLI has access to.) `px` is in voxel pitches, `0.00625` world units. Every
vertex-set comparison — the de-bend table and the `as_scanned` surface figures both — has a **rigid
mean offset removed first** (`demo._rms_after_translation`), because the chart's origin is arbitrary
once extraction has trimmed the ends; no rotation is fitted, only the translation.

**De-bend round trip vs ground truth** (rigid mean offset removed; normalised by `BL = 0.8000`):

| | RMS | max |
|---|---|---|
| all vertices | **0.389% BL** (0.50 px) | 3.535% BL (4.52 px) |
| body only | **0.157% BL** (0.20 px) | 0.676% BL (0.86 px) |

**Extracted centerline vs the exact C-curve**: mean `0.00052 BL` (0.07 px), max `0.00160 BL` (0.20 px).

**Fin labelling vs construction truth** — 8/8 islands found, **100% purity on every island**, and
**zero** body vertices given a fin label:

| fin | found | truth | purity | recall |
|---|---|---|---|---|
| anal | 54 | 63 | 100.0% | 85.7% |
| caudal_lower | 75 | 81 | 100.0% | 92.6% |
| caudal_upper | 168 | 168 | 100.0% | 100.0% |
| dorsal | 95 | 108 | 100.0% | 88.0% |
| pectoral_L | 88 | 110 | 100.0% | 80.0% |
| pectoral_R | 88 | 110 | 100.0% | 80.0% |
| pelvic_L | 60 | 80 | 100.0% | 75.0% |
| pelvic_R | 62 | 80 | 100.0% | 77.5% |

Recall is 75–100% because a fin's **root** vertices sit at body radius and stay labelled `body` by
design — the label marks the protruding blade; `station_range` and `insertion_centroid` are what the
rig binds the base with.

**Texture / topology**: faces identical after de-bend `True`; UVs identical after de-bend `True`; UV
error after GLB write + reload `0.00e+00`.

**Rig**: 29 joints = 13 schema spine + 16 fin joints. Weights `6290 × 29`, ≤ 3 influences per vertex
(mean 1.87 — two spine joints everywhere, plus the fin root on the 3 rings of body vertices inside
each fin-base blend), rows sum to 1 to `2.2e-16`. Fin-base seam edges change length by at most
**0.34% BL** under the cruise clip, against **2.13% BL** with the blend off. Fin parenting resolved
by geometry:

| fin | spine parent |
|---|---|
| pectoral_L, pectoral_R | `spine_05_trunk_03` |
| pelvic_L, pelvic_R | `spine_09_trunk_07` |
| dorsal | `spine_10_precaudal` |
| anal | `spine_11_caudal_axis_1` |
| caudal_upper, caudal_lower | `spine_12_caudal_axis_2` |

**`as_scanned`** (same `BL = 0.8000`, same rigid mean offset removed): spine joints land on the
scanned centerline to max `0.00117 BL` (0.15 px); skinned surface vs the scan RMS `0.547% BL`, max
`2.839% BL`.

**Cruise kinematics**: 0.90 Hz, wavelength 0.90 BL, prescribed tail amplitude 0.110 BL, posed
0.103 BL (0.94×). DCT bending-mode energy **0.976 in 4 modes, 0.996 in 6**. Implied longitudinal skin
strain: anterior **5.5%**, mid-body **10.8%**, posterior **18.4%** (literature bracket 3.9–13.0%).

**Escape kinematics**: peak curvature **3.696 /BL** (`ESCAPE_PEAK_CURVATURE_UNCAPPED_PER_BL = 6.0`
capped by `ESCAPE_MAX_TOTAL_TURN_DEG = 180`), total turning of the midline **170.4°** on this rig
(176.6° on `motion.default_spine_fractions()`), last spine joint **0.580 BL** from the snout, posed
caudal lobe **0.493 BL** from the posed head.

> **Peak curvature is not what tells an escape from a cruise, and the README used to imply it was.**
> Quoting `motion.py`: *the most curved point on a cruising anguilliform swimmer is its own tail tip,
> and that stays comparable. The signature is that during a C-start the whole body — head and
> anterior trunk included — bends the same way at once, so net turning is an order of magnitude
> larger than a cruise's, whose travelling S-wave cancels itself.* Measured on the defaults: whole-body
> peak curvature is `3.70 /BL` escape against `5.70 /BL` cruise (the cruise's own tail tip wins), while
> **net turning is 176.6° against 29.9°** and over the head-and-anterior-trunk window `s ∈ [0, 0.35]`
> the escape's peak curvature is `3.70` against the cruise's `1.48`. `test_motion.py` asserts on the
> two quantities that separate, not on the one that does not.

**Clips written**: `cruise` 134 frames / 4.44 s / looping, `turn` 134 / 4.44 s / looping, `escape` 16
frames / 0.49 s / one-shot, `as_scanned` 46 frames / 1.50 s / one-shot. `as_scanned` writes `29`
rotation channels and exactly **one** translation channel — the root, the only joint that moves.

**glTF validation**: input `sevengill_synthetic_bent.glb`, rest `sevengill_rest.glb` and output
`sevengill_rigged.glb` — **0 errors, 0 warnings each**.

Total demo wall time **2.3 s**. Full prototype test suite: **177 tests, 22 s**.

### Report contents (`--report DIR`)

| file | contents |
|---|---|
| `centerline.json` | scanned + straight centerlines, per-station radius (the EDT, ready for bone thickness), voxel pitch, thick-core `tau`, head/tail widths, sagitta |
| `fins.json` | per-vertex label counts, per-fin vertex count, station range, `s` range **as a fraction of chart length**, insertion centroid, `phi` centroid in degrees, and the radius envelope |
| `skeleton.json` | joint names, parents, kinds, arc-length fractions, rest positions, fin → (root, tip) map, fin → spine-parent map, clip inventory, and the `as_scanned` per-joint error |
| `weights.json` | shape, row-sum extremes, influence histogram, per-joint vertex mass, any unweighted joint |
| `contact_strip.png` | orthographic PIL contact strip: **row 1** the scan (top XY + side XZ) with the extracted centerline in red; **row 2** the de-bent rest pose, same views, skeleton overlaid (spine blue, fin bones orange); **row 3** six frames of the cruise clip skinned through `rig.lbs`, posed spine drawn in red, all at one shared scale |

---

## 7. Known limitations

- **`--core-radius-frac` is a prior, and priors mislead.** Fin naming uses anatomical sectors —
  `|phi| ≤ 45°` dorsal midline, `|phi| ≥ 160°` ventral, laterals split pectoral/pelvic at `s = 0.50`,
  median islands past `s = 0.85` are caudal. On a sevengill this is right. On a mesh with an unusual
  fin set, a dorsal fin far forward, or a caudal that survives the core threshold, it will produce
  confidently wrong names. `fins.json` and the contact strip exist so you can see it happen; fix it by
  re-running with a different `--core-radius-frac` or by pinning `fin_info[name]["parent"]`.
- **The chart stops at the peduncle.** The caudal lobes are transported rigidly by the terminal frame.
  That preserves the heterocercal upsweep through the de-bend, but it also means the tail is not
  independently charted, and a mesh whose caudal is fleshy enough to survive the core threshold will
  inflate `info["tail_width"]` and can flip the head-first call. Check the printed head/tail widths.
- **Nearest-segment ambiguity at corners.** For large `r` near a corner of a bent centerline the
  nearest-segment rule is genuinely ambiguous: about 4% of vertices (all fin tips) re-land on a
  neighbouring segment, shifting `s` by up to `r × (turn per segment)`. It scales exactly as
  `1/n_stations` (6.1e-3 / 3.0e-3 / 1.5e-3 / 7.5e-4 BL at `n = 64/128/256/512`) and is currently an
  order of magnitude below centerline-extraction error. At a much finer voxel pitch it becomes the
  next term to attack, with spline upsampling.
- **No self-contact handling, and the default C-start is capped so it does not need any.** Nothing
  detects or resolves the tail intersecting the head; the clip is kinematic, not simulated. The
  original default (`ESCAPE_PEAK_CURVATURE_UNCAPPED_PER_BL = 6.0 /BL`) turned the midline **276.6°**
  end to end — an O, not a C: the last spine joint came back to **0.175 BL** of the snout and the
  posed caudal lobe to **0.038 BL** of the posed head, i.e. the shipped default *always* violated
  this very caveat. `motion.ESCAPE_MAX_TOTAL_TURN_DEG = 180.0` now caps it, and
  `motion.escape_peak_curvature_cap()` turns that cap into the shipped peak of **3.696 /BL** in
  closed form (total turning is `peak × ∫E(s) ds` at the stage-1 extreme, and `∫E = head_ramp/2 +
  (1 − head_ramp) = 0.85`). Both are named module constants and an explicit
  `EscapeParams(peak_curvature_per_bl=...)` still gets whatever the caller asks for — the cap is on
  the *default*, not on the model.
- **LBS candy-wrapper — measured, and not where this file used to say.** Linear blend skinning
  collapses volume where two bones meet at a large angle. At the escape peak (frame 7, `t = 0.229 s`,
  the shipped 3.696 /BL default) on the demo mesh, over the **16 547** trunk edges (both endpoints
  labelled `body`):

  | quantity | shipped default | uncapped 6.0 /BL |
  |---|---|---|
  | worst edge ratio (posed / rest) | **0.155** | 0.115 |
  | trunk edges shrinking > 40% | **19** (0.115%) | 305 (1.843%) |
  | worst face area ratio (10 948 body faces) | **0.110** | 0.111 |

  And the collapse does **not** cluster at the peduncle. The worst 100 edges sit almost entirely in
  the **mid-trunk**, between `spine_04_trunk_02` and `spine_07_trunk_05` — 75 of 100 in those three
  segments (32 + 28 + 15), with only 12 in `spine_11_caudal_axis_1 → spine_12_caudal_axis_2`. That is
  where the C-start's curvature envelope has ramped fully in *and* the body is still thick, so the
  cross-section has the most volume to lose. Reproduce all of it with
  `python scripts/measure_lbs_artifact.py --uncapped`. Dual-quaternion skinning would fix it and is
  not implemented — glTF's basic skinning is LBS anyway, so a DQS rig would need the viewer to
  cooperate.
- **No muscle or skin-strain model.** The surface is carried by the bones and nothing else. Real shark
  skin is a stressed-fibre pressure vessel: Donley & Shadwick (2003) measured red-muscle longitudinal
  strain of **±3.9 / 6.6 / 4.8%** at three body positions at ~1 BL/s in leopard shark, and the derived
  skin figure is roughly 10–12% per flank. `motion.implied_skin_strain()` reports what the *kinematics*
  imply — 5.5 / 10.8 / 18.4% at `s = 0.25 / 0.50 / 0.75` — so the mismatch is at least visible and
  auditable, but nothing constrains the mesh to it.
- **The posterior 18.4% exceeding Donley's 4.8% is expected, not a bug.** A leopard shark is
  subcarangiform and peaks mid-body; a monotonically growing anguilliform envelope peaks at the tail.
  No hexanchiform kinematics has ever been published — every default in `motion.py` says "plausible,
  not measured" in its own docstring.
- **`--seconds` is nominal for looping modes.** `motion.make_clip` refuses to emit a loop that pops, so
  a looping clip is rounded to a whole number of tail beats (4 s at 0.9 Hz → 4.44 s). `escape`'s length
  comes from its own stage durations and ignores `--seconds` entirely.
- **Amplitudes are in units of the *chart* length, not the mesh length.** The spine spans snout →
  peduncle (0.727 of a 1.009-long mesh here), so a `0.11 BL` tail amplitude is `0.11 ×` the *charted*
  body. The caudal lobe, rigidly attached past the last spine joint, adds excursion beyond that. If
  you need a specific tail-tip amplitude in world units, solve `tail_amplitude_bl` against
  `motion.tail_tip_amplitude(...)["fk_bl"]`.
- **Not carried through the exporter**: vertex colours (`COLOR_0`) and any material channel beyond
  `baseColorFactor` / `baseColorTexture` / metallic / roughness. A UV-textured Meshy GLB is fine; a
  vertex-coloured input would lose its colour.

---

## 8. Note on the leopard shark project

The owner has a **separate leopard shark 3D project** whose conventions were **not available in this
session** — nothing here was written against it, and no attempt was made to guess at them. If you want
to port those conventions in (or export from here into that project), these are the interfaces to
align, in the order they bite:

1. **World axes and handedness.** This prototype is snout `+X`, tail `-X`, dorsal `+Z`, animal's left
   `+Y`, right-handed — the `create_shark_armature.py` convention. Set once in
   `mesh3d.canonical_frames()` and `mesh3d.straight_centerline()`; every other module reads it from
   there. A Y-up project needs `--up 0 1 0` on input **and** a change to those two functions if the
   *output* is to be Y-up too.
2. **`phi` zero and sign.** `phi = atan2(v·B, v·N)`, measured from dorsal `+Z` toward left `+Y`:
   `phi = 0` dorsal, `+90°` left flank, `180°` ventral, in `(-π, π]`. A project that measures `phi`
   from the ventral midline or in the opposite sense will mirror every fin label.
3. **Arc-length direction.** `s` runs head → tail, so the body tangent is `-X` and
   `motion.BODY_AXIS_YAW_SIGN = -1.0`. Flipping `s` flips that sign and the sign of every yaw curve.
4. **Joint names and count.** The spine is `skeleton_sevengill.SPINE_JOINTS` — 13 joints,
   `spine_00_cranium` … `spine_12_caudal_axis_2`. A leopard shark rig with a different chain length
   needs `rig.spine_arclength_fractions` and `motion.default_spine_fractions` re-derived together, and
   `motion.dct_energy_fraction` will refuse a non-13-joint spine because the DCT bending basis is
   defined on the schema's 12 segments.
5. **Arc-length fractions.** `rig.DEFAULT_PRECAUDAL_FRACTION = 0.78` plus
   `HEAD_PRECAUDAL_FRACTIONS` / `CAUDAL_REMAINDER_FRACTIONS`. These are `[UNVERIFIED]` and were chosen
   for monotonicity against the schema's pinned `MIDLINE_AXIS_FRACTIONS`; they are the single most
   likely thing to differ between two species' rigs. Both the precaudal fraction and the whole
   13-vector are overridable (`build_skeleton(spine_fractions=...)`).
6. **Fin naming vocabulary.** `mesh3d.FIN_LABELS` uses `pectoral_L` / `pectoral_R` (not `_left` /
   `_right`). Fin names are *not* fixed by the rig — whatever keys `fin_info` uses become the labels
   and the joint names — but `motion`'s fin drives resolve by longest **family prefix**, so
   `pectoral_left`, `pectoral_l` and `pectoral` all match the `pectoral` family and an unrecognised fin
   silently stays at identity.
7. **Skinning matrix semantics.** `rig.forward_kinematics` returns
   `W_j = W_parent · T(p_j) · R_j · T(-p_j)`, and **`W_j` *is* the skinning matrix** — identity
   rotations reproduce the rest mesh bit-for-bit. Note that
   `shark-pose-3d/shark_pose/model_3d/skinning.py` multiplies by `rest_inv = T(-p)` a *second* time and
   therefore subtracts the joint position twice; that bug was deliberately **not** mirrored here and is
   worth filing upstream.
8. **UV V-axis flip.** trimesh measures V from the bottom of the image, glTF from the top, and trimesh
   flips on load. `gltf_export.write_skinned_glb` therefore flips V on write, so
   load → de-bend → write → load is a UV identity. Any other GLB writer in the pipeline must match this
   or textures invert.
9. **Quaternion order.** glTF `(x, y, z, w)`, Hamilton product, unit norm enforced in float32 (the
   validator rejects quaternions off the unit sphere). Blender and scipy both use `(w, x, y, z)`
   internally in places.
10. **Mode vocabulary.** `cruise`, `turn`, `escape`, `rest`, `glide`, `breach`, `strike` — the first
    five implemented, the last two present for interface parity with
    `shark_pose/synthetic/blender/pose_sampler.py` and raising `NotImplementedError`.

---

## 9. Files

| file | contract |
|---|---|
| `mesh3d.py` | the chart: `load_mesh`, `extract_centerline_3d`, `tube_frames`, `tube_coords`/`tube_to_points`, `debend`/`rebend`, `detect_fins`. CLI: `python mesh3d.py IN.glb -o OUT.glb` |
| `synth.py` | the procedural sevengill with ground truth: `make_sevengill`, `c_curve`/`s_curve`, `bend`, `export_glb`, `preview_png` |
| `rig.py` | `build_skeleton`, `fin_info_from_detection`, `compute_weights`, `forward_kinematics`, `lbs`, quaternion helpers. Imports `skeleton_sevengill` from phase1b |
| `gltf_export.py` | `write_skinned_glb`, `validate_glb` (shells out to the Khronos validator) |
| `motion.py` | the swimming model: `WaveParams`, `MODE_CONFIG`, `curvature`, `joint_yaw_angles`, `make_clip`, plus diagnostics (`phase_report`, `dct_energy_fraction`, `implied_skin_strain`, `tail_tip_amplitude`). `python motion.py` prints a kinematics report |
| **`rig_sevengill.py`** | **the CLI**: `run_pipeline`, `solve_as_scanned`, `write_report`, `render_contact_strip` |
| **`demo.py`** | **the end-to-end synthetic demo**; writes `demo/` and prints every number in §6 |
| `scripts/measure_lbs_artifact.py` | the numbers in §7's LBS paragraph: worst trunk edge and face ratios at the escape peak and which spine segments they cluster in. `--uncapped` adds the pre-cap 6 /BL contrast |
| `tests/` | pytest — 177 tests, ~22 s, including `test_e2e.py` which runs the whole demo once and interrogates the GLB it writes |

Reuse, not vendoring: `frames.rotation_minimizing_frames` is imported from
`prototypes/02-centerline-chart`; `skeleton_sevengill` from `phase1b/p0-sevengill-schema`
(`$SEVENGILL_SCHEMA_DIR` overrides); `lbs` semantics and the `MODE_CONFIG` names mirror
`/home/user/shark-pose-3d`.
