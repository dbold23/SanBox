# Prototype 04 — Sevengill de-bend, rig and swim

A **mesh-agnostic** pipeline that takes a photogrammetric GLB of an elongate animal, recovers its 3D
centerline, **straightens it to a canonical rest pose**, rigs it with the phase1b sevengill skeleton,
and writes one Khronos-clean skinned GLB carrying real swimming animations.

```
load_mesh → extract_centerline_3d → tube_frames → tube_coords → debend
          → detect_fins → build_skeleton → compute_weights → make_clip → write_skinned_glb
```

De-bending preserves UVs, material and texture, so the straightened body still wears the scan's own
photo-projected skin. §9 cashes that in: `texture_identity.py` reads that skin into prototype 05's
canonical chart, de-lights it, and fits **catalogue individual #0** to it — the real animal enters the
synthetic corpus with no photo pipeline at all.

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

#### The stand-in's fins are solid

The fins used to be single-sided zero-thickness plates, which is not what the Meshy scan has and not
what a rig has to hold together. Each fin is now a **closed two-sided loft** of a symmetric
NACA-00xx section: a `sqrt(x)` nose, maximum thickness `FIN_THICKNESS_RATIO = 12%` of the local chord
at the root, thinning to a tip lid, with small facets closing the nose and the tail so that no two
section vertices coincide.

It stays **welded** — one mesh-graph component, nothing floating, nothing orphaned — because the root
loop is a *slit cut into the body grid itself*, not a patch stuck on top of it:

1. The fin's root column is split into two **lips**, pushed apart in `phi` by the local section
   half-thickness. At `phi = 0` the duplicated UV-seam column already supplies the two; everywhere
   else one lip keeps the original vertex id and the other is a new one, and the body quads on that
   side are rewired to it.
2. Its `FIN_WARP_MARGIN`-plus neighbours are **slid outward** by a linear re-spacing of the ring, so
   the quads between lip and neighbour stay evenly wide instead of folding over. `v` follows the same
   displacement, so the texture stays glued to the surface rather than shearing.
3. One triangle closes the notch the slit leaves at each end of the root (at the ends of the body,
   the cap apex plays that part).
4. The blade lofts outward as closed sections and is capped by a lid across the outermost one — no
   extra vertices, and every edge of that section used exactly once.

`trimesh.is_watertight` is `False` on the mesh as built, for exactly one reason, unchanged by any of
this: the body carries a **duplicated column at the UV seam** so the texture wraps without a
degenerate parameterisation, and those pairs are topologically apart though geometrically together.
Merge them by position and the mesh is a closed, consistently wound, **genus-0 solid** — no boundary
edge, no edge shared by three faces, positive volume. `test_the_solid_fins_close_into_one_manifold_solid`
asserts all of it, including that the coincident-vertex count is the seam **and nothing else** (the
two fins that sit on the seam pull it open over their own stations, so it is shorter than the station
count by exactly their length).

Measured by `synth.fin_section_report`, on the mesh as built rather than as asked for:

| fin | root chord | root thickness | % of chord | nose facet | trailing edge | tip | thinnest |
|---|---|---|---|---|---|---|---|
| pectoral_L / _R | 0.07207 | 0.00852 | **11.8%** | 2.18% c | 0.58% c | 0.00120 | 3.0e-4 |
| pelvic_L / _R | 0.06486 | 0.00808 | **12.5%** | 2.40% c | 0.51% c | 0.00122 | 3.0e-4 |
| dorsal | 0.07928 | 0.01006 | **12.7%** | 2.53% c | 0.49% c | 0.00116 | 3.0e-4 |
| anal | 0.05766 | 0.00733 | **12.7%** | 2.51% c | 0.50% c | 0.00109 | 2.9e-4 |
| caudal_upper | 0.07928 | 0.01028 | **13.0%** | 2.71% c | 0.48% c | 0.00150 | 3.0e-4 |
| caudal_lower | 0.05766 | 0.00727 | **12.6%** | 2.50% c | 0.50% c | 0.00109 | 2.9e-4 |

The measured ratio runs a little over the requested 12% because the two lips are opened by an *angle*
sized at the fin's mid-chord radius and the body is thicker at the front of the chord than the back.
The nose facet is 3.8–5.6× the trailing-edge facet: the leading edge is round and blunt, the trailing
edge thin — but never zero, floored at `FIN_MIN_HALF_THICKNESS = 1.5e-4 BL` per side so no section
degenerates into a knife edge that float32 export could close.

Nothing downstream needed a change. `FIN_SPECS` keys and the `metadata["fins"]` contract
(`u0`, `u1`, `phi_root`, `span`, `station_range`, `insertion_centroid`) are what they were; the entry
gained `volumetric`, `root_chord`, `root_thickness`, `root_thickness_ratio` and `section_pairs`
(the `(span row, chord station, 2)` pairing of each blade vertex with its twin, which is what
`fin_section_report` measures and what the label test checks). `vertex_labels` still marks exactly the
blade: both sides of every lofted section carry the fin's name, and the root lips — body-grid
vertices at body radius — stay `body`, which is what `detect_fins` documents and depends on.
`make_sevengill(solid_fins=False)` still builds the old sheets, which is what the A/B numbers below
are measured against.

One thing *upstream* did need a change, and it was a latent bug rather than a consequence.
`mesh3d.estimate_roll` drops whole stations touched by a detected fin — a fin's root row keeps the
`body` label by design and is then the fattest thing at its station, which would make the roll fit
track the fins instead of the dorsal ridge. It was blocking only the **detected blade's** station
range, and the fin runs a station or two further at each end. Those leftover end stations, one per
fin, were already choosing between the left and the right blade of a bilaterally symmetric animal at
equal radius to 1e-6 — a coin flip, and `np.unwrap` turns one flip into a `pi` offset over the whole
tail of the fit. Solid fins flipped the coin: the 90° roll fixture read **285°**. Sweeping
`FIN_THICKNESS_RATIO` across 0.10–0.14 flips it either way, on the old sheets as well as the new
solids, so it was never a property of the fins. `_ROLL_FIN_STATION_PAD = 2` pads the block, and the
fit reads **82°** for every one of those configurations, sheets and solids alike.

One case cannot have a slit of its own: two fins asking for the **same strip of skin**. Only a
hand-built `FIN_SPECS` reaches it — `test_mesh3d` stacks a second caudal lobe on the first to test
island merging — and the later fin then falls back to the old zero-thickness sheet and records
`volumetric: False`, rather than cutting a second slit through the first fin's lips and tearing the
surface open. Every fin of the shipped stand-in is volumetric.

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

Procedural sevengill: **7150 verts, 14120 faces, 8 fin islands**, UV-textured, bent into a 120° C
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
| all vertices | **0.494% BL** (0.63 px) | 3.500% BL (4.48 px) |
| body only | **0.157% BL** (0.20 px) | 0.674% BL (0.86 px) |

**Extracted centerline vs the exact C-curve**: mean `0.00051 BL` (0.07 px), max `0.00159 BL` (0.20 px).

**Fin labelling vs construction truth** — 8/8 islands found, **100% purity on every island**, and
**zero** body vertices given a fin label:

| fin | found | truth | purity | recall | (recall, sheet fins) |
|---|---|---|---|---|---|
| anal | 108 | 126 | 100.0% | 85.7% | 85.7% |
| caudal_lower | 144 | 162 | 100.0% | 88.9% | 92.6% |
| caudal_upper | 336 | 336 | 100.0% | 100.0% | 100.0% |
| dorsal | 182 | 216 | 100.0% | 84.3% | 88.0% |
| pectoral_L | 176 | 220 | 100.0% | 80.0% | 80.0% |
| pectoral_R | 176 | 220 | 100.0% | 80.0% | 80.0% |
| pelvic_L | 120 | 160 | 100.0% | 75.0% | 75.0% |
| pelvic_R | 120 | 160 | 100.0% | 75.0% | 77.5% |

Recall is 75–100% because a fin's **root** vertices sit at body radius and stay labelled `body` by
design — the label marks the protruding blade; `station_range` and `insertion_centroid` are what the
rig binds the base with.

Solid fins doubled the vertex count of every blade without moving a single one of them in `r`, so
purity is unchanged (still exactly 100%, still zero body vertices given a fin label) and recall moves
by at most 3.7 points — down on `caudal_lower` and `dorsal`, unchanged elsewhere. The mechanism is
worth naming because it is the detector's one real soft spot: the per-station radius envelope is a
percentile, the terminal chart station collects **everything** that overhangs the chart (both caudal
lobes, 448 of its 549 vertices now against 224 of 323 before), and a station whose population is
mostly blade has a percentile that sits in the blade. The margin gap is the difference between the
threshold and the fin base, and it narrowed at that one station and nowhere else.

**Texture / topology**: faces identical after de-bend `True`; UVs identical after de-bend `True`; UV
error after GLB write + reload `0.00e+00`. (The UV check is against an absolute `1e-6`, not
`np.allclose`'s default relative tolerance: the GLB stores UVs as float32 and *v*-flipped, so a `v`
near 0 comes back with ~6e-8 of absolute error and a relative error far above `rtol=1e-5`. The slit
lips put `v` values that close to 0 on the mesh for the first time; the quantisation was always
there.)

**Rig**: 29 joints = 13 schema spine + 16 fin joints. Weights `7150 × 29`, ≤ 3 influences per vertex
(mean 1.89 — two spine joints everywhere, plus the fin root on the 3 rings of body vertices inside
each fin-base blend), rows sum to 1 to `2.2e-16`. Fin-base seam edges change length by at most
**0.45% BL** under the cruise clip, against **1.94% BL** with the blend off (346 seam edges now, not
174: the seam is the boundary of a two-sided blade). Fin parenting resolved
by geometry:

| fin | spine parent |
|---|---|
| pectoral_L, pectoral_R | `spine_05_trunk_03` |
| pelvic_L, pelvic_R | `spine_09_trunk_07` |
| dorsal | `spine_10_precaudal` |
| anal | `spine_11_caudal_axis_1` |
| caudal_upper, caudal_lower | `spine_12_caudal_axis_2` |

**`as_scanned`** (same `BL = 0.8000`, same rigid mean offset removed): spine joints land on the
scanned centerline to max `0.00117 BL` (0.15 px); skinned surface vs the scan RMS `0.604% BL`, max
`2.807% BL`.

**Cruise kinematics**: 0.90 Hz, wavelength 0.90 BL, prescribed tail amplitude 0.110 BL, posed
0.103 BL (0.94×). DCT bending-mode energy **0.976 in 4 modes, 0.996 in 6**. Implied longitudinal skin
strain: anterior **5.5%**, mid-body **10.8%**, posterior **18.4%** (literature bracket 3.9–13.0%).

**Escape kinematics**: peak curvature **3.696 /BL** (`ESCAPE_PEAK_CURVATURE_UNCAPPED_PER_BL = 6.0`
capped by `ESCAPE_MAX_TOTAL_TURN_DEG = 180`), total turning of the midline **170.4°** on this rig
(176.6° on `motion.default_spine_fractions()`), last spine joint **0.580 BL** from the snout, posed
caudal lobe **0.492 BL** from the posed head.

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

Total demo wall time **3.1 s**. Full prototype test suite: **236 tests, 75 s**.

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
  the shipped 3.696 /BL default) on the demo mesh, over the **17 035** trunk edges (both endpoints
  labelled `body`):

  | quantity | shipped default | uncapped 6.0 /BL |
  |---|---|---|
  | worst edge ratio (posed / rest) | **0.204** | 0.114 |
  | trunk edges shrinking > 40% | **26** (0.153%) | 315 (1.849%) |
  | worst face area ratio (11 239 body faces) | **0.159** | 0.131 |

  And the collapse does **not** cluster at the peduncle. The worst 100 edges sit almost entirely in
  the **mid-trunk**, between `spine_04_trunk_02` and `spine_07_trunk_05` — 75 of 100 in those three
  segments (31 + 31 + 13), with only 15 in `spine_11_caudal_axis_1 → spine_12_caudal_axis_2`. That is
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

## 9. Real texture as individual #0

The Meshy GLB is a photograph wrapped around a mesh. Everything the rest of this prototype does —
centerline, de-bend, rig, animate — treats that photograph as freight: `debend` moves vertices and
leaves `faces`, `visual` (UVs, material, image) and `metadata` alone, which is the whole reason the
straightened rest pose still wears the animal's own skin. `texture_identity.py` cashes that in.

**The claim: the real animal enters the catalogue as individual #0 with no photo step.**
Prototype 05 builds synthetic identities in a canonical `(s, phi)` chart and can also *copy* one —
`pattern.copy_from_chart` takes a chart image of a real animal's speckling and returns an ordinary
`Individual`. The missing piece was always "where does that chart image come from", and the assumed
answer was a photo pipeline: detect the animal in an image, rectify it to the chart, read off the
freckles. **That pipeline is not needed here.** The scan already carries the photograph, already
registered to the surface, and the de-bent mesh already has an exact `(s, phi)` for every vertex. So
the chart is *read off the mesh* — `bake.mesh_texture_to_chart` — and `copy_from_chart` fits a spot
table to it. No detector, no rectification, no annotation, no scale ambiguity. Individual #0 is the
scanned animal, and from that point on it is an ordinary catalogue entry: `drift.resight` ages it,
`pattern.render_chart` re-renders it, and it can be baked back onto the same body as a new GLB.

```
IN.glb → load_mesh → extract_centerline_3d → tube_coords → debend        (§2, unchanged)
       → chart_coords            04's (s, r, phi)  →  05's (s∈[0,1], φ∈[-π,π))
       → mesh_texture_to_chart   the scan's own skin, in chart space      chart_real.png
       → de-light                bake's low-frequency luminance divide    chart_delighted.png
       → copy_from_chart         individual #0                            individual0.json
       → resight ×N / randomize ×M → render_chart → bake_chart_to_texture → GLB + validator
```

```
python texture_identity.py --glb IN.glb --out DIR [--n-resights 4 --years 3] [--n-random 3] [--seed 0]
```

With no `--glb` it runs on `synth.make_sevengill(textured=True)`, whose procedural skin stands in for
the Meshy photo texture. Outputs in `DIR`: `chart_real.png`, `chart_delighted.png`, `chart_skin.png`,
`individual0.json` (the spot table), `individual0.glb`, `resight_NN.glb`, `random_NN.glb`,
`textures/*.png`, `contact_sheet.png` (real | de-lit | individual #0 | 2 resights | 2 randoms, each as
chart **and** as a side view of the straightened body wearing that texture) and `summary.json`.

### De-lighting is what keeps the scan's shadows out of the identity layer

A photo-projected texture has the capture's lighting **baked into the albedo**: a bright dorsal
highlight where the key light hit, a dark ventral falloff, a soft shadow cast by the first dorsal fin
onto the flank. Nothing in the file distinguishes those from pigment. Two things go wrong if they are
left in.

1. **Individual #0 is fitted from a threshold.** `copy_from_chart` segments the chart at an Otsu cut.
   A shading gradient moves the whole ventral half of the chart across that cut, so the "spots" it
   finds on one flank are shadow and the spots it misses on the other are real. The identity you get
   is a portrait of the lighting rig.
2. **Every synthetic sibling inherits the same lighting.** The random individuals are baked onto a base
   albedo taken from this same scan. If that base still carries the capture's shading, then *every*
   animal in the corpus shares one lighting fingerprint — a perfectly stable, perfectly
   identity-free feature. That is exactly the shortcut this programme exists to avoid (prototype 01's
   whole finding was that a model will happily learn the confound instead of the animal).

So the texture is de-lit before anything reads identity from it. The estimator is **not**
re-implemented here: `delight_texture` calls `bake.bake_chart_to_texture` with a *unit* multiplier
chart, so the only thing that runs is the `base_albedo` de-light a normal 05 bake does anyway — the
same spot-robust Legendre×Fourier fit of log-luminance on the surface, the same `[0.25, 4]` gain
clamp, the same warnings. Fitting **in chart space** rather than in UV space is load-bearing: a blur in
the atlas averages texels that are adjacent in the image and far apart on the animal, so it invents a
discontinuity exactly at the atlas seam.

Two textures come out of that step and the difference matters:

| | shading | freckles | used for |
|---|---|---|---|
| `albedo` | removed | **kept** | reading individual #0 |
| `skin` | removed | **removed** | the base a *synthetic* individual is baked onto |

`skin` is `albedo` with the identity layer flattened by the same spot-robust smooth
(`bake.estimate_lowfreq_luminance`, whose asymmetric reweighting exists precisely to keep dark marks
out of the smooth level). Baking a random individual onto `albedo` instead would give every synthetic
animal the real one's freckles *plus* its own.

### What de-lighting removes, and what it cannot

**It removes** anything smoother than the cutoff — `sigma_s = 0.10` body lengths, three girth
harmonics, a 4th-order polynomial fore-aft. That covers the terms that actually dominate a photo
capture: the dorsal-to-ventral key-light gradient (a pure `cos φ`, which the basis represents
*exactly* and therefore removes completely), a fore-aft falloff from a light closer to the head than
the tail, and any soft ambient occlusion in the flanks. Measured on the synthetic stand-in, the
low-frequency swing of the surface luminance drops from **1.132 to 0.0019** — a 99.8 % reduction, and
the dorsal/ventral luminance ratio goes from 0.888 to 1.007 — while the spot-scale contrast of the
chart only falls from **0.0466 to 0.0363**, i.e. **78 % of the identity layer survives**. The gain
clamp did not bind on a single texel.

**It cannot remove, and you must not pretend otherwise:**

- **Hard shadow edges.** A cast shadow with a crisp boundary — a fin edge, a hand, the tank rim — is
  *high* frequency across that boundary. The divide is a smooth field; the edge passes straight
  through it and lands in the identity layer as a long dark mark. `copy_from_chart` will happily fit
  a spot (or a scar-shaped blob) to it, and no downstream stage can tell it from pigment.
- **Specular highlights.** A wet shark is glossy. A specular lobe is *view-dependent* and often
  clipped to white; dividing by a smooth luminance estimate cannot recover albedo under a blown-out
  highlight (the gain is clamped at 4, and 4 × 0 is still 0), and a small tight highlight is not
  low-frequency in the first place. It survives as a bright patch that the threshold reads as
  *unmarked skin*, punching holes in the pattern.
- **Coloured illuminants.** The gain is a scalar applied to all three channels, so a green-cast
  underwater capture stays green-cast.
- **Countershading.** Sevengill dorsum-dark / ventrum-pale albedo is *itself* a low-frequency term and
  is flattened along with the shading. This is deliberate — it belongs to the species, not to the
  individual or to the capture — and 05 puts it back in chart space after the bake. But it does mean
  the de-lit texture looks flat, and that is correct, not a bug.

**This is the single biggest risk on this path.** The de-bend, the chart and the bake are arithmetic
with known error bars; the identity you get out is only as clean as the assumption that the Meshy
texture's lighting is smooth. Before trusting individual #0 from a real scan, **look at
`chart_delighted.png`**: any long, sharp, dark streak that follows a fin outline is a shadow edge, and
any blown-white patch is a specular. Both must be masked (add a region to the exclusion geometry) or
re-captured, because no amount of de-lighting will fix them. A genuinely 3D de-lighting — inverse
rendering from several views with a known light — beats this whenever more than one view of the same
surface exists; this module does not have that and does not pretend to.

### The convention bridge (04 → 05)

The two prototypes agree on the zero and the sign of `phi` and disagree on the half-open end. 04 uses
`(-π, π]`, so **ventral is `+π`**; 05 uses `[-π, π)`, so **ventral is `-π`**. That is not a corner case:
it is every vertex of the ventral seam column of a UV-unwrapped tube. `chart_coords` converts with
`bake.wrap_to_pi` and nothing else — identity on `(-π, π)`, `+π ↦ -π` — and the tests assert the seam
value explicitly rather than trusting the reader.

`s` is the other half. 04's `TubeCoords.s` is an arc length **in metres along the chart**, and it
deliberately leaves `[0, L]`: `extract_centerline_3d` stops at the peduncle, so the caudal fin
overhangs the far end (measured: `s` runs to 1.018 on a 0.731 m chart). 05 wants a fraction with
`0 = snout tip`, `1 = caudal terminus`. The default `normalize="extent"` rescales the *observed* range
onto `[0, 1]`, which puts both anatomical ends where 05's convention says they are.
`normalize="chart"` divides by the chart length and clips, which is the recipe in
`bake.bake_chart_to_texture`'s docstring; it is exact on the body but folds the whole heterocercal tail
onto `s = 1`, so it is not the default here.

Layouts are transposes of each other and both are in play: **bake layout** `(n_s, n_phi)` for
`bake.py`, **pattern layout** `(H_phi, W_s)` for `render_chart`, `build_exclusion_mask` and
`drift.similarity`. Every function in `texture_identity.py` states which one it takes. Three of
`copy_from_chart`'s auto-detections are pinned rather than trusted, because each one fails *quietly*:
`axis_order="s_major"` (auto guesses from the aspect ratio), `chart_semantics="albedo"` (auto guesses
from the mean, and a de-lit sevengill chart sits at 0.46 — below the 0.5 cut, so auto would read it as
a darkness map and **invert the pattern**), and `regions` (passed explicitly so individual #0 carries
the same exclusion geometry as every randomised individual).

### Measured (synthetic stand-in, seed 0, 256-texel texture, 240 × 128 chart)

Measured on `synth.make_sevengill(textured=True)` as it stands in this repo — 7150 vertices,
14120 faces. `synth.py` is a fixture, not a contract: if it changes, re-run
`python texture_identity.py --out DIR --tex-size 256` and read `summary.json`, which carries every
number below. The tests assert on ratios, not on these absolutes.

| quantity | value |
|---|---|
| straighten (load → centerline → de-bend → chart → fins) | 0.85 s, all 8 fins found |
| chart span | `s` runs −0.038 → 0.977 m on a 0.731 m chart, i.e. the caudal fin overhangs by 0.25 BL |
| low-frequency luminance swing, before → after de-lighting | 1.132 → 0.0019 |
| dorsal/ventral luminance ratio, before → after | 0.888 → 1.007 |
| chart spot-scale contrast, real → de-lit | 0.0466 → 0.0363 (78 % kept) |
| chart spot-scale contrast, de-lit → flattened `skin` base | 0.0363 → 0.0009 (2 % left) |
| gain clamp bound on | 0 % of texels |
| chart cells actually measured (fin texels dropped) | 85 % — the rest are `NaN`, i.e. *unobserved* |
| individual #0 | **55 spots** from 61 connected components, Otsu threshold 0.595 |
| render → re-fit round trip | 36 spots, `recoverable_spot_count` = 36 (exact) |
| `drift.similarity` to resights at 9 / 18 / 27 / 36 months | 0.973 / 0.913 / 0.841 / 0.770 |
| `drift.similarity` to 3 random individuals | −0.009 / 0.007 / 0.028 |
| same pipeline on a C-120 bend of the same mesh | same identity recovered (similarity > 0.6, spot count within 20 %) |
| same pipeline on `demo/sevengill_synthetic_bent.glb`, unmodified | runs, all GLBs validate |
| whole run (1 + 4 + 3 GLBs, all validated) | 13.5 s |

The C-bend rows are the load-bearing ones: **the C-curve is a pose, not an identity.** Feeding the
straight mesh and a 120° bend of that same mesh through the same unmodified pipeline recovers the same
animal, because de-bending happens before anything reads the skin. (The comparison is run against a
bend of the *current* mesh rather than against the `demo/` GLB on disk, which is a build artefact that
can lag `synth.py`; the demo GLB is still exercised, as the input a real C-posed file would be.)

**Fin texels are dropped from the chart READ, not from the bake.** `mesh_texture_to_chart` splats every
covered texel into the chart cell at its own `(s, φ)` — and a fin blade has an `(s, φ)` too: the body
cell it happens to project onto. Left alone, a pelvic fin's albedo is averaged into the skin beneath
it. `body_texel_alpha` builds a per-texel alpha from `detect_fins`' per-vertex labels (a face is kept
only if all three vertices are `body`, so the root ring goes with the blade) and hands it to the read as
the texture's alpha channel, which `mesh_texture_to_chart` already honours. The cells the fins were
hiding then come back as `NaN` — **unobserved**, which is the truth — instead of carrying a fin's
colour. Baking is unaffected: a fin should wear the chart value at its own `(s, φ)`.

Every GLB the module writes goes through the Khronos validator with **zero errors**, keeps the
straightened mesh's vertex count, face array and UVs (round trip to < 1e-6), and differs from its
siblings only in the base-colour image. `write_textured_glb` uses trimesh's own GLB writer rather than
`gltf_export.write_skinned_glb` — there is no skeleton here, and trimesh already round-trips the UV
V-axis flip that §8's item 8 warns about (measured: < 1e-6 over the whole atlas).

### Limits of this path, beyond the lighting

- **Spot counts are not identities.** `copy_from_chart` merges touching marks and drops sub-pixel
  ones; the fitted radius is systematically ~10–15 % small because a threshold cuts a soft-edged mark
  inside its nominal boundary. The `provenance` dict records the threshold and both counts so the loss
  is visible rather than assumed away.
- **Chart resolution caps what can be recovered.** 240 × 128 is isotropic in 05's scaled chart metric
  at `phi_scale = 0.085`; a finer chart than the texture supports holds interpolation, not measurement
  (`mesh_texture_to_chart` returns the per-cell coverage so this is checkable).
- **Working texture size is capped at 1024** (`--tex-size`). Rasterising a 12 k-face atlas at 4096
  costs minutes and feeds a 240 × 128 chart; the cap is reported in `summary.json`.
- **The exclusion geometry is the schema's, and its station values are `[UNVERIFIED]`** — see
  `exclusions.default_stations`. Eyes, nares, the mouth/jaw band and the seven gill slits are excluded
  from both the fit and every render; if those stations are wrong, identity is being read off the
  wrong band of the animal.
- **One texture, one individual.** Nothing here fuses several scans of the same animal, and a single
  photo-projected texture only ever sees the surface the cameras saw. Chart cells the atlas never
  covered come back as `NaN` and are treated as *unobserved*, not as unmarked skin — but they are still
  a hole in individual #0.

---

## 10. Files

| file | contract |
|---|---|
| `mesh3d.py` | the chart: `load_mesh`, `extract_centerline_3d`, `tube_frames`, `tube_coords`/`tube_to_points`, `debend`/`rebend`, `detect_fins`. CLI: `python mesh3d.py IN.glb -o OUT.glb` |
| `synth.py` | the procedural sevengill with ground truth: `make_sevengill` (solid, welded, two-sided fins; `solid_fins=False` for the old sheets), `fin_section_report`, `c_curve`/`s_curve`, `bend`, `export_glb`, `preview_png` |
| `rig.py` | `build_skeleton`, `fin_info_from_detection`, `compute_weights`, `forward_kinematics`, `lbs`, quaternion helpers. Imports `skeleton_sevengill` from phase1b |
| `gltf_export.py` | `write_skinned_glb`, `validate_glb` (shells out to the Khronos validator) |
| `motion.py` | the swimming model: `WaveParams`, `MODE_CONFIG`, `curvature`, `joint_yaw_angles`, `make_clip`, plus diagnostics (`phase_report`, `dct_energy_fraction`, `implied_skin_strain`, `tail_tip_amplitude`). `python motion.py` prints a kinematics report |
| **`rig_sevengill.py`** | **the CLI**: `run_pipeline`, `solve_as_scanned`, `write_report`, `render_contact_strip` |
| **`demo.py`** | **the end-to-end synthetic demo**; writes `demo/` and prints every number in §6 |
| `scripts/measure_lbs_artifact.py` | the numbers in §7's LBS paragraph: worst trunk edge and face ratios at the escape peak and which spine segments they cluster in. `--uncapped` adds the pre-cap 6 /BL contrast |
| **`texture_identity.py`** | **the scan's own skin → catalogue individual #0** (§9): `chart_coords` (04 → 05 convention bridge), `straighten`, `delight_texture`, `fit_individual`, `resight_series`, `random_individuals`, `bake_individual`, `write_textured_glb`, `render_side`, `contact_sheet`. CLI: `python texture_identity.py --glb IN.glb --out DIR`. Imports prototype 05 read-only through a `sys.path` shim (`$SEVENGILL_P05_DIR` overrides) |
| `tests/` | pytest — 234 tests, ~80 s, including `test_e2e.py` (runs the whole demo once and interrogates the GLB it writes) and `test_texture_identity.py` (runs the identity pipeline twice, on the straight mesh and on the C-curved GLB) |

Reuse, not vendoring: `bake` / `pattern` / `drift` / `exclusions` / `render` are imported from
`prototypes/05-synthetic-identities` **read-only** (nothing under it is written);
`frames.rotation_minimizing_frames` is imported from
`prototypes/02-centerline-chart`; `skeleton_sevengill` from `phase1b/p0-sevengill-schema`
(`$SEVENGILL_SCHEMA_DIR` overrides); `lbs` semantics and the `MODE_CONFIG` names mirror
`/home/user/shark-pose-3d`.
