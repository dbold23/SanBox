# RATIONALE — sevengill keypoint schema S1 and the serial-spine skeleton (P0-a)

Deliverables: `keypoints_sevengill_v1.yaml` (30 points), `skeleton_sevengill.py` (35 joints on a
13-joint serial spine) + `tests/` (44 tests), `patches/` + `APPLY.md`, `COMPAT.md`.

## Each design decision, and the evidence it comes from

**Author, don't port.** Direct answer 3 of `01-evidence-and-answers.md` tallies Schema v2 against
*Notorynchus*: 2 of 16 points do not exist, 1 is fundamentally ambiguous, 2 are anatomically
relocated, 2 degrade badly, 2 (including the tree root) are ill-posed on a bending body — roughly
9 of 16 compromised, about 5 transferring cleanly. A remap of a majority-broken schema is not a
remap. Counting off the delta table below: **16 of S1's 30 points are the image of a v2 point**
(several redefined, relocated or demoted; v2 id 12 `anal_fin_tip` splits into two S1 points, which
is why 15 surviving v2 rows yield 16 S1 ids), **14 are new**, and one v2 point
(`second_dorsal_tip`) is dropped outright.

**A small Type I/II anchor set, and fin insertions instead of fin tips.** The same answer: Type III
points slide and are formally non-homologous, and more than half of Schema v2 is Type III. Every
`*_tip` in v2 is Type III; every `*_origin` / `*_insertion` in S1 is a fin–body tissue junction,
Type I. That is the entire reason S1 reaches 16 Type I points against v2's 4–6, on a schema only
1.9× larger. Tips are retained as tier-3 shape channels because dorsal height and pectoral length
are still wanted measurements — they are simply never joints and never re-ID anchors.

The exact arithmetic, because it is easy to overstate: S1's 30 points are **19 Type I/II anchors**
(16 Type I + 3 Type II) **+ 4 Type III fin tips + 7 Type III midline semilandmarks**. The
**anchor** count is 19, not 23 — the 4 fin tips are shape/measurement channels and are excluded
from the anchor set by the same argument that demoted them. The 7 midline points are Type III by
construction and are a curve, not a correspondence.

**Seven gill slits, as two bracketing landmarks with dorsal and ventral termini.** `keypoints_16.yaml`
id 3 reads verbatim *"Back edge of last (5th) gill slit"*. Slits 2–6 are deliberately not
keypoints: their only correspondence-carrying property is their **order**, which bracketing slits 1
and 7 already fixes, and ten more near-identical collinear creases would degrade pose-model
association for nothing. The ordered antero-posterior sequence (slits 1–7 → pectoral → … → anal →
precaudal pit) is recorded in the yaml as its own block, because that is the cheap
deformation-tolerant correspondence mechanism the colon-registration literature supplies, and —
unlike Schema v2 — every point it names is fully defined for this species.

The sequence stops short of one claim. **Pelvic, dorsal and cloaca are deliberately NOT ordered
relative to one another.** They sit between pectoral and anal, and that bracket is all S1 asserts.
The scan's prose reads "seven gill slits → single dorsal → pelvics → cloaca"; an earlier draft of
this schema asserted pelvic → dorsal → cloaca, reversing the first pair; and that draft also
contradicted the yaml's own id-13 definition, which puts the cloaca on the mid-ventral line
*between the pelvic fin bases*, i.e. at the pelvic station rather than behind the dorsal. The
dorsal sits "over or behind the pelvics" — a range, not a point — and the offset moves with growth.
Any specific order among the three is **[UNVERIFIED]**, so the yaml carries an
`unordered_posterior_trio` block forbidding validators, matching costs and sequence-consistency
checks from enforcing one. If the order matters downstream it is a per-individual measurement, not
a schema constraint.

**A single dorsal, relocated, not renamed.** The fin sits far posterior over or behind the pelvics.
S1 parents `dorsal_fin_origin` to the same spine station as `pelvic_origin`, which is what makes the
relocation structural rather than cosmetic: any rest pose, bone length or shape prior learned at the
lamnid station is invalid, and the skeleton now says so.

**A deterministic 7-point trunk midline.** The campaign verdict makes flank matching the primary
Phase 1B arm ("identity distributed"; headless ≈ body in all four cells; the pre-registered
head advantage reversed to −9.1 once trained). Flank matching under midline rectification needs a
centerline, so the schema has to carry one. "Evenly spaced" is not a definition, so the yaml gives
the machine procedure (SAM 2 mask → `prototypes/02-centerline-chart/centerline.py`
`extract_centerline` → truncate at snout and precaudal pit → arc-length fractions k/8) and a human
fallback with its bias stated: the fallback ticks run along the dorsal outline, which is the convex
fibre on one side of a bend, so it inherits the ±3.9–6.6% longitudinal strain bracket measured by
sonomicrometry in a swimming leopard shark [SEARCH-grade; do not promote]. Fallback records are
flagged so they can be excluded from strain measurement.

Uniform fractions in the schema, non-uniform DOF downstream. The programme design calls for
allocating bending DOF non-uniformly (sparse anterior, dense posterior). That belongs in the
skeleton, which resamples the fitted spline; baking a kinematic assumption into a measurement
protocol makes the protocol unfalsifiable.

**Arc length on the centerline, chart terminating at the precaudal pit.** Both are spec decisions 1
and 4 of the recommended architecture: *s* on the centerline is what keeps the coordinate
bend-invariant (TubULAR integrates along the surface and concedes in its own README that surface
deformation changes the mapping), and aft of the precaudal pit the section compresses into a keel
and stops being star-shaped about the medial axis, so the tube assumption fails. The yaml's `chart`
block records `arc_length_origin: gill_slit_1_dorsal_origin` (spec: anchor at a gill slit,
re-anchor on the gill contours to kill drift) and excludes the fins from the identity surface —
which disposes of the second-dorsal problem instead of patching it.

**The nares are a keypoint because of field practice, not theory.** The operating La Jolla programme
has matched sevengills since 2010 on the freckle patch between the nares and the gill slits. The
campaign demotes that region to the secondary arm but does not delete it, so S1 carries
`naris_anterior_margin` and `mouth_rictus` explicitly to define and rectify that crop.

**No mirror augmentation.** Measured, not assumed: once same-session opposite-flank contamination
was removed, Melops cross-flank Rank-1 was **0.70%** zero-shot — left and right flanks behave as
separate identities. A mirrored left flank is a fabricated right flank; it injects false-positive
pairs and silently inverts both the `side` field and the antero-posterior direction the chart
depends on. `flip_idx` is identity because the schema is single-sided, and that is precisely the
trap — it would make a flip *index-safe* — so the flip is disabled outright (`fliplr: 0.0`).

**A serial spine, not a star.** `shark-pose-3d/core/skeleton.py` parents nearly everything to
`body_midpoint_dorsal`, leaving `gill_slit` and `caudal_notch` as the only two lateral-bending DOF
for the whole animal: adequate for a thunniform lamnid, wrong model class for an anguilliform
hexanchiform. `skeleton_sevengill.py` gives 13 serial spine joints / 12 axial segments, inside the
10–20 range the design calls for, compressible to `NUM_BENDING_MODES = 6` (top of the 4–6 bracket;
four eigenworm modes cover >95% of *C. elegans* posture variance, three PCs >90% of teleost larval
tail shape). The root moves from a Type III constructed mid-body point — defined against a body axis
that is a different curve every frame, and whose noise propagates into all downstream joints — to
`spine_00_cranium` in the rigid anterior region, adjacent to the chart's arc-length origin. The
axial chain continues **into the upper caudal lobe** rather than branching symmetrically to both
tips, because the tail is strongly heterocercal.

The bending basis shipped is an analytic DCT-II placeholder, labelled as such in the module: there
is no annotated sevengill midline corpus to fit an eigen-basis to. The projection API does not
change when it is replaced by PCA over real profiles. Only `n_segments - 1` = **11** modes exist,
not 12: DCT-II mode *k = n_segments* is evaluated at odd multiples of π/2 and is identically zero
at every sample point, so normalising it divides by float noise (measured `max|BBᵀ − I| = 0.607` at
`n_modes = 12`). `build_bending_basis` rejects that argument rather than returning a garbage row.
Reconstruction is lossy in both directions — global heading by design, and modes 7–11 by
truncation: measured relative L2 residual **1.8%** on a constant-curvature arc, **5.7%** on a
one-wavelength undulation, **15.4%** on a two-wavelength one, **66%** on white noise.

**The yaml's drawing graph is computed, not drawn.** `skeleton_edges` is the true contraction of
`KINEMATIC_TREE` onto the 30 keypoints — each keypoint attaches to the nearest ancestor joint that
is itself a keypoint — and the test suite recomputes it and asserts equality. This matters because
the hand-drawn predecessor had 29 edges forming a tidy spanning tree, of which **11 joined siblings
or cousins** (eye → gill slit 1, gill slit 1 → gill slit 7, snout → midline 01) and so drew as
bones relations the skeleton does not contain. The honest contraction is a **forest: 19 edges, 11
roots** (ids 0–9 and 23). It is disconnected because the entire cranial and branchial block hangs
off `spine_00_cranium`, `spine_01_branchial_1` and `spine_02_branchial_7`, none of which is the
image of a keypoint — the joint-tree root is unlabelled, so no connected contraction exists. That
is a real property of the skeleton and it is now visible rather than papered over; an annotation
tool that wants a connected head guide must either promote a cranial spine station to a keypoint or
draw its own guide lines and not call them bones.

## Annotator Schema-v2 delta

Against `sharkscarannotator/annotation/models.py:283` (`SHARK_KEYPOINT_SEQUENCE`, 16 points), which
`shark-morphometrics/shark_pose_project/config/keypoints_16.yaml` mirrors.

| v2 | v2 name | S1 | S1 name | verdict |
|---|---|---|---|---|
| 0 | `snout_tip` | 0 | `snout_tip` | **maps, degraded** — broad blunt head, no curvature maximum; Type II in name only |
| 1 | `eye_center` | 2 | `eye_center` | **maps clean** |
| 2 | `gill_slit_front` | 5 | `gill_slit_1_dorsal_origin` | **redefined** — slit 1 named explicitly, terminus not edge midpoint |
| 3 | `gill_slit_back` | 7 | `gill_slit_7_dorsal_origin` | **redefined** — v2 text says "5th gill slit"; factually wrong for Hexanchiformes |
| 4 | `pectoral_base_front` | 9 | `pectoral_origin` | **maps clean**, renamed to protocol register |
| 5 | `pectoral_fin_tip` | 11 | `pectoral_fin_tip` | **demoted** tier 2 → tier 3, shape-only, never a joint |
| 6 | `pectoral_base_back` | 10 | `pectoral_insertion` | **maps clean**, renamed |
| 7 | `dorsal_fin_tip` | 16 | `dorsal_fin_apex` | **relocated + demoted** to tier 3 |
| 8 | `dorsal_base_front` | 14 | `dorsal_fin_origin` | **relocated** — far posterior, over/behind the pelvics |
| 9 | `dorsal_base_back` | 15 | `dorsal_fin_insertion` | **relocated**, demoted to tier 3 |
| 10 | `second_dorsal_tip` | — | — | **dropped** — no second dorsal fin |
| 11 | `pelvic_fin_tip` | 12 | `pelvic_origin` | **replaced** — Type III tip → Type I insertion |
| 12 | `anal_fin_tip` | 17, 18 | `anal_fin_origin`, `anal_fin_insertion` | **replaced** — v2 tip is body-occluded and among the worst v2 points |
| 13 | `caudal_notch` | 19 | `precaudal_pit` | **renamed** to avoid collision with the subterminal notch; now the chart terminator |
| 14 | `caudal_upper_tip` | 21 | `caudal_upper_lobe_tip` | **maps, degraded** — long lobe, moves through the tailbeat, often out of frame |
| 15 | `caudal_lower_tip` | 22 | `caudal_lower_lobe_tip` | **demoted** tier 1 → tier 3 — lower lobe weakly developed, no repeatable referent |

**New in S1 (14):** `naris_anterior_margin`, `mouth_rictus`, `spiracle`,
`gill_slit_1_ventral_terminus`, `gill_slit_7_ventral_terminus`, `cloaca`,
`caudal_subterminal_notch`, and `midline_01`…`midline_07`. (`pelvic_origin`, `anal_fin_origin`,
`anal_fin_insertion` are counted above as replacements rather than additions.) 16 + 14 = 30.

**Dropped (1 from the annotator list, 4 from the `shark-pose-3d` variant):** `second_dorsal_tip`;
and additionally `second_dorsal_base`, `body_midpoint_dorsal`, `body_midpoint_ventral` where the
pose repo's variant carries them.

⚠️ **The three repos do not agree on what "Schema v2" is.** `sharkscarannotator` and
`shark-morphometrics` share one 16-name list (`gill_slit_front/back`, `dorsal_base_front/back`,
`pectoral_base_back`, `second_dorsal_tip`). `shark-pose-3d/core/skeleton.py:15-32` carries a
*different* 16 — one `gill_slit`, `second_dorsal_base` **and** `second_dorsal_tip`,
`body_midpoint_dorsal`, `body_midpoint_ventral`, and no `pectoral_base_back` or `dorsal_base_back`.
Any migration that assumes a single canonical v2 will silently mis-map. This was found while
building the delta table and is not documented anywhere in the repos.

## Open questions for the PI

1. **Sign-off.** S1 is a v1 draft and has not been reviewed by anyone who has handled the animal.
   The named reviewer is David A. Ebert (Pacific Shark Research Center, Moss Landing Marine
   Laboratories, ~20 minutes from CSUMB). Do not spend annotator hours against S1 first.
2. **Is 30 points too many?** The lab's v2 model reaches Pose mAP50 0.578 on 16 *distinctive*
   landmarks. Seven collinear, visually near-identical midline points are a known-hard association
   case and no source addresses SLEAP/DeepLabCut accuracy on them. `N_midline` is a schema
   hyperparameter — it can drop to 4 without touching a single anchor id. Measure per-point PCK on
   the first batch before committing.
3. **Fin stations are provisional.** No published *N. cepedianus* fin-station proportions were
   retrieved [UNVERIFIED]. `KINEMATIC_TREE` currently parents pelvic/dorsal/cloaca to
   `spine_07_trunk_05` (t = 0.625) and anal to `spine_08_trunk_06` (t = 0.750) from qualitative
   anatomy alone. Re-derive from the first ~50 annotated frames; each is a one-line edit.
4. **Can annotators count to seven in the water?** If the seventh slit is not reliably countable at
   working visibility, id 7 degrades to "posteriormost discernible slit, with the count recorded",
   which changes its Bookstein type from I to III and weakens the ordered-sequence argument.
5. **Which `body_zone` atlas?** `config/scars.yaml` defines white-shark body zones against
   `second_dorsal_base`/`second_dorsal_tip` (lines 159, 166). The zone atlas cannot be reused as-is
   and needs its own P0 decision; S1 does not address it.
6. **Does `skeleton_sevengill.py` land in `shark-pose-3d` at all?** Delivered as a design artifact,
   not a patch. If the programme descopes the 3D leg, the serial spine is still wanted — the
   Phase 1B rectifier needs the midline chain — but it would live next to the rectifier rather than
   in SharkSMPL.
7. **P0-d still gates the premise.** Ontogenetic stability of the speckling is unmeasured for this
   species, and sevengills expand several-fold linearly between birth and maturity. No schema fixes
   a pattern that changes.
