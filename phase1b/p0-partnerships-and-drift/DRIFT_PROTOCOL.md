# DRIFT_PROTOCOL — P0-d pre-registered measurement plan

**Question.** Does *Notorynchus cepedianus* speckling stay matchable over years, and how fast does the
match decay with elapsed time and with growth?

**Why it is P0.** `docs/sevengill-canonical-reid/03-candidate-approaches.md` P0-d: there is **no
published study** of whether sevengill speckling is stable through growth, and the nearest transferable
evidence is a warning (epaulette sharks: ~86% for mature animals, juvenile/neonate patterns
**unreliable**). The whole programme assumes a durable pattern. This measurement tests the assumption
on data that already exists, before any sevengill field season is funded.

**Status of this document.** It is a **pre-registration**. It is written and frozen *before* any archive
export is requested, so that no threshold can be chosen after seeing a number. Deviations go in
§10 with a reason and a date; they are never silently edited in.

**What it costs.** One data-sharing agreement, one export, a laptop, and the code that already exists in
`prototypes/01-melops-ablation/`. No new model, no new training run, no field time.

---

## 1. The data we are asking for

The Ocean Sanctuaries Sevengill Shark ID Project (La Jolla, running since 2010, now on
Sharkbook/Wildbook) is the only long-run California sevengill photo archive; matching is on the
nares-to-gill freckle patch, with human-confirmed matches on top of Wildbook's pattern-matching
algorithms [SEARCH]. Sharkbook currently runs **MiewID** and carries a per-annotation **left/right
viewpoint** field — a July 2026 Wild Me forum thread reports that automatic left/right detection
regressed after a platform upgrade and now needs manual edits, so **expect missing or hand-entered
viewpoints in any 2026 export** [SEARCH]. That is a data-cleaning fact we plan around (§3, side rule),
not a blocker.

The request (see `OUTREACH_OCEAN_SANCTUARIES.md`) is for the **confirmed-resight subset**: every
encounter belonging to an individual with ≥2 human-confirmed encounters, plus the unmatched-queue
counts (§7.7) and, where the platform records it, the **discovery route of each confirmed link** —
manual versus algorithm-suggested (§4 rule 8, §7.11). Images stay contributor-copyright; we hold them
under agreement, publish no images without per-photographer permission, and hand back the analysis and
the tooling.

**Minimum viable size.** The Phase 1B kill criterion is stated at **40+ individuals**; Ocean Sanctuaries
is reported at **~76 matched individuals over 2010–2020** [SEARCH, advocacy/press — order of magnitude
only, not a number to publish]. If the delivered confirmed-resight subset contains **< 40 individuals
with ≥2 sightings**, the decision thresholds in §5 do not fire; the run is reported as a
**power failure of the archive, not a result about sevengills**, and the next action is a second archive
(Sharkbook's South African / Cape Town sevengill data, or Otago's Stewart Island BUV catalogue —
149 individuals across 357 encounters [SEARCH]).

---

## 2. Design decision: reuse the Melops machinery unchanged

Everything below runs on code that has already been run to a verdict on a 9,861-individual fish
catalogue (`prototypes/01-melops-ablation/results/CAMPAIGN.md`). We add **no analysis code**. The entire
integration is: *write one CSV in the layout `melops_data.load_melops` already reads.*

`melops_data.py` takes either the real `wildlife_datasets` loader **or** a plain `metadata.csv` in
`root`, whose required columns are exactly:

```
image_id, identity, path, date, side, bbox_body, bbox_head, bbox_headless
```

(`melops_data._REQUIRED_PLAIN_COLUMNS`). Produce that file and `diagnose.py`'s recapture-gap curve runs
on day one, with the same split, the same same-side matching rules, the same buckets and the same
JSON/Markdown output as run 3 of the Melops campaign.

---

## 3. The mapping: a Sharkbook export → the catalogue contract

One row per **(annotation, media asset)**, i.e. one row per usable crop of one animal in one photo.

| catalogue column | source in a Wildbook/Sharkbook export | rule (binding) |
|---|---|---|
| `image_id` | `Encounter.catalogNumber` + media-asset basename + annotation index | `"<catalogNumber>__<assetBasename>__a<k>"`, ASCII, no spaces. **Must be unique** — `melops_data._check_catalogue` raises on a duplicate, and `protocol._check_input_frame` raises again on a duplicate `path`. |
| `identity` | `MarkedIndividual.individualID` (fall back: `Encounter.individualID`) | string; adjudicated per §4. Never the encounter id. |
| `path` | local relative path of the downloaded asset under `root` | one asset may back several annotations (two sharks in frame); in that case the rows share the file, so **write one derived crop file per row** and give each its own `path`. Duplicate `path` is a hard protocol error. |
| `date` | `Encounter.year`, `Encounter.month`, `Encounter.day` | zero-padded `YYYY-MM-DD`. Full day precision is **required** for the primary analysis; a row missing `day` is dropped and counted (§7 imputation is a sensitivity arm only). |
| `side` | annotation `viewpoint` | see the side rule below. |
| `bbox_body` | annotation bounding box | `"left,top,width,height"` floats, in the pixel space of the file named in `path` (Melops semantics, LTWH). |
| `bbox_head` | derived, see the arm remap below | same format. |
| `bbox_headless` | derived, see the arm remap below | same format. |

**Side convention (binding).** The catalogue contract is `side ∈ {"L","R"}` and
`orientation == side`; `_normalize_side` already maps `left/l → L`, `right/r → R`, and anything else
falls through and is rejected by `_check_catalogue`. So:

* a viewpoint string **containing** `left` → `L`; **containing** `right` → `R`;
* pure `front` / `back` / `up` / `down`, empty, or ambiguous → **dropped and counted**;
* a compound viewpoint (`frontleft`, `upright`) keeps its lateral component — this is the incumbent's
  own convention and is what Sharkbook's matcher partitions on;
* where the export's viewpoint is missing (the 2026 regression above), an annotator codes it by eye
  **blind to identity** — the coder sees the crop and nothing else — and the hand-coded fraction is
  reported;
* **`(identity, side)` is the unit of analysis.** A left and a right image of the same animal are two
  units and are never matched against each other. This is not fastidiousness: on Melops, 80% of the
  apparent cross-flank successes (5,222 of 6,503) were same-session opposite flanks, and once that
  contamination was removed cross-flank Rank-1 fell to **0.70%** zero-shot (CAMPAIGN.md, run 2).

**Arm remap (binding, and it must appear as a legend under every results table).** We keep the three
column *names* and change their sevengill *meaning*, so that `--arm head|body|headless` keeps working
with no code change:

| column | Melops meaning | **sevengill meaning here** | role per CAMPAIGN.md |
|---|---|---|---|
| `bbox_body` | whole fish | whole animal (the Sharkbook annotation box) | context arm |
| `bbox_head` | head crop | **whole head**: snout tip → posterior edge of **gill slit 7**, full dorsoventral extent | **secondary** — the direct analogue of the Melops head-crop arm |
| `bbox_headless` | body minus head | posterior edge of gill slit 7 → precaudal pit (flank) | **primary** — flank matching |

**`bbox_head` is the whole-head arm, not the incumbent's freckle patch — the two are different crops
and this document never uses one name for the other.** `bbox_head` exists so that the anterior/posterior
contrast measured on Melops (head crop vs body-minus-head) transfers to sevengills with the same code
and the same semantics; it is the Melops head-crop analogue, and its extent is set by the *anatomy split
point* (gill slit 7), not by what Ocean Sanctuaries matches on.

The incumbent's region is a **narrower, distinct, additional crop**: the nares-to-gill freckle patch,
named in Schema S1 as `chart.head_patch_bounds`
(`naris_anterior_margin → gill_slit_1_dorsal_origin`, `phase1b/p0-sevengill-schema/keypoints_sevengill_v1.yaml`).
It is strictly contained in `bbox_head` and stops at gill slit **1**, where `bbox_head` runs on to gill
slit **7**. It is **not** one of the three contract columns, so it is not an arm of the primary run; it
is produced as a **fourth named crop** (`head_patch`) once the S1 detector or the §3 blind annotation
supplies the two landmarks, and it is run as a **fourth export root** through the same `--arm head`
code path. Until then the incumbent-region comparison is not available and no result may be described
as measuring it.

Flank primary / head secondary is the CAMPAIGN.md verdict: headless ≈ body in all four Melops cells
(gaps 0.2 / 1.0 / 0.0 / 1.9 points) and the trained models favoured the flank by up to 9.1 points. The
whole-head arm is retained because it is the Melops comparison replayed on a new body plan; the
`head_patch` crop, when it exists, is the one that speaks to the incumbent, and it is the comparison
that tells Ocean Sanctuaries something they do not already know.

**How the split between the two boxes is drawn.** From the gill-slit-7 landmark of Schema S1
(`phase1b/p0-sevengill-schema/keypoints_sevengill_v1.yaml`) once the detector is trained. Until then:
one annotator drags a **single vertical split line per crop** (~5 s/frame), stored as `split_x`, and the
two boxes are the annotation box cut at `split_x`. The split is annotated **blind to identity and before
any embedding is computed**, and the same `split_x` serves both arms, so no arm can be advantaged by the
cut.

**Size covariate (needed for §5's growth axis and §7's assortativity control).** `melops_data._normalize`
keeps only the contract columns, so size must travel in the sidecar the length readout already reads:
`readout_length_controlled.py` loads `<root>/Melops_metadata.txt` (`sep=None`, sniffed) and uses columns
**`filename_year`** and **`length`**. Write that file with `filename_year = image_id` and
`length =` estimated total length in **cm** (diver estimate or the archive's measurement field; the unit
only has to be internally consistent — the readout uses ratios and ±10% bands). Rows with no length are
excluded from the length readouts only, and counted (`n_queries_missing_length`).

**Validation gate (run this before anything else; it is five seconds and it is the whole integration
test):**

```sh
cd /home/user/SanBox/prototypes/01-melops-ablation
python -c "import melops_data, sys; d=melops_data.load_melops(sys.argv[1]); \
print(len(d), d['identity'].nunique(), sorted(d['side'].unique()), d['date'].min(), d['date'].max())" \
  /path/to/sevengill_export
```

If it raises, the export is not yet in contract. The exceptions are the specification: duplicate
`image_id`, `orientation != side`, `side` outside `{L,R}`, unparseable `date`, degenerate bbox.

---

## 4. ID adjudication rule (pre-registered)

The identities are the archive's, not ours. We adjudicate only *which* of their links we admit:

1. **Human-confirmed only.** An identity link enters the primary analysis only if the archive records it
   as reviewer-confirmed. An unreviewed algorithmic candidate is **not** an identity. If the export does
   not carry a confirmation flag, we request it explicitly; if it cannot be supplied, we require **two
   independent confirmations** from project records and report the count. *(Using unreviewed MiewID
   suggestions as ground truth would make this study measure MiewID's self-consistency, which is
   circular and worthless.)*
2. **Pattern-based links only in the primary.** A link confirmed on the freckle/spot pattern counts. A
   link resting on **scars alone** is excluded from the primary and reported as a labelled secondary
   stratum — elasmobranch scars are a decaying channel (a ~20 cm blacktip reef bite wound closed almost
   completely within 3 days; a wound and scar were undetectable within 179 days [SEARCH]), so a
   scar-only link and a pattern link are different measurements.
3. **Provisional IDs out.** Any identity the project itself marks uncertain, provisional or "possible"
   is excluded; the excluded count is reported.
4. **Singletons stay.** Individuals with one encounter remain in the catalogue — the split needs them
   for open-set realism (they become novel queries) — but they contribute no gap bucket.
5. **Left/right links are metadata.** If the archive links a left and a right catalogue entry as one
   animal, we record it and ignore it for matching (§3 side rule).
6. **Freeze before embedding.** The identity table is hashed and frozen before a single embedding is
   computed. No identity is revised after any similarity score is seen. If the archive owners revise an
   ID later, the analysis is re-run in full and **both** versions are reported.
7. **Their adjudication is a ceiling, and we say so.** Photo-ID ground truth is human consensus, not
   hardware. The honest calibration is white sharks: **~85% concordance between fin photo-ID and
   microsatellite genotypes over five years** [SEARCH]. Melops, by contrast, had PIT-tag ground truth.
   Every number in this study inherits an unquantified label-noise floor and must be reported as such.
8. **The ground truth is MiewID-mediated, and that biases us optimistic.** Rule 1 admits only
   human-confirmed links — but on Sharkbook a human confirms by *reviewing a candidate list MiewID
   proposed*. A reviewer rarely discovers a resight the matcher never surfaced. So the confirmed-resight
   subset is, to an unknown degree, **the set of pairs MiewID already scores highly**, and the pairs an
   embedding scores *poorly* — exactly the pairs a drift measurement most needs — are systematically
   **under-represented**. The direction is not ambiguous: **this inflates apparent stability**, and it
   inflates it most on the backbone most correlated with MiewID (i.e. the `miewid` backbone itself, and
   to a lesser extent any modern embedding). This is **not** the same bias as §7.7: §7.7 is about *which
   animals* got matched at all, this is about *which pairs of a matched animal* got surfaced for
   confirmation. It is carried as confound §7.11 with its control, and it is stated as a limitation
   beside every drift number.

---

## 5. Pre-registered metrics and thresholds

### 5.1 Protocol (identical to the Melops runs)

```
split      protocol.one_shot_open_set_split(df, cutoff_fraction=0.5, seed=0,
                                            same_date_policy="exclude")
matching   same-side only, L2-normalised cosine, one-shot gallery
readout    diagnose.recapture_gap_section  ->  buckets 0-30 / 31-180 / 181-365 /
                                               366-730 / 731+ days
per bucket n, mean true-mate cosine similarity, Rank-1
always reported  n_gallery, n_known, n_novel, n_same_date_excluded
backbones  `megadescriptor` (MegaDescriptor-L-384, primary), `miewid` (second),
           `hist` (dependency-free smoke), `random` (seeded null arm)
arms       headless (flank, primary) / head (whole head, secondary) / body (whole animal, context)
           [+ head_patch (S1 nares-to-gill-slit-1 freckle patch) as a fourth export root,
            only once the S1 landmarks exist — see the §3 arm remap]
```

This is exactly the curve that produced the Melops pattern-stability result — true-mate similarity
**0.605 → 0.474** from <30 days to 2+ years (CAMPAIGN.md run 2, finding 3). That curve is the comparison
object: a wild fish, handled, non-bending, PIT-tagged. The sevengill curve is the same measurement on a
free-swimming, bending elasmobranch.

### 5.2 Power floor (pre-registered, so no bucket is over-read)

Wilson 95% CIs at n = 154 are roughly **±7 points** (CAMPAIGN.md sign-off). Therefore:

* a bucket with **n_known < 30** is reported but is **not decision-bearing**;
* the decision uses the **pooled ≤180 d** pre-window (0–30 + 31–180) versus the **pooled ≥181 d**
  kill window (181–365 + 366–730 + 731+), which is the Phase 1B "≥6 months" split; the **pooled
  ≥366 d** long-gap sub-window (366–730 + 731+) is the separability read used by RED-A;
* every reported delta carries its Wilson CI and the two n's. Unpaired z is used and named as
  conservative, as in the campaign.

### 5.3 Decision table (primary — pattern drift)

Read on the **head (whole-head) arm and the headless (flank) arm separately**, at ≥40 individuals — and
on the `head_patch` crop as a third, separately-reported read once the S1 landmarks make it available
(§3 arm remap).

**The evaluation window (binding).** The Phase 1B kill criterion is *"query sightings ≥6 months after
gallery"*. Six months is **≥181 days**, so the kill window is the **pooled 181–365 + 366–730 + 731+**
buckets, evaluated as one proportion — not the 366-day-and-up buckets alone. The 181–365 bucket is
inside the kill window, not a no-man's-land between the pre-window and the long-gap read.

**Every bucket's role (the table below is exhaustive over the five `recapture_gap_section` buckets;
no bucket is undecided and none is discarded).**

| bucket | pooled into | which decision it feeds, and how |
|---|---|---|
| **0–30 d** | pre-window `≤180 d` | AMBER precondition (pre-window Rank-1 ≥ 50%); **sole trigger for RED-B** (Rank-1 < 50% here = pipeline failure, not drift); first point of the monotonicity check |
| **31–180 d** | pre-window `≤180 d` | AMBER precondition, pooled with 0–30 d; monotonicity check |
| **181–365 d** | kill window `≥181 d` | **GREEN / AMBER**, pooled with 366–730 and 731+ — this bucket is *inside* the Phase 1B ≥6-month window and carries equal weight there; monotonicity check. It does **not** feed RED-A |
| **366–730 d** | kill window `≥181 d` **and** long-gap `≥366 d` | GREEN / AMBER via the pooled ≥181 d proportion; **RED-A** via the pooled ≥366 d separability read; monotonicity check |
| **731+ d** | kill window `≥181 d` **and** long-gap `≥366 d` | as 366–730 d; also the terminal point of the monotonicity check |

Per-bucket Rank-1, n and Wilson CI are reported for all five buckets regardless; a bucket with
`n_known < 30` is reported but is not decision-bearing on its own (§5.2) — it still contributes its
counts to the pooled window it belongs to, because pooling is what the power floor exists to buy.

| outcome | condition | what it means | what happens |
|---|---|---|---|
| **GREEN** | Rank-1 **≥ 50%** in the **pooled ≥181 d** kill window (181–365 + 366–730 + 731+) | the pattern survives the Phase 1B ≥6-month window at the stated bar | premise holds; Phase 1B proceeds as written |
| **AMBER** | pooled ≤180 d ≥ 50% but **pooled ≥181 d < 50%**, decay monotone across the five buckets | pattern is real but drifts | programme survives **with a re-enrolment cadence** set at the longest bucket still ≥50% that also clears the §5.2 floor of `n_known ≥ 30`; that cadence becomes a field-protocol requirement and a stated limitation of the whole method |
| **RED-A (cheap kill of the premise)** | in the pooled ≥366 d long-gap window the true-mate similarity is not separable from the impostor distribution (window-wise open-set AUROC ≤ 0.5, or mean true-mate sim ≤ the 95th percentile of same-side impostor max-sim) | multi-year pattern identity is not there | **the programme's premise fails.** No amount of pose normalization fixes a pattern that changes. Report it, publish it (it is the first quantitative sevengill drift measurement either way), and stop. |
| **RED-B (kill of *our pipeline*, not the premise)** | Rank-1 < 50% even in the **0–30 d** bucket, on the head arm, at ≥40 individuals | we are failing where drift *cannot* be the explanation — same animal, same weeks, on links the archive itself has human-confirmed | **do not read this as a sevengill result.** It is an imaging/crop/embedding failure; diagnose with `diagnose.py`'s contact sheets before anything else is concluded |

**Order of reading, and the fallthrough.** RED-B is checked first (a pipeline failure invalidates every
other reading), then RED-A, then GREEN, then AMBER. Any result matching none of the four rows — most
plausibly pooled ≤180 d < 50% without RED-B firing, or non-monotone decay — is recorded as
**UNRESOLVED** with the failing condition named, and is **not** read as a pass; the next action is the
§7 controls and the contact sheets, not a threshold revision.

The 50% figure is not chosen here: it is the Phase 1B kill criterion already in
`03-candidate-approaches.md` — *"Rank-1 below 50% on a time-separated split (query sightings ≥6 months
after gallery) at 40+ individuals."* The pooled ≥181 d window **is** that ≥6-month time-separated split;
the five gap buckets resolve it into elapsed-time strata for diagnosis without changing the criterion it
is pooled back into. So this study **is** the Phase 1B kill test run early on someone else's data.

External anchors to quote beside the result, never as targets: MegaDescriptor **62.02%** on WhaleSharkID
(a spot-patterned elasmobranch on a less deformable body — a ceiling); **90.3%** with classic I3S over up
to 496 days on blue-spotted ribbontail ray; **~85%** photo-ID-vs-genotype concordance in white sharks;
and the adoption bar biologists state, **top-10 ≥ 95%** [all SEARCH].

**Mandatory bias statement on every verdict.** Whichever row fires, the reported verdict carries, in the
same breath: (i) the discovery-route strata of §7.11 alongside the pooled number, and (ii) the sentence
that the archive's ground truth is MiewID-mediated and therefore **biases the estimate optimistic — the
stability reported here is an upper bound** (§4 rule 8, §7.7, §7.11). A GREEN read without that sentence
is an overclaim; a RED read is, if anything, strengthened by it.

### 5.4 What kills the programme cheapest

**RED-A on both arms.** It costs one export and one afternoon of compute, it needs no sevengill field
season, no template, no licence, and no new model — and it forecloses Approaches 1B, 2 and 3
simultaneously, because all three assume a multi-year-stable pattern. This is deliberately the first
thing we try to break.

---

## 6. Secondary pre-registration: what rectification buys (the thing Melops could not test)

**This is the part that is different from Melops, and it is the reason this archive is worth more to us
than a second wrasse dataset.** Melops fish are handled, shot against a standardised board, and **do not
bend in frame**; the campaign's standing caveat is explicit that what pose normalization buys on a
bending body remains unmeasured. La Jolla sevengills are photographed **free-swimming and bending**. This
archive is therefore the **first data in the programme that can measure rectification at all.**

### 6.1 Arms

| arm | pipeline |
|---|---|
| **A — raw crop** | `bbox` applied → resize → backbone (exactly the primary drift run) |
| **B — midline-rectified** | SAM2 mask → `prototypes/02-centerline-chart/centerline.extract_centerline(mask, n_stations)` → `chart.rectify(image, centerline, half_width, n_s, n_r, mask)` → fixed `(n_s × n_r)` strip → same backbone |

Same split, same seed, same identities, same queries, same backbone weights. Only the pixels differ.

### 6.2 Threshold (pre-registered, from the spec)

> *"if midline rectification does not beat the raw-crop baseline by **≥3 Rank-1 points** in the same
> experiment, the rectification idea is dead"* — `03-candidate-approaches.md`, Phase 1B kill.

Applied on the **headless/flank arm**, on the same time-separated split. Because both arms score the
identical query set, the comparison is **paired**: report McNemar on the discordant pairs and the paired
95% CI, not two independent proportions.

**The kill rule (binding, and it is the spec's rule verbatim and alone).** Arm B must beat arm A by
**≥3 Rank-1 points overall in the same experiment**. If it does not, **the rectification idea is dead
for Phase 1B.** There is no second route to survival: no stratum, tercile, subgroup or arm may be
substituted for the overall comparison, and this rule is fixed here, before any number is seen,
precisely so that one cannot be.

**Bend-stratified readout (descriptive, NON-BINDING).** `extract_centerline` yields the polyline, so
total turn (radians) is computable per frame at no cost. Stratify queries into curvature terciles and
report arm A, arm B and their paired delta per tercile. The pre-registered directional expectation —
stated in advance so that it too cannot be chosen afterwards — is that **rectification's gain increases
with curvature**. This readout is **reported for diagnosis; it has no bearing on the kill decision.**
A ≥3-point gain confined to the top tercile does **not** rescue a rectifier that missed the overall bar;
it is recorded as a hypothesis about where a future rectifier might pay off, and it dies with the arm
for Phase 1B.

### 6.3 Calibration and honesty notes

* Prototype 02 already measured its own ceiling on synthetic ground truth: with bend only, spot positions
  recover to **0.07% of body length**; with literature-bracket ±5% skin strain the residual is
  **~0.7% BL mean / 1.4% BL max**, antisymmetric between flanks, matching the beam model with slope 1.019
  (`prototypes/02-centerline-chart/README.md`). The strain residual is a property of shark skin and **no
  chart removes it**. So a null result here is a real possibility that was predicted in advance.
* **Failure-mode exclusions must be arm-symmetric.** `extract_centerline` is documented to produce a
  confidently wrong shortcut on a fused self-touching bend, and warns on blob-like masks. Every frame
  where the extractor warns or fails is **excluded from both arms**, and the excluded count and its
  per-bucket distribution are reported. Excluding a hard frame from B only would manufacture the effect.
* Rectification consumes a mask. Mask quality is a confound in its own right: report SAM2 mask
  area-stability and exclude frames with no single dominant component, again from both arms.

---

## 7. Confounds to control (each with the control, most from the Melops lesson)

1. **Size assortativity — the headline Melops lesson.** The campaign found the size-assortativity index
   essentially immovable (**0.338 → 0.342** across a full fine-tune; run 3 finding 3) — training added
   pattern signal *on top of* the size bias rather than removing it. Sevengills grow from ~50 cm to
   170 cm (M) / 224 cm (F) at maturity, so size is a strong nuisance variable that is **correlated with
   the very axis we are measuring**. Control: run `readout_length_controlled.py` (assortativity index,
   length-stratified Rank-1 terciles, **±10% length-banded impostor AUROC**) and report the banded result
   beside every raw result. A drift curve without the length-banded companion is not reportable.
2. **"True mates are old photos."** The campaign's neighbour-date structure (argmax-gallery median
   **310 days** away for known vs **1,046** for novel) showed a years-later mate losing to
   contemporaneous, size-matched impostors — a structural property of one-shot open-set matching under
   drift, predicted there to "apply verbatim to any multi-year sevengill catalogue." Control: report, per
   gap bucket, the **median elapsed days between a query and its argmax gallery entry**. If long-gap
   failures land on contemporaneous impostors, the curve is measuring gallery composition as much as
   skin.
3. **Gallery size grows over the archive's decade**, and Rank-1 falls with gallery size. Long-gap queries
   therefore face bigger galleries by construction. Control: `diagnose.small_gallery_calibration`
   (K enrolled + K novel, 3 seeds) run **per gap bucket at a fixed K**, so buckets are compared at equal
   gallery size. This is the most important single control and it is already written and tested.
4. **Catalogue density.** Density moved Melops more than any modelling choice (Rank-1 1.9 → 15.3
   zero-shot from density alone). Control: run the dense-subset ablation exactly as run 3 Leg A —
   units with **≥4 images**, filter applied to the catalogue **before** the split and identically for
   every arm (`_filter_dense_units`; for the gap curve, via a pre-filtered second export root — see §8),
   `n_units` and `n_images` retained reported. Density differs systematically between well-loved
   individuals and one-off sightings, so this is also a covariate, not just an operating point.
5. **Same-session near-duplicates.** `same_date_policy="exclude"` (default), and
   `n_same_date_excluded` reported. Without it the 0–30 d bucket measures burst photography.
6. **Ontogenetic instability.** Juvenile/neonate patterns are the documented failure case in the nearest
   analogue (epaulette sharks), and San Francisco and Humboldt Bays are nurseries, so a California corpus
   is enriched for exactly those size classes. Control: stratify the drift curve by estimated TL
   (juvenile / sub-adult / adult against 170 cm M, 224 cm F [SEARCH]) and report the growth-tercile
   stratification from readout 2. **Pre-registered expectation:** drift is faster in the smallest
   stratum. If it is, that is a field-protocol finding (enrol adults) and not a programme kill.
7. **Survivorship / selection bias — the one that would most flatter us.** The confirmed-resight subset
   is, by construction, the animals whose patterns *were* matchable by the incumbent. Measuring drift
   only there over-estimates stability. Controls: (a) request and report the **size of the unmatched
   queue** and the fraction of encounters never assigned an ID; (b) report the result explicitly as an
   **upper bound on stability**; (c) if the archive can supply photos of *known* individuals that the
   incumbent failed to match, those are the most informative frames in the dataset and are analysed as a
   labelled stratum.
8. **Image-quality drift over a decade** (camera generations, GoPro eras, visibility). Control: report
   per-year median crop long-axis in px; pre-register a **resolution-matched sensitivity analysis** at
   ≥256 px flank (the spec's floor); include crop resolution as a covariate in the reading.
9. **Photographer/site effects.** Report contributions per photographer and per site; a bucket dominated
   by one contributor is flagged in the results table.
10. **Date precision.** Primary analysis is day-precision rows only. Sensitivity arm: month-precision
    rows imputed to the 15th, reported separately, never pooled into the primary.
11. **MiewID-mediated ground-truth selection bias — the other one that flatters us** (§4 rule 8). The
    archive's human-confirmed links are largely *confirmations of MiewID-proposed candidates*, so
    resight pairs that MiewID scores poorly are under-represented in the ground truth, and the measured
    drift curve is **biased optimistic** — it partly measures which pairs an embedding could already
    find. Controls: (a) **stratify resight pairs by discovery route** wherever the export records it —
    `manual` (a human found the match unaided: bookmarked animal, diver's own note, name-search,
    retrospective review) versus `algorithm-suggested` (confirmed off a MiewID/Wildbook candidate list)
    — and **report both strata separately as well as pooled**, since the manual stratum is the only one
    not conditioned on a matcher's score; (b) if the export carries no route field, request it, and if
    it cannot be supplied, say so and treat the entire curve as the algorithm-suggested stratum;
    (c) report the stratum sizes and the manual-stratum share, because a small manual stratum bounds
    how much of the bias this control can actually remove; (d) **state the bias direction explicitly —
    "stability estimated here is an upper bound" — wherever the drift curve is interpreted**: in §5.3's
    verdict, in every results-table legend, in the §9 hand-back, and in any abstract. The backbones that
    partly escape this are `hist` and `random`, whose scores are uncorrelated with MiewID's ranking; a
    much larger manual-vs-suggested gap on `megadescriptor` or `miewid` than on `hist` is direct
    evidence of the bias operating, and is reported as such.

---

## 8. Runbook (day one, once the export validates)

```sh
cd /home/user/SanBox/prototypes/01-melops-ablation
ROOT=/path/to/sevengill_export          # contains metadata.csv, Melops_metadata.txt, crops/

# 0. contract gate (§3) - must not raise
python -c "import melops_data,sys; print(len(melops_data.load_melops(sys.argv[1])))" $ROOT

# 1. dependency-free smoke: does the whole curve compute end to end?
python diagnose.py --data melops --root $ROOT --backbone hist --arm headless \
    --out results/sevengill-smoke --seed 0

# 2. primary drift curve, all three arms (--data melops takes the plain
#    metadata.csv path in melops_data.load_melops; no wildlife-datasets needed)
for ARM in headless head body; do
  python diagnose.py --data melops --root $ROOT --backbone megadescriptor \
      --arm $ARM --emb-cache emb_cache_sevengill --calibration-k 30 \
      --out results/sevengill-drift-$ARM --seed 0
done

# 3. size / assortativity controls (reads the cache written above; never embeds)
for ARM in headless head; do
  python readout_length_controlled.py --root $ROOT --backbone megadescriptor \
      --arm $ARM --emb-cache emb_cache_sevengill --band 0.10 \
      --out results/sevengill-length-$ARM.json --seed 0
done

# 4. full four-arm Rank-1 matrix incl. the cross-flank arm, for comparability
#    with the Melops campaign table
python run_ablation.py --data melops --root $ROOT --backbone megadescriptor \
    --arms head,body,headless,cross_orientation --emb-cache emb_cache_sevengill \
    --out results/sevengill-arms --seed 0
```

**Two CLI facts to plan around (verified against the code, not assumed):**

* `--calibration-k` defaults to **500** enrolled + 500 novel units, which a ~40–80 individual sevengill
  catalogue cannot supply. Set it to a K the archive can actually fill (30 is a sane start); the tool
  clamps K down to what the split supports and **prints the K it actually used** — record that K, because
  the Melops calibration number is not comparable at a different K.
* **`diagnose.py` has no `--dense-min-images` flag** — only `run_ablation.py` and
  `readout_length_controlled.py` do (`_filter_dense_units`, run 3 Leg A). So the density control (§7.4)
  for the gap curve is produced as a **second export root** whose `metadata.csv` is pre-filtered to
  `(identity, side)` units with ≥4 images. Filtering at export time gives exactly the property the
  campaign required — the filter is applied to the catalogue *before* the split and identically for every
  arm, because all arms read the same rows. Report `n_units` and `n_images` retained.

Artifacts per run: `diagnostics.json`, `diagnostics.md`, `contact_sheet_<arm>.png`. **Look at the contact
sheets before reading any number** — on Melops they were the check that head crops contain heads; here
they are the check that the `bbox_head` box runs snout tip → gill slit 7 and so *contains* (does not
equal) the S1 nares-to-gill-slit-1 freckle patch, and that `bbox_headless` starts behind gill slit 7.
A RED-B reading with bad contact sheets is a cropping bug, not a finding.

Rectification arm (§6) additionally needs `prototypes/02-centerline-chart/` on the path and SAM2 masks;
the strips are written as ordinary image files with their own `metadata.csv` (same `image_id`s, same
`identity`/`date`/`side`, bbox = whole strip), so **arm B is a second catalogue root and needs no new
analysis code either**.

---

## 9. What we hand back to Ocean Sanctuaries

Committed in advance, because it is the consideration in the agreement, not a courtesy:

1. The **first quantitative measurement of sevengill pattern drift** — the recapture-gap curve on their
   own catalogue, with CIs and every control in §7, and with the limitation stated plainly: because the
   confirmed links are largely MiewID-proposed, the measured stability is an **upper bound** (§4 rule 8,
   §7.11).
2. A **per-individual drift report**: which of their animals are drifting, and which multi-year links are
   the weakest, ranked. Directly actionable for their matching queue.
3. The **export → analysis tooling**, and the mapping in §3, so they can re-run it annually themselves.
4. Co-authorship on any publication, on their terms, and no image published without the photographer's
   permission.

---

## 10. Deviations log

*(empty at pre-registration; every departure from §3–§7 is recorded here with a date and a reason,
before the affected number is quoted anywhere)*

| date | section | deviation | reason |
|---|---|---|---|
| — | — | — | — |

---

### Evidence-grade note

Grades follow `docs/sevengill-canonical-reid/README.md` and are not promoted. `[PRIMARY]` facts in this
document are the ones read first-hand from code in this repository (`melops_data.py`, `protocol.py`,
`diagnose.py`, `readout_length_controlled.py`, `prototypes/02-centerline-chart/`) and the campaign
results. Everything about the Ocean Sanctuaries archive's size, contents, field names and current
platform state is `[SEARCH]` — `sevengillsharksightings.org`, `wildbook.docs.wildme.org` and
`sketchfab.com` were all **egress-blocked** in this environment, so no archive page or Wildbook schema
page was fetched first-hand. **The §3 column names are therefore a specification to confirm against the
first real export, not a verified schema** [UNVERIFIED]: confirm them in the first ten minutes of
contact, and treat any mismatch as a §10 deviation rather than a surprise.
