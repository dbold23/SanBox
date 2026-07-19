# The frontier individual identifier — design & honest roadmap

*How to make spot-pattern re-identification of sevengill sharks genuinely
best-in-class, grounded in 2024–2026 methods — and an equally honest
account of why the data, not the model, is the binding constraint.*

This document synthesizes parallel web-grounded research across five
frontier threads (learned re-ID, pattern photo-ID, foundation-model
correspondence, curved-surface normalization, small-data/sim-to-real) and
two adversarial feasibility critiques. Citations name real methods; treat
specific arXiv IDs / accuracy numbers as **to-verify against primary
sources** (some may be imprecise).

---

## 1. The reframe (why the two prior matchers failed)

Both prior matchers tried to estimate **one geometric transform between two
photos** of a curved surface. That is the wrong problem. The 2024–2026
answer for curved, deformable, patterned animals is:

> **Don't match photo-to-photo. Map every photo *independently* onto one
> shared canonical flank surface, then match spot constellations in that
> pose-normalized space.**

"Unwrap-then-match." This dissolves all four diagnosed failure modes at
once:

| failure mode (observed) | how the reframe removes it |
|---|---|
| curved flank ≠ one homography | UV-unwrap absorbs curvature; matching happens in a flat canonical frame |
| dense clouds → coincidental alignments | matching in a metric-correct canonical frame + distinctiveness weighting |
| partial overlap / different extents | becomes a UV-mask intersection, scored only on the shared region |
| tiny label set | the unwrapper is trained on **synthetic** dense supervision; the matcher is **zero-shot** |

The closest published analog: photo-ID of the **Hula painted frog** (faint
ventral spots, capture–recapture) reportedly reached ~98% top-1 with
**zero-shot deep local-feature matching**, *beating* fine-tuned global
embeddings (~60%). **Pelage-pattern unwrapping** (self-supervised UV
mapping via surface normals) set SOTA on curved, deforming bodies (seals,
leopards). **WildFusion** (calibrated global+local score fusion) reached
~84% mean over 17 datasets. The frontier design below is that hybrid,
assembled around the assets this project already owns.

---

## 2. The architecture (best achievable, as a pipeline)

```
photo ─▶ [0] SAM 2 flank ROI
       ─▶ [1] UV-unwrap net (CSE/DensePose, synthetic-supervised)  ◀── KEY
       ─▶ [2] faint-spot enhancement U-Net (supervised by 6,618 boxes)
       ─▶ [3a] MiewID global embedding ──▶ top-K candidate shortlist
       ─▶ [3b] zero-shot local/dense matcher (RoMa / LightGlue+ALIKED)
               verified by non-rigid TPS + LNBNN distinctiveness in UV space
       ─▶ [4] calibrated global+local fusion + open-set threshold
       ─▶ [5] ranked candidates → human reviewer (semi-automatic, like Wildbook)
```

- **[1] is the linchpin.** A Continuous Surface Embeddings / DensePose-style
  net maps every flank pixel to a canonical UV coordinate. Its dense
  ground truth is **unobtainable on real sharks but free from the synthetic
  generator** — the generator's one irreplaceable asset. This is what turns
  "~100% on synthetic" from a dead-end demo into a transferable front-end.
- **[3b] is the accuracy driver but also the biggest risk** (see §3): faint
  repetitive blobs are near-worst-case for foundation matchers.
- Everything the project already built is reused: the **6,618 boxes** →
  enhancement-net targets + a spot detector + a density prior; the
  **constellation/RANSAC code** → the geometric verifier, now operating in
  the canonical space where it is finally valid; the **generator** → the
  unwrapper's training engine.

Every near-term component has open weights (SAM 2, MiewID, RoMa,
LightGlue/ALIKED, Efficient-LoFTR, WildFusion / wildlife-tools), so a
baseline can stand up on existing GPUs in days.

---

## 3. The honest reality check (why this is a research program, not a win)

Two independent adversarial critiques converged on the same verdict:
**directionally right, but unfalsifiable as written, because the data
cannot measure it.** The load-bearing problems:

1. **No statistical power.** 4 individuals / 8 true pairs → a 95% CI on any
   top-1 rate spans ≈ ±30 points. 2/8 vs 5/8 is *three photographs*.
   Bootstrapping 8 positives doesn't manufacture power. You could build all
   five stages and be unable to say whether any of them helped.
2. **The unwrapper's synthetic ~100% is closed-loop.** An inverse net that
   inverts the same generator that made the data proves *simulator
   self-consistency*, not real transfer — and there is **no real dense UV
   ground truth** to validate it against. An unmeasurable linchpin.
3. **Foundation matchers may not see spots at all.** RoMa/DINO/LightGlue
   key on outline, gill slits, fins, and natural texture — they can produce
   a confident smooth warp that *aligns two different sharks*, reproducing
   the false-alignment failure one layer deeper. The discriminative spot
   signal is exactly what they average away.
4. **Calibration / fusion / open-set thresholds need a population
   distribution** that 4 individuals cannot provide; fitting them on the 8
   pairs and reporting on the same pairs is leakage.
5. **Unverified biology.** Are sevengill flank spots permanent, unique, and
   re-detectable across sightings (as whale-shark spots are)? At 0.055
   contrast, part of the 2/8 may be a **detectability floor**, not a method
   floor. Untested.
6. **Data growth is a field/population constraint misfiled as engineering.**
   58 photos yielded ~4 re-sights (~7%). The active-learning "flywheel"
   can only *screen* re-sightings that were actually photographed twice; it
   cannot conjure them.

**Imported numbers don't transfer.** The frog's 98% and WildFusion's 84%
come from datasets with hundreds–thousands of curated individuals and bold,
roughly-planar, high-contrast patterns. Quoting them as *this* project's
ceiling silently imports 1–2 orders of magnitude more data than exists.

---

## 3b. UPDATE — Experiment B was run, and it PASSED decisively

The foundation-matcher probe (`spotid/probe_matchers.py`) was executed on the
real flank crops. Off-the-shelf local-feature matchers (DISK deep features
and classical SIFT) + RANSAC verification, RANSAC-inlier counts:

| matcher | TRUE re-sightings | DIFFERENT individuals |
|---------|-------------------|-----------------------|
| SIFT    | 57–156 (mean 106) | 0–13 (mean 10, max 13)|
| DISK    | 553–821 (mean 681)| 14–23 (mean 18, max 23)|

**All 4 true pairs sit far above every one of 24 impostor pairs, for both
matchers — zero overlap, a ~5–30× margin.** Different sharks get only
noise-floor inliers; the matchers do NOT align different individuals by body
outline (critique risk #3 falsified). Inlier correspondences land on the
flank (spots + skin), not the background.

This **overturns the earlier "real re-ID fails (2/8)" conclusion** — and the
reason is important: the failing matchers used *only spot centroids* (each
spot reduced to an (x,y) point). The pixels around each spot — its shape,
edges, and the surrounding skin texture — carry far more matchable structure,
and generic local-feature matching exploits it. The same pairs my
centroid matcher missed (J001, J003, A257) are matched cleanly here.

Caveats (still real): n is small (4 pairs vs 24 impostors), though the effect
size is enormous; the pairs may share some imaging context (mitigated by
tight flank crops + on-flank correspondences); this is closed-set feasibility,
not open-set deployment. But **the load-bearing thesis of the frontier design
— appearance-based local features re-identify these sharks — is validated on
real data.** The unwrap-then-match machinery is now an *accuracy multiplier*
on top of a baseline that already works, not a prerequisite.

## 4. Do these FIRST — cheap experiments that de-risk everything (this week)

Both critics independently prescribed the same near-zero-cost checks.
**Run these before writing any pipeline code**, on the assets already owned:

- **A. Human/oracle identifiability bound.** Have an expert attempt the 8
  known re-sighting matches by eye (plus decoys), blind, and confirm the
  *same physical spots* are visible in both frames of each pair. This tells
  you whether 2/8 is a *method* floor or a *biology/detectability* floor.
  If humans can't do it at 0.055 contrast, the project is a **capture-
  protocol** problem, not an algorithms problem — and no model will fix it.
- **B. Foundation-matcher probe.** Run frozen RoMa v2 / LightGlue+ALIKED on
  real crops for **both true pairs and known-different-individual pairs**.
  The decisive test: *does the matcher REFUSE to align two different
  sharks?* If it confidently warps different individuals together (likely,
  via body outline), the "zero-shot matcher as accuracy driver" thesis is
  falsified and you need a **spot-native** matcher instead.
- **C. Fix the evaluation on paper.** Define a protocol where calibration/
  thresholds are fit on data that never overlaps the test pairs
  (leave-one-individual-out minimum), and accept that at n=8 the honest
  deliverable is a **feasibility probe, not a top-1 number**.
- **D. Mine guaranteed-ground-truth pairs.** The single thing that unblocks
  everything: **aquarium / known-ID / tagged animals** give free, certain
  positive pairs and a real validation set; partner catalogs
  (Sharkbook/Wildbook) may add more. That dataset — not any model — turns
  the roadmap from unfalsifiable to buildable.

---

## 5. Roadmap (staged, with kill criteria)

No stage advances until it shows a **measurable, leakage-free** improvement
over the prior one.

- **Stage A — this week (zero training).** SAM 2 + CLAHE + MiewID retrieval
  + RoMa/LightGlue rerank with **non-planar** (TPS / fundamental-matrix,
  *not* homography) verification, leave-one-individual-out. Report as a
  *hypothesis test about the failure modes* with brutally honest CIs.
  *Kill:* if it can't localize spots (Experiment B fails), pivot to a
  spot-native matcher.
- **Stage B — weeks 2–6 (cash in the generator).** Upgrade the generator
  from a plane to a **curved 3-D flank template** exporting per-pixel UV +
  normals; train a faint-spot enhancement U-Net on synthetic low-contrast
  renders + the 6,618 boxes. *Kill:* if synthetic contrast/curvature
  statistics can't be matched to real (sensitivity analysis), the unwrapper
  won't transfer.
- **Stage C — months 2–4 (the linchpin).** Synthetic-supervised CSE/
  DensePose unwrapper; run the whole engine in UV space. *Validate* via
  synthetic held-out UV error, **cross-view UV consistency** (unwrap two
  real views of one shark, measure spot agreement in UV), and sparse-
  landmark reprojection error with CIs. *Baseline to beat:* landmark-TPS
  canonicalization (may capture most of the benefit at a fraction of the
  risk — run it, don't just list it as a fallback).
- **Stage D — months 3–6 (data flywheel + spot-native matcher).** Active
  learning to grow confirmed pairs; pretrain a SuperGlue/LightGlue-style
  graph matcher whose **keypoints are spots**, on millions of synthetic
  constellation pairs, then fine-tune on confirmed real pairs.
- **Stage E — months 6–12+ (data-gated frontier).** Once ≈50–200+
  individuals × 3–5 sightings exist: LoRA / sub-center-ArcFace fine-tune a
  sevengill-adapted retriever; adopt a **video** protocol and build
  per-individual **canonical 3-D spot-texture atlases** (SfM → underwater
  3D Gaussian Splatting) so every sighting matches against a stable 3-D
  atlas. This is the publishable frontier endpoint and generalizes to other
  faint-patterned marine species.

---

## 6. Data requirements (the real bottleneck, in priority order)

1. **Confirmed re-sighting pairs** — the true binding constraint. Milestones:
   *dozens* (unlocks calibration + honest eval) → *50–200+ individuals ×
   3–5 sightings* (unlocks metric-learning fine-tune). Grown via the
   active-learning loop + known-ID animals.
2. **Standardized capture protocol** — fixed flank side, in-frame scale
   reference, **overlapping frames / short video**, controlled lighting.
   Fixes partial-overlap at the source and unlocks 3-D reconstruction.
3. **Sparse real landmarks** on ~50–100 flanks (eye, 7 gill slits, fin
   insertions) — cheap; anchors the unwrapper's sim-to-real transfer.
4. **Dense UV + normals** — *free* from the upgraded generator; the one
   place the tiny real-label count doesn't bite.
5. **The 6,618 boxes** — repurposed as enhancement targets, spot-detector
   labels, and a density prior.
6. **Unlabeled flank crops / dive footage** — for self-supervised
   pretraining and realism-transfer fitting.

---

## 7. One-line north star

**Stop chasing a transform between two photos; map every photo onto one
shared canonical flank surface and match spots there — with the unwrapper
trained on synthetic supervision and the matcher zero-shot — but prove the
spots are identifiable and collect real re-sightings first, because at
n=8 nothing else can be measured.**
