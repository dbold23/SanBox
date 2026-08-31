# Run 1 analysis — zero-shot Melops ablation (2026-08-31)

**Bottom line: this run is INCONCLUSIVE, and its three `report.md` verdicts ("identity
distributed — Approach 2 earns a hearing") must not be quoted.** They fired vacuously through a
gap in the verdict logic that has since been closed (`compute_verdict` now gates on a 15-point
decision floor and would report INCONCLUSIVE for exactly these numbers). The raw metrics remain
valid measurements of *zero-shot* performance; they do not answer the head-vs-body question.

## What the numbers say

| backbone | head | body | headless | cross* | chance floor |
|---|---|---|---|---|---|
| MegaDescriptor-L-384 | 1.93 | 1.00 | 1.16 | 18.64* | 0.0096 |
| DINOv2 ViT-S/14 | 1.08 | 1.23 | 0.92 | 19.99* | 0.0096 |
| hist (floor) | 0.27 | 0.08 | 0.04 | 0.18* | 0.0096 |

Rank-1 percentage points; gallery 10,410 one-shot units; 2,596 known / 11,276 novel queries;
295 same-session near-duplicates excluded per same-side arm.

1. **Signal exists but is ~20× below the decision regime.** 1–2 points is 100–200× the chance
   floor — the embeddings are not blind — but the kill rule needs head ≥ 15 points to be
   *expressible*, and "within 5 points" of a ~1-point base is noise, not evidence that identity
   is distributed. Per-arm correct counts are 50 / 26 / 30 images (MegaDescriptor): head > headless
   is ~2.2σ — weak evidence pointing, if anywhere, in the *kill* direction, at 1/20 the magnitude
   the rule requires.

2. **\*The cross-orientation column is contaminated — do not use it.** All three runs predate
   commit `73eba5a` (`n_same_date_excluded = 0` in every cross arm proves it), so the 18.6–20%
   figures are inflated by same-handling-session opposite flanks: the same fish, wet, on the same
   board, seconds apart. Re-run with the fixed `cross_orientation_split` before drawing any
   cross-flank conclusion.

3. **Open-set AUROC is *below* 0.5 (0.31–0.41) on every same-side arm, every backbone.** Known
   queries score systematically lower max-similarity than novel queries. With identical impostor
   pools this should be ≥ 0.5 (a known query's max includes its true mate), so a confound is
   operating — the leading suspect is temporal/acquisition drift: known queries span all years
   (any later image of a pre-cutoff unit), novel queries are late-years only. Until the stratified
   diagnostic explains this, treat every number in this run as provisional.

4. **Why zero-shot collapsed, and why that was foreseeable.** The 0.35 one-shot anchor
   (arXiv:2301.00596, NLDL 2023) comes from a model trained on this population; MegaDescriptor has
   never seen a corkwing wrasse, and its own baseline sweep shows aquatic collapse (WhaleSharkID
   62% vs 99.9% captive-tank fish). The spec called for "zero-shot **and fine-tuned**" — the
   fine-tuned arm is the decisive one and has not run.

## What run 2 must do (in order)

1. **Diagnostics first** (`diagnose.py`, added after this run): recapture-gap curve (true-mate
   similarity and Rank-1 vs elapsed time — doubles as a pattern-stability measurement, which is
   independently valuable), AUROC stratified by query year (explains or falsifies the temporal
   confound), a 500-identity small-gallery calibration (comparability with the 0.35 anchor), and a
   crop contact sheet (verifies the head/headless bboxes actually contain what they claim).
2. **Re-run cross_orientation** on the fixed split.
3. **Fine-tune arm**: fine-tune MegaDescriptor (ArcFace or triplet head) on a disjoint 60% of
   identities, evaluate the same four-arm ablation on the held-out 40%. Only if this lifts the
   operating point above the decision floor does the head-vs-headless question become answerable.
4. Re-read the verdict — the floor gate now reports INCONCLUSIVE instead of a vacuous answer.

## What this run did establish

- The full pipeline runs end-to-end on real Melops at scale (24.5k images, ~1,000 s per arm per
  backbone on the rented GPU), the side-normalization fix (`left/right` → `L/R`) is in, and the
  protocol guards held (295 same-session near-duplicates caught and excluded per arm).
- Zero-shot species-agnostic embeddings are nowhere near a usable operating point on this
  population — consistent with the literature scan's aquatic-gap finding, and now measured
  first-hand.
