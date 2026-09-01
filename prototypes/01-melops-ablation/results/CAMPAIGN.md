# Melops campaign — reading after run 2, and the run-3 decision

**Status: the head-vs-body question remains unanswered, and the planned experiment has still not
actually run.** The "fine-tune arm" that executed was a fallback — full-backbone training OOMed even
at 48 GB, so only the last 20 tensors trained, for 10 short epochs, with the loss still descending
(22.5 → 18.45, no plateau) on a pathological class regime (2,341 ArcFace classes over 5,427 images,
~2.3 per class). Its apples-to-apples gain was ≤0.2 Rank-1 points and the assortativity index did not
move (0.338 → 0.331), so under the pre-registered reading rule the nuisance-suppression argument
failed **for this fallback** — not for fine-tuning as such. Do not cite run 2 as "fine-tuning fails
on Melops."

## What run 2 established (citable)

1. **Cross-flank matching is near-impossible once the contamination is removed.** The fixed
   `cross_orientation` arm: Rank-1 **0.70%** zero-shot (was 18.6% contaminated; 5,222 of 6,503 known
   cross queries — 80% — were same-session opposite flanks). Left and right flanks are separate
   identities in practice, as the protocol's (identity, side) unit design assumed.
2. **The AUROC inversion is not temporal and not fully size-explained.** Within-year AUROC stays
   0.22–0.47; restricting impostors to ±10% body length lifts AUROC only to 0.35 (zero-shot) / 0.41
   (fine-tuned), still below 0.5. Remaining hypothesis, supported by the neighbor-date structure
   (argmax-gallery median 310 days away for known vs 1,046 for novel): **true mates are old photos**
   — a years-later mate loses to contemporaneous, size-matched impostors. This is a structural
   property of one-shot open-set matching under appearance drift, and it will apply verbatim to any
   multi-year sevengill catalogue.
3. **Pattern stability decays measurably**: true-mate similarity 0.605 → 0.474 from <30 days to 2+
   years; even same-month resights only reach 5.2% Rank-1 at the 10,410 gallery.
4. **A directional residual, pre-registered here, uncitable as a verdict**: head > headless by
   ~1.5–1.7 points in every measurement so far (both models, both gallery scales). Floor-level under
   the decision rule. Recorded so that if a higher operating point confirms it, the record shows it
   was predicted — and if not, that it was never over-claimed.

## Run 3 — the last Melops round, with a stopping rule

Two cheap legs, one night, then the leg ends whatever happens:

**Leg A — dense-subset ablation (no training; protocol variant, signed off).** The Melops regime of
~2.5 images/individual with mass singletons is exactly what the sevengill plan (8–15 images per
individual) is designed to avoid. So ask the question in the regime the programme will actually
occupy: restrict the catalogue to (identity, side) units with **≥ 4 images**, then run the standard
one-shot open-set split and all four arms, zero-shot. Rules: the subset filter applies to the
catalogue BEFORE the split and identically for every arm; report n_units and n_images retained; no
other protocol change; same decision floor. This raises the operating point without touching a
weight and directly simulates the planned sevengill catalogue density.

**Leg B — the fine-tune that was actually planned.** Full-backbone training with the memory
engineered properly: bf16 autocast + gradient checkpointing + batch 16 with 4-step gradient
accumulation (or LoRA adapters if checkpointing still OOMs; or full fine-tune of
MegaDescriptor-B-224 as the fallback-of-record). Train to plateau with early stopping (patience 3 on
a fixed probe subset's Rank-1, not on loss), min-images-per-unit ≥ 2 as already approved. Re-run the
four arms and both readouts (assortativity, length-matched AUROC).

**Stopping rule (pre-registered):** if after run 3 every crop arm is still below the 15-point floor
on BOTH the standard and the dense-subset ablation, the Melops leg is **exhausted** — no further
runs. The campaign then stands as: a validated protocol and tooling stack, a pattern-stability decay
curve, the flank-independence result, the size/temporal confound analysis, and a measured bound on
species-agnostic + light-fine-tune transfer to fine-grained wild fish. The programme redirects to
Phase 1B (sevengill data with ≥8 images/individual, rigid-anterior patch matcher as the primary arm,
midline rectification as the secondary), and the head>headless residual travels there as a
pre-registered hypothesis, not a result.
