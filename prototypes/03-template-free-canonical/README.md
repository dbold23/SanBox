# Prototype 03 — Template-free canonical shape from video

**Status: BLOCKED — deliberately not started.** Per the kill criterion in
[`docs/sevengill-canonical-reid/03-candidate-approaches.md`](../../docs/sevengill-canonical-reid/03-candidate-approaches.md),
this approach is blocked outright until a labelled sevengill catalogue exists. It structurally requires
the identity labels that prototypes 01 and 02 exist to generate, so it cannot come first, whatever its
appeal.

## What it would be

Per-individual canonical shape and appearance optimized directly from multi-minute tracked video
(BANMo / lab4d-style differentiable rendering with a canonical space), skipping the fixed template.
The canonical space then carries the pattern comparison.

## Unblock conditions (all three, in order)

1. **Prototype 01 verdict = "identity distributed"** — headless ≈ body within 5 Rank-1 points on the
   Melops one-shot open-set ablation. If identity is concentrated in the rigid anterior, nothing
   downstream of a patch matcher is worth building.
2. **A labelled sevengill catalogue exists** — 40+ individuals × ~10 images × both flanks with
   adjudicated identities (Approach 1 Phase 1B produces this as a by-product).
3. **Prototype 02's chart demonstrably fails for a diagnosable reason** that per-instance optimization
   would fix (e.g. template mismatch dominating the residual) — otherwise the cheaper chart wins.

## Prerequisites it inherits from P0

- Species column migration in SharkScarAnnotator (no `species` field exists today).
- A sevengill keypoint schema (seven gill slits; no second dorsal; single posterior dorsal).
- Multi-minute tracked sequences per individual — the SharkScarAnnotator track-propagation pipeline is
  the collection tool.

Do not add code here until the three unblock conditions are met and recorded.
