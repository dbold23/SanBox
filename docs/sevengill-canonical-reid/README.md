# Pose-invariant re-ID via canonical 3D surface mapping — feasibility for sevengill sharks

Literature scan, August 2026. Question posed: *can pose-invariant re-identification via canonical 3D
surface mapping work for sevengill sharks (*Notorynchus cepedianus*), where the identity signal is scar
and pigmentation pattern on an elongate, laterally-bending body?*

| | |
|---|---|
| **Scope** | Peer-reviewed venues + arXiv, 2018–present unless foundational; released code prioritised, repo and licence recorded |
| **Corpus** | 236 candidates → **188 verified/corrected**, 55 rejected as unverifiable |
| **Method** | 13 parallel search angles → per-paper fact-check → 3 adversarial refuters → synthesis → 2 critics → repair pass → final audit |

## The short answer

**No one has done this on a fish or an elasmobranch — and the reason it hasn't been done is more
interesting than the gap.**

1. **The negative holds.** There is no paper, dataset, or repository in which observations of a fish,
   shark or ray are mapped to a canonical 3D surface or UV parameterization for individual ID. Three
   adversarial refuters attacked it independently and all failed. It also holds at code level: the
   complete DensePose-CSE mesh registry is **13 meshes — one human and twelve placental mammals, zero
   fish** (fetched from the reference implementation; see `appendix/`).

2. **The isometry assumption is false, and that is the finding that matters.** Unwrapping silently
   assumes the deformation is near-isometric, so a pattern is invariant in UV space. It isn't:
   sonomicrometry in a leopard shark at a leisurely 1 BL/s gives **±3.9–6.6% longitudinal strain**,
   the greatest bending strain occurs *nearest the skin*, and shark skin is most extensible in exactly
   the direction arc-length runs — with stiffness actively modulated by swimming speed.

3. **The measured payoff from pose normalization is small everywhere it has ever been measured.**
   +7.9 Rank-1 in the best-instrumented human 3D ablation; +1.99 for explicit 3D on Market-1501;
   approximately zero for 3D frontalization on faces; +8.8 on Grevy's zebra — on a dataset a generic
   2D baseline already scores 1.0000 on.

4. **The field routes around the problem, and it works.** Whale-shark ID crops a region posterior to
   the gill slits. The salmon system anchors patches to the lateral line: 0.609 → 0.860 cross-camera
   mAP, no 3D. Hughes & Burghardt identify great whites from a 1-D fin contour. And the **existing
   sevengill programme has been matching individuals in La Jolla since 2010 on the nares-to-gill
   freckle patch — the most rigid, least-bending region of the animal.**

5. **There is one real, non-obvious advantage, and it is why this isn't a flat "no".** Medicine has
   shipped this exact operation on bent tubes for twenty-five years, in production, under BSD and MIT
   licences — virtual-colonoscopy conformal flattening and VMTK-style centerline
   (arc-length × circumferential-angle) unfolding. The design that makes it work transfers to a shark
   almost unchanged, and is *better posed* here than in medicine, because you have a rest-pose template
   and can compute the chart once, offline, then carry it through skinning.

> **Outcome (2026-09-01): the Melops ablation was run and the leg is closed with a verdict.**
> Identity in a fine-grained wild fish is **not concentrated in the rigid head** — headless ≈ body in
> every cell tested, and the zero-shot head advantage reversed under fine-tuning (it was generic-
> feature bias, not biology). Catalogue density mattered more than any modelling choice (Rank-1 1.9 →
> 15.3 zero-shot from density alone; 29.2 dense + trained). Approach 2 has earned its hearing;
> Phase 1B proceeds with flank-based matching primary. Full record, statistical bounds and caveats:
> `prototypes/01-melops-ablation/results/CAMPAIGN.md`.

**Recommended first move is not the 3D pipeline.** It is a four-week ablation on `Melops` (corkwing
wrasse: 24,578 images / 9,861 individuals / 7 years / CC BY 4.0, shipping body, head *and* headless
crops) to settle whether the identity signal lives in the deformable body at all. If it lives in the
head and gills — as a decade of sevengill field practice suggests — the right thing to build is a very
good rigid-patch matcher, a species column, and a partnership with the people who already have the
photographs.

## Contents

| File | What it is |
|---|---|
| [`01-evidence-and-answers.md`](01-evidence-and-answers.md) | **(a)** The evidence tables, grouped by question, plus direct answers to all six questions |
| [`02-unsolved-elongate-bodies.md`](02-unsolved-elongate-bodies.md) | **(b)** What is unsolved for elongate, bending bodies specifically |
| [`03-candidate-approaches.md`](03-candidate-approaches.md) | **(c)** Three approaches to prototype, each with data requirements, failure modes and a kill criterion |
| [`appendix/verified-corpus.json`](appendix/verified-corpus.json) | 188 enriched records: representation, supervision, data requirement, repo, licence |
| [`appendix/rejected-unverifiable.json`](appendix/rejected-unverifiable.json) | 55 records that failed verification — **do not cite these** |
| [`appendix/detectron2-densepose-cse-modelzoo.md`](appendix/detectron2-densepose-cse-modelzoo.md) | Third-party (Apache-2.0) copy of Meta's DensePose-CSE docs, retained as the primary source for the zero-fish-mesh finding |

## How much to trust this

Read the *How to read this — methodology and limits* block at the top of
[`01-evidence-and-answers.md`](01-evidence-and-answers.md) before quoting anything. The short version:

- **The network policy in the environment this ran in blocked arxiv.org, doi.org, Crossref, Semantic
  Scholar, OpenAlex, PMC, CVF, Springer and HuggingFace at the CONNECT layer.** GitHub and
  `raw.githubusercontent.com` were the only reliably fetchable primary sources.
- **Consequence: licensing and code facts are strong** (read first-hand from repos); **bibliographic
  facts — author lists, page ranges, DOIs — are weaker** and are graded accordingly.
- **Forward-citation chasing was requested and never performed**, because every citation index was
  blocked. No statement here about who cites whom is a citation-graph result.
- The exhaustive full-text negative covers **~2025-03 onward**; earlier work was keyword-searched, not
  full-text grepped.
- Every claim carries an evidence grade: `[PRIMARY]` (fetched it), `[SEARCH]` (corroborated by multiple
  search hits), `[SECONDARY]` (mirror/aggregator), `[MEMORY]`, `[UNVERIFIED]`. **Do not promote a grade
  when quoting.**

State the central finding as *"no released implementation and no paper we could reach"* — never as
*"no work exists."*

## Relevance to the existing codebases

Verified by direct inspection, and load-bearing for **(b)** and **(c)**:

- `shark-pose-3d` has **no UV atlas anywhere in the model code**. The "91 dense surface landmarks" are a
  placeholder — `shark_smpl.py:109-111` builds a one-hot regressor over `torch.linspace` vertex indices.
- `core/skeleton.py` is a **star, not a spine**: nearly everything parents directly to
  `body_midpoint_dorsal`, leaving `gill_slit` and `caudal_notch` as the only body-axis bending joints —
  **~2 lateral-bending DOF for the whole animal.** Adequate for a thunniform white shark; wrong model
  class for an anguilliform sevengill.
- `shark-morphometrics/config/keypoints_16.yaml` id 3 reads *"Back edge of last (5th) gill slit"* —
  hardcoded to five. Sevengills have seven. Ids 10–11 are `second_dorsal_*`, which sevengills lack.
- `SharkScarAnnotator` has **no species column** anywhere in its schema. A sevengill programme needs a
  migration before it needs a model.
- The 12 body zones in `annotation/models.py` are already a coarse discrete atlas of the body surface,
  and annotators are labelling into it today.
