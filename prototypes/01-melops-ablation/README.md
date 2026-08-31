# Prototype 01 — Melops rigid-part vs deformable-body ablation (Phase 1A)

Implements the Phase 1A experiment from Approach 1 of
`docs/sevengill-canonical-reid/03-candidate-approaches.md`: does fish identity
live in a small rigid anterior patch (head), or in the whole deformable body?
One-shot, open-set, time-separated, side-partitioned protocol over the Melops
corkwing-wrasse corpus, across four arms: **head / body / headless /
cross_orientation**.

Runs in two modes with identical code paths:

* **synthetic** — a deterministic PIL-rendered miniature corpus with
  controllable head/body identity-signal strengths (used by the tests to prove
  the ablation *detects* both a head-concentrated and a distributed corpus);
* **melops** — the real dataset via `wildlife-datasets`, on a machine with
  open egress.

## Do-not-overread block (verbatim intent of the spec)

> Melops fish are *board-mounted* — handled, shot against a standardised white
> board with a colour reference card, not laterally bending in frame. Its
> `orientation` field is a two-valued side flip, not a body-axis bend
> parameter. This experiment settles **where identity lives** (rigid part vs
> deformable flank). It **cannot** measure what pose normalization / unwrapping
> buys on a bending body, and no existing fish or elasmobranch dataset can.
> Do not let a clean result here be read as evidence that unwrapping works.

## Quick smoke (zero optional deps: numpy, pandas, Pillow, pytest only)

```bash
cd prototypes/01-melops-ablation
python -m pytest tests/ -q                    # < 15 s
python run_ablation.py --data synthetic --root corpus/ --backbone hist \
    --arms head,body,headless,cross_orientation --out results/
```

`results/results.json` and `results/report.md` carry per-arm Rank-1 / Rank-5 /
mAP, open-set novelty AUROC, a rejection-threshold curve, and the
kill-criterion verdict:

* head Rank-1 exceeds headless Rank-1 by **>= 15 points** →
  `KILL / redirect: identity concentrated in rigid part - build a patch matcher, not a surface`
* |headless − body| **<= 5 points** →
  `identity distributed - Approach 2 earns a hearing`
* otherwise → `intermediate - widen data before deciding`

## Running the real thing (lab Mac, Python 3.9, open egress)

All modules start with `from __future__ import annotations` and avoid 3.10+
syntax; they import cleanly on 3.9 (checked with `vermin`: minimum 3.7).

1. **Install**

   ```bash
   pip install numpy pandas pillow scipy pytest
   pip install wildlife-datasets            # data loader (AGPL-3.0 — see licences)
   pip install torch torchvision timm       # deep backbones (optional)
   ```

2. **Download Melops** (Zenodo record 17404087, ~27.7 GB class of archives;
   the wrapper pulls the body-crop archive + metadata by default):

   ```python
   from wildlife_datasets import datasets
   datasets.Melops.get_data("/data/Melops")
   ```

   Note (from `melops.py` itself): `get_data` downloads **only the body-crop
   archive plus metadata**. Head and headless views are produced by applying
   the shipped `bbox_head` / `bbox_headless` columns to the body crops
   (`bbox="head"` / `"headless"`); the separate head/headless image archives
   on Zenodo must be requested explicitly if you want the pre-cropped files.
   One metadata image (`P8316190_2020`) is missing from the body archive and
   is dropped with a warning.

3. **Run the ablation**

   ```bash
   python run_ablation.py --data melops --root /data/Melops \
       --backbone megadescriptor --arms head,body,headless,cross_orientation \
       --out results-melops/
   ```

   Backbone model strings (exact, from the spec):
   * `megadescriptor` → timm `hf-hub:BVRA/MegaDescriptor-L-384` at 384 px
   * `dinov2` → timm `vit_small_patch14_dinov2.lvd142m` (ViT-S/14, 518 px)
   * `miewid` → `conservationxlabs/miewid-msv2` at 440 px — **licence
     unsettled**, see below
   * `hist` (numpy-only sanity arm) and `random` (chance floor) always work.

4. **Expected operating point**: the prior method paper on the same population
   (arXiv:2301.00596, NLDL 2023) reports **one-shot 0.35**, 5-shot 0.56,
   100-shot 0.88 **[SEARCH-grade, confirm against the PDF before quoting]**.
   One-shot ~0.35 is the realistic Rank-1 ballpark for the body arm — do not
   expect synthetic-corpus numbers.

5. **Runtime expectations**: Melops is 24,578 images / 9,861 individuals
   (~2.5 images/individual — singletons are the norm and the protocol is built
   to survive them). CPU embedding with MegaDescriptor-L-384 is roughly
   ~1 s/image → plan on hours per arm on CPU, minutes on a GPU. The `hist`
   backbone runs the full corpus in minutes on CPU and is the right first
   end-to-end check. The synthetic smoke (tests + CLI) is under 60 s total.

## Protocol contract (what the tests enforce)

* Gallery = the **earliest** sighting of each `(identity, side)` unit whose
  first sighting precedes the cutoff — one image per unit (one-shot).
* Queries = all later sightings of enrolled units (known) plus every sighting
  of units first seen after the cutoff (novel, open set).
* **No image is ever both gallery and query**; identity leakage and image
  reuse raise `ProtocolViolation`, they are not warnings.
* **Matching is side-partitioned**: a query is only ever compared against
  same-side gallery entries. The `cross_orientation` arm (enroll L, query R)
  is **the only arm that crosses sides, by design**, and is evaluated through
  an explicit `cross_side=True` gate that verifies the frames really are
  single-and-opposite-sided.
* **Same-session near-duplicates are excluded by default**: an enrolled
  unit's other photos from the *same date* as its gallery image are dropped
  from the query set (`same_date_policy="exclude"`), because on real Melops
  they are same-handling-session near-duplicates that inflate known-query
  metrics. The excluded count is reported per arm as
  `n_same_date_excluded` in `results.json`; pass
  `same_date_policy="include"` to measure the inflation directly.
* **Dates must parse**: a NaT/missing date raises `ProtocolViolation`
  rather than being silently binned.
* Metrics: Rank-1 / Rank-5 / mAP over known queries (one-shot gallery ⇒ AP =
  1/rank), open-set novelty AUROC on max cosine similarity, and Rank-1 at
  rejection thresholds swept over max-similarity quantiles.
* Every stochastic step (synthetic rendering, date-tie breaking, random
  embedder) takes an explicit seed; runs are bit-deterministic.

## Licences (checked 2026-08-31, this session)

| Asset | Licence | Source |
|---|---|---|
| Melops dataset | **CC BY 4.0** | `melops.py` summary block [PRIMARY] |
| `wildlife-datasets` (loader) | **AGPL-3.0** — GNU Affero GPL v3, "Copyright (C) 2026 Lukáš Adam, Vojtěch Čermák and Lukáš Picek" | fetched `WildlifeDatasets/wildlife-datasets/main/LICENSE` this session [PRIMARY] |
| `wildlife-tools` | **MIT** — "Copyright (c) 2023 Vojtěch Čermák" | fetched `WildlifeDatasets/wildlife-tools/main/LICENSE` this session [PRIMARY] |
| MiewID (`wbia-plugin-miew-id`) | **UNSETTLED** — no OSI licence file, only "Copyright Conservation X Labs" per the report; settle terms with Conservation X Labs before building on the weights | spec §Approach 1 licence flags |

The AGPL finding matters: the spec's earlier pass guessed MIT for
`wildlife-datasets` and flagged it unverified — it is actually **AGPL-3.0**
(strong network copyleft). Importing it at runtime is fine for an internal
experiment; do **not** vendor its code into anything the lab intends to
release under a permissive licence. This prototype therefore keeps
`wildlife-datasets` as an optional, import-guarded dependency and ships its
own plain-CSV fallback loader.

## Files

| File | Contract |
|---|---|
| `melops_data.py` | `load_melops(root, bbox)` adapter (real loader when importable, plain `metadata.csv` fallback, identical column semantics: `orientation == side`, LTWH `bbox_{body,head,headless}` in body-crop pixels); `make_synthetic(...)` with `head_signal` / `body_signal` knobs |
| `embedders.py` | `hist`, `random` (numpy-only), `megadescriptor` / `dinov2` / `miewid` (torch/timm-guarded, exact model strings) |
| `protocol.py` | `one_shot_open_set_split`, `cross_orientation_split`, `evaluate`; all invariants raise `ProtocolViolation` |
| `run_ablation.py` | CLI; writes `results.json` + `report.md` with the verdict and the do-not-overread caveat |
| `tests/` | protocol invariants, chance-floor/above-chance metric sanity, verdict-detection on constructed corpora, CLI smoke |
