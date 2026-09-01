# DIGITALLIFE3D_LICENCE — the P0-b gate on Approach 2

**P0-b as written** (`03-candidate-approaches.md`): *"The lab's only shark mesh is the DigitalLife3D
great white (3,912 verts). **Its licence is UNKNOWN** … If the terms prohibit derivative works or
commercial/redistributive use, Approach 2 does not exist in its current form."*

**Result of this pass: the gate moves from UNKNOWN to AMBER — probably CC BY-NC 4.0, which would permit
everything Approach 2 needs for research, with one real downstream constraint. It is not yet GREEN,
because the licence string has still not been read first-hand.** Two of the three things that make it
GREEN cost five minutes each on an unrestricted machine; the third is the email in §5.

---

## 1. The single most useful finding: the answer was already in the lab's own repo

`/home/user/shark-pose-3d` records the asset's identity **and its licence**, first-hand:

* `tasks/progress.md:16` — `- [x] Great white shark mesh (DigitalLife3D Model 92A, 3912 verts, CC-BY-NC)`
* `tasks/progress.md:45` — `- Only one mesh template (DigitalLife3D great white, 3912 verts)`
* `scripts/process_shark_mesh.py:506` and `scripts/assign_weights_and_export.py:159` both stamp the
  exported OBJ with `# Great White Shark (DigitalLife3D Model 92A)`

`[PRIMARY]` — read directly from the clone at commit `f4bb27c`. So the mesh is **Model 92A**, and
whoever downloaded it recorded **CC-BY-NC** at the time. That does not close the gate on its own (a
progress-log annotation is not a licence grant, and it is undated and unsourced), but it is
independently corroborated in §2 and it means P0-b was never as dark as the scan believed.

## 2. What the Digital Life Project publishes, with grades

| # | claim | grade |
|---|---|---|
| 1 | The **Digital Life Project** is a non-profit initiative at **UMass Amherst**, started **November 2016**, creating high-resolution 3D models of living organisms; director **Duncan Irschick** | `[SEARCH]`, many independent hits (UMass news, phys.org, Yale e360) |
| 2 | Stated policy: models are **open access for non-profit scientific, educational and artistic use, including downloads**; low-resolution models are **free for creative and educational non-profit work** and are distributed via their **Sketchfab** account | `[SEARCH]`, repeated near-verbatim across several independent result snippets |
| 3 | **Commercial** use, and access to **high-resolution** models: contact director **Duncan Irschick, `duncan@umass.edu`**, who consults the **UMass Tech Transfer** officer | `[SEARCH]`, repeated across independent snippets |
| 4 | The shark assets are Sketchfab **"Model 92A — Great White Shark"** (animated) and **"Model 92C — White Shark Mesh"** (mesh only), both by `DigitalLife3D`, both made with the **Save Our Seas Foundation**, both built as a **composite adult female** from open-access images plus SOSF imagery/video | `[SEARCH]` |
| 5 | Both shark models are listed on Sketchfab under **CC Attribution-NonCommercial** | `[SEARCH]`, **single-source snippet** — the model page itself could not be opened |
| 6 | The great-white mesh in this lab is **Model 92A**, 3,912 verts, recorded as **CC-BY-NC** | `[PRIMARY]` (lab repo, §1) — corroborates 5 from a completely independent direction |

**Why nothing here is `[PRIMARY]` for the licence itself:** `digitallife3d.org` and `sketchfab.com` are
both **egress-blocked** by this environment's proxy (verified — both fetches returned `EGRESS_BLOCKED`),
exactly as `digitallife3d.org` was in the original scan. Every statement in rows 1–5 comes from search
result text, not from a page this session read `[UNVERIFIED as PRIMARY]`.

## 3. What CC BY-NC 4.0 would actually permit — mapped onto Approach 2

Approach 2 consists of *using and modifying* this mesh. Under **CC BY-NC 4.0** (assuming row 5/6 holds):

| Approach 2 activity | permitted under CC BY-NC 4.0? |
|---|---|
| Research use in a non-commercial academic project | **Yes** |
| **Retopology** into a sevengill template (seven gill slits, one posterior dorsal, long-lobed heterocercal caudal, blunt head) | **Yes** — "NC" restricts *purpose*, not modification. There is **no `ND`** in the licence name, so derivatives are allowed |
| Computing a centerline, cutting `(s, φ)` charts, deriving LBS weights from it | **Yes** — all derivative works |
| **Redistributing** the derived sevengill template to other researchers | **Yes, with attribution, non-commercially** — and the derivative inherits NC |
| Publication figures, thesis, conference renders | **Yes**, with attribution |
| Releasing the derived template under a permissive licence (MIT/CC BY) for the community | **NO.** NC propagates. This is the one real constraint |
| Any commercially-funded or commercially-licensed downstream use | **NO** without a separate grant from UMass (row 3) |

**So: if row 5/6 holds, Approach 2 is unblocked as a research programme.** The blocker the scan feared —
"terms prohibit derivative works" — is not the shape of the risk. The actual risk is *downstream
encumbrance*: an NC-derived sevengill template cannot be released openly, which conflicts with the
programme's stated intent to hand tooling back to partners and to publish reusable artifacts. That is
worth one email to fix in advance rather than discovering at release time.

## 4. Two live risks and one latent licence conflict

1. **The licence string has not been read first-hand.** Sketchfab's *display* licence and the licence on
   the *downloadable* file can differ, and a page can be re-licensed. **Five-minute fix:** from an
   unrestricted machine, open the two model pages and copy the licence text verbatim into this file,
   upgrading row 5 to `[PRIMARY]`.
   * `https://sketchfab.com/3d-models/model-92a-great-white-shark-702e7b53637f4ded9ca479a8124e810d`
   * `https://sketchfab.com/3d-models/model-92c-white-shark-mesh-94a0e9663ead46c08de4cf8473ff9822`
2. **Upstream provenance.** 92A/92C are described as a **composite** built from various open-access
   images plus Save Our Seas Foundation imagery and video (row 4). A composite can carry obligations
   Digital Life's own licence does not mention. The permission email should name **SOSF** explicitly and
   ask whether any separate attribution or consent is expected.
3. **Latent conflict inside `shark-pose-3d` — flag it to the PI now.** The repo's `README.md` declares
   **`## License` / `MIT`** for the whole repository `[PRIMARY]`, while its template mesh is CC-BY-NC.
   Today this is harmless: `git ls-files` confirms **no mesh, OBJ or derived `.npy` is committed** — only
   `scripts/process_shark_mesh.py`, which regenerates them locally `[PRIMARY]`. The conflict becomes real
   the moment `shark_template_mesh.obj`, `joint_regressor.npy` or any rendered dataset derived from the
   mesh is committed or released under the repo's blanket MIT. **Action:** before any such commit, add an
   `ASSETS.md` stating that mesh-derived artifacts are CC BY-NC 4.0 © Digital Life Project / UMass
   Amherst, with SOSF acknowledged, and that the MIT grant covers code only. The OBJ writer already
   stamps attribution (`# Great White Shark (DigitalLife3D Model 92A)`) — extend that line to carry the
   licence too, which is a one-line change in both export scripts.

## 5. Permission-request draft (136 words) — **DRAFT, not sent**

Send from Daniel Sambold, CSUMB. Route: `duncan@umass.edu` per row 3 `[SEARCH]` — **confirm the address
on `digitallife3d.org` before sending**; if the site now routes enquiries through a form, use the form.

> **Subject: Permission request — research use and retopology of Model 92A (Great White Shark)**
>
> Dear Dr. Irschick,
>
> I am Daniel Sambold at CSU Monterey Bay. We are building an individual photo-identification method for
> broadnose sevengill sharks, and would like written permission covering four uses of Digital Life's
> white shark model (Model 92A / 92C):
>
> 1. **Research use** in a non-commercial academic project.
> 2. **Modification** — retopologising the mesh into an anatomically correct sevengill template: seven
>    gill slits, a single posterior dorsal, no second dorsal.
> 3. **Redistribution of that derived template** to other researchers, with attribution to Digital Life
>    and the Save Our Seas Foundation, under whatever terms you specify.
> 4. **Publication figures** in papers, theses and talks.
>
> Digital Life would be credited in every figure and release. If the Sketchfab CC Attribution-NonCommercial
> listing already covers all four, a one-line confirmation is all we need.
>
> Thank you,
> Daniel Sambold — `dsambold@csumb.edu`

## 6. Gate status

| | |
|---|---|
| **Verdict** | **AMBER** — Approach 2 may be *planned* and *scoped*; no retopology hours are authorised until GREEN |
| **GREEN requires** | (i) the Sketchfab licence string read first-hand and pasted here, **and** (ii) a written reply to §5 covering redistribution of derivatives, **or** an explicit PI decision to proceed on CC BY-NC alone with the §4.3 asset-licensing note in place |
| **RED would require** | the model turning out to be `ND` (no derivatives), or the download carrying different terms from the display page. Then Approach 2 needs a commissioned or scanned mesh instead — check MorphoSource/oVert first (`Imageomics/pyMorphoSource`, MIT), noting CT resolves calcified cartilage, not skin |
| **Cost of staying AMBER** | low for now: Approach 2 is gated behind Phase 1B results anyway (`03-candidate-approaches.md` sequencing, months 9–18), so the licence answer is not on the critical path this quarter — but the email is free and the reply may take weeks, so **send it now** |

---

## Owner decision — 2026-09-01

The project owner has designated this use as **personal, non-commercial research** and directed the
programme to proceed under CC-BY-NC as-is. The gate moves **AMBER → CLEARED for internal research
use**. The permission-request draft above becomes optional courtesy (recommended before any
publication that redistributes a derived mesh), not a blocker.

Three obligations remain in force under CC-BY-NC and are now the compliance checklist:

1. **Attribution** — credit the Digital Life Project (Model 92A) wherever the mesh or a derivative
   appears: code, figures, acknowledgements.
2. **NC travels with derivatives** — a retopologised sevengill template derived from the mesh is
   itself CC-BY-NC. It may be used and shared for research, but never relicensed (not MIT, not
   CC-BY) and never used commercially.
3. **Do not commit the mesh or a derivative into an MIT-licensed repo without a licence carve-out**
   — `shark-pose-3d` is MIT; if the template lands there, add a `LICENSE-ASSETS` note stating the
   mesh-derived assets are CC-BY-NC and excluded from the MIT grant.

Revisit only if the programme's outputs stop being non-commercial (a paid product, a funded service
deliverable) or a derived mesh is redistributed publicly — then send the request letter first.
