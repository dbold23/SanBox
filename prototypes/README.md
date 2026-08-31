# Prototypes — sevengill canonical-surface re-ID programme

Implementation of the three candidate approaches from
[`docs/sevengill-canonical-reid/`](../docs/sevengill-canonical-reid/). Sequencing and kill criteria are
defined there; this directory is the code.

| # | Prototype | Status | Decides |
|---|---|---|---|
| [01](01-melops-ablation/) | Rigid-part vs deformable-body ablation (Melops) | **Runnable** — pipeline + synthetic smoke here; real run needs a machine with open egress (Zenodo + HuggingFace) | *Where identity lives.* Head − headless ≥ 15 Rank-1 pts → build a patch matcher, stop the 3D programme. |
| [02](02-centerline-chart/) | Centerline (arc-length × angle) chart + strain harness | **Runnable here** — pure numpy/scipy; doubles as the Phase 1B midline rectifier | *What bending costs.* Quantifies the residual a centerline chart cannot remove under injected ±5% strain. |
| [03](03-template-free-canonical/) | Template-free canonical shape from video | **Blocked by design** | Nothing yet — see its README for the three unblock conditions. |

## Order of operations

1. Run 01 on synthetic here (`--data synthetic`) to validate the protocol, then on real Melops from a
   machine with open egress. The kill criterion fires in weeks.
2. 02's rectifier is useful regardless of 01's outcome (it is the Phase 1B midline rectifier); its full
   chart earns budget only if 01 says identity is distributed.
3. 03 stays empty until its unblock conditions are recorded.

## Environment notes

- All code targets **Python 3.9** (`from __future__ import annotations`, no 3.10-only syntax) because
  the lab's Mac toolchain is 3.9 — see `shark-pose-3d/tasks/lessons.md`.
- Core paths need only `numpy pandas pillow scipy`; `torch`/`timm` are optional and import-guarded
  (needed only for the real backbones: MegaDescriptor-L-384, DINOv2, MiewID).
- This session's network policy blocks zenodo.org and huggingface.co, so real data and weights cannot
  be pulled here; the READMEs carry exact runbooks for an open-egress machine.
