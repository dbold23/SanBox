# COMPAT — what `shark-morphometrics` hardcodes, and what the patch did and did not touch

Repo inspected: `/home/user/shark-morphometrics` at `cf1e7b4` (branch `sevengill-schema-s1`).
All paths below are relative to `shark_pose_project/`.

## What the patch changes

Additive only. No existing file is modified.

| added | why |
|---|---|
| `config/keypoints_sevengill_v1.yaml` | Schema S1, 30 points, Hexanchiformes. Sits beside `keypoints_16.yaml`; does not replace it. |
| `scripts/shared/schema.py` | Species-parameterised loader + validator. New code can ask for a schema by key instead of pasting a name list. Existing imports of `shared/constants.py` are untouched, so no current script changes behaviour. |
| `tests/test_keypoint_schemas.py` | 15 pytest tests. Validates both yamls, and pins the invariant that matters: Schema v2's 16 names, order and permissive flip default are unchanged, while the sevengill schema has no `second_dorsal_*`, brackets seven gill slits, and forbids mirror augmentation. |

The repo had **no test directory and no test runner** before this patch. `pyyaml>=6.0` is already in
`requirements.txt`, so the loader adds no dependency.

## What the patch deliberately does NOT change, and why

The v2 keypoint name list is **duplicated verbatim in 23 modules**. `scripts/shared/constants.py`
declares itself "Canonical source for keypoint schema … Import from here instead of duplicating",
and 22 files ignore it. Rewriting those call sites to read the yaml is a 23-file edit against a
repo with **zero test coverage** and no reproducible dataset in this environment — there is no way
to demonstrate such an edit is safe, and a silent break in the white-shark training pipeline is a
much worse outcome than a documented duplication. So the duplications are inventoried here rather
than patched.

### Inventory: duplicated Schema-v2 name lists

Each of these declares its own copy of the 16 names. `constants.py:14` is the intended canonical
one; the other 22 are copies of it.

| file | line | symbol |
|---|---|---|
| `scripts/shared/constants.py` | 14 | `KP_NAMES` (canonical) |
| `deploy/import_skeleton.py` | 22 | `KEYPOINTS` |
| `scripts/active_learning_select.py` | 43 | `KP_NAMES` (+ `N_KP = len(KP_NAMES)  # 16` at :50) |
| `scripts/blender_keypoints.py` | 40 | `KP_NAMES` |
| `scripts/compare_models.py` | 22 | `KP_NAMES` |
| `scripts/cvat_to_yolo.py` | 34 | `EXPECTED_KEYPOINTS` |
| `scripts/eda_annotations.py` | 16 | `KP_NAMES` |
| `scripts/export_annotation_morphometrics.py` | 21 | `KP_NAMES` |
| `scripts/export_v2_dataset.py` | 28 | `KEYPOINT_ORDER` |
| `scripts/extract_and_preannotate.py` | 33 | `KP_NAMES` |
| `scripts/package_for_roboflow.py` | 38 | `KP_NAMES` |
| `scripts/plot_morphometrics.py` | 25 | `KP_NAMES` |
| `scripts/preannotate_pool.py` | 33 | `KP_NAMES` |
| `scripts/render_measurement_overlays.py` | 25 | `KP_NAMES` |
| `scripts/render_pose_overlays.py` | 30 | `KP_NAMES` |
| `scripts/sam_demo.py` | 122 | `KP_NAMES` |
| `scripts/scarapp_to_yolo.py` | 28 | `KEYPOINT_ORDER` |
| `scripts/select_gap_frames.py` | 36 | `KP_NAMES` |
| `scripts/validate_annotations.py` | 35 | `KP_NAMES` |
| `scripts/validate_model.py` | 35 | `KP_NAMES` |
| `scripts/visualize_predictions.py` | 25 | `KP_NAMES` |
| `scripts/visualize_v3_failures.py` | 23 | `KP_NAMES` |
| `scripts/yolo_to_cvat.py` | 23 | `KP_NAMES` |

### Inventory: hardcoded cardinality 16

| file | line | what |
|---|---|---|
| `config/keypoints_16.yaml` | 152–154 | `kpt_shape: [16, 3]`, `flip_idx: [0..15]`, `num_keypoints: 16` |
| `scripts/export_v2_dataset.py` | 88 | `yolo_kps = [[0.0, 0.0, 0.0] for _ in range(16)]` |
| `scripts/export_v2_dataset.py` | 277–278 | `"kpt_shape": [16, 3]`, `"flip_idx": list(range(16))` |
| `scripts/split_dataset.py` | 109–110 | `"kpt_shape": [15, 3]`, `"flip_idx": list(range(16))` — **pre-existing inconsistency, see below** |
| `scripts/cvat_to_yolo.py` | 60, 67 | `return list(range(16))` as the fallback index mapping |
| `scripts/train_v4_no_negatives.sh` | 52, 56 | `'kpt_shape': [16, 3]`, `'flip_idx': list(range(16))` |
| `scripts/frame_quality_scorer.py` | 53 | docstring: "fraction of all 16 keypoints visible" |
| `scripts/compute_morphometrics.py` | 42 | docstring: "from a set of 16 keypoints (schema v2)" |
| `scripts/compute_morphometrics_batch.py` | 51 | docstring: "from 16 keypoints" |
| `scripts/render_measurement_overlays.py` | 147 | on-image caption `f"{len(kps)}/16 keypoints visible"` |
| `scripts/package_for_roboflow.py` | 170 | comment/branch "Pad if fewer than 16 keypoints" |
| `progress.json` | 151 | schema note recording the 15→16 migration |

### Inventory: references to `second_dorsal_*`, which does not exist on *Notorynchus*

Beyond the 23 name lists above:

| file | line | what |
|---|---|---|
| `config/keypoints_16.yaml` | 76–81 | keypoint id 10 `second_dorsal_tip` |
| `config/keypoints_16.yaml` | 29 | id 3 description: **"Back edge of last (5th) gill slit"** — wrong slit count for this order |
| `config/scars.yaml` | 159, 166 | body-zone boundaries defined against `second_dorsal_base` / `second_dorsal_tip` |
| `config/cvat_skeleton.json` | 12–13, 31–33, 43–44 | CVAT skeleton nodes, edges and tier strings |
| `deploy/annotator_guide.md` | 26 | annotator instruction table |
| `deploy/import_skeleton.py` | 41 | skeleton edge `[9, 10]` |
| `scripts/blender_morphometrics.py` | 50 | 3D rest position for keypoint 10 |
| `scripts/generate_preannotations.py` | 54 | name list inline in a dict literal |
| `scripts/migrate_yolo_labels_v2.py` | 14, 41 | v1→v2 index remap table |
| `scripts/yolo_to_cvat.py` | 41 | skeleton edge |

`config/scars.yaml` is the one that will bite hardest: the white-shark **body-zone** definitions are
anchored to a second dorsal fin, so the zone atlas cannot be reused for sevengills as-is. That is a
separate P0 item, not a keypoint-schema item, and it is not addressed by this patch.

### One shape difference a skeleton consumer must handle: S1's edge list is a FOREST

`keypoints_16.yaml`'s skeleton is a connected graph. Schema S1's `skeleton_edges` is **not**: it is
the computed contraction of the sevengill kinematic tree onto the keypoint set, and it comes out as
**19 edges over 30 keypoints in 11 components** (roots listed in `skeleton_contraction_roots`: ids
0–9 and 23; ids 1–8 are isolated vertices). The cause is structural, not an omission — the whole
cranial and branchial block hangs off unlabelled spine stations, so no connected contraction of
that tree exists. See `RATIONALE.md`.

Consequences for the tools that consume a skeleton, none of which this patch touches because none
of them reads the sevengill yaml yet:

| file | line | assumption to check before pointing it at S1 |
|---|---|---|
| `deploy/import_skeleton.py` | 22, 41 | builds a CVAT skeleton from a hardcoded v2 edge list; will need the S1 edges, and must tolerate isolated nodes |
| `config/cvat_skeleton.json` | 12–13, 31–33, 43–44 | v2 nodes/edges/tiers; a sevengill equivalent has to be generated, not hand-edited |
| `scripts/yolo_to_cvat.py` | 41 | same, on the export side |

`scripts/shared/schema.py` itself imposes no connectivity requirement — `validate()` only checks
that each edge references two distinct real ids — so the loader is already correct for a forest.

## A pre-existing bug found in passing (not fixed here)

`scripts/split_dataset.py:109-110` writes a dataset yaml with `kpt_shape: [15, 3]` and
`flip_idx: list(range(16))`. The two disagree, and the repo migrated to 16 keypoints
(`progress.json:151`, `scripts/migrate_yolo_labels_v2.py:139`). Any ultralytics run reading a
`data.yaml` from this script trains a 15-keypoint head against 16-keypoint labels, or fails on the
flip index. Left untouched because it is on the white-shark path, is unrelated to the sevengill
schema, and cannot be verified in this environment. Flagged for the repo owner.

## Migration route, when someone does want to parameterize

The cheap, safe order — each step independently testable:

1. Make `scripts/shared/constants.py` derive `KP_NAMES` from `load_schema("white_shark").names`
   instead of a literal, keeping every other symbol byte-identical. One file, and
   `tests/test_keypoint_schemas.py` already pins the expected names and order.
2. Replace the 22 duplicate lists with `from shared.constants import KP_NAMES`, one file per
   commit. Mechanical, and each is independently revertible.
3. Only then thread a `--species` argument through the export and training scripts.

Do not attempt step 3 before steps 1 and 2, and do not attempt any of it without first getting the
white-shark pipeline under a smoke test.
