# APPLY — the `shark-morphometrics` patch

One patch, additive only, against a clean `main` of `shark-morphometrics`
(base commit `cf1e7b4`, "Initial commit: white shark 16-keypoint pose detector and morphometrics
pipeline"). No existing file is modified, so the white-shark pipeline is unchanged.

    patches/0001-Add-sevengill-keypoint-schema-S1-and-a-species-param.patch

## Apply

Work on a branch. Never push from this environment.

```sh
cd /home/user/shark-morphometrics
git checkout main
git checkout -b sevengill-schema-s1
git am /home/user/SanBox/phase1b/p0-sevengill-schema/patches/0001-*.patch
```

If `git am` stops for any reason, `git am --abort` returns the tree to where it was.

## Run the tests that prove it

```sh
cd /home/user/shark-morphometrics/shark_pose_project
python -m pytest tests/ -q
```

Expected: `15 passed`. Requires only `pytest` and `pyyaml`; `pyyaml>=6.0` is already in
`shark_pose_project/requirements.txt` and no new dependency is introduced. The repo had no test
directory before this patch, so these are the first tests in it.

Verified in this environment on a throwaway clone of `main`: `git am` applied cleanly and the
suite reported `15 passed in 0.10s`.

## What the tests actually check

* both config yamls load and pass structural validation (ids contiguous, edges and morphometric
  pairs reference real ids, declared `kpt_shape` / `num_keypoints` / `flip_idx` agree with the
  keypoint count);
* **Schema v2 is untouched** — the 16 white-shark names, their order, `caudal_notch` still at
  index 13, and the pre-existing permissive horizontal-flip default;
* the sevengill schema has no `second_dorsal_*`, brackets gill slits 1 and 7, no longer carries
  `gill_slit_back`, forbids mirror augmentation, and exposes an ordered midline chain at
  fractions k/8.

Two properties of the shipped sevengill yaml are worth knowing before writing a consumer, and are
pinned by the design-artifact tests below rather than by the repo suite:

* `skeleton_edges` is a **forest, not a connected skeleton** — 19 edges over 30 keypoints in 11
  components, with the roots listed in `skeleton_contraction_roots`. It is computed as the
  contraction of the sevengill kinematic tree onto the keypoints, and the tree's root is an
  unlabelled cranial spine station, so no connected contraction exists. `COMPAT.md` lists the
  skeleton-consuming scripts this affects.
* `ordered_ap_sequence` asserts **no order among `pelvic_origin`, `dorsal_fin_origin` and
  `cloaca`**. They are bracketed by pectoral and anal and nothing more; any specific order is
  `[UNVERIFIED]` and must not be enforced by a validator or a matching cost.

## Design-artifact tests (not part of the patch)

`skeleton_sevengill.py` is a candidate for `shark-pose-3d`, not a patch — integration there is a
later decision. Its tests live in this deliverable directory:

```sh
cd /home/user/SanBox/phase1b/p0-sevengill-schema
python -m pytest tests/ -q
```

Expected: `44 passed`. Requires `pytest`, `torch` and `pyyaml`. These are the tests that recompute
`skeleton_edges` from `KINEMATIC_TREE` and assert equality, pin the unordered pelvic/dorsal/cloaca
trio, and check the bending basis is orthonormal at every valid `n_modes`.

## Not patched

23 modules duplicate the Schema-v2 name list and a dozen more hardcode the count 16. They are
inventoried by file and line in `COMPAT.md`, together with a pre-existing `kpt_shape: [15, 3]` /
`flip_idx: range(16)` inconsistency in `scripts/split_dataset.py`, and a safe migration order.
None of it is rewritten here because the repo carries no test coverage against which such an edit
could be shown safe.
