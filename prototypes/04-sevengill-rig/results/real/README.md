# Prototype 04 on the real Meshy GLB (run 2026-09-01, BEFORE the fixes)

> Superseded by `results/real_v8/README.md` (2026-09-02): the pipeline fixes for the terraced fins, seam lines, folded dorsal tip, dropped tail and mis-named caudal are described and measured there.  This directory is kept as the before state.

Input: `Meshy_AI_Frilled_Shark_0901203940_texture.glb` (123,112,964 bytes, md5 0bf748b16c457186a7385afcf35c12eb),
copied to `assets/sevengill.glb`. Not committed (over 100 MB). The rigged output `sevengill_rigged.glb`
(171 MB) and the eight identity GLBs (45 MB each) are not committed either; every command below regenerates them.

Environment: Python 3.13.11, numpy 2.5.2, scipy 1.18.1, trimesh 5.1.0, torch 2.13.0, timm 1.0.29, node 26, gltf-validator via
`npm i gltf-validator` in `prototypes/04-sevengill-rig`. Branch head a959282.

## Deviations from the runbook (no pipeline module was modified)

| what | why | what was done instead |
|---|---|---|
| `gltf_export._DEFAULT_NODE_PATH` points at another machine's scratchpad | demo.py raised `gltf-validator not found under '/tmp/claude-0/-home-user-SanBox/.../node_modules'` | `export GLTF_VALIDATOR_NODE_PATH=<repo>/prototypes/04-sevengill-rig/node_modules` (an override the module already honours) |
| `--auto-up` is not a `rig_sevengill.py` flag (only `mesh3d.py` has it) | exact step-6 command fails: `rig_sevengill.py: error: unrecognized arguments: --auto-up` | ran `mesh3d.py --up 0 1 0 --auto-up` first (it kept +Y: "negating it does not clear the flags"), then `rig_sevengill.py ... --up 0 1 0` |
| `pattern.DEFAULT_SCHEMA_PATH` is hardcoded to `/home/user/SanBox/phase1b/p0-sevengill-schema/keypoints_sevengill_v1.yaml` | exact step-7 command fails with `FileNotFoundError` (see `identity_cli.log`); the same path breaks 35 of the 236 tests | `check/scripts/run_identity_with_schema.py` calls `texture_identity.run(...)` with the CLI's exact defaults plus `schema_path=<this checkout's yaml>` |

Tests with the validator path set (`pytest.log`): **200 passed, 1 skipped, 2 failed, 33 errors** (236 total). All 35 failures/errors are in
`tests/test_texture_identity.py` and raise the `/home/user/SanBox/...keypoints_sevengill_v1.yaml` FileNotFoundError; the 1 skip is
`tests/test_synth.py`, which hardcodes the *other* foreign path (the `/tmp/claude-0/-home-user-SanBox/.../scratchpad/node_modules`
validator location, lines 22-25) and ignores the env override, so "236 passed" is unreachable here without editing a test file.
`demo.py` (`demo.log`): input / rest / rigged GLBs all **0 errors, 0 warnings**.

Extra verification that is not in the runbook: `check/scripts/verify_rig_real.py` re-runs `run_pipeline` in-process (same arguments,
`validate=False`, no GLB written), reproduces the same fin counts and warnings, applies demo.py's step-7 surface comparison to the real
mesh (`check/as_scanned_surface_check.json`) and draws the fin-label overlays. The identity driver also sets `warnings.simplefilter("always")`,
which the CLI does not.

## Mesh (step 5)

1,013,814 vertices, 1,961,876 faces, extents (0.3572, 0.1200, 0.4352) m, visual kind `texture`, UVs present.
Single node, identity transform, Y-up (dorsal +Y), 239 connected components, not watertight. Base colour 8192x8192 JPEG,
plus a 4096 normal-type map and a 4096 metallic-roughness map.

## Rig (step 6) -- `rig_run.log`, `report/`

- centerline: 64 stations, length **0.5044 m**, voxel pitch 0.00340 m, head/tail width 0.0231/0.0093 m (head-first call correct), tau 0.00715
- de-bend: sagitta 0.1295 m = **25.7 %** of chart length; rest extents [0.6678, 0.2314, 0.1298] m
- warnings that fired (verbatim in `rig_run.log`): `check_anatomy: up vector probably flipped` -- a **false positive** of the caudal-span heuristic: negating +Y does not clear it, dorsal lands at phi 5.6 deg and pectoral_L on +Y; it fires because the terminal caudal lobe was mis-split (below). `estimate_roll: the body is rolled about its own axis by -108.2 deg end to end (-3.742 rad per unit s, r2 0.68)` -- treat with care: the fin phi centroids show no accumulating twist along the body (dorsal +5.6 deg at s 0.77, caudal_upper -3.5 deg at s 0.93, L/R pairs within 6-12 deg of mirror), so the linear roll fit is not a clean measurement of body torsion. Five `detect_fins: 2 disjoint islands classify as ...` demotions. The `expected fin ... not found` warning did NOT fire in the rig run; it fires (anal, caudal_upper, caudal_lower) only when the mesh is charted with the wrong axis (+Z).
- fins: 12 islands = 8 named + 4 unassigned. Six names are anatomically right (dorsal, anal, pelvic_L/R, pectoral_L/R; L/R not mirrored, dorsal posterior over/behind the pelvics as it should be for this species). The caudal is wrong: `caudal_upper` (1,119 verts, a sliver on top of the peduncle) and `caudal_lower` (12,080, the ventral base of the lobe, reaching past the chart end) are not the fin, while the long lobe (30,569 verts, `unassigned_island_3`, station 62, s = 1.09-1.27 of chart length, phi centroid 154 deg) stays labelled body and in the rest pose hangs about 23 deg below the axis and toward +Y -- inverted for a heterocercal tail, consistent with a hooked/rolled terminal frame carrying it rigidly. Asymmetries worth knowing: pelvic_L is at stations 39-43 but pelvic_R at 43-46 (about 29 mm stagger) and pelvic_R sits almost on the ventral midline; pectoral_R has 24 % more vertices than pectoral_L. In `check/labels_*.png` the unassigned islands are drawn body-grey (they carry the body label), not the yellow the legend promises.
- rig: 37 joints (13 spine + 24 fin; the 4 unassigned islands got zero-weight root/tip joints), weights (1013814, 37), max 3 influences. No spine joint lies aft of `spine_12_caudal_axis_2` (x = -0.230) while the mesh reaches x = -0.388, so the terminal 24 % of the body is bound to one joint.
- clips: cruise 134 f / 4.44 s (four 1.11 s beats), turn 134 / 4.44, escape 16 / 0.49, rest 101 / 3.33, as_scanned 46 / 1.50 (spine joint error max 0.00118 m = 0.234 % of the 0.5044 m chart length, 0.177 % of the 0.6678 m straight extent)
- GLB 170,973.9 kB, validator **0 errors, 0 warnings**
- `check/as_scanned_surface_check.json` (demo.py step-7 logic on this mesh): LBS-posed rig vs scan **RMS 1.70 % BL, max 10.5 % BL** (BL = 0.6678 m straight X extent, 3.3 voxels RMS); body-only RMS 1.64 %. Synthetic reference in the README is 0.60 % RMS. The 10.5 % max is almost certainly the single-joint caudal lobe.

## Texture -> identity (step 7) -- `identity_driver.log`, `identity/`

- individual #0: **59 spots** (threshold 0.435, 120 components; `individual0.json` provenance `oversized_dropped: 4`); warning verbatim from `identity_driver.log`: `copy_from_chart dropped 4 oversized component(s) (largest 4960 px = 16.1% of the chart; limit 2.0%) - unobserved or shadowed regions are not spots`
- the straighten step inside this path found only 5 fins (dorsal, pectoral_L/R, pelvic_L/R; `summary.json` straighten.fins_found), so anal and both caudal patches were never excluded from the read
- visual checks asked for in the runbook: **there ARE black regions on the baked meshes** -- a solid black wedge on the snout tip and a black blob at the dorsal-fin base that bleeds onto the fin's lower leading edge (about 2 % of body pixels in the individual #0 and resight rows of `contact_sheet.png`); no whole fin is black. Spot areas run 4-556 chart px (median 13, 17 % above 100 px); 37 of 59 spots sit at the eccentricity clamp 2.4; round trip recovers 37 / refits 32 of the 59.
- resight similarity 0.978 / 0.928 / 0.863 / 0.789 (9, 18, 27, 36 months); random 0.010 / 0.035 / 0.032; 307 s
- **Caveat that outranks the numbers:** the Meshy atlas is many dozens of irregular UV islands, each baked with its own illumination (`check/texture_real_1024.jpg`); read through the UVs the skin is a brightness mosaic (`identity/chart_real.png`). De-lighting removed 98.5 % of the low-frequency swing (0.184 -> 0.0028) but only 0.7 % of the high-frequency energy (0.1390 -> 0.1381), because island edges are high-frequency steps. The fitted "spots" are island shards and shadow blobs, not freckles. Real freckles are visible at native resolution (`check/texture_real_crop_native_1024.jpg`; 1032 of 1052 dark components are under 8 px native, median 2 px) and are sub-pixel after the 8192 -> 1024 cap. The resight/random similarity spread is a self-consistency check of the drift simulator (resights are perturbations of individual #0's own spot list) and says nothing about the animal. Dorsoventral contrast 1.03 -> 1.04 also shows the chart carries no countershading signal.

## Viewer

`viewer/index.html` (three.js, serve `results/real` over HTTP) and 18 saved frames in `viewer/frames/`. Frames 03-05 (cruise t = 0, 1.1, 2.2 s)
are all at the same beat phase (the beat is 1.111 s); frames 13-18 sample one beat at t = 0.00 / 0.28 / 0.56 / 0.83 s.

Verdict: in profile the rig reads as a sevengill (broad flat head, wide mouth, seven gill slits, spotted skin, one dorsal fin far back over the
pelvics, anal fin behind it). Cruise is a large lateral wave -- tail-tip excursion about 26 % BL peak-to-peak, growing from a nearly static
snout (contact_strip.png row 3) -- turn is a sustained C, escape a strong C-start, and as_scanned re-poses the straight rig onto the scan's C.
What is wrong is the tail: the terminal 24 % (the whole caudal lobe) rides one joint, keeps the scan's curl and droops below the axis, so it
swings as a rigid curled paddle instead of a heterocercal caudal.
