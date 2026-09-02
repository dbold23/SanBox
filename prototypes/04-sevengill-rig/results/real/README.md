# Prototype 04 on the real Meshy GLB (run 2026-09-01)

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

Tests with the validator path set: **200 passed, 1 skipped, 2 failed, 33 errors** (236 total); every failure/error is the hardcoded schema path.
`demo.py`: input / rest / rigged GLBs all **0 errors, 0 warnings**.

## Mesh (step 5)

1,013,814 vertices, 1,961,876 faces, extents (0.3572, 0.1200, 0.4352) m, visual kind `texture`, UVs present.
Single node, identity transform, Y-up (dorsal +Y), 239 connected components, not watertight. Base colour 8192x8192 JPEG,
plus a 4096 normal-type map and a 4096 metallic-roughness map.

## Rig (step 6) -- `rig_run.log`, `report/`

- centerline: 64 stations, length **0.5044 m**, voxel pitch 0.00340 m, head/tail width 0.0231/0.0093 m (head-first call correct), tau 0.00715
- de-bend: sagitta 0.1295 m = **25.7 %** of chart length; rest extents [0.6678, 0.2314, 0.1298] m
- warnings that fired (verbatim in `rig_run.log`): `check_anatomy: up vector probably flipped` (not a true flip: negating +Y does not clear it; the dorsal fin lands at phi 5.6 deg); `estimate_roll: the body is rolled about its own axis by -108.2 deg end to end (-3.742 rad per unit s, r2 0.68)`; five `detect_fins: 2 disjoint islands classify as ...` demotions. The `expected fin ... not found` warning did NOT fire in the rig run (all 8 names were assigned); it fires for anal, caudal_upper, caudal_lower only when the mesh is charted with the wrong axis (+Z).
- fins: 12 islands = 8 named + 4 unassigned. Six names are anatomically right (dorsal, anal, pelvic_L/R, pectoral_L/R; L/R not mirrored). The caudal is wrong: `caudal_upper` (1,119 verts) and `caudal_lower` (12,080) are patches at the peduncle, while the long lobe (30,569 verts, `unassigned_island_3`, s = 1.09-1.27 of chart length) stays labelled body and, in the rest pose, hangs ventrally and to +Y because the terminal frame is rolled. See `check/labels_*.png`.
- rig: 37 joints (13 spine + 24 fin; the 4 unassigned islands got zero-weight root/tip joints), weights (1013814, 37), max 3 influences
- clips: cruise 134 f / 4.44 s, turn 134 / 4.44, escape 16 / 0.49, rest 101 / 3.33, as_scanned 46 / 1.50 (spine joint error max 0.00118 m)
- GLB 170,973.9 kB, validator **0 errors, 0 warnings**
- `check/as_scanned_surface_check.json` (demo.py step-7 logic on this mesh): LBS-posed rig vs scan **RMS 1.70 % BL, max 10.5 % BL** (BL = 0.6678 m straight extent); body-only RMS 1.64 %

## Texture -> identity (step 7) -- `identity_driver.log`, `identity/`

- individual #0: **59 spots** (threshold 0.435, 120 components); warning verbatim: `copy_from_chart dropped 4 oversized component(s) (largest 4960 px = 16.1% of the chart; limit 2.0%) - unobserved or shadowed regions are not spots`
- resight similarity 0.978 / 0.928 / 0.863 / 0.789 (9, 18, 27, 36 months); random 0.010 / 0.035 / 0.032; 307 s
- **Caveat that outranks the numbers:** the Meshy atlas is hundreds of separately lit UV islands (`check/texture_real_1024.jpg`); read through the UVs the skin is a brightness mosaic (`identity/chart_real.png`), so the fitted "spots" are island shards, not freckles. Real freckles are visible at native resolution (`check/texture_real_crop_native_1024.jpg`) but are sub-pixel after the 8192 -> 1024 cap.

## Viewer

`viewer/index.html` (three.js, serve `results/real` over HTTP) and 12 saved frames in `viewer/frames/`.
