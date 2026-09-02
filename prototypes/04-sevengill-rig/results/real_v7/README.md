# Prototype 04 on the real Meshy GLB, after the fixes (final run v7, 2026-09-02)

Same input mesh as `results/real/` (`assets/sevengill.glb`), except that the base-colour atlas has the
right eye mirrored onto the left eye (`assets/sevengill_eyefix.glb`, made by
`results/real/check/scripts/mirror_eye_texture.py`; geometry, UVs and the other two textures are
identical).  Same command as before, `--up 0 1 0`.  Neither GLB is committed (123 MB / 153 MB in,
171 MB out).

## What was wrong and what changed

| symptom (user report) | cause | fix |
|---|---|---|
| terraced, stretched left pectoral; right pectoral compressed | fins were charted through the tube: on a 25 % sagitta C-curve a blade on the inside of the bend spans 2.9x more arc length than its base | fin islands are carried rigidly in the frame at their insertion and blended into the chart over half a body radius above the detection threshold (`mesh3d.map_mesh`, used by `debend`, `rebend`, `synth.bend`) |
| faint seam lines across the body at every station | the chart was a 64-segment polyline: at each corner a vertex at radius r jumps by r x turn | the chart runs on a cubic B-spline approximant of the stations, upsampled 8x, s reported in station chord length (`DEFAULT_UPSAMPLE`, `densify_centerline`) |
| dorsal tip folded back, overhang at its base; pelvic and anal fins "sliced in half"; a swaying triangle on the peduncle | the fin tip joint was the island vertex farthest from the insertion, which on a long low fin is the front base corner, so the fin drive hinged the blade fore-aft | tip = vertex protruding farthest from the body axis on the insertion's side (`rig.fin_info_from_detection`) |
| tail tip pointing down | the medial path climbed into the upper caudal lobe on its last 3 stations (terminal tangent 46 deg up); straightening removed that pitch and the rigidly carried lobe rotated down with it.  The rotation-minimising frame itself stays within 3 deg of world-up, so the "-108 deg roll" warning was not the cause | end hooks are trimmed in `extract_centerline_3d` (turn > 3x median within the last 5 % of stations) |
| caudal lobe unnamed, anal fin split, "up vector flipped" warning | the Meshy mesh delivers fins as several disconnected shells; each was named alone, and a 10-vertex sliver at the right phi beat the 29,540-vertex lobe in the prior tie-break | touching same-sector shells are merged before naming; an island that *starts* past 0.85 of the chart is caudal whatever its sector; slivers under 5 % of the largest contender lose collisions |
| left eye textured in the wrong place | Meshy atlas | right-eye texels copied onto a 22 mm disc around the true left eye: the mirrored left head is registered onto the right head by ICP (the head is not mirror-symmetric, pure mirroring landed 3.7 mm off, one iris width), every atlas texel of the left disc is sampled at its registered position on the right flank (texture only) |

Not fixed, because they are in the mesh, not the pipeline: the right pectoral has 36 % more blade area
than the left (measured on the untouched scan), and the pelvic fins are staggered by 29 mm of arc
length on the scan itself.

## Numbers

| quantity | before (`results/real`) | after (`results/real_v7`) |
|---|---|---|
| chart length | 0.5044 m | 0.4803 m (hook of 3 stations trimmed) |
| rest extents | 0.668 x 0.231 x 0.130 m | 0.682 x 0.230 x 0.108 m |
| named fins | 8, but `caudal_upper` = 1,119-vertex sliver, lobe (30,569) unnamed | 8, `caudal_upper` = 29,528-vertex lobe, `anal` 5,372 (both shells) |
| unassigned islands | 4 | 3 (two snout lumps, one 10-vertex sliver) |
| joints | 37 | 35 (13 spine + 22 fin); caudal_upper bone points aft (-0.092, +0.044, -0.013 m) |
| as_scanned spine joint error max | 0.00118 m | 0.00084 m |
| as_scanned surface RMS / max vs scan | 1.70 % / 10.5 % BL | 0.75 % / 2.9 % BL |
| body-only RMS | 1.64 % BL | 0.46 % BL |
| validator | 0 / 0 | 0 / 0 |
| synthetic ground truth (demo.py): de-bend RMS | 0.49 % BL (body 0.16 %) | 0.49 % BL (body 0.14 %) |
| synthetic: skinned surface vs scan | 0.60 % BL | 0.60 % BL |
| tests | 200 passed + 35 schema-path errors | 199 non-identity tests pass, 7 new (same 35 errors elsewhere) |

Warnings that still fire: `estimate_roll` (-139 deg, r2 0.68; see the `check_anatomy` note) -- the estimator folds the fattest-vertex
phi with period pi and the fin phi centroids show no accumulating twist, so it is noise on this mesh;
one `detect_fins` demotion (the 10-vertex sliver).

## Files

`report/` (pipeline report), `check/` (label overlays, surface check, zooms), `frames/` (three.js viewer
frames incl. both eyes, tail, dorsal), `eye/` (eye-patch before/after renders and the detected centres),
`rig_run.log`, `check.log`.  The viewer at `results/real/viewer/index.html` takes `?glb=../v7/sevengill_rigged.glb`.

Review: an independent three-lens review of the diff (invertibility, regression/test adequacy, design/docs) found three bugs that were fixed before this run -- `rebend` reusing fin records built for another centerline or inherited from the input mesh, `_merge_shells` able to fuse a pelvic fin with the anal fin, and the end-hook trim cutting a genuine lateral tail flex (it now judges the sagittal turn only) -- plus the caudal tip rule, the blend threshold index and the upsample threading in `map_mesh`.  Seven tests were added for the new behaviour.
