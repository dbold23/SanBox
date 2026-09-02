"""Verification only (no pipeline module is modified): recompute the rig in-process with
the same arguments as the CLI run, then (a) LBS-pose the rest mesh with the as_scanned
clip and measure the surface RMS against the scan exactly as demo.py step 7 does, and
(b) render fin-label-coloured views of the rest pose and the scan so the eight fin
names can be checked by eye."""
import os, sys, json, time, warnings
warnings.simplefilter("always")
HERE = os.environ["RIG_DIR"]; sys.path.insert(0, HERE); os.chdir(HERE)
import numpy as np
from PIL import Image, ImageDraw
import rig_sevengill, rig, demo

OUT = "results/real/check"; os.makedirs(OUT, exist_ok=True)
t0 = time.time()
res = rig_sevengill.run_pipeline(
    "assets/sevengill.glb", out=None, motions=("cruise", "turn", "escape", "rest"),
    fps=30.0, seconds=4.0, n_stations=64, up=(0.0, 1.0, 0.0), keep_bent=True,
    report=None, validate=False, seed=0, verbose=False)
print("pipeline recomputed in %.0f s" % (time.time() - t0), flush=True)

scan = np.asarray(res.mesh.vertices, float)
rest = np.asarray(res.straight_mesh.vertices, float)
faces = np.asarray(res.straight_mesh.faces, np.int64)
L = float(res.centerline_info["length"]); pitch = float(res.centerline_info["pitch"])
BL = float(rest[:, 0].max() - rest[:, 0].min())          # straightened X extent
labels = np.asarray(res.detection.labels, dtype=object)
body = labels == "body"

# (a) as_scanned: LBS-posed rest mesh vs the scan (demo.py step 7, verbatim logic)
as_clip = res.clips[rig_sevengill.AS_SCANNED_NAME]
rot = rig.quat_to_rotmat(as_clip.quats[-1])
posed = rig.lbs(rest, res.weights, res.skeleton, rot) + as_clip.meta["root_translation"]
rms_all, max_all = demo._rms_after_translation(posed, scan)
rms_body, max_body = demo._rms_after_translation(posed[body], scan[body])
joint_err = np.asarray(as_clip.meta["joint_error"], float)
summary = {
    "chart_length_m": L, "voxel_pitch_m": pitch, "straight_x_extent_m_BL": BL,
    "n_vertices": int(len(rest)), "n_body_vertices": int(body.sum()),
    "as_scanned_spine_joint_error_max_m": float(joint_err.max()),
    "as_scanned_spine_joint_error_max_BL": float(joint_err.max() / BL),
    "as_scanned_surface_rms_m": rms_all, "as_scanned_surface_max_m": max_all,
    "as_scanned_surface_rms_pct_BL": 100.0 * rms_all / BL,
    "as_scanned_surface_max_pct_BL": 100.0 * max_all / BL,
    "as_scanned_surface_rms_pct_BL_body_only": 100.0 * rms_body / BL,
    "as_scanned_surface_max_pct_BL_body_only": 100.0 * max_body / BL,
    "as_scanned_surface_rms_px": rms_all / pitch,
    "rest_extents_m": [float(x) for x in res.straight_mesh.extents],
    "scan_extents_m": [float(x) for x in res.mesh.extents],
    "label_counts": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
    "timings_s": {k: float(v) for k, v in res.timings.items()},
}
json.dump(summary, open(os.path.join(OUT, "as_scanned_surface_check.json"), "w"), indent=2)
print(json.dumps(summary, indent=2), flush=True)

# (b) fin-label-coloured orthographic renders (painter's algorithm, same projector as the report)
PALETTE = {
    "body": (170, 175, 180), "dorsal": (220, 40, 40), "anal": (240, 150, 30),
    "caudal_upper": (30, 90, 220), "caudal_lower": (60, 190, 230),
    "pectoral_L": (30, 160, 60), "pectoral_R": (150, 220, 60),
    "pelvic_L": (150, 40, 180), "pelvic_R": (230, 100, 220),
}
def colour_of(lab):
    if lab in PALETTE: return PALETTE[lab]
    return (250, 230, 40)  # unassigned islands: yellow
face_lab = labels[faces[:, 0]]
face_rgb = np.array([colour_of(l) for l in face_lab], float)

def render(verts, axes, size, title, path, light=(0.4, 0.5, 0.75)):
    img = Image.new("RGB", size, (250, 250, 248)); draw = ImageDraw.Draw(img)
    lo, hi = verts.min(0), verts.max(0)
    proj = rig_sevengill._Projector(lo, hi, axes, size)
    px = proj(verts); tri = verts[faces]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    lam = np.abs(nrm @ (np.asarray(light) / np.linalg.norm(light)))
    shade = np.clip(0.35 + 0.65 * lam, 0, 1)
    depth = tri[:, :, proj.depth_axis].mean(axis=1)
    order = np.argsort(depth)
    rgb = (face_rgb * shade[:, None]).astype(int)
    for k in order:
        draw.polygon([tuple(px[i]) for i in faces[k]], fill=tuple(int(c) for c in rgb[k]))
    # axes + legend
    ax = "XYZ"; draw.text((8, 6), "%s   horizontal=%s  vertical=%s" % (title, ax[axes[0]], ax[axes[1]]), fill=(30, 30, 30))
    y = 24
    for name in list(PALETTE) + ["unassigned_island_*"]:
        draw.rectangle([8, y, 22, y + 12], fill=colour_of(name)); draw.text((28, y), name, fill=(30, 30, 30)); y += 15
    img.save(path); print("wrote", path, flush=True)

SIZE = (1400, 560)
render(rest, (0, 2), SIZE, "REST (de-bent) side view, snout should be +X, dorsal +Z", os.path.join(OUT, "labels_rest_side_XZ.png"))
render(rest, (0, 1), SIZE, "REST (de-bent) top view, +Y is left flank", os.path.join(OUT, "labels_rest_top_XY.png"))
render(rest, (1, 2), (700, 560), "REST (de-bent) front view, looking down -X", os.path.join(OUT, "labels_rest_front_YZ.png"))
render(scan, (0, 1), SIZE, "SCAN as loaded, side view (mesh is Y-up)", os.path.join(OUT, "labels_scan_side_XY.png"))
render(scan, (0, 2), SIZE, "SCAN as loaded, top-down view (X,Z)", os.path.join(OUT, "labels_scan_top_XZ.png"))
print("done in %.0f s" % (time.time() - t0))
