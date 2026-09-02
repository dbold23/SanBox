"""Chart the real mesh with given n_stations / core_radius_frac; print tail + fin diagnostics; render label crops."""
import os, sys, warnings, json
warnings.simplefilter("ignore")
sys.path.insert(0, os.environ["RIG_DIR"]); os.chdir(os.environ["RIG_DIR"])
import numpy as np, mesh3d, rig_sevengill
from PIL import Image, ImageDraw
n = int(sys.argv[1]); frac = float(sys.argv[2]); tag = "n%d_f%.2f" % (n, frac)
OUT = "results/real/experiments"; os.makedirs(OUT, exist_ok=True)
mesh = mesh3d.load_mesh("assets/sevengill.glb")
cl, info = mesh3d.extract_centerline_3d(mesh, n_stations=n, core_radius_frac=frac)
frames, coords, det0 = mesh3d._chart(mesh, cl, (0.0, 1.0, 0.0), check=False)
straight, target = mesh3d.debend(mesh, cl, frames, check_roll=False)
scoords = mesh3d.tube_coords(straight, target, mesh3d.tube_frames(target, (0.0, 0.0, 1.0))) if hasattr(mesh3d, "tube_frames") else None
det = mesh3d.detect_fins(straight, scoords, check=False) if scoords is not None else det0
T = np.asarray(frames[0]); up = np.array([0, 1.0, 0])
rad=np.asarray(info["radius"]); print("[%s] radius profile: min %.4f at station %d of %d; last 8: %s; T.up every 4th of last 24: %s" % (tag, rad.min(), int(rad.argmin()), len(rad), np.round(rad[-8:],4), np.round((T @ up)[-24::4],2)))
print("[%s] centerline length %.4f m, %d stations, tau %.5f, T.up at last 3 stations %s, rest extents %s" % (tag, info["length"], len(cl), info["tau"], np.round(T[-3:] @ up, 2), np.round(straight.extents, 4)))
V = np.asarray(straight.vertices); lab = np.asarray(det.labels, dtype=object)
for name in sorted(det.fins):
    f = det.fins[name]; v = V[np.asarray(f["vertex_indices"])]
    print("  %-20s %6d verts st %s s/L %s phi %6.1f  X[%.3f,%.3f] Z[%.3f,%.3f]" % (name, f["n_vertices"], f["station_range"], tuple(np.round(np.asarray(f["s_range"]) / info["length"], 3)), np.degrees(f["phi_centroid"]), v[:,0].min(), v[:,0].max(), v[:,2].min(), v[:,2].max()))
# renders: pectoral_L crop (top view), tail crop (side view), dorsal crop (side)
PAL = {"body": (170,175,180), "dorsal": (220,40,40), "anal": (240,150,30), "caudal_upper": (30,90,220), "caudal_lower": (60,190,230), "pectoral_L": (30,160,60), "pectoral_R": (150,220,60), "pelvic_L": (150,40,180), "pelvic_R": (230,100,220)}
faces = np.asarray(straight.faces); fl = lab[faces[:,0]]
rgb = np.array([PAL.get(l, (250,230,40)) for l in fl], float)
def render(axes, lo, hi, size, path):
    img = Image.new("RGB", size, (250,250,248)); d = ImageDraw.Draw(img)
    proj = rig_sevengill._Projector(lo, hi, axes, size, pad=0.02)
    tri = V[faces]; inside = np.all((tri[:, :, list(axes)] >= np.array(lo)[list(axes)]) & (tri[:, :, list(axes)] <= np.array(hi)[list(axes)]), axis=(1,2))
    idx = np.nonzero(inside)[0]
    nrm = np.cross(tri[idx,1]-tri[idx,0], tri[idx,2]-tri[idx,0]); nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    lam = np.abs(nrm @ (np.array([0.4,0.5,0.75])/np.linalg.norm([0.4,0.5,0.75]))); shade = np.clip(0.35+0.65*lam, 0, 1)
    depth = tri[idx][:, :, proj.depth_axis].mean(1); order = idx[np.argsort(depth)]
    px = proj(V); col = (rgb * 1.0)
    sh = dict(zip(idx, shade))
    for k in order:
        c = (col[k] * sh[k]).astype(int); d.polygon([tuple(px[i]) for i in faces[k]], fill=tuple(int(x) for x in c))
    d.text((6, 4), "%s %s" % (tag, path.split('/')[-1]), fill=(20,20,20)); img.save(path)
# left pectoral, viewed from above (X,Y), box around the island
pl = V[lab == "pectoral_L"] if (lab == "pectoral_L").any() else V[V[:,1] > 0.04]
lo = pl.min(0) - 0.02; hi = pl.max(0) + 0.02
render((0, 1), lo, hi, (900, 700), "%s/%s_pectoralL_top.png" % (OUT, tag))
# tail, side view (X,Z): x < -0.18
sel = V[V[:,0] < -0.17]; lo = sel.min(0) - 0.01; hi = sel.max(0) + 0.01
render((0, 2), lo, hi, (1000, 600), "%s/%s_tail_side.png" % (OUT, tag))
# dorsal fin, side view
dv = V[lab == "dorsal"] if (lab == "dorsal").any() else V[(V[:,2] > 0.03)]
lo = dv.min(0) - 0.03; hi = dv.max(0) + 0.03
render((0, 2), lo, hi, (800, 500), "%s/%s_dorsal_side.png" % (OUT, tag))
print("[%s] renders written" % tag)
