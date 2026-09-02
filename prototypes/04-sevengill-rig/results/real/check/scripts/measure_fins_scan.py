"""Measure fin islands on the SCAN (bent mesh) so chart distortion cannot be blamed: face area, span, s-range, at two margins."""
import os, sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, os.environ["RIG_DIR"]); os.chdir(os.environ["RIG_DIR"])
import numpy as np, mesh3d
mesh = mesh3d.load_mesh("assets/sevengill.glb")
cl, info = mesh3d.extract_centerline_3d(mesh, n_stations=64, core_radius_frac=0.17)
frames = mesh3d.tube_frames(cl, (0.0, 1.0, 0.0)); coords = mesh3d.tube_coords(mesh, cl, frames)
V = np.asarray(mesh.vertices); F = np.asarray(mesh.faces)
fa = 0.5 * np.linalg.norm(np.cross(V[F[:,1]]-V[F[:,0]], V[F[:,2]]-V[F[:,0]]), axis=1)
for margin in (0.30, 0.15):
    det = mesh3d.detect_fins(mesh, coords, margin=margin, check=False)
    lab = np.asarray(det.labels, dtype=object); fl = lab[F]
    print("== margin %.2f ==" % margin)
    for name in ("pectoral_L", "pectoral_R", "pelvic_L", "pelvic_R", "dorsal", "anal"):
        if name not in det.fins: print("  %-11s not found" % name); continue
        f = det.fins[name]; idx = np.asarray(f["vertex_indices"]); pts = V[idx]
        area = fa[np.all(fl == name, axis=1)].sum()
        ins = np.asarray(f["insertion_centroid"]); span = np.linalg.norm(pts - ins, axis=1).max()
        s = coords.s[idx]; r = coords.r[idx]
        print("  %-11s verts %6d  area %.5f m2  span %.3f m  s-range %.3f-%.3f (mid %.3f)  r max %.3f  stations %s" % (name, len(idx), area, span, s.min(), s.max(), 0.5*(s.min()+s.max()), r.max(), f["station_range"]))
