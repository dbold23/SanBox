"""Mirror the (correct) right-eye texturing onto the left eye of the Meshy asset.

Works in the straight rest pose (snout +X, dorsal +Z, animal's left +Y), which is
left/right symmetric up to the model's own asymmetries.  Steps:
  1. per-vertex luminance from the base-colour atlas -> the right eye = darkest
     compact cluster on the head's right flank (a dark pupil in a pale ring);
  2. its mirror image (y -> -y) is the left-eye centre;
  3. every atlas texel inside the left-eye disc (rasterised through the UV
     triangles of the faces there) is replaced by the colour the RIGHT flank
     shows at the mirrored 3D position (closest surface point -> its UV -> atlas);
  4. the patched atlas is written back into a copy of the asset GLB.
Usage: python mirror_eye_texture.py RIGGED_OR_REST.glb ASSET.glb OUT_ASSET.glb OUT_DIR [--radius-mm 14]
"""
import os, sys, io, json
import numpy as np, trimesh
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

rest_path, asset_path, out_asset, out_dir = sys.argv[1:5]
radius = float(sys.argv[sys.argv.index("--radius-mm") + 1]) / 1000.0 if "--radius-mm" in sys.argv else 0.014
os.makedirs(out_dir, exist_ok=True)
rest = trimesh.load(rest_path, force="mesh")
V = np.asarray(rest.vertices, float); F = np.asarray(rest.faces); UV = np.asarray(rest.visual.uv, float)
asset = trimesh.load(asset_path)                       # scene, keeps all three textures
geom_name, geom = next(iter(asset.geometry.items()))
tex = geom.visual.material.baseColorTexture.convert("RGB"); W_, H_ = tex.size
img = np.asarray(tex).astype(np.float32)
assert len(geom.vertices) == len(V), "asset/rest vertex count mismatch"

def uv_to_px(uv):
    return np.column_stack([uv[:, 0] * (W_ - 1), (1.0 - uv[:, 1]) * (H_ - 1)])
def sample(uv):
    p = uv_to_px(uv); x = np.clip(np.round(p[:, 0]).astype(int), 0, W_ - 1); y = np.clip(np.round(p[:, 1]).astype(int), 0, H_ - 1)
    return img[y, x]

BL = V[:, 0].max() - V[:, 0].min()
lum = sample(UV).mean(1)
head = V[:, 0] > V[:, 0].max() - 0.22 * BL              # anterior 22% of the body
# The eye sits on the flank ABOVE the mouth line and behind the snout tip; the
# mouth corner (dark, ventral) is excluded by the z > 0 constraint.
right = head & (V[:, 1] < -0.004) & (V[:, 2] > 0.0) & (V[:, 0] < V[:, 0].max() - 0.03)
# a pupil is a small dark blob inside a bright ring: score = darkness of the
# 4 mm neighbourhood minus darkness of the 4-9 mm annulus
cand = np.flatnonzero(right); tree = cKDTree(V[cand])
thr = np.percentile(lum[cand], 3.0); dark = cand[lum[cand] <= thr]
best, best_score = None, -1e9
for p in V[dark]:
    inner = tree.query_ball_point(p, 0.004); outer = tree.query_ball_point(p, 0.009)
    if len(inner) < 20 or len(outer) <= len(inner): continue
    ring = np.setdiff1d(outer, inner)
    score = lum[cand[ring]].mean() - lum[cand[inner]].mean()
    if score > best_score: best_score, best = score, p
seed = best
disc = dark[np.linalg.norm(V[dark] - seed, axis=1) < 0.006]
eye_R = V[disc].mean(0)
print("pupil/ring contrast score %.1f" % best_score)
eye_L_mirror = eye_R * np.array([1.0, -1.0, 1.0])
# The head is not exactly mirror-symmetric (mirrored left points sit ~3.7 mm off
# the right surface, about one iris width).  Register the mirrored LEFT head
# onto the RIGHT head with ICP; the sampling map is then p -> T(mirror(p)) and
# the true left eye is mirror(T^-1(eye_R)).
MIR = np.array([1.0, -1.0, 1.0])
selL = head & (V[:, 1] > 0.004) & (np.linalg.norm(V - eye_L_mirror, axis=1) < 0.06)
selR = head & (V[:, 1] < -0.004) & (np.linalg.norm(V - eye_R, axis=1) < 0.06)
rng = np.random.default_rng(0)
srcL = V[rng.choice(np.flatnonzero(selL), min(6000, selL.sum()), replace=False)] * MIR
tgtR = V[rng.choice(np.flatnonzero(selR), min(20000, selR.sum()), replace=False)]
T_icp, _, cost = trimesh.registration.icp(srcL, tgtR, scale=False, max_iterations=50)
eye_L = (np.linalg.inv(T_icp) @ np.append(eye_R, 1.0))[:3] * MIR
print("ICP mirrored-left -> right: residual %.5f m; left eye moved %.4f m from the pure mirror" % (cost, np.linalg.norm(eye_L - eye_L_mirror)))
print("right eye centre (rest frame) %s, %d dark verts; left eye centre %s" % (np.round(eye_R, 4), len(disc), np.round(eye_L, 4)))

# faces whose vertices lie in the left-eye disc / right-eye disc
inL = np.linalg.norm(V - eye_L, axis=1) < radius
inR = np.linalg.norm(V - eye_R, axis=1) < radius * 1.6
facesL = np.flatnonzero(inL[F].any(1)); facesR = np.flatnonzero(inR[F].all(1))
print("left-eye faces %d, right-eye faces %d" % (len(facesL), len(facesR)))
rightmesh = trimesh.Trimesh(V, F[facesR], process=False)

# rasterise the left-eye faces in atlas space: for each texel, barycentric -> 3D -> mirror -> right surface -> UV -> colour
patched = img.copy(); n_tex = 0; dists = []
for fi in facesL:
    tri_uv = uv_to_px(UV[F[fi]]); x0, y0 = np.floor(tri_uv.min(0)).astype(int); x1, y1 = np.ceil(tri_uv.max(0)).astype(int)
    if x1 - x0 > 400 or y1 - y0 > 400:      # a degenerate/huge UV triangle: skip
        continue
    xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1)); P = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    a, b, c = tri_uv; v0, v1, v2 = b - a, c - a, P - a
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1; d20, d21 = v2 @ v0, v2 @ v1
    den = d00 * d11 - d01 * d01
    if abs(den) < 1e-9: continue
    w1 = (d11 * d20 - d01 * d21) / den; w2 = (d00 * d21 - d01 * d20) / den; w0 = 1 - w1 - w2
    inside = (w0 >= -0.02) & (w1 >= -0.02) & (w2 >= -0.02)
    if not inside.any(): continue
    Pw = P[inside]; bary = np.column_stack([w0, w1, w2])[inside]
    pos3 = bary @ V[F[fi]]
    if np.linalg.norm(pos3 - eye_L, axis=1).min() > radius: continue
    keep = np.linalg.norm(pos3 - eye_L, axis=1) < radius
    Pw, pos3 = Pw[keep], pos3[keep]
    mirrored = trimesh.transform_points(pos3 * MIR, T_icp)
    closest, dist, tri_id = trimesh.proximity.closest_point(rightmesh, mirrored)
    dists.append(dist)
    fr = F[facesR[tri_id]]
    # barycentric of the closest point on the right face -> UV
    A, B, C = V[fr[:, 0]], V[fr[:, 1]], V[fr[:, 2]]
    v0, v1, v2 = B - A, C - A, closest - A
    d00 = (v0 * v0).sum(1); d01 = (v0 * v1).sum(1); d11 = (v1 * v1).sum(1); d20 = (v2 * v0).sum(1); d21 = (v2 * v1).sum(1)
    den = np.maximum(d00 * d11 - d01 * d01, 1e-18)
    b1 = (d11 * d20 - d01 * d21) / den; b2 = (d00 * d21 - d01 * d20) / den; b0 = 1 - b1 - b2
    uv_r = b0[:, None] * UV[fr[:, 0]] + b1[:, None] * UV[fr[:, 1]] + b2[:, None] * UV[fr[:, 2]]
    col = sample(uv_r)
    xi = np.clip(np.round(Pw[:, 0]).astype(int), 0, W_ - 1); yi = np.clip(np.round(Pw[:, 1]).astype(int), 0, H_ - 1)
    patched[yi, xi] = col; n_tex += len(xi)
print("patched %d texels; mapped points to right surface: mean %.4f m, p95 %.4f m" % (n_tex, np.concatenate(dists).mean(), np.percentile(np.concatenate(dists), 95)))
out_img = Image.fromarray(np.clip(patched, 0, 255).astype(np.uint8))
geom.visual.material.baseColorTexture = out_img
asset.export(out_asset)
print("wrote", out_asset, os.path.getsize(out_asset) // 1_000_000, "MB")
# before/after crops of both eyes for review (per-vertex colour rendering of the head, side views)
def render_side(colors, sign, path, mark=None):
    sel = head & (np.sign(V[:, 1]) == sign) if sign != 0 else head
    idx = np.flatnonzero(inL | inR | (head & (np.abs(V[:, 1]) > 0)))
    pts = V[head]; cols = colors[head]
    proj = np.column_stack([pts[:, 0], pts[:, 2]]); size = 900
    lo, hi = proj.min(0), proj.max(0); sc = (size - 20) / max(hi - lo); px = (proj - lo) * sc + 10
    order = np.argsort(sign * pts[:, 1])  # far side first
    im = Image.new("RGB", (size, int((hi - lo)[1] * sc) + 20), (30, 40, 45)); d = ImageDraw.Draw(im)
    for k in order:
        if sign * pts[k, 1] < 0: continue
        x, y = px[k]; c = tuple(int(v) for v in cols[k]); d.ellipse([x - 1.2, im.size[1] - y - 1.2, x + 1.2, im.size[1] - y + 1.2], fill=c)
    if mark is not None:
        mx, my = (np.array([mark[0], mark[2]]) - lo) * sc + 10; rr = radius * sc
        d.ellipse([mx - rr, im.size[1] - my - rr, mx + rr, im.size[1] - my + rr], outline=(255, 40, 40), width=3)
    im.save(path)
vcol_before = sample(UV); img = patched; vcol_after = sample(UV)
render_side(vcol_before, -1, os.path.join(out_dir, "eye_right_flank_before.png"), mark=eye_R)
render_side(vcol_before, +1, os.path.join(out_dir, "eye_left_flank_before.png"), mark=eye_L)
render_side(vcol_after, +1, os.path.join(out_dir, "eye_left_flank_after.png"), mark=eye_L)
json.dump({"eye_R_rest": eye_R.tolist(), "eye_L_rest": eye_L.tolist(), "radius_m": radius, "texels": n_tex, "faces_left": int(len(facesL))}, open(os.path.join(out_dir, "eye_patch.json"), "w"), indent=2)
print("renders written to", out_dir)
