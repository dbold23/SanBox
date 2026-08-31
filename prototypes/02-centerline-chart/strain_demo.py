"""Strain measurement harness: how much does skin strain corrupt the chart?

Builds a synthetic elongate "fish" with a seeded spot texture in body-frame
(s, r) coordinates; renders it straight and bent through a known
constant-curvature centerline; imposes a controllable multiplicative
arc-length stretch that ramps linearly across the body from +epsilon at the
convex outer fibre to -epsilon at the concave outer fibre (strain field
eps * r / W, zero on the midline - the bending-beam profile). Then runs the
REAL pipeline on both renders (mask -> extract_centerline -> rectify),
detects spot centroids in both charts, matches them nearest-neighbour, and
reports per-spot displacement in chart space.

epsilon default 0.05: the midpoint of the +/-3.9-6.6% longitudinal strain
bracket measured by sonomicrometry in a swimming leopard shark
(Donley & Shadwick, J. Exp. Biol. 206(7), 2003) - see README.

With epsilon = 0 the residual displacement is the pipeline's own error
(bend-invariance, demonstrated); with epsilon = 0.05 it is the irreducible
error strain imposes on ANY centerline chart (finding 02, quantified).

Run:  python strain_demo.py    -> results/metrics.json + results/panel_*.png
"""

from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from centerline import arc_length, extract_centerline
from chart import rectify

__all__ = [
    "PARAMS", "half_width_profile", "make_spots", "render_straight",
    "render_bent", "run_pipeline", "detect_spots", "measure_epsilon", "main",
]

PARAMS = dict(
    L=560.0,        # body length, px
    W=64.0,         # nominal max half-width, px (also the chart half_width)
    theta=1.2,      # total bend angle, rad (constant curvature C-bend)
    margin=24,      # render margin, px
    n_s=384,        # chart stations
    n_r=96,         # chart offsets
    n_stations=256, # centerline extraction stations
    n_spots=40,
    spot_radius=4.5,
    min_sep=30.0,   # body-frame spot separation, px
    body_val=0.78,
    spot_val=0.12,
    bg_val=0.15,
)


def half_width_profile(s, L, W):
    """Fish silhouette half-width w(s): blunt wide head, tapering tail.

    Wider over the anterior 10% than the posterior 10%, so the
    widest-end-first rule in extract_centerline orients head-first.
    """
    u = np.clip(np.asarray(s, dtype=float) / L, 0.0, 1.0)
    return W * np.sin(np.pi * u ** 0.8) ** 0.7 * (1.0 - 0.35 * u)


def make_spots(seed, params=PARAMS):
    """Seeded Poisson-disk-ish spot centres in body frame; (k, 2) of (s, r)."""
    rng = np.random.default_rng(seed)
    L, W = params["L"], params["W"]
    radius, sep = params["spot_radius"], params["min_sep"]
    spots = []
    for _ in range(8000):
        if len(spots) >= params["n_spots"]:
            break
        s = rng.uniform(0.10 * L, 0.90 * L)
        w = half_width_profile(s, L, W)
        if w < 2.2 * radius:
            continue
        r = rng.uniform(-0.72, 0.72) * w
        if abs(r) + radius + 2.0 > 0.92 * w:
            continue
        if spots and np.min(np.hypot(*(np.array(spots) - [s, r]).T)) < sep:
            continue
        spots.append((s, r))
    return np.array(spots)


def _texture(s, r, spots, params):
    """Body-frame texture value at (s, r) arrays: light body, dark spots."""
    val = np.full(np.broadcast(s, r).shape, params["body_val"])
    radius = params["spot_radius"]
    for sk, rk in spots:
        d = np.hypot(s - sk, r - rk)
        cov = np.clip(radius + 0.5 - d, 0.0, 1.0)  # 1px soft edge
        val = val * (1 - cov) + params["spot_val"] * cov
    return val


def _compose(s_body, r, spots, params):
    L, W = params["L"], params["W"]
    inside = (s_body >= 0) & (s_body <= L)
    w = half_width_profile(np.where(inside, s_body, 0.0), L, W)
    mask = inside & (np.abs(r) <= w)
    img = np.full(mask.shape, params["bg_val"])
    img[mask] = _texture(s_body[mask], r[mask], spots, params)
    return img, mask


def render_straight(spots, params=PARAMS):
    """Rest pose: body frame == world frame (head at low x)."""
    L, W, m = params["L"], params["W"], params["margin"]
    h = int(2 * W + 2 * m)
    wimg = int(L + 2 * m)
    Y, X = np.mgrid[0:h, 0:wimg].astype(float)
    s_body = X - m
    r = Y - h / 2.0
    return _compose(s_body, r, spots, params)


def render_bent(spots, epsilon, params=PARAMS):
    """Bent pose on a constant-curvature arc, with strain field eps * r / W.

    Body point (s, r) maps to C0 + (R + r) * (cos a, sin a) with
    a = a0 - s_bent / R and s_bent = s * (1 + eps * r / W): the +r side is
    radially outward, i.e. the CONVEX side, and its material is stretched
    tail-ward by up to +eps at the outer fibre; the -r (concave) side is
    compressed head-ward by up to -eps. Rendered by exact closed-form
    inversion per pixel.
    """
    L, W, m = params["L"], params["W"], params["margin"]
    theta = params["theta"]
    R = L / theta
    a0 = np.pi / 2 + theta / 2

    # Bounding box of the mapped silhouette.
    ss = np.linspace(0, L, 400)
    aa = a0 - ss / R
    edge = np.concatenate([
        (R + W + 4) * np.column_stack([np.cos(aa), np.sin(aa)]),
        (R - W - 4) * np.column_stack([np.cos(aa), np.sin(aa)]),
    ])
    c0 = m - edge.min(axis=0)
    extent = edge.max(axis=0) - edge.min(axis=0)
    wimg, h = int(extent[0] + 2 * m), int(extent[1] + 2 * m)

    Y, X = np.mgrid[0:h, 0:wimg].astype(float)
    dx, dy = X - c0[0], Y - c0[1]
    rho = np.hypot(dx, dy)
    r = rho - R
    a = np.arctan2(dy, dx)          # arc stays within (0, pi): no wrap
    s_bent = (a0 - a) * R
    s_body = s_bent / (1.0 + epsilon * r / W)
    return _compose(s_body, r, spots, params)


def run_pipeline(image, mask, params=PARAMS):
    """The measured pipeline: mask -> centerline -> rectified strip."""
    cl = extract_centerline(mask, n_stations=params["n_stations"])
    strip = rectify(image, cl, params["W"], params["n_s"], params["n_r"], mask=mask)
    return cl, strip


def detect_spots(strip, params=PARAMS, min_size=8):
    """Thresholded connected-component spot centroids; (k, 2) of (s_idx, r_idx)."""
    filled = np.where(np.isnan(strip), 1.0, strip)
    dark = filled < 0.5 * (params["body_val"] + params["spot_val"])
    labels, n = ndimage.label(dark, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return np.empty((0, 2))
    sizes = ndimage.sum_labels(np.ones_like(labels), labels, index=np.arange(1, n + 1))
    keep = np.flatnonzero(sizes >= min_size) + 1
    if len(keep) == 0:
        return np.empty((0, 2))
    coms = ndimage.center_of_mass(dark, labels, keep)
    return np.array(coms, dtype=float)


def _to_px(det, L_chart, params):
    """Chart indices -> physical (s_px, r_px)."""
    s = det[:, 0] * L_chart / (params["n_s"] - 1)
    r = -params["W"] + det[:, 1] * 2 * params["W"] / (params["n_r"] - 1)
    return np.column_stack([s, r])


def _greedy_match(a, b, gate):
    """One-to-one greedy nearest-neighbour matching; list of (ia, ib)."""
    if len(a) == 0 or len(b) == 0:
        return []
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    pairs = [(d[i, j], i, j) for i in range(len(a)) for j in range(len(b)) if d[i, j] <= gate]
    pairs.sort()
    used_a, used_b, out = set(), set(), []
    for _, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        out.append((i, j))
    return out


def _resolve_r_sign(det_px, gt, gate):
    """The chart's r sign depends on extracted head->tail normal handedness.

    Try both signs against ground truth, keep the one with more matches
    (then lower cost). Returns (sigma, matches with sigma applied).
    """
    best = None
    for sigma in (1.0, -1.0):
        cand = det_px * np.array([1.0, sigma])
        m = _greedy_match(cand, gt, gate)
        cost = float(np.mean([np.linalg.norm(cand[i] - gt[j]) for i, j in m])) if m else np.inf
        key = (-len(m), cost, -sigma)
        if best is None or key < best[0]:
            best = (key, sigma, m)
    return best[1], best[2]


def measure_epsilon(epsilon, seed=0, params=PARAMS):
    """Full measurement for one strain level. Returns (stats, artifacts)."""
    spots = make_spots(seed, params)
    L, W = params["L"], params["W"]

    img_rest, mask_rest = render_straight(spots, params)
    img_bent, mask_bent = render_bent(spots, epsilon, params)
    cl_rest, strip_rest = run_pipeline(img_rest, mask_rest, params)
    cl_bent, strip_bent = run_pipeline(img_bent, mask_bent, params)
    L_rest = float(arc_length(cl_rest)[-1])
    L_bent = float(arc_length(cl_bent)[-1])

    det_rest = detect_spots(strip_rest, params)
    det_bent = detect_spots(strip_bent, params)
    px_rest = _to_px(det_rest, L_rest, params)
    px_bent = _to_px(det_bent, L_bent, params)

    sig_rest, gt_match = _resolve_r_sign(px_rest, spots, gate=0.05 * L)
    sig_bent, _ = _resolve_r_sign(px_bent, spots, gate=0.10 * L)
    px_rest = px_rest * np.array([1.0, sig_rest])
    px_bent = px_bent * np.array([1.0, sig_bent])

    # Rest detection -> ground-truth spot (for side labels and prediction).
    gt_of_rest = {i: j for i, j in gt_match}

    # The headline matching: rest chart <-> bent chart, nearest neighbour.
    pairs = _greedy_match(px_rest, px_bent, gate=0.10 * L)

    rows = []
    for i, j in pairs:
        if i not in gt_of_rest:
            continue
        k = gt_of_rest[i]
        sk, rk = spots[k]
        ds = px_bent[j, 0] - px_rest[i, 0]
        dr = px_bent[j, 1] - px_rest[i, 1]
        pred = epsilon * (rk / W) * sk   # (lambda - 1) * s
        rows.append(dict(s=sk, r=rk, ds=ds, dr=dr, pred=pred,
                         side="convex" if rk > 0 else "concave"))

    ds = np.array([q["ds"] for q in rows])
    dr = np.array([q["dr"] for q in rows])
    pred = np.array([q["pred"] for q in rows])
    sides = np.array([q["side"] for q in rows])

    def side_stats(name):
        sel = sides == name
        if not sel.any():
            return dict(n=0)
        return dict(
            n=int(sel.sum()),
            mean_signed_ds_px=float(ds[sel].mean()),
            mean_abs_ds_px=float(np.abs(ds[sel]).mean()),
            max_abs_ds_px=float(np.abs(ds[sel]).max()),
            mean_abs_ds_pct_bl=float(np.abs(ds[sel]).mean() / L * 100),
            max_abs_ds_pct_bl=float(np.abs(ds[sel]).max() / L * 100),
        )

    slope, intercept = (None, None)
    if len(rows) >= 2 and float(np.var(pred)) > 1e-12:
        fit = np.polyfit(pred, ds, 1)
        slope, intercept = float(fit[0]), float(fit[1])

    stats = dict(
        epsilon=epsilon,
        seed=seed,
        n_spots_truth=int(len(spots)),
        n_detected_rest=int(len(det_rest)),
        n_detected_bent=int(len(det_bent)),
        n_matched=int(len(rows)),
        chart_len_rest_px=L_rest,
        chart_len_bent_px=L_bent,
        mean_abs_ds_px=float(np.abs(ds).mean()),
        max_abs_ds_px=float(np.abs(ds).max()),
        mean_abs_ds_pct_bl=float(np.abs(ds).mean() / L * 100),
        max_abs_ds_pct_bl=float(np.abs(ds).max() / L * 100),
        mean_abs_dr_px=float(np.abs(dr).mean()),
        max_abs_dr_px=float(np.abs(dr).max()),
        convex=side_stats("convex"),
        concave=side_stats("concave"),
        fit_slope_measured_vs_predicted=slope,
        fit_intercept_px=intercept,
    )
    artifacts = dict(
        img_rest=img_rest, img_bent=img_bent,
        strip_rest=strip_rest, strip_bent=strip_bent,
        det_rest=det_rest, det_bent=det_bent, pairs=pairs, rows=rows,
        spots=spots, r_flip_display=(sig_rest * sig_bent < 0),
    )
    return stats, artifacts


# ---------------------------------------------------------------- rendering

def _gray_rgb(img01):
    g = np.clip(np.nan_to_num(img01, nan=0.0) * 255, 0, 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def _strip_rgb(strip):
    rgb = _gray_rgb(strip)
    rgb[np.isnan(strip)] = (45, 45, 70)
    return rgb


def make_panel(stats, art, path):
    """Rest render, bent render, and the two strips with matched spots linked."""
    pad = 12
    rest = Image.fromarray(_gray_rgb(art["img_rest"]))
    bent = Image.fromarray(_gray_rgb(art["img_bent"]))
    # Display the bent strip in the rest strip's r handedness so link
    # lines read straight; the measurement resolves the sign analytically.
    strip_bent = art["strip_bent"]
    det_bent = art["det_bent"].copy()
    if art["r_flip_display"] and len(det_bent):
        strip_bent = strip_bent[:, ::-1]
        det_bent[:, 1] = (strip_bent.shape[1] - 1) - det_bent[:, 1]
    st_r = Image.fromarray(np.transpose(_strip_rgb(art["strip_rest"]), (1, 0, 2)))
    st_b = Image.fromarray(np.transpose(_strip_rgb(strip_bent), (1, 0, 2)))

    w = max(rest.width + bent.width + 3 * pad, st_r.width + 2 * pad) + pad
    h = pad + max(rest.height, bent.height) + pad + 16 + st_r.height + pad + 16 + st_b.height + pad + 24
    canvas = Image.new("RGB", (w, h), (25, 25, 30))
    draw = ImageDraw.Draw(canvas)

    canvas.paste(rest, (pad, pad + 14))
    canvas.paste(bent, (2 * pad + rest.width, pad + 14))
    y1 = pad + max(rest.height, bent.height) + pad + 16
    canvas.paste(st_r, (pad, y1))
    y2 = y1 + st_r.height + pad + 16
    canvas.paste(st_b, (pad, y2))

    draw.text((pad, 2), "rest pose", fill=(230, 230, 230))
    draw.text((2 * pad + rest.width, 2), "bent pose  eps=%.2f" % stats["epsilon"],
              fill=(230, 230, 230))
    draw.text((pad, y1 - 13), "rest chart (s ->, r v)", fill=(230, 230, 230))
    draw.text((pad, y2 - 13), "bent chart (matched spots linked)", fill=(230, 230, 230))

    def dot(xy, color):
        x, y = xy
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], outline=color, width=1)

    for i, j in art["pairs"]:
        si, ri = art["det_rest"][i]
        sj, rj = det_bent[j]
        p1 = (pad + si, y1 + ri)
        p2 = (pad + sj, y2 + rj)
        draw.line([p1, p2], fill=(240, 200, 60), width=1)
        dot(p1, (90, 200, 255))
        dot(p2, (255, 110, 110))

    draw.text((pad, h - 20),
              "mean|ds|=%.2fpx (%.2f%%BL)  max|ds|=%.2fpx  matched=%d" % (
                  stats["mean_abs_ds_px"], stats["mean_abs_ds_pct_bl"],
                  stats["max_abs_ds_px"], stats["n_matched"]),
              fill=(240, 200, 60))
    canvas.save(path)


def main(seed=0, out_dir=None):
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)

    runs = {}
    for eps in (0.0, 0.05):
        stats, art = measure_epsilon(eps, seed=seed)
        runs["eps_%.2f" % eps] = stats
        make_panel(stats, art, os.path.join(out_dir, "panel_eps_%.2f.png" % eps))

    metrics = dict(
        config=dict(PARAMS, seed=seed),
        epsilon_default_source=(
            "0.05 = midpoint of the +/-3.9-6.6% longitudinal skin-strain bracket "
            "from leopard-shark sonomicrometry (Donley & Shadwick, J. Exp. Biol. "
            "206(7), 2003) [SEARCH-grade per docs/sevengill-canonical-reid]"
        ),
        runs=runs,
    )
    path = os.path.join(out_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
