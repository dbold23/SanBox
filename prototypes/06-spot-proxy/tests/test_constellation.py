"""Tests for the non-RGB spot-constellation matcher (prototype 06).

Run with the MAIN checkout venv (python 3.9):
    "/Volumes/External Dive 2TB/projects/marine-cv/7Gill/.venv/bin/python" \
        -m pytest prototypes/06-spot-proxy/tests/test_constellation.py -q

The identity-recovery test is the expensive one (~40 s): 20 identities x 20
gallery entries x 4 flips of RANSAC. Everything else is sub-second.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_P06 = os.path.dirname(_HERE)
_P02 = os.path.join(os.path.dirname(_P06), "02-centerline-chart")
for _p in (_P06, _P02):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import constellation as C  # noqa: E402
import eval_constellation as E  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _ellipse_mask(w=900, h=300, cx=450, cy=150, a=380, b=110):
    yy, xx = np.mgrid[0:h, 0:w]
    return (((xx - cx) / float(a)) ** 2 + ((yy - cy) / float(b)) ** 2) <= 1.0


def _ellipse_detection(n_spots=40, seed=0, w=900, h=300, a=380, b=110, frac=0.9):
    """A straight horizontal ellipse body with spots inside it, contract-shaped.

    ``frac`` is how close to the rim the spots may sit (1.0 = the boundary).
    """
    rng = np.random.default_rng(seed)
    cx, cy = w / 2.0, h / 2.0
    t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    poly = np.column_stack([cx + a * np.cos(t), cy + b * np.sin(t)])
    pts = []
    while len(pts) < n_spots:
        x = rng.uniform(cx - a, cx + a)
        y = rng.uniform(cy - b, cy + b)
        if ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= frac ** 2:
            pts.append((x, y))
    spots = [{"x": x - 8, "y": y - 8, "w": 16.0, "h": 16.0, "cx": x, "cy": y, "conf": 0.9}
             for x, y in pts]
    return {"width": w, "height": h, "body_polygon": poly.tolist(),
            "obstruction_polygons": None, "spots": spots}


# --------------------------------------------------------------------------- #
# 1. prototype-02 rectification of a straight ellipse                          #
# --------------------------------------------------------------------------- #
def test_ellipse_chart_is_monotone_in_s_and_bounded_in_r():
    """s must increase along the ellipse's long axis and |r| must stay <= 1."""
    det = _ellipse_detection(n_spots=60, seed=1)
    ss = C.extract_spotset(det)
    assert ss is not None
    assert ss.frame == "chart", ss.meta.get("drops")

    x = np.array([sp["cx"] for sp in det["spots"]])
    # the centerline runs along +x or -x; either way s must be a monotone
    # function of x, i.e. |Spearman rho| == 1 up to rectification noise.
    order_x = np.argsort(x)
    d = np.diff(ss.s[order_x])
    rho = np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(ss.s)))[0, 1]
    assert abs(rho) > 0.999, "s is not monotone along the axis (rho=%.4f)" % rho
    # Monotone up to the rectification floor: the centerline of a rasterised
    # ellipse is not exactly the axis, so two spots at the same x but opposite
    # r can swap. Measured on this mask: 1 inversion in 60 spots, 8e-4 of the
    # body length. Anything above 1% of the body length is a real defect.
    back = -d if d.mean() < 0 else d
    assert back.min() > -0.01, "s reverses by %.4f of the body length" % (-back.min())

    assert np.all(np.abs(ss.r) <= 1.0 + 1e-9), float(np.abs(ss.r).max())
    # the true |r| for these spots is <= 0.9 by construction; the ray-marched
    # half width must not be so wrong that everything piles up on the rim.
    assert ss.meta["max_abs_r_raw"] <= 1.05, ss.meta["max_abs_r_raw"]
    assert np.abs(ss.r).max() > 0.5, "no spot reached the flank - r is over-normalised"


def test_ellipse_r_sign_tracks_the_side_of_the_axis():
    det = _ellipse_detection(n_spots=60, seed=2)
    ss = C.extract_spotset(det)
    y = np.array([sp["cy"] for sp in det["spots"]])
    above = y < det["height"] / 2.0
    # all spots on one side of the axis must share an r sign
    assert (np.sign(ss.r[above]).mean() ** 2) > 0.9
    assert np.sign(ss.r[above]).mean() * np.sign(ss.r[~above]).mean() < 0


def test_edt_r_norm_over_normalises_near_the_tips():
    """Why _station_half_widths exists: the EDT is not the lateral half width.

    It only bites for spots near the snout/tail, where the EDT measures the
    distance to the TIP rather than the lateral half width. Measured on this
    ellipse with spots out to 98% of the rim: max |r| 0.98 (ray) vs 1.33 (EDT),
    with 2 spots pushed outside the body by the EDT normalisation.
    """
    det = _ellipse_detection(n_spots=120, seed=3, frac=0.98)
    ray = C.extract_spotset(det, r_norm="ray")
    edt = C.extract_spotset(det, r_norm="edt")
    assert ray.meta["max_abs_r_raw"] <= 1.0
    assert edt.meta["max_abs_r_raw"] > 1.2
    assert not [d for d in ray.meta["drops"] if d["reason"] == "r_clipped_to_unit"]
    assert [d for d in edt.meta["drops"] if d["reason"] == "r_clipped_to_unit"]


def _tapered_detection(length=800.0, hw0=150.0, hw1=40.0, n=24, frac=0.85,
                       w=900, h=400, n_slices=40):
    """A wedge whose half width tapers ``hw0`` -> ``hw1``, spots at ``frac`` of it.

    The point of the taper: a spot at 85% of the LOCAL half width is r = 0.85
    everywhere. Normalising by the body's GLOBAL minor half extent instead makes
    the same spot read 0.85 * hw1 / hw0 = 0.23 at the narrow end -- which is what
    ``_pca_frame`` used to do, and it is why chart-frame and pca-frame sets could
    not be compared.
    """
    cy = h / 2.0
    x0 = (w - length) / 2.0
    xs = np.linspace(x0, x0 + length, n_slices)
    hw = np.linspace(hw0, hw1, n_slices)
    poly = ([[float(x), float(cy - t)] for x, t in zip(xs, hw)]
            + [[float(x), float(cy + t)] for x, t in zip(xs[::-1], hw[::-1])])
    spots = []
    for i in range(n):
        u = (i + 0.5) / n
        x = x0 + u * length
        t = float(np.interp(x, xs, hw))
        y = cy + (frac * t if i % 2 == 0 else -frac * t)
        spots.append({"x": x - 6, "y": y - 6, "w": 12.0, "h": 12.0,
                      "cx": float(x), "cy": float(y), "conf": 0.9})
    return {"width": w, "height": h, "body_polygon": poly,
            "obstruction_polygons": None, "spots": spots}


def test_pca_frame_r_is_the_local_half_width_not_the_global_extent(monkeypatch):
    det = _tapered_detection()

    def boom(*a, **kw):
        raise RuntimeError("force the pca fallback")

    monkeypatch.setattr(C.cl02, "extract_centerline", boom)
    ss = C.extract_spotset(det)
    assert ss is not None and ss.frame == "pca"
    r = np.abs(ss.r)
    # every spot sits at 0.85 of its own local half width
    assert r.min() > 0.70, "narrow end under-normalised: min |r| = %.3f" % r.min()
    assert r.max() < 1.0, r.max()
    # the narrow half of the body must read the same as the wide half; under the
    # old global normaliser the narrow end came out ~3.7x smaller
    s_order = np.argsort(ss.s)
    wide, narrow = r[s_order[:len(r) // 2]], r[s_order[len(r) // 2:]]
    assert abs(wide.mean() - narrow.mean()) < 0.10, (wide.mean(), narrow.mean())


def test_pca_frame_matches_the_chart_frame_on_the_same_body(monkeypatch):
    """The two frames must put the same spot at the same r, or they cannot be
    compared -- which is what ``_pair_eval`` does for every cross-frame pair.

    The agreement is checked on the *median* spot, not the worst one: prototype
    02's medial axis hooks at the tips, so the first and last spot of a wedge
    come out at |r| 0.12 and 0.62 against a true 0.85. That is a chart-frame end
    artefact (the same one ``_chart_is_unusable`` guards with its magnitude
    test), not a disagreement about what r means.
    """
    det = _tapered_detection()
    chart = C.extract_spotset(det)
    if chart is None or chart.frame != "chart":
        pytest.skip("prototype 02 declined this mask; nothing to compare")

    def boom(*a, **kw):
        raise RuntimeError("force the pca fallback")

    monkeypatch.setattr(C.cl02, "extract_centerline", boom)
    pca = C.extract_spotset(det)
    assert pca.frame == "pca"
    # same spot order, so compare elementwise; the two frames may disagree about
    # which end is s=0 and about the sign of r, so compare |corr| and |r|
    assert abs(np.corrcoef(chart.s, pca.s)[0, 1]) > 0.99
    d = np.abs(np.abs(chart.r) - np.abs(pca.r))
    assert float(np.median(d)) < 0.05, float(np.median(d))
    # the old global-extent normaliser put the narrow end 3.7x low, so the
    # median disagreement was ~0.3 -- an order of magnitude worse than this
    assert float(np.mean(d < 0.10)) > 0.8, float(np.mean(d < 0.10))


# --------------------------------------------------------------------------- #
# 2. mask construction                                                         #
# --------------------------------------------------------------------------- #
def test_build_body_mask_scales_and_punches_obstructions():
    poly = [[0, 0], [4000, 0], [4000, 2000], [0, 2000]]
    ob = [[[1000, 500], [2000, 500], [2000, 1500], [1000, 1500]]]
    mask, scale = C.build_body_mask(poly, ob, 4032, 3024, max_side=1024)
    assert max(mask.shape) <= 1024
    assert abs(scale - 1024.0 / 4032.0) < 1e-6
    hole = mask[int(1000 * scale):int(1400 * scale), int(1200 * scale):int(1800 * scale)]
    assert not hole.any(), "obstruction was not punched out"
    with pytest.raises(ValueError):
        C.build_body_mask([[0, 0], [1, 1]])


def test_pca_fallback_when_centerline_fails(monkeypatch):
    """Any prototype-02 failure must fall back to the PCA frame and log it."""
    det = _ellipse_detection(n_spots=40, seed=6)

    def boom(*a, **kw):
        raise RuntimeError("synthetic centerline failure")

    monkeypatch.setattr(C.cl02, "extract_centerline", boom)
    ss = C.extract_spotset(det)
    assert ss is not None
    assert ss.frame == "pca"
    assert any(d["reason"] == "centerline_failed" for d in ss.meta["drops"])
    assert np.all(np.abs(ss.r) <= 1.0)
    # the fallback frame must still order the ellipse along its long axis
    x = np.array([sp["cx"] for sp in det["spots"]])
    rho = np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(ss.s)))[0, 1]
    assert abs(rho) > 0.999, rho


def test_degenerate_masks_are_reported_not_crashed():
    """Too few spots, or a torn polygon, return None rather than raising."""
    det = _ellipse_detection(n_spots=40, seed=7)
    assert C.extract_spotset(det, min_spots=100) is None
    assert C.extract_spotset({"width": 100, "height": 100, "spots": [],
                              "body_polygon": det["body_polygon"]}) is None
    assert C.extract_spotset({"width": 100, "height": 100, "body_polygon": [[1, 1]],
                              "spots": det["spots"]}) is None


# --------------------------------------------------------------------------- #
# 3. descriptors and flips                                                     #
# --------------------------------------------------------------------------- #
def test_descriptors_are_l1_normalised_and_translation_invariant():
    rng = np.random.default_rng(0)
    base = E.make_identity(rng)
    a = C.SpotSet(base[:, 0], base[:, 1], np.full(len(base), 0.02),
                  np.ones(len(base)), frame="toy", aspect=1.0)
    ha, mnn = C.descriptors(a)
    assert ha.shape == (len(a), C.N_RADIAL * C.N_ANGULAR)
    assert np.allclose(ha.sum(axis=1), 1.0)
    assert mnn > 0
    # a rigid shift of the whole set leaves every descriptor unchanged
    b = C.SpotSet(base[:, 0] + 0.05, base[:, 1] - 0.02, a.size, a.conf,
                  frame="toy", aspect=1.0)
    hb, _ = C.descriptors(b)
    assert np.abs(ha - hb).max() < 1e-9


def test_flipped_round_trips():
    rng = np.random.default_rng(1)
    base = E.make_identity(rng)
    a = C.SpotSet(base[:, 0], base[:, 1], np.full(len(base), 0.02),
                  np.ones(len(base)), frame="toy", aspect=1.0)
    f = a.flipped(True, True)
    assert np.allclose(f.s, 1.0 - a.s) and np.allclose(f.r, -a.r)
    assert np.allclose(f.flipped(True, True).s, a.s)
    assert a.flipped(False, False) is a


def test_match_score_is_symmetric_in_its_arguments():
    """score(a, b) must equal score(b, a) exactly.

    The directed registration is not symmetric and does not become symmetric
    with a bigger RANSAC budget: measured mean |score(a,b) - score(b,a)| over
    276 toy pairs was 0.018 with a max of 0.102, against a different-individual
    mean of 0.25, and 0.029 / 0.022 / 0.024 at n_iters 400 / 2000 / 8000. Every
    ``--real``/``--synth`` number therefore used to depend on the order the
    jsonl happened to be in, because ``_pair_eval`` scores only i < j.
    """
    gallery, queries, _ = E.build_toy_corpus(n_ids=6, seed=0)
    sets = gallery + queries
    worst = 0.0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            ab = C.match_score(sets[i], sets[j])
            ba = C.match_score(sets[j], sets[i])
            assert ab["score"] == ba["score"], (i, j, ab["score"], ba["score"])
            assert ab["n_inliers"] == ba["n_inliers"]
            assert ab["n_a"] == len(sets[i]) and ab["n_b"] == len(sets[j])
            assert ba["n_a"] == len(sets[j]) and ba["n_b"] == len(sets[i])
            worst = max(worst, abs(ab["score_ab"] - ab["score_ba"]))
    # and the raw directed scores really do disagree, i.e. the symmetrisation is
    # doing work rather than papering over an already-symmetric function
    assert worst > 0.01, worst


def test_symmetric_false_returns_the_directed_score():
    gallery, queries, _ = E.build_toy_corpus(n_ids=4, seed=3)
    a, b = queries[0], gallery[1]
    directed = C.match_score(a, b, symmetric=False)
    both = C.match_score(a, b)
    assert directed["direction"] == "ab"
    assert directed["score"] == both["score_ab"]
    assert both["score"] == max(both["score_ab"], both["score_ba"])
    assert both["score_mean"] == pytest.approx(
        0.5 * (both["score_ab"] + both["score_ba"]))
    assert both["score"] == both["n_inliers"] / float(min(both["n_a"], both["n_b"]))


def test_one_far_outside_spot_condemns_the_chart_frame():
    """The out-FRACTION test alone misses a single wildly misplaced spot.

    Real example, catalog image 271: 4 of 97 spots outside the body, fraction
    0.041 -- comfortably under ``_MAX_R_OUT_FRAC`` = 0.10 -- but max |r| 30.1,
    because at a tip the medial axis turns ~90 degrees and a spot 60 px *along*
    the body is charted as a 60 px *lateral* offset at a 2 px half width.
    Clipped to +-1, such a spot becomes a fake rim inlier.
    """
    mask = _ellipse_mask()
    meta = {"length_px": 760.0}
    fine = np.concatenate([np.full(96, 0.5), [0.99]])
    assert C._chart_is_unusable(mask, meta, fine) == []
    one_bad = np.concatenate([np.full(93, 0.5), [1.2, 1.2, 1.2, 30.1]])
    reasons = C._chart_is_unusable(mask, meta, one_bad)
    assert any(r.startswith("spot_far_outside_body") for r in reasons), reasons
    assert not any(r.startswith("spots_outside_body") for r in reasons), reasons


def test_interior_obstruction_is_refilled_and_said_so():
    """``_largest_filled`` undoes an interior punch-out; that must be logged.

    ``build_body_mask`` punches the obstruction and ``binary_fill_holes`` puts
    it straight back, so the pipeline's "polygon (minus obstructions)" is only
    true for occluders that cross the silhouette. Measured on the real corpus:
    28 of 40 sampled obstructed records were fully restored.
    """
    det = _ellipse_detection(n_spots=40, seed=11)
    ob = [[[380, 110], [520, 110], [520, 190], [380, 190]]]
    punched = dict(det, obstruction_polygons=ob)
    clean_mask, _ = C.build_body_mask(det["body_polygon"], None,
                                      det["width"], det["height"])
    punched_mask, _ = C.build_body_mask(det["body_polygon"], ob,
                                        det["width"], det["height"])
    assert punched_mask.sum() < clean_mask.sum(), "the fixture punches nothing"

    ss = C.extract_spotset(punched)
    assert ss is not None
    refill = [d for d in ss.meta["drops"] if d["reason"] == "obstruction_refilled"]
    assert refill, ss.meta["drops"]
    assert refill[0]["n_px"] == int(clean_mask.sum() - punched_mask.sum())
    # and the mask really is the un-punched body again
    assert ss.meta["mask_area_px"] == int(C._largest_filled(clean_mask).sum())


def test_match_score_is_perfect_on_an_identical_set():
    rng = np.random.default_rng(2)
    base = E.make_identity(rng)
    a = C.SpotSet(base[:, 0], base[:, 1], np.full(len(base), 0.02),
                  np.ones(len(base)), frame="toy", aspect=1.0)
    res = C.match_score(a, a)
    assert res["flip"] == (False, False)
    assert res["score"] > 0.95, res
    assert res["n_inliers"] == res["n_a"] == res["n_b"] or res["score"] > 0.95


# --------------------------------------------------------------------------- #
# 4. identity recovery on the toy generator (the headline assertion)           #
# --------------------------------------------------------------------------- #
def test_toy_identity_recovery_rank1():
    """20 identities, 2% jitter, 20% dropout, 20% clutter, random flips.

    Pooled over three consecutive seeds (60 queries), not one, because a single
    20-identity draw resolves Rank-1 only to +/-0.05: the per-seed values for
    seeds 0-4 are 0.95 1.00 0.95 1.00 1.00 for the default ``axis`` model and
    1.00 1.00 0.90 1.00 1.00 for ``s_affine``, so any single-seed threshold at
    0.95 is a coin flip on the seed rather than a statement about the matcher.
    """
    n_ids, seeds = 20, (0, 1, 2)
    r1 = r5 = n = 0
    pos, neg, flip_hits = [], [], []
    for seed in seeds:
        gallery, queries, true_flips = E.build_toy_corpus(
            n_ids=n_ids, jitter=0.02, dropout=0.2, clutter=0.2, seed=seed)
        sc, _, flips = E.score_matrix(queries, gallery, seed=seed)
        m = E.rank_metrics(sc, np.arange(n_ids))
        assert m["rank1"] >= 0.90, (seed, m)     # per-seed floor
        r1 += m["rank1"] * n_ids
        r5 += m["rank5"] * n_ids
        n += n_ids
        pos.extend(sc[np.arange(n_ids), np.arange(n_ids)])
        neg.extend(sc[~np.eye(n_ids, dtype=bool)])
        flip_hits.extend(flips[i, i] == true_flips[i] for i in range(n_ids))

    assert r1 / n >= 0.95, "pooled rank-1 %.3f" % (r1 / n)
    assert r5 / n >= 0.98, "pooled rank-5 %.3f" % (r5 / n)
    assert E.auroc(pos, neg) >= 0.97, E.auroc(pos, neg)
    # the flip that brings the gallery set into the query's orientation
    assert np.mean(flip_hits) >= 0.95, float(np.mean(flip_hits))


def test_toy_degrades_gracefully_not_silently():
    """4% jitter is past the resolution limit; the matcher must LOOK broken there.

    The gate is 0.6 x the median spot spacing, so a different animal of the same
    spot density already scores ~0.2 by chance. A same-individual score that has
    fallen to that level is the honest signal that the frame is too noisy to
    match, and the benchmark has to show it rather than hide it.
    """
    g4, q4, _ = E.build_toy_corpus(n_ids=12, jitter=0.04, dropout=0.2,
                                   clutter=0.2, seed=0)
    sc4, _, _ = E.score_matrix(q4, g4, seed=0)
    tc = np.arange(12)
    same4 = sc4[tc, tc].mean()
    diff4 = sc4[~np.eye(12, dtype=bool)].mean()
    assert same4 < 2.0 * diff4, (same4, diff4)

    g1, q1, _ = E.build_toy_corpus(n_ids=12, jitter=0.005, dropout=0.2,
                                   clutter=0.2, seed=0)
    sc1, _, _ = E.score_matrix(q1, g1, seed=0)
    same1 = sc1[tc, tc].mean()
    diff1 = sc1[~np.eye(12, dtype=bool)].mean()
    # measured: same 0.675 vs different 0.21 at 0.5% jitter, against
    # same 0.29 vs different 0.24 at 4%.
    assert same1 > 2.5 * diff1, (same1, diff1)
    assert same1 > 2.0 * same4, (same1, same4)


def test_rank_returns_sorted_hits_with_ids():
    gallery, queries, _ = E.build_toy_corpus(n_ids=6, jitter=0.01, dropout=0.2,
                                             clutter=0.2, seed=3)
    ids = ["id%d" % i for i in range(6)]
    hits = C.rank(queries[0], gallery, ids=ids)
    assert len(hits) == 6
    assert [h["score"] for h in hits] == sorted([h["score"] for h in hits], reverse=True)
    assert hits[0]["id"] == "id0"


# --------------------------------------------------------------------------- #
# 5. the metrics themselves                                                    #
# --------------------------------------------------------------------------- #
def test_auroc_and_cohens_d_reference_values():
    assert E.auroc([1, 2, 3], [0, 0, 0]) == 1.0
    assert E.auroc([0, 0, 0], [1, 2, 3]) == 0.0
    assert E.auroc([1, 1], [1, 1]) == 0.5  # all ties
    d = E.cohens_d([1, 2, 3, 4], [1, 2, 3, 4])
    assert abs(d) < 1e-12


def test_toy_sighting_truth_is_consistent():
    rng = np.random.default_rng(4)
    base = E.make_identity(rng)
    ss, truth = E.make_sighting(rng, base, jitter=0.0, dropout=0.0, clutter=0.0,
                                warp_amp=0.0, allow_flip=False)
    assert truth["n_clutter"] == 0
    assert len(ss) == len(base) == truth["n_true"]
    assert np.allclose(np.sort(ss.s), np.sort(base[:, 0]), atol=1e-6)


def test_monotone_warp_never_reorders():
    rng = np.random.default_rng(5)
    grid = np.linspace(0, 1, 500)
    for _ in range(20):
        f = E._monotone_warp(rng, 0.06)
        assert np.all(np.diff(f(grid)) > -1e-12)
