from __future__ import annotations

import numpy as np

import strain_demo
from chart import chart_to_image, image_to_chart, rectify
from strain_demo import PARAMS


def _sine_centerline():
    t = np.linspace(0, 1, 400)
    return np.column_stack([40 + 400 * t, 120 + 35 * np.sin(2 * np.pi * t)])


def test_chart_image_round_trip():
    cl = _sine_centerline()
    hw, n_s, n_r = 20.0, 256, 41
    rng = np.random.default_rng(7)
    s_idx = rng.uniform(5, n_s - 6, size=50)
    r_idx = rng.uniform(3, n_r - 4, size=50)

    pts = chart_to_image(cl, hw, n_s, n_r, s_idx, r_idx)
    back = image_to_chart(cl, hw, n_s, n_r, pts)

    assert np.abs(back[:, 0] - s_idx).max() < 0.15   # s index
    assert np.abs(back[:, 1] - r_idx).max() < 0.15   # r index


def test_rectify_masks_outside_samples():
    img = np.ones((60, 200))
    mask = np.zeros((60, 200), dtype=bool)
    mask[20:40, 10:190] = True
    cl = np.column_stack([np.linspace(15, 185, 100), np.full(100, 30.0)])
    strip = rectify(img, cl, half_width=25.0, n_s=64, n_r=33, mask=mask)
    assert strip.shape == (64, 33)
    mid = strip[:, 16]
    assert np.isfinite(mid).all()                   # on-body samples valid
    assert np.isnan(strip[:, 0]).all()              # r = -25 is off-body
    assert np.isnan(strip[:, -1]).all()             # r = +25 is off-body


def _recovered_spot_error(image, mask, spots, params):
    """Pipeline the render, detect chart spots, compare to ground truth.

    Returns (raw_err, gauge_free_err) in px. The chart's s-origin sits at the
    extracted snout tip, which lands a few px short of the analytic s=0 (a
    pure translation gauge; in the real system it is pinned by anatomy, e.g.
    gill slit 1). gauge_free_err removes the mean (ds, dr) offset.
    """
    cl, strip = strain_demo.run_pipeline(image, mask, params)
    det = strain_demo.detect_spots(strip, params)
    from centerline import arc_length

    px = strain_demo._to_px(det, float(arc_length(cl)[-1]), params)
    sigma, matches = strain_demo._resolve_r_sign(px, spots, gate=0.05 * params["L"])
    assert len(matches) >= 0.9 * len(spots)
    px = px * np.array([1.0, sigma])
    delta = np.array([px[i] - spots[j] for i, j in matches])
    raw = np.linalg.norm(delta, axis=1)
    gauge_free = np.linalg.norm(delta - delta.mean(axis=0), axis=1)
    return raw, gauge_free


def test_unbent_round_trip_recovers_spots():
    spots = strain_demo.make_spots(0, PARAMS)
    img, mask = strain_demo.render_straight(spots, PARAMS)
    raw, gauge_free = _recovered_spot_error(img, mask, spots, PARAMS)
    # Bend-invariance baseline: chart position == body position, up to a
    # small global origin offset (< 1% BL) and sub-px per-spot scatter.
    assert raw.mean() < 0.01 * PARAMS["L"]
    assert gauge_free.mean() < 1.0
    assert gauge_free.max() < 2.5


def test_bent_eps0_recovers_spots_within_looser_tolerance():
    spots = strain_demo.make_spots(0, PARAMS)
    img, mask = strain_demo.render_bent(spots, 0.0, PARAMS)
    raw, gauge_free = _recovered_spot_error(img, mask, spots, PARAMS)
    assert raw.mean() < 0.012 * PARAMS["L"]
    assert gauge_free.mean() < 1.5
    assert gauge_free.max() < 4.0
