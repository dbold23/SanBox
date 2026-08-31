from __future__ import annotations

import numpy as np
import pytest
from conftest import dist_to_curve, rasterize_tube

from centerline import arc_length, extract_centerline, resample_polyline


def _dense(fn, t0, t1, n=2000):
    t = np.linspace(t0, t1, n)
    return np.column_stack(fn(t))


TUBES = {
    "straight": (lambda t: (30 + 240 * t, np.full_like(t, 40.0)), 12.0, (80, 300)),
    "arc": (lambda t: (150 + 120 * np.cos(np.pi * (1.0 + 1.0 * t)),
                       200 + 120 * np.sin(np.pi * (1.0 + 1.0 * t))), 10.0, (235, 300)),
    "s_curve": (lambda t: (30 + 340 * t, 70 + 28 * np.sin(2 * np.pi * t)), 11.0, (140, 400)),
}


@pytest.mark.parametrize("name", sorted(TUBES))
def test_centerline_tracks_analytic_tube(name):
    fn, w, shape = TUBES[name]
    truth = _dense(fn, 0.0, 1.0)
    mask = rasterize_tube(truth, w, shape)

    cl = extract_centerline(mask, n_stations=200)

    # Interior stations (ends excluded) sit on the true centerline.
    interior = cl[16:-16]
    dev = dist_to_curve(interior, truth)
    assert dev.max() < 2.5
    assert np.median(dev) < 1.5

    # Endpoints reach near the tube ends (round caps extend by ~w).
    ends = np.array([truth[0], truth[-1]])
    d_first = np.linalg.norm(ends - cl[0], axis=1).min()
    d_last = np.linalg.norm(ends - cl[-1], axis=1).min()
    assert d_first < 1.6 * w
    assert d_last < 1.6 * w
    # The two extracted endpoints reach *different* tube ends.
    assert np.linalg.norm(cl[0] - cl[-1]) > 0.7 * np.linalg.norm(ends[0] - ends[1])


@pytest.mark.parametrize("name", sorted(TUBES))
def test_arc_length_monotone_and_uniform(name):
    fn, w, shape = TUBES[name]
    mask = rasterize_tube(_dense(fn, 0.0, 1.0), w, shape)
    cl = extract_centerline(mask, n_stations=150)
    s = arc_length(cl)
    steps = np.diff(s)
    assert (steps > 0).all()                      # strictly monotone
    assert steps.max() / steps.min() < 1.01       # uniform resampling


def test_orientation_widest_end_first():
    # A wedge tube: wide at x=30, narrow at x=270. Head (wide end) must be first.
    t = np.linspace(0, 1, 2000)
    truth = np.column_stack([30 + 240 * t, np.full_like(t, 45.0)])
    h, w = 90, 300
    yy, xx = np.mgrid[0:h, 0:w]
    width = 18.0 - 12.0 * np.clip((xx - 30) / 240.0, 0, 1)
    mask = (np.abs(yy - 45.0) <= width) & (xx >= 30) & (xx <= 270)
    cl = extract_centerline(mask, n_stations=100)
    assert cl[0, 0] < cl[-1, 0]  # starts at the wide (low-x) end

    # And the mirrored mask flips deterministically.
    cl_m = extract_centerline(mask[:, ::-1], n_stations=100)
    assert cl_m[0, 0] > cl_m[-1, 0]


def test_robust_to_ragged_mask():
    fn, w, shape = TUBES["s_curve"]
    truth = _dense(fn, 0.0, 1.0)
    mask = rasterize_tube(truth, w, shape).copy()
    rng = np.random.default_rng(3)
    # Pepper holes inside and specks outside; add a detached blob.
    holes = rng.integers(0, [shape[0], shape[1]], size=(300, 2))
    mask[holes[:, 0], holes[:, 1]] = False
    mask[5:15, 5:15] = True
    cl = extract_centerline(mask, n_stations=200)
    dev = dist_to_curve(cl[16:-16], truth)
    assert dev.max() < 3.5


def test_resample_polyline_endpoints_and_count():
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0]])
    out = resample_polyline(pts, 31)
    assert out.shape == (31, 2)
    assert np.allclose(out[0], pts[0])
    assert np.allclose(out[-1], pts[-1])
    s = arc_length(out)
    assert np.allclose(np.diff(s), s[-1] / 30, rtol=1e-6)


def test_single_pixel_mask_raises_cleanly():
    import numpy as np
    import pytest
    from centerline import extract_centerline

    mask = np.zeros((20, 20), dtype=bool)
    mask[10, 10] = True
    with pytest.raises(ValueError, match="no centerline exists"):
        extract_centerline(mask)


def test_non_tubular_mask_warns():
    import numpy as np
    import pytest
    from centerline import extract_centerline

    # A solid disc: no tube structure at all, medial path is far shorter than
    # area / (2 * mean path half-width) predicts (measured ratio ~0.65 vs
    # 0.96+ for genuine tubes). A fused hairpin is NOT detectable this way -
    # see the KNOWN LIMITATION note in extract_centerline.
    yy, xx = np.mgrid[0:80, 0:80]
    mask = (yy - 40) ** 2 + (xx - 40) ** 2 < 35 ** 2
    with pytest.warns(RuntimeWarning, match="not look tubular"):
        extract_centerline(mask)


def test_healthy_tube_does_not_warn():
    import warnings as _w
    import numpy as np
    from centerline import extract_centerline

    yy, xx = np.mgrid[0:60, 0:300]
    tube = (np.abs(yy - 30) < 12) & (xx > 10) & (xx < 290)
    with _w.catch_warnings():
        _w.simplefilter("error", RuntimeWarning)
        extract_centerline(tube)
