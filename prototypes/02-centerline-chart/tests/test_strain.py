from __future__ import annotations

import numpy as np

from strain_demo import PARAMS


def test_eps0_measures_near_zero(demo_measurements):
    stats, _ = demo_measurements[0.0]
    assert stats["n_matched"] >= 0.9 * stats["n_spots_truth"]
    # Bend-invariance: residual chart displacement well under 0.5% of BL.
    assert stats["mean_abs_ds_pct_bl"] < 0.5
    assert stats["max_abs_ds_pct_bl"] < 1.0
    assert stats["mean_abs_dr_px"] < 1.5


def test_eps5pct_displacement_consistent_with_strain(demo_measurements):
    stats0, _ = demo_measurements[0.0]
    stats, art = demo_measurements[0.05]
    assert stats["n_matched"] >= 0.9 * stats["n_spots_truth"]

    # Strain must dominate the eps=0 pipeline floor.
    assert stats["mean_abs_ds_px"] > 3.0 * stats0["mean_abs_ds_px"]

    # Per-spot: measured ds tracks the beam-model prediction
    # ds = eps * (r / W) * s within a factor band, not exactly.
    rows = art["rows"]
    pred = np.array([q["pred"] for q in rows])
    ds = np.array([q["ds"] for q in rows])
    strong = np.abs(pred) > 3.0
    assert strong.sum() >= 5
    ratio = ds[strong] / pred[strong]
    assert 0.5 < np.median(ratio) < 1.6
    # Aggregate consistency: the regression slope is near 1.
    assert 0.6 < stats["fit_slope_measured_vs_predicted"] < 1.5
    # Scale sanity: max displacement is a few percent of body length,
    # bounded by eps * 0.72 (max |r|/W of a spot) * BL.
    assert stats["max_abs_ds_px"] < 0.05 * 0.9 * PARAMS["L"]


def test_convex_and_concave_sides_displace_oppositely(demo_measurements):
    stats, _ = demo_measurements[0.05]
    convex = stats["convex"]
    concave = stats["concave"]
    assert convex["n"] >= 5 and concave["n"] >= 5
    # Stretched (convex) side spots slide tail-ward (+s), compressed
    # (concave) side spots slide head-ward (-s).
    assert convex["mean_signed_ds_px"] > 1.0
    assert concave["mean_signed_ds_px"] < -1.0


def test_metrics_and_panels_written(tmp_path):
    import strain_demo

    metrics = strain_demo.main(seed=0, out_dir=str(tmp_path))
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "panel_eps_0.00.png").exists()
    assert (tmp_path / "panel_eps_0.05.png").exists()
    import json

    on_disk = json.loads((tmp_path / "metrics.json").read_text())
    assert on_disk["runs"]["eps_0.05"]["n_matched"] == \
        metrics["runs"]["eps_0.05"]["n_matched"]
