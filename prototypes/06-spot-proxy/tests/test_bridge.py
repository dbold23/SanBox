"""Tests for the generator/detector bridge (prototype 06).

Covers ``synth_features.py`` (the detection-to-render-ground-truth matcher and
the record shape it writes) and ``compare_features.py`` (the KS comparison and
the calibration objective).

Run with the MAIN checkout venv (python 3.9)::

    "/Volumes/External Dive 2TB/projects/marine-cv/7Gill/.venv/bin/python" \\
        -m pytest prototypes/06-spot-proxy/tests/test_bridge.py -q

Everything here is sub-second: nothing loads a model or renders a frame.  The
tests that need a real corpus on disk skip themselves when it is absent.
"""

from __future__ import annotations

import json
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

import compare_features as CF  # noqa: E402
import eval_constellation as E  # noqa: E402
import synth_features as SF  # noqa: E402

_REAL_JSONL = os.path.join(_P06, "results", "real", "detections.jsonl")
_SMOKE_JSONL = os.path.join(_P06, "results", "synth_smoke", "detections.jsonl")


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _det(cx, cy, conf, wh=8.0):
    return {"x": cx - wh / 2.0, "y": cy - wh / 2.0, "w": wh, "h": wh,
            "cx": float(cx), "cy": float(cy), "conf": float(conf)}


def _gt(cx, cy, radius_px, visible=True, sid=0):
    return {"id": sid, "cx": float(cx), "cy": float(cy),
            "radius_px": float(radius_px), "visible": bool(visible),
            "s": 0.5, "phi": 0.0, "rendered_darkness": 0.8, "n_pixels": 10}


# --------------------------------------------------------------------------- #
# 1. the TP/FP matcher, hand-built                                             #
# --------------------------------------------------------------------------- #
def test_matcher_hand_built_case():
    """A layout whose every assignment is worked out by hand.

    GT (radius_px -> tolerance = max(radius, 6)):
        g0 (100, 100) r=10 -> tol 10
        g1 (300, 100) r= 2 -> tol  6   (floor bites)
        g2 (500, 100) r=20 -> tol 20
        g3 (700, 100) r= 5 -> tol  6   (nothing near it)

    detections:
        d0 (104, 100) conf .90  -> 4 px from g0            TP
        d1 (109, 100) conf .80  -> 9 px from g0, but g0 is
                                   already claimed         FP
        d2 (305, 100) conf .70  -> 5 px from g1 (tol 6)    TP
        d3 (308, 100) conf .60  -> 8 px from g1, > tol     FP
        d4 (515, 100) conf .50  -> 15 px from g2 (tol 20)  TP
        d5 (900, 900) conf .40  -> nowhere near anything   FP
    """
    gt = [_gt(100, 100, 10, sid=0), _gt(300, 100, 2, sid=1),
          _gt(500, 100, 20, sid=2), _gt(700, 100, 5, sid=3)]
    det = [_det(104, 100, 0.90), _det(109, 100, 0.80), _det(305, 100, 0.70),
           _det(308, 100, 0.60), _det(515, 100, 0.50), _det(900, 900, 0.40)]

    m = SF.match_detections(det, gt, min_tol_px=6.0)

    assert m["n_det"] == 6
    assert m["n_gt"] == 4
    assert m["tp"] == 3
    assert sorted(p[0] for p in m["pairs"]) == [0, 2, 4]
    assert sorted(p[1] for p in m["pairs"]) == [0, 1, 2]
    assert m["fp"] == [1, 3, 5]
    assert m["missed"] == [3]
    assert m["precision"] == pytest.approx(3.0 / 6.0)
    assert m["recall"] == pytest.approx(3.0 / 4.0)
    assert m["tp_conf"] == pytest.approx([0.90, 0.70, 0.50])
    assert m["fp_conf"] == pytest.approx([0.80, 0.60, 0.40])
    # distances carried through
    dists = dict((p[0], p[2]) for p in m["pairs"])
    assert dists[0] == pytest.approx(4.0)
    assert dists[2] == pytest.approx(5.0)
    assert dists[4] == pytest.approx(15.0)


def test_matcher_is_one_to_one_and_greedy_by_confidence():
    """Two detections on one GT spot: the confident one wins, the other is a FP."""
    gt = [_gt(100, 100, 12)]
    lo_first = [_det(103, 100, 0.30), _det(97, 100, 0.90)]
    m = SF.match_detections(lo_first, gt)
    assert m["tp"] == 1 and m["fp"] == [0]          # index 1 (conf .90) took it
    assert m["pairs"][0][0] == 1
    # the *nearer* low-confidence detection does not displace it
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(1.0)


def test_matcher_picks_the_nearest_eligible_gt():
    """One detection, two eligible GT spots -> it claims the closer one."""
    gt = [_gt(100, 100, 30, sid=0), _gt(112, 100, 30, sid=1)]
    m = SF.match_detections([_det(110, 100, 0.5)], gt)
    assert m["tp"] == 1
    assert m["pairs"][0][1] == 1                    # 2 px away, not 10
    assert m["missed"] == [0]


def test_matcher_tolerance_floor():
    """``min_tol_px`` is a floor on a sub-pixel GT spot's match radius."""
    gt = [_gt(100, 100, 1.0)]
    det = [_det(105, 100, 0.5)]                     # 5 px away
    assert SF.match_detections(det, gt, min_tol_px=6.0)["tp"] == 1
    assert SF.match_detections(det, gt, min_tol_px=2.0)["tp"] == 0
    # and the GT radius wins when it is larger than the floor
    assert SF.match_detections([_det(115, 100, 0.5)], [_gt(100, 100, 20.0)],
                               min_tol_px=6.0)["tp"] == 1


def test_matcher_degenerate_sides():
    """No detections -> precision undefined; no GT -> recall undefined."""
    gt = [_gt(100, 100, 10)]
    m0 = SF.match_detections([], gt)
    assert m0["tp"] == 0 and m0["precision"] is None
    assert m0["recall"] == pytest.approx(0.0) and m0["missed"] == [0]

    m1 = SF.match_detections([_det(1, 1, 0.9)], [])
    assert m1["tp"] == 0 and m1["recall"] is None
    assert m1["precision"] == pytest.approx(0.0) and m1["fp"] == [0]

    m2 = SF.match_detections([], [])
    assert m2["precision"] is None and m2["recall"] is None


def test_visible_gt_filters_invisible_and_centreless():
    spots = [_gt(10, 10, 3, visible=True, sid=0),
             _gt(20, 20, 3, visible=False, sid=1),
             {"id": 2, "cx": None, "cy": None, "radius_px": None, "visible": True}]
    keep = SF.visible_gt(spots)
    assert [g["id"] for g in keep] == [0]


# --------------------------------------------------------------------------- #
# 2. compare_features: identical inputs must give D = 0 everywhere             #
# --------------------------------------------------------------------------- #
def _write_slice(src, dst, n):
    kept = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            fout.write(line)
            kept += 1
            if kept >= n:
                break
    return kept


@pytest.mark.skipif(not os.path.exists(_REAL_JSONL), reason="no real detections.jsonl")
def test_compare_identical_files_gives_zero_everywhere(tmp_path):
    a = str(tmp_path / "a.jsonl")
    b = str(tmp_path / "b.jsonl")
    n = _write_slice(_REAL_JSONL, a, 60)
    assert n == 60
    _write_slice(_REAL_JSONL, b, 60)

    left = CF.load_side(a)
    right = CF.load_side(b)
    summary = CF.compare(left, right)

    for thr, block in summary["thresholds"].items():
        for group in ("per_image", "pooled"):
            for name, entry in block[group].items():
                assert entry["ks_D"] == pytest.approx(0.0), (thr, group, name)
                assert not entry["degenerate"], (thr, group, name)
                assert entry["n_real"] == entry["n_synth"], (thr, group, name)
                for q in ("q25", "q50", "q75"):
                    assert entry["real"][q] == pytest.approx(entry["synth"][q]) \
                        or entry["real"][q] is entry["synth"][q]
    assert CF.objective(summary) == pytest.approx(0.0)
    assert CF.geometry_objective(summary) == pytest.approx(0.0)
    assert summary["objective"]["value"] == pytest.approx(0.0)
    assert summary["geometry_objective"]["value"] == pytest.approx(0.0)


def test_objective_is_the_documented_average():
    """The objective is exactly the two half-means, equally weighted."""
    block = {
        "per_image": dict((k, {"ks_D": d}) for k, d in
                          (("density", 0.4), ("size_q50", 0.2), ("nn_median", 0.6),
                           ("conf_q50", 0.8), ("aspect", 0.1), ("area_norm", 0.3),
                           ("bbox_width_frac", 0.5))),
        "pooled": dict((k, {"ks_D": d}) for k, d in
                       (("size", 0.1), ("nn", 0.2), ("conf", 0.3))),
    }
    summary = {"thresholds": {"0.25": block}}
    assert CF.objective(summary) == pytest.approx(0.5 * 0.5 + 0.5 * 0.2)
    assert CF.geometry_objective(summary) == pytest.approx((0.1 + 0.3 + 0.5) / 3.0)
    d = CF.objective(summary, detail=True)
    assert d["per_image_mean_D"] == pytest.approx(0.5)
    assert d["pooled_mean_D"] == pytest.approx(0.2)
    assert set(d["per_image_D"]) == set(CF.CAL_PER_IMAGE)


def test_empty_sample_scores_the_worst_possible_D():
    """A renderer that emits no spots must not score well by default."""
    e = CF.ks_entry(np.array([0.1, 0.2, 0.3]), np.zeros(0))
    assert e["ks_D"] == 1.0 and e["degenerate"] and e["ks_p"] is None
    assert e["n_synth"] == 0 and e["synth"]["q50"] is None


def test_threshold_filters_the_spot_statistics():
    """Raising the confidence floor drops spots and moves every spot statistic."""
    feats = {
        "ok": True,
        "frame": {"D_minor": 100.0},
        "spots_uv": [[0.0, 0.0, 0.02, 0.30],
                     [0.1, 0.0, 0.04, 0.45],
                     [0.3, 0.0, 0.06, 0.55]],
        "scalars": {"area_norm": 1.5, "aspect": 2.0, "bbox_width_frac": 0.9,
                    "body_conf": 0.8},
    }
    lo, sp_lo = CF.image_features(feats, 0.25)
    hi, sp_hi = CF.image_features(feats, 0.50)
    assert lo["n_spots"] == 3 and hi["n_spots"] == 1
    assert lo["density"] == pytest.approx(3 / 1.5)
    assert hi["density"] == pytest.approx(1 / 1.5)
    assert lo["size_q50"] == pytest.approx(0.04)
    assert hi["size_q50"] == pytest.approx(0.06)
    assert lo["conf_q50"] == pytest.approx(0.45)
    assert lo["nn_median"] == pytest.approx(0.1)   # NNs are 0.1, 0.1, 0.2
    assert hi["nn_median"] is None                 # one spot: no neighbour
    assert sp_lo["conf"].size == 3 and sp_hi["conf"].size == 1
    # geometry is threshold-independent
    for k in CF.GEOM_SCALARS:
        assert lo[k] == hi[k]


def test_no_body_record_yields_no_spot_statistics():
    feats = {"ok": False, "frame": None, "spots_uv": [],
             "scalars": {"area_norm": None, "aspect": None,
                         "bbox_width_frac": 0.5, "body_conf": None,
                         "n_spots": 7}}
    scal, spots = CF.image_features(feats, 0.25)
    assert scal["n_spots"] is None and scal["density"] is None
    assert scal["bbox_width_frac"] == pytest.approx(0.5)
    assert all(spots[k].size == 0 for k in CF.POOLED_FEATURES)


@pytest.mark.skipif(not os.path.exists(_REAL_JSONL), reason="no real detections.jsonl")
def test_recompute_at_the_detector_floor_reproduces_the_stored_scalars():
    """At conf >= 0.25 (the detector's own floor) nothing is filtered, so the
    recomputed statistics must equal what ``osea_contract.features`` stored."""
    checked = 0
    with open(_REAL_JSONL) as fh:
        for line in fh:
            rec = json.loads(line)
            f = rec["feats"]
            if not f["ok"]:
                continue
            sc = f["scalars"]
            scal, _ = CF.image_features(f, 0.25)
            assert int(scal["n_spots"]) == int(sc["n_spots"])
            if sc["density"] is not None:
                assert scal["density"] == pytest.approx(sc["density"])
            if sc["nn_median"] is not None:
                assert scal["nn_median"] == pytest.approx(sc["nn_median"])
            if (sc["size"] or {}).get("q50") is not None:
                assert scal["size_q50"] == pytest.approx(sc["size"]["q50"])
            checked += 1
            if checked >= 40:
                break
    assert checked == 40


# --------------------------------------------------------------------------- #
# 3. record shape round-trips through eval_constellation._to_detection         #
# --------------------------------------------------------------------------- #
def _synthetic_record():
    """One record in exactly the shape ``synth_features.run`` writes."""
    det = {
        "body_polygon": [[10.0, 10.0], [90.0, 10.0], [90.0, 40.0], [10.0, 40.0]],
        "body_bbox": {"x": 10, "y": 10, "w": 80, "h": 30},
        "body_conf": 0.61,
        "obstruction_polygons": [],
        "obstruction_count": 0,
        "head_polygon": None, "head_bbox": None, "head_conf": None,
        "spots": [_det(30, 25, 0.51), _det(60, 30, 0.42)],
        "spot_count": 2,
        "image_width": 120, "image_height": 60,
    }
    rec = {
        "image_id": "syn0000_00", "filename": "syn0000_00.jpg",
        "rel_path": "body/syn0000_00.jpg",
        "individual_code": "syn0000", "encounter_id": "2020-01-01",
        "exif_ts": None, "width": 120, "height": 60,
        "det": det,
        "feats": {"ok": True, "frame": {"D_minor": 30.0}, "spots_uv": [],
                  "scalars": {"n_spots": 2}},
        "identity": "syn0000", "sighting": 0, "date": "2020-01-01", "side": "L",
        "pose": {"yaw_deg": 3.0}, "camera": {"elevation_deg": 20.0},
    }
    return rec, det


def test_record_round_trips_through_to_detection():
    rec, det = _synthetic_record()
    flat = E._to_detection(json.loads(json.dumps(rec)))
    assert flat["body_polygon"] == det["body_polygon"]
    assert flat["spots"] == det["spots"]
    assert flat["obstruction_polygons"] == det["obstruction_polygons"]
    assert flat["width"] == 120 and flat["height"] == 60
    # identity / labelling fields survive for the pair evaluation
    assert flat["image_id"] == "syn0000_00"
    assert flat["individual_code"] == "syn0000"
    assert flat["encounter_id"] == "2020-01-01"
    assert flat["side"] == "L"
    # truth fields ride along untouched
    assert flat["identity"] == "syn0000" and flat["sighting"] == 0
    assert flat["pose"]["yaw_deg"] == pytest.approx(3.0)
    # "det" and "feats" are consumed, not duplicated
    assert "det" not in flat and "feats" not in flat


@pytest.mark.skipif(not (os.path.exists(_SMOKE_JSONL) and os.path.exists(_REAL_JSONL)),
                    reason="run synth_features.py on results/synth_smoke first")
def test_written_synth_records_match_the_real_record_shape():
    with open(_REAL_JSONL) as fh:
        real = json.loads(fh.readline())
    with open(_SMOKE_JSONL) as fh:
        synth = json.loads(fh.readline())

    shared = ("image_id", "filename", "rel_path", "individual_code",
              "encounter_id", "exif_ts", "width", "height", "det", "feats")
    for k in shared:
        assert k in real and k in synth, k
    assert set(synth["det"]) == set(real["det"])
    assert set(synth["feats"]) == set(real["feats"])
    for k in SF.TRUTH_FIELDS:
        assert k in synth, k

    # and both sides normalise the same way
    fr = E._to_detection(real)
    fs = E._to_detection(synth)
    for k in ("body_polygon", "spots", "width", "height"):
        assert k in fr and k in fs
    assert fs["width"] == synth["det"]["image_width"]
    assert fs["spots"] == synth["det"]["spots"]


@pytest.mark.skipif(not os.path.exists(_SMOKE_JSONL),
                    reason="run synth_features.py on results/synth_smoke first")
def test_smoke_corpus_loads_through_compare_features():
    side = CF.load_side(_SMOKE_JSONL)
    assert side["n_records"] == 6
    assert side["n_ok"] + side["n_no_body"] == side["n_records"]
    for t in CF.THRESHOLDS:
        for k in CF.PER_IMAGE_FEATURES:
            assert side["per_image"][t][k].shape == (6,)
