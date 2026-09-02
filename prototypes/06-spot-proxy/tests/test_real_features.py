"""Tests for the real-photo pass (``real_features.py``).

Two things, both of which were wrong and are cheap to pin down:

1. ``Accumulator`` -- the per-image scalar table must be conditioned on a usable
   body polygon. An image with no body has no spot field to measure, so its
   structural ``n_spots = 0`` is a detector failure, not a spot-field fact, and
   putting 61 of them into the table moved the published ``n_spots`` q05 from 39
   to 0 and its q50 from 112 to 107.
2. ``slim_record`` -- the tracked variant of ``detections.jsonl``. It must drop
   the body polygon and shrink the spot boxes while leaving ``feats`` byte
   identical, because ``summary.json`` is supposed to be recomputable from it.

Run: "MAIN/.venv/bin/python" -m pytest P06/tests -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import osea_contract as oc  # noqa: E402
import real_features as rf  # noqa: E402


def _det(n_spots=5, body=True, obstructions=0):
    """A contract-shaped detection dict, with or without a body polygon."""
    spots = [{"x": 10.0 * i, "y": 10.0, "w": 6.0, "h": 6.0,
              "cx": 10.0 * i + 3.0, "cy": 13.0, "conf": 0.5}
             for i in range(n_spots)]
    poly = None
    if body:
        t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        poly = np.column_stack([200 + 150 * np.cos(t), 100 + 60 * np.sin(t)]).tolist()
    ob = [[[0, 0], [20, 0], [20, 20], [0, 20]] for _ in range(obstructions)]
    return {"body_polygon": poly,
            "body_bbox": {"x": 50, "y": 40, "w": 300, "h": 120} if body else None,
            "body_conf": 0.9 if body else None,
            "obstruction_polygons": ob, "obstruction_count": len(ob),
            "head_polygon": None, "head_bbox": None, "head_conf": None,
            "spots": spots, "spot_count": len(spots),
            "image_width": 640, "image_height": 480,
            "spots_raw_count": n_spots if body else None,
            "spots_max_det": 300, "spots_truncated": False}


# --------------------------------------------------------------------------- #
# 1. the scalar table is conditioned on a body                                 #
# --------------------------------------------------------------------------- #
def test_no_body_image_is_missing_from_the_count_quantiles():
    acc = rf.Accumulator()
    for _ in range(4):
        acc.add(oc.features(_det(n_spots=40)))
    no_body = oc.features(_det(n_spots=0, body=False, obstructions=2))
    assert no_body["ok"] is False
    assert no_body["scalars"]["n_spots"] == 0            # structurally, not measured
    assert no_body["scalars"]["obstruction_count"] == 2
    acc.add(no_body)

    table = acc.quantile_table()
    assert acc.n_images == 5 and acc.n_body == 4 and acc.n_no_body == 1
    for key in rf.BODY_CONDITIONED_COUNTS:
        assert table[key]["n"] == 4, key
        assert table[key]["n_missing"] == 1, key
    # the structural zero must not drag the low quantile down
    assert table["n_spots"]["q005"] == 40.0
    # and it really is NaN in the raw array, not simply absent
    assert np.isnan(acc.scalars["n_spots"][-1])
    assert np.isnan(acc.scalars["obstruction_count"][-1])


def test_a_real_zero_spot_body_image_still_counts():
    """The exclusion is about the *body*, not about the spot count."""
    acc = rf.Accumulator()
    acc.add(oc.features(_det(n_spots=40)))
    acc.add(oc.features(_det(n_spots=0)))               # a body, genuinely no spots
    table = acc.quantile_table()
    assert table["n_spots"]["n"] == 2 and table["n_spots"]["n_missing"] == 0
    assert table["n_spots"]["q000"] == 0.0


def test_only_two_scalars_were_ever_non_missing_without_a_body():
    """``BODY_CONDITIONED_COUNTS`` must name exactly the keys the fix changes.

    ``run_image`` never runs the spot model without a body, so the shape a
    no-body record really takes is "no body, no spots"; all 61 such images in
    the catalog have ``n_spots == 0``.
    """
    flat = oc.flat_scalars(oc.features(_det(n_spots=0, body=False, obstructions=1)))
    present = sorted(k for k, v in flat.items() if v is not None)
    assert present == sorted(rf.BODY_CONDITIONED_COUNTS), present


# --------------------------------------------------------------------------- #
# 2. the tracked slim record                                                   #
# --------------------------------------------------------------------------- #
def _record(**kw):
    det = _det(**kw)
    feats = oc.features(det)
    return {"image_id": 7, "filename": "x.jpg", "rel_path": "data/x.jpg",
            "individual_code": "AOTB_A001", "encounter_id": 3, "exif_ts": None,
            "side": "L", "width": 640, "height": 480, "det": det,
            "feats": {k: v for k, v in feats.items() if k != "spots_raw"}}


def test_slim_record_drops_the_polygon_and_keeps_everything_else():
    rec = _record(n_spots=6)
    slim = rf.slim_record(rec)

    assert slim["det"]["body_polygon"] is None
    assert slim["det"]["body_polygon_dropped"] is True
    assert slim["det"]["body_bbox"] == rec["det"]["body_bbox"]
    assert slim["feats"] == rec["feats"]
    for key in ("image_id", "individual_code", "encounter_id", "side",
                "width", "height"):
        assert slim[key] == rec[key]
    # the original record must not be mutated -- the caller writes both files
    assert rec["det"]["body_polygon"] is not None
    assert isinstance(rec["det"]["spots"][0], dict)


def test_slim_spots_round_trip_to_the_same_numbers():
    rec = _record(n_spots=6)
    slim = rf.slim_record(rec)
    assert slim["det"]["spots_format"] == list(rf.SLIM_SPOT_KEYS)
    for full, thin in zip(rec["det"]["spots"], slim["det"]["spots"]):
        assert thin == [full[k] for k in rf.SLIM_SPOT_KEYS]
        # x/y are dropped because they are recoverable
        assert full["x"] == full["cx"] - full["w"] / 2.0
        assert full["y"] == full["cy"] - full["h"] / 2.0


def test_slim_record_is_json_serialisable_and_smaller():
    rec = _record(n_spots=60)
    full_bytes = len(json.dumps(rec, separators=(",", ":")))
    slim_bytes = len(json.dumps(rf.slim_record(rec), separators=(",", ":")))
    assert slim_bytes < full_bytes
