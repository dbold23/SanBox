"""Tests for the benchmark corpus plan and the matcher's benchmark protocol.

Two blocks, one per thing added for the calibrated corpus:

1. ``synth_render.plan_same_side_sightings`` / ``identity_timeline`` -- the
   corpus plan that replaces 05's field-catalogue ``plan_sightings``. The smoke
   corpus proved why it is needed: both of its same-individual pairs were
   L-vs-R, so they shared no spots and the AUROC measured nothing.
2. ``eval_constellation``'s same-side rank protocol, drift buckets and the
   ``--conf-min`` spot filter.

Neither block renders or loads a model; the whole file is sub-second.

Run with the MAIN checkout venv (python 3.9):
    "/Volumes/External Dive 2TB/projects/marine-cv/7Gill/.venv/bin/python" \
        -W ignore -m pytest prototypes/06-spot-proxy/tests/test_corpus_eval.py -q
"""

from __future__ import annotations

import collections
import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_P06 = os.path.dirname(_HERE)
if _P06 not in sys.path:
    sys.path.insert(0, _P06)

import eval_constellation as E  # noqa: E402
import synth_render as SR  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. the corpus plan                                                           #
# --------------------------------------------------------------------------- #
def _plan(seed, n=4, years=3.0, min_same_side=2):
    return SR.plan_same_side_sightings(np.random.default_rng(seed), n, years,
                                       start="2019-03-01",
                                       min_same_side=min_same_side)


def test_the_plan_returns_exactly_the_requested_number_of_sightings():
    """05's plan draws 2..n+2 sightings, or 1 for a deliberate singleton.

    A corpus of "40 identities x 4 sightings" must actually be 160 frames with
    4 per animal, or the pair counts in the report do not mean what they say.
    """
    for n in (1, 2, 3, 4, 6):
        for seed in range(25):
            assert len(_plan(seed, n=n)) == n


def test_every_identity_gets_at_least_two_same_side_sightings():
    for seed in range(200):
        sides = [sd for _, sd in _plan(seed)]
        assert collections.Counter(sides).most_common(1)[0][1] >= 2


def test_min_same_side_is_honoured_at_other_values_and_clamps_above_n():
    for k in (1, 2, 3, 4):
        for seed in range(40):
            sides = [sd for _, sd in _plan(seed, n=4, min_same_side=k)]
            assert collections.Counter(sides).most_common(1)[0][1] >= k
    # asking for more same-side sightings than there are sightings is not an
    # error, it just makes every sighting the same side
    sides = [sd for _, sd in _plan(0, n=3, min_same_side=9)]
    assert len(set(sides)) == 1


def test_the_repair_only_ever_adds_primary_side_sightings():
    """The side repair must not manufacture a *second* flank.

    It flips minority-side draws onto the primary; it never flips the other
    way, so the number of distinct sides can only go down.
    """
    for seed in range(60):
        rng_a = np.random.default_rng(seed)
        rng_b = np.random.default_rng(seed)
        strict = [sd for _, sd in SR.plan_same_side_sightings(
            rng_a, 4, 3.0, min_same_side=3)]
        loose = [sd for _, sd in SR.plan_same_side_sightings(
            rng_b, 4, 3.0, min_same_side=0)]
        # the side draws happen before the repair, so `loose` IS the raw draw
        strict_major = collections.Counter(strict).most_common(1)[0]
        loose_major = collections.Counter(loose).most_common(1)[0][1]
        assert strict_major[1] >= loose_major
        assert set(strict) <= set(loose), "the repair never invents a flank"
        for a, b in zip(strict, loose):
            assert a == b or a == strict_major[0]


def test_dates_are_strictly_increasing_and_inside_the_window():
    for seed in range(50):
        plan = _plan(seed, n=4, years=3.0)
        d = [np.datetime64(x) for x, _ in plan]
        assert all(d[i] < d[i + 1] for i in range(len(d) - 1)), "no same-day pairs"
        t0 = np.datetime64("2019-03-01")
        span = int((d[-1] - t0).astype("timedelta64[D]").astype(int))
        assert 0 <= span <= int(round(3.0 * 365.25)) + 2


def test_gaps_populate_every_recapture_bucket_over_the_corpus():
    """Log-uniform gaps, inherited from 05, must fill the short AND long buckets."""
    gaps = []
    for seed in range(200):
        d = [np.datetime64(x) for x, _ in _plan(seed)]
        gaps.extend(int((b - a).astype("timedelta64[D]").astype(int))
                    for i, a in enumerate(d) for b in d[i + 1:])
    gaps = np.asarray(gaps)
    for lo, hi, _ in E.DRIFT_BUCKETS:
        assert ((gaps >= lo) & (gaps <= hi)).sum() > 0, "empty bucket %d-%d" % (lo, hi)


def test_the_plan_is_deterministic_in_its_generator():
    assert _plan(3) == _plan(3)
    assert _plan(3) != _plan(4)


@pytest.mark.parametrize("index", [0, 1, 7])
def test_identity_timeline_applies_prototype_05_drift_between_dates(index):
    """The pattern a sighting shows must be 05's DRIFTED pattern for its date.

    Not a re-draw: the identity is generated once at the first date and
    ``drift.resight`` walks it forward, so consecutive states are different
    objects carrying a growing animal's spots.
    """
    ctx = SR.pattern_context(SR.load_config())
    identity, length_cm, states = SR.identity_timeline(
        ctx, seed=0, index=index, n_sightings=4, years=3.0,
        start_date="2019-03-01", min_same_side=2)
    assert identity == "syn%04d" % index
    assert len(states) == 4
    dates = [d for d, _, _ in states]
    assert len(set(dates)) == 4
    inds = [ind for _, _, ind in states]
    assert all(a is not b for a, b in zip(inds, inds[1:])), "resight not applied"
    assert str(inds[0].date) == dates[0] and str(inds[-1].date) == dates[-1]
    assert inds[-1].length_cm > inds[0].length_cm, "the animal must grow"


def test_identity_timeline_matches_05_seeding_for_identity_and_length():
    """Same seeds as ``make_dataset.individual_timeline``, so the same animal.

    Only the PLAN differs; the identity string and the drawn length must come
    out of the same generator streams, or the two corpora are of different
    populations and their numbers are not comparable.
    """
    import make_dataset

    ctx = SR.pattern_context(SR.load_config())
    for index in (0, 5):
        a = SR.identity_timeline(ctx, seed=0, index=index, n_sightings=4,
                                 years=3.0, start_date="2019-03-01")
        b = make_dataset.individual_timeline(ctx, seed=0, index=index,
                                             sightings_per_individual=4, years=3.0,
                                             start_date="2019-03-01")
        assert a[0] == b[0]
        assert a[1] == pytest.approx(b[1])


def test_the_config_default_leaves_05s_plan_in_place():
    """``corpus.min_same_side`` defaults to None so nothing already rendered moves."""
    assert SR.DEFAULT_CONFIG["corpus"]["min_same_side"] is None
    cfg = SR.load_config(overrides={"corpus": {"min_same_side": 2}})
    assert cfg["corpus"]["min_same_side"] == 2
    assert SR.DEFAULT_CONFIG["corpus"]["min_same_side"] is None


# --------------------------------------------------------------------------- #
# 2. the same-side rank protocol                                               #
# --------------------------------------------------------------------------- #
def _rank(score, labels, sides, encs):
    return E._rank_eval(np.asarray(score, float), labels, sides, encs)


def test_rank_eval_on_a_hand_worked_matrix():
    """Four images, two animals, one flank, four encounters.

    scores (symmetric, diagonal unused):
            0     1     2     3
      0    -    0.9   0.4   0.2      0,1 = animal A ; 2,3 = animal B
      1   0.9    -    0.1   0.3
      2   0.4   0.1    -    0.8
      3   0.2   0.3   0.8    -

    Query 0: gallery {1 (correct, 0.9), 2 (0.4), 3 (0.2)} -> rank 1.
    Query 1: correct 0 at 0.9, wrong 3 at 0.3, 2 at 0.1     -> rank 1.
    Query 2: correct 3 at 0.8, wrong 0 at 0.4, 1 at 0.1     -> rank 1.
    Query 3: correct 2 at 0.8, wrong 1 at 0.3, 0 at 0.2     -> rank 1.
    """
    sc = [[0, .9, .4, .2], [.9, 0, .1, .3], [.4, .1, 0, .8], [.2, .3, .8, 0]]
    out = _rank(sc, ["A", "A", "B", "B"], ["L"] * 4, ["e0", "e1", "e2", "e3"])
    assert out["n_query"] == 4 and out["n_unscorable"] == 0
    assert out["rank1"] == 1.0 and out["rank5"] == 1.0
    assert out["mean_rank"] == 1.0 and out["mrr"] == 1.0
    assert out["gallery_size_median"] == 3.0


def test_a_wrong_entry_above_the_correct_one_costs_exactly_one_rank():
    """Query 0's correct entry (1, at 0.4) sits below one distractor (2, at 0.7)."""
    sc = [[0, .4, .7, .1], [.4, 0, .1, .2], [.7, .1, 0, .8], [.1, .2, .8, 0]]
    out = _rank(sc, ["A", "A", "B", "B"], ["L"] * 4, ["e0", "e1", "e2", "e3"])
    assert out["n_query"] == 4
    assert out["rank1"] == 0.75          # queries 1, 2, 3 are rank 1
    assert out["mean_rank"] == pytest.approx((2 + 1 + 1 + 1) / 4.0)
    assert out["mrr"] == pytest.approx((0.5 + 1 + 1 + 1) / 4.0)


def test_ties_at_the_top_are_counted_against_the_query_and_reported():
    """The score is n_inliers/min(n_a,n_b), so ties are real and must not be hidden.

    Query 0's correct entry (1) ties with distractor 2 at 0.5. Pessimistic rank
    is 2; optimistic is 1.
    """
    sc = [[0, .5, .5, .1], [.5, 0, .1, .2], [.5, .1, 0, .9], [.1, .2, .9, 0]]
    out = _rank(sc, ["A", "A", "B", "B"], ["L"] * 4, ["e0", "e1", "e2", "e3"])
    assert out["rank1"] == 0.75
    assert out["rank1_optimistic"] == 1.0
    assert out["n_query_tied_at_top"] == 1


def test_opposite_side_entries_are_not_in_the_gallery_at_all():
    """An R photo cannot be retrieved by an L query and is not a distractor either.

    Image 1 is the only same-individual entry for query 0 and it is on the other
    flank, so query 0 is unscorable; image 2 is a distractor on the wrong flank
    and must not appear in anybody's gallery.
    """
    sc = [[0, .9, .9, .9], [.9, 0, .9, .9], [.9, .9, 0, .9], [.9, .9, .9, 0]]
    out = _rank(sc, ["A", "A", "B", "B"], ["L", "R", "R", "L"],
                ["e0", "e1", "e2", "e3"])
    # every query has at most one same-side entry and it is the wrong animal
    assert out["n_query"] == 0 and out["n_unscorable"] == 4


def test_same_encounter_entries_are_excluded_from_the_gallery():
    """Near-duplicates are the leakage that makes the real arm unusable.

    Images 0 and 1 are one animal in ONE encounter; 2 and 3 are another animal
    in another. Nothing is retrievable across encounters, so nothing is
    scorable -- the protocol must say so rather than report Rank-1 1.0 off
    near-duplicates.
    """
    sc = [[0, .9, .1, .1], [.9, 0, .1, .1], [.1, .1, 0, .9], [.1, .1, .9, 0]]
    out = _rank(sc, ["A", "A", "B", "B"], ["L"] * 4, ["e0", "e0", "e1", "e1"])
    assert out["n_query"] == 0 and out["n_unscorable"] == 4


def test_rank5_and_rank10_track_the_gallery_depth():
    """One correct entry buried under seven distractors: rank 8."""
    n = 10
    sc = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                sc[i, j] = 0.1
    sc[0, 1] = sc[1, 0] = 0.5            # the correct pair
    for j in range(2, 9):                # seven distractors above it
        sc[0, j] = sc[j, 0] = 0.9
    labels = ["A", "A"] + ["B%d" % k for k in range(2, n)]
    out = _rank(sc, labels, ["L"] * n, ["e%d" % k for k in range(n)])
    assert out["n_query"] == 2           # only A's two images have a correct entry
    q0 = 8                               # 7 distractors above + itself
    assert out["mean_rank"] == pytest.approx((q0 + 1) / 2.0)
    assert out["rank1"] == 0.5 and out["rank5"] == 0.5 and out["rank10"] == 1.0


# --------------------------------------------------------------------------- #
# 3. drift buckets                                                             #
# --------------------------------------------------------------------------- #
def test_elapsed_days_is_symmetric_and_tolerates_non_dates():
    assert E._elapsed_days("2019-03-01", "2019-03-31") == 30
    assert E._elapsed_days("2019-03-31", "2019-03-01") == 30
    assert E._elapsed_days("2019-03-01", "2021-03-01") == 731
    assert E._elapsed_days("AOTB_2019_08", "AOTB_2020_01") is None
    assert E._elapsed_days(None, "2019-03-01") is None


def test_a_catalog_encounter_id_is_never_read_as_a_year():
    """The regression this guard exists for.

    ``np.datetime64("21")`` is the year 21, so the first real run turned
    encounter ids 2 and 21 into a 6940-day recapture interval for AOTB_A014.
    Small integers and bare numeric strings must be refused outright.
    """
    assert E._elapsed_days(2, 21) is None
    assert E._elapsed_days("2", "21") is None
    assert E._elapsed_days(2019, 2020) is None
    assert E._as_date(2) is None and E._as_date("21") is None
    assert E._as_date(True) is None


def test_a_posix_timestamp_is_accepted_because_that_is_what_exif_ts_holds():
    """`results/real/detections.jsonl` carries `exif_ts` as epoch seconds."""
    assert E._as_date(1583935256) == np.datetime64("2020-03-11")
    # 2019-08-28 10:49:02 UTC -> 2020-03-11 is 196 days
    assert E._elapsed_days(1566989342, 1583935256) == 196
    assert E._elapsed_days(1566989342, "2019-08-28") == 0


def test_drift_buckets_partition_the_positives_by_elapsed_time():
    pos = [{"score": 0.8, "elapsed_days": 10}, {"score": 0.6, "elapsed_days": 182},
           {"score": 0.5, "elapsed_days": 183}, {"score": 0.4, "elapsed_days": 500},
           {"score": 0.3, "elapsed_days": 900}, {"score": 0.9, "elapsed_days": None}]
    out = E._drift_eval(pos, [0.1, 0.2, 0.05])
    assert [out[n]["n_pairs"] for _, _, n in E.DRIFT_BUCKETS] == [2, 1, 1, 1]
    assert out["0-6 months"]["mean_score"] == pytest.approx(0.7)
    assert out["2+ years"]["mean_score"] == pytest.approx(0.3)
    assert out["n_pairs_without_dates"] == 1
    # median of the FIVE dated pairs (10, 182, 183, 500, 900)
    assert out["elapsed_days"] == {"min": 10, "median": 183.0, "max": 900}
    # every bucket's AUROC is against the SAME negative population
    assert out["0-6 months"]["auroc_vs_all_negatives"] == 1.0
    assert out["6-12 months"]["auroc_vs_all_negatives"] == 1.0


def test_drift_buckets_cover_the_line_with_no_gaps_or_overlaps():
    edges = [(lo, hi) for lo, hi, _ in E.DRIFT_BUCKETS]
    assert edges[0][0] == 0
    for (lo_a, hi_a), (lo_b, _) in zip(edges, edges[1:]):
        assert lo_b == hi_a + 1


# --------------------------------------------------------------------------- #
# 4. the --conf-min spot filter                                                #
# --------------------------------------------------------------------------- #
def _record(tmp_path, confs, image_id="i0"):
    w, h = 900, 300
    cx, cy, a, b = 450.0, 150.0, 380.0, 110.0
    t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    poly = np.column_stack([cx + a * np.cos(t), cy + b * np.sin(t)]).round(1).tolist()
    rng = np.random.default_rng(0)
    spots = []
    for c in confs:
        while True:
            x = rng.uniform(cx - a, cx + a)
            y = rng.uniform(cy - b, cy + b)
            if ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 0.7 ** 2:
                break
        spots.append({"x": x - 8, "y": y - 8, "w": 16.0, "h": 16.0,
                      "cx": x, "cy": y, "conf": float(c)})
    path = tmp_path / "det.jsonl"
    with open(str(path), "w") as fh:
        fh.write(json.dumps({"image_id": image_id, "individual_code": "A",
                             "encounter_id": "e0", "side": "L", "date": "2019-03-01",
                             "width": w, "height": h, "body_polygon": poly,
                             "obstruction_polygons": None, "spots": spots}) + "\n")
    return path


def test_conf_min_filters_spots_before_rectification(tmp_path):
    confs = [0.26, 0.31, 0.39, 0.41, 0.55, 0.62, 0.70, 0.80]
    path = _record(tmp_path, confs)

    sets, records, drops = E.load_detections(str(path))
    assert E.load_detections.last_spot_filter is None
    assert len(sets[0]) == len(confs)

    sets, records, drops = E.load_detections(str(path), conf_min=0.40)
    stats = E.load_detections.last_spot_filter
    assert stats == {"conf_min": 0.40, "spots_in": 8, "spots_kept": 5,
                     "kept_frac": 5 / 8.0}
    assert len(sets[0]) == 5, "the SpotSet itself must be built from the kept spots"
    assert not drops, "a spot filter is not an image drop"
    assert records[0]["conf_min"] == 0.40
    assert all(sp["conf"] >= 0.40 for sp in records[0]["spots"])


def test_conf_min_is_inclusive_at_the_threshold(tmp_path):
    path = _record(tmp_path, [0.40, 0.40, 0.3999, 0.5])
    sets, _, _ = E.load_detections(str(path), conf_min=0.40)
    assert E.load_detections.last_spot_filter["spots_kept"] == 3


def test_conf_min_above_every_spot_drops_the_image_not_the_run(tmp_path):
    path = _record(tmp_path, [0.26, 0.30, 0.31])
    sets, records, drops = E.load_detections(str(path), conf_min=0.90)
    assert sets == [] and records == []
    assert len(drops) == 1 and "too few spots" in drops[0]["reason"]
    assert E.load_detections.last_spot_filter["spots_kept"] == 0


# --------------------------------------------------------------------------- #
# 5. the --frames report                                                       #
# --------------------------------------------------------------------------- #
def test_run_frames_tallies_frames_and_rejection_kinds(tmp_path, monkeypatch):
    """The frame report is the source for "the chart frame fails on real masks".

    Two records: a long thin ellipse that prototype 02 can chart, and a fat blob
    that it cannot. The tally must name the frame each one landed in and split
    the chart rejection reasons by KIND -- the reasons carry their measured
    value in parentheses (`spots_outside_body(0.12)`), and tallying the raw
    string gives one bucket per value instead of one per reason.
    """
    import eval_constellation as EE

    def rec(w, h, a, b, n, seed, image_id):
        rng = np.random.default_rng(seed)
        cx, cy = w / 2.0, h / 2.0
        t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        poly = np.column_stack([cx + a * np.cos(t), cy + b * np.sin(t)])
        spots = []
        while len(spots) < n:
            x = rng.uniform(cx - a, cx + a)
            y = rng.uniform(cy - b, cy + b)
            if ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 0.81:
                spots.append({"x": x - 8, "y": y - 8, "w": 16.0, "h": 16.0,
                              "cx": x, "cy": y, "conf": 0.8})
        return {"image_id": image_id, "individual_code": "A", "encounter_id": "e",
                "side": "L", "width": w, "height": h,
                "body_polygon": [[round(x, 1), round(y, 1)] for x, y in poly],
                "obstruction_polygons": None, "spots": spots}

    path = tmp_path / "det.jsonl"
    with open(str(path), "w") as fh:
        fh.write(json.dumps(rec(1200, 260, 540, 90, 60, 0, "thin")) + "\n")
        fh.write(json.dumps(rec(700, 600, 300, 260, 60, 1, "blob")) + "\n")

    monkeypatch.setattr(EE, "RESULTS", str(tmp_path))
    out = EE.run_frames(str(path), out_prefix="t")

    assert out["mode"] == "frames"
    assert out["n_records_read"] == 2 and out["n_rectified"] == 2
    assert set(out["frames"]) <= {"chart", "pca"}
    assert sum(out["frames"].values()) == 2
    assert out["pca_fallback_frac"] == out["frames"].get("pca", 0) / 2.0
    # every rejection key is a bare reason NAME, never a name plus its value
    for key in out["chart_rejection_reasons"]:
        assert "(" not in key and ")" not in key
    assert out["max_abs_r_raw"]["max"] >= out["max_abs_r_raw"]["min"]
    assert os.path.exists(os.path.join(str(tmp_path), "t_summary.json"))
    assert json.load(open(os.path.join(str(tmp_path), "t_summary.json")))["mode"] == "frames"


def test_run_frames_puts_the_fat_blob_in_the_pca_fallback(tmp_path, monkeypatch):
    """A near-circular mask is exactly what prototype 02 cannot chart."""
    import eval_constellation as EE

    rng = np.random.default_rng(2)
    w = h = 700
    cx = cy = 350.0
    a = b = 300.0
    t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    poly = np.column_stack([cx + a * np.cos(t), cy + b * np.sin(t)])
    spots = []
    while len(spots) < 50:
        x, y = rng.uniform(cx - a, cx + a), rng.uniform(cy - b, cy + b)
        if ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 0.81:
            spots.append({"x": x - 8, "y": y - 8, "w": 16.0, "h": 16.0,
                          "cx": x, "cy": y, "conf": 0.8})
    path = tmp_path / "blob.jsonl"
    with open(str(path), "w") as fh:
        fh.write(json.dumps({"image_id": "blob", "individual_code": "A",
                             "encounter_id": "e", "side": "L", "width": w, "height": h,
                             "body_polygon": [[round(x, 1), round(y, 1)] for x, y in poly],
                             "obstruction_polygons": None, "spots": spots}) + "\n")
    monkeypatch.setattr(EE, "RESULTS", str(tmp_path))
    out = EE.run_frames(str(path), out_prefix="blob")
    assert out["frames"].get("pca") == 1, out["frames"]
    assert out["chart_rejection_reasons"], "the rejection must say why"
