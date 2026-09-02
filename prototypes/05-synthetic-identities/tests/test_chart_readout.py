"""Contract tests for the chart-space readout.

The readout is the script every chart-space number in README.md is quoted
from, so what has to be pinned is not "the answer is 0.87" -- that is a
property of the corpus -- but the properties that make the answer MEAN
something:

* the unwrap really is the oracle chart (a pixel lands in the cell its own
  ``(s, phi)`` names, and nowhere else);
* an under-covered pair is reported UNDEFINED, never as a low score;
* excluded cells never enter a score;
* the split is prototype 01's, not a local imitation;
* the sensitivity flags actually move the answer -- which is the whole reason
  they exist as flags.

One small corpus is generated once per session and shared.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_MELOPS = os.path.abspath(os.path.join(_ROOT, "..", "01-melops-ablation"))

import chart_readout as CR  # noqa: E402
import exclusions as E  # noqa: E402
import make_dataset  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.path.isdir(_MELOPS),
    reason="prototypes/01-melops-ablation not present (the split is imported "
           "from it, never reimplemented)",
)


# Small enough to stay inside the test budget, large enough that the open-set
# split has a gallery, known queries and novel queries.
CORPUS_KW = dict(
    n_individuals=8,
    sightings_per_individual=6,
    years=4,
    resolution=(112, 224),
    tex_size=96,
    chart_resolution=(80, 160),
    n_spots=180,
    n_stations=48,
    n_around=32,
    shadow_map_size=256,
    seed=0,
)

READOUT_RESOLUTION = (64, 120)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("readout_corpus"))
    summary = make_dataset.generate(root, **CORPUS_KW)
    return root, summary


@pytest.fixture(scope="module")
def params():
    return CR.ReadoutParams(chart_resolution=READOUT_RESOLUTION,
                            min_joint_coverage=0.02)


@pytest.fixture(scope="module")
def readout(corpus, params):
    root, _ = corpus
    return CR.run(root, params=params, with_truth=True, with_bands=True)


# ---------------------------------------------------------------------------
# 1. The unwrap is the ORACLE chart
# ---------------------------------------------------------------------------

def test_a_pixel_lands_in_the_cell_its_own_chart_gt_names():
    """The unwrap must be a scatter through the GT, not a resampling of it.

    A synthetic frame with two known pixels at two known ``(s, phi)`` must
    light exactly the two cells those coordinates index and no others -- if
    this drifts, every downstream number is measuring the wrong skin.
    """
    res = (16, 32)
    chart_s = np.full((3, 3), np.nan)
    chart_phi = np.full((3, 3), np.nan)
    select = np.zeros((3, 3), dtype=bool)
    rgb = np.zeros((3, 3, 3))

    # s = 0.25 -> column 8 of 32;  phi = 0 (dorsal) -> row 8 of 16.
    chart_s[0, 0], chart_phi[0, 0], select[0, 0] = 0.25, 0.0, True
    rgb[0, 0] = 1.0
    # s = 0.75 -> column 24;  phi = +pi/2 (the animal's LEFT) -> row 12.
    chart_s[2, 2], chart_phi[2, 2], select[2, 2] = 0.75, math.pi / 2.0, True
    rgb[2, 2] = 0.5

    _values, cov = CR.unwrap(rgb, chart_s, chart_phi, select, res,
                             highpass_frac=0.05)
    lit = sorted(zip(*np.nonzero(cov)))
    assert lit == [(8, 8), (12, 24)], lit


def test_unselected_and_non_finite_pixels_never_reach_the_chart():
    res = (8, 16)
    chart_s = np.array([[0.5, np.nan], [0.5, 0.5]])
    chart_phi = np.zeros((2, 2))
    select = np.array([[True, True], [False, True]])
    rgb = np.ones((2, 2, 3))
    _v, cov = CR.unwrap(rgb, chart_s, chart_phi, select, res, highpass_frac=0.05)
    # only (0,0) and (1,1) are selected AND finite, and both name the same cell
    assert int(cov.sum()) == 1


def test_the_high_pass_removes_a_ramp_but_not_the_speckle():
    """Shading is low-frequency in chart space; the pattern is not."""
    h, w = 64, 120
    cov = np.ones((h, w), dtype=bool)
    s_ramp = np.linspace(0.0, 1.0, w)[None, :] * np.ones((h, 1))
    rng = np.random.default_rng(0)
    speckle = rng.standard_normal((h, w)) * 0.1
    flat = CR.highpass(s_ramp, cov, radius=int(round(0.02 * w)))
    assert float(np.abs(flat).max()) < 0.05, "a linear ramp must be flattened"
    kept = CR.highpass(s_ramp + speckle, cov, radius=int(round(0.02 * w)))
    # the speckle survives: it correlates with itself far better than the ramp
    assert CR._pearson(kept.ravel(), speckle.ravel()) > 0.85


def test_excluded_cells_are_never_covered_and_never_scored(params):
    """"Excluding the eye and the mouth" has to hold in the READOUT too."""
    excl = CR.exclusion_mask(params.chart_resolution)
    assert excl.any() and not excl.all()
    h, w = params.chart_resolution
    S, PHI = E.chart_meshgrid((h, w))
    rgb = np.ones((h, w, 3))
    select = np.ones((h, w), dtype=bool)
    _v, cov = CR.unwrap(rgb, S, PHI, select, (h, w), exclusion=excl,
                        highpass_frac=params.highpass_frac)
    assert not (cov & excl).any()


# ---------------------------------------------------------------------------
# 2. An undefined pair is undefined
# ---------------------------------------------------------------------------

def test_a_pair_below_the_coverage_floor_is_nan_not_a_low_score():
    """"They never saw the same skin" is not evidence of a different animal."""
    val = np.zeros((8, 16))
    val[0, :4] = [1.0, -1.0, 1.0, -1.0]
    a_cov = np.zeros((8, 16), dtype=bool)
    a_cov[0, :4] = True
    b_cov = a_cov.copy()
    score, n = CR.chart_ncc(val, a_cov, val, b_cov, min_cells=4)
    assert n == 4 and abs(score - 1.0) < 1e-12
    score, n = CR.chart_ncc(val, a_cov, val, b_cov, min_cells=5)
    assert n == 4 and math.isnan(score)


def test_ncc_uses_only_the_jointly_covered_cells():
    val_a = np.zeros((4, 8))
    val_b = np.zeros((4, 8))
    val_a[0, :4] = [1.0, -1.0, 1.0, -1.0]
    val_b[0, :4] = [1.0, -1.0, 1.0, -1.0]
    val_a[1, :] = 5.0          # a's private cells disagree wildly
    val_b[1, :] = -5.0
    a_cov = np.zeros((4, 8), dtype=bool)
    b_cov = np.zeros((4, 8), dtype=bool)
    a_cov[0, :4] = a_cov[1, :] = True
    b_cov[0, :4] = True
    score, n = CR.chart_ncc(val_a, a_cov, val_b, b_cov, min_cells=1)
    assert n == 4
    assert abs(score - 1.0) < 1e-12


def test_a_band_restricts_the_cells_the_score_is_taken_over():
    val_a = np.zeros((4, 8))
    val_b = np.zeros((4, 8))
    val_a[0, :4] = [1.0, -1.0, 1.0, -1.0]
    val_b[0, :4] = [-1.0, 1.0, -1.0, 1.0]      # anti-correlated here
    val_a[2, :4] = [1.0, -1.0, 1.0, -1.0]
    val_b[2, :4] = [1.0, -1.0, 1.0, -1.0]      # correlated here
    cov = np.zeros((4, 8), dtype=bool)
    cov[0, :4] = cov[2, :4] = True
    band = np.zeros((4, 8), dtype=bool)
    band[2, :] = True
    assert CR.chart_ncc(val_a, cov, val_b, cov, 1, band=band)[0] == pytest.approx(1.0)
    band_other = np.zeros((4, 8), dtype=bool)
    band_other[0, :] = True
    assert CR.chart_ncc(val_a, cov, val_b, cov, 1,
                        band=band_other)[0] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# 3. The corpus readout
# ---------------------------------------------------------------------------

def test_the_readout_unwraps_the_corpus_and_beats_chance(readout):
    ident = readout["identity"]
    assert readout["n_unwrapped"] == readout["n_images_in_metadata"]
    assert ident["n_scored"] > 0
    assert ident["n_gallery"] > 0 and ident["n_novel_queries"] > 0
    # A tiny 8-animal corpus has a high chance rate (few same-side gallery
    # entries), so the bar is a multiple of the MEASURED chance, not 0.025.
    assert ident["rank1"] > 2.0 * ident["chance"], ident
    assert ident["rank1"] > 0.5, ident
    assert ident["separation"] > 0.1, ident


def test_the_split_is_prototype_01s_own(corpus, params):
    """Imported, not reimplemented: the counts must match protocol.py exactly."""
    if _MELOPS not in sys.path:
        sys.path.insert(0, _MELOPS)
    import pandas as pd
    import protocol

    root, _ = corpus
    meta = pd.read_csv(os.path.join(root, "metadata.csv"))
    gallery_df, query_df = protocol.one_shot_open_set_split(
        meta, cutoff_fraction=0.5, seed=0)
    excl = CR.exclusion_mask(params.chart_resolution)
    truth = CR.load_corpus(root)[1]
    values, coverage, _ = CR.unwrap_corpus(root, meta, truth, params, excl)
    res = CR.rank1_open_set(meta, values, coverage, params, min_cells=1,
                            split_seed=0, cutoff_fraction=0.5)
    assert res["n_gallery"] == len(gallery_df)
    assert res["n_known_queries"] == int(query_df["is_known"].astype(bool).sum())
    assert res["n_novel_queries"] == int((~query_df["is_known"].astype(bool)).sum())


def test_matching_is_side_partitioned(corpus, params):
    """A left flank is never scored against a right one (Schema S1: 0.70 %)."""
    import pandas as pd

    root, _ = corpus
    meta = pd.read_csv(os.path.join(root, "metadata.csv"))
    excl = CR.exclusion_mask(params.chart_resolution)
    truth = CR.load_corpus(root)[1]
    values, coverage, _ = CR.unwrap_corpus(root, meta, truth, params, excl)
    side = dict(zip(meta["image_id"].astype(str), meta["side"].astype(str)))
    # Every pair the recapture curve scores must agree on side, and every
    # gallery candidate a query is compared against does too (the Rank-1 path
    # filters on side before it ever computes an NCC).
    gap = CR.recapture_gap(meta, values, coverage, params, min_cells=1)
    assert gap["n_pairs"] > 0
    for pair in gap["pairs"]:
        assert side[pair["image_a"]] == side[pair["image_b"]]


def test_the_recapture_gap_tracks_the_drift_it_was_given(readout):
    """The TRUE chart NCC comes from drift.similarity on the rebuilt states.

    If the rebuild ever diverges from what ``make_dataset`` rendered, the
    "true" column stops being true -- so it is checked against the one thing
    the generator guarantees: drift is monotone in elapsed time.
    """
    gap = readout["recapture_gap"]
    assert readout["true_similarity_available"] is True
    assert gap["n_scored_pairs"] > 0
    true_means = [b["mean_true_ncc"] for b in gap["buckets"]
                  if b["n_scored"] > 0 and np.isfinite(b["mean_true_ncc"])]
    assert len(true_means) >= 3
    assert true_means == sorted(true_means, reverse=True), true_means
    assert gap["spearman_true_vs_elapsed"] < -0.5, gap


def test_the_rebuilt_states_are_the_states_that_were_rendered(corpus):
    """``individual_timeline`` is the generator's own path, not a copy of it."""
    root, summary = corpus
    args = summary["args"]
    context = make_dataset.build_pattern_context(
        head_signal=args["head_signal"], flank_signal=args["flank_signal"],
        n_spots=args["n_spots"], n_common=args["n_common"],
        chart_resolution=tuple(args["chart_resolution"]))
    identity, _length, states = make_dataset.individual_timeline(
        context, args["seed"], 0,
        sightings_per_individual=args["sightings_per_individual"],
        years=args["years"], start_date=args["start_date"])
    truth = CR.load_corpus(root)[1]
    rendered = {(r["identity"], r["date"]): r for r in truth.values()}
    seen = 0
    for date, _side, ind in states:
        rec = rendered.get((identity, str(date)))
        if rec is None:                       # a dropped frame; there are few
            continue
        seen += 1
        assert rec["n_spots"] == len(ind.spots), (identity, date)
        assert abs(rec["length_cm"] - float(ind.length_cm)) < 1e-9
    assert seen > 0


def test_the_band_table_covers_the_whole_body_and_the_three_bands(readout):
    names = [row["band"] for row in readout["bands"]]
    assert names[0] == "whole body"
    assert len(names) == 4
    cells = {row["band"]: row["n_cells"] for row in readout["bands"]}
    assert (sum(v for k, v in cells.items() if k != "whole body")
            == cells["whole body"])


# ---------------------------------------------------------------------------
# 4. The sensitivity flags must actually be sensitive
# ---------------------------------------------------------------------------

def test_the_coverage_floor_changes_how_many_pairs_are_defined(corpus, params):
    root, _ = corpus
    loose = CR.run(root, params=params.replace(min_joint_coverage=0.01),
                   with_truth=False, with_bands=False)
    tight = CR.run(root, params=params.replace(min_joint_coverage=0.25),
                   with_truth=False, with_bands=False)
    assert (loose["recapture_gap"]["n_scored_pairs"]
            > tight["recapture_gap"]["n_scored_pairs"])
    assert loose["min_joint_cells"] < tight["min_joint_cells"]


def test_aliasing_the_chart_costs_separation(corpus, params):
    """Below Nyquist for the spot diameter the readout throws the signal away."""
    root, _ = corpus
    fine = CR.run(root, params=params, with_truth=False, with_bands=False)
    coarse = CR.run(root, params=params.replace(chart_resolution=(16, 30)),
                    with_truth=False, with_bands=False)
    assert (coarse["identity"]["separation"]
            < fine["identity"]["separation"]), (coarse["identity"],
                                                fine["identity"])


def test_the_identity_mask_flag_changes_which_pixels_are_unwrapped(corpus, params):
    root, _ = corpus
    on = CR.run(root, params=params, with_truth=False, with_bands=False)
    off = CR.run(root, params=params.replace(use_identity_mask=False),
                 with_truth=False, with_bands=False)
    assert (off["coverage_fraction"]["mean"]
            > on["coverage_fraction"]["mean"]), (off["coverage_fraction"],
                                                 on["coverage_fraction"])


def test_the_readout_is_deterministic(corpus, params):
    root, _ = corpus
    import json

    a = CR.run(root, params=params, with_truth=False, with_bands=False)
    b = CR.run(root, params=params, with_truth=False, with_bands=False)
    # json, not ==: an undefined correlation is NaN and NaN != NaN, but two
    # runs must still agree that it is undefined.
    assert (json.dumps(a["identity"], sort_keys=True)
            == json.dumps(b["identity"], sort_keys=True))
    assert (json.dumps(a["recapture_gap"], sort_keys=True)
            == json.dumps(b["recapture_gap"], sort_keys=True))


def test_bad_parameters_are_refused_not_clamped():
    with pytest.raises(ValueError):
        CR.ReadoutParams(min_joint_coverage=1.5)
    with pytest.raises(ValueError):
        CR.ReadoutParams(highpass_frac=0.0)
    with pytest.raises(ValueError):
        CR.ReadoutParams(chart_resolution=(2, 2))


# ---------------------------------------------------------------------------
# 5. The CLI a README command line has to keep working
# ---------------------------------------------------------------------------

def test_the_cli_writes_readout_json_and_prints_a_markdown_table(corpus, tmp_path,
                                                                capsys):
    root, _ = corpus
    out = str(tmp_path / "readout.json")
    rc = CR.main(["--data", root, "--chart-resolution", "64x120",
                  "--min-joint-coverage", "0.02", "--no-truth", "--out", out,
                  "--quiet"])
    assert rc == 0
    assert os.path.exists(out)
    printed = capsys.readouterr().out
    assert "one-shot open-set Rank-1" in printed
    assert "| gap (days) |" in printed

    import json
    record = json.load(open(out))
    assert record["params"]["chart_resolution"] == [64, 120]
    assert record["identity"]["n_scored"] > 0
