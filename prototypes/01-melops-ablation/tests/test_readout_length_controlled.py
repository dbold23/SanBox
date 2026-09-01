"""Tests for the length-controlled readout (pure functions) and the
training-side min-images-per-unit filter added for run 2."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import finetune
import readout_length_controlled as ro


def _frame(units_with_counts):
    rows = []
    i = 0
    for (identity, side), count in units_with_counts.items():
        for _ in range(count):
            rows.append(
                {"image_id": "img%d" % i, "identity": identity, "side": side,
                 "path": "p%d.jpg" % i, "date": "2020-01-01"}
            )
            i += 1
    return pd.DataFrame(rows)


def test_min_images_filter_drops_small_units_only():
    df = _frame({("a", "L"): 3, ("a", "R"): 1, ("b", "L"): 2, ("c", "L"): 1})
    out, n_units, n_rows = finetune._filter_min_images_per_unit(df, 2)
    assert n_units == 2 and n_rows == 2
    kept = set(zip(out["identity"], out["side"]))
    assert kept == {("a", "L"), ("b", "L")}


def test_min_images_filter_k1_is_noop():
    df = _frame({("a", "L"): 1})
    out, n_units, n_rows = finetune._filter_min_images_per_unit(df, 1)
    assert len(out) == 1 and n_units == 0 and n_rows == 0


def test_min_images_filter_raises_when_everything_drops():
    df = _frame({("a", "L"): 1, ("b", "R"): 1})
    with pytest.raises(finetune.ProtocolViolation):
        finetune._filter_min_images_per_unit(df, 2)


def test_assortativity_index_positive_when_matching_is_size_assorted():
    glen = np.array([100.0, 200.0])
    qlen = np.array([101.0, 199.0, 102.0, 198.0])
    S = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
    out = ro.compute_assortativity(S, qlen, glen, np.random.default_rng(0))
    assert out["index"] > 0.5
    assert out["n_queries_used"] == 4 and out["n_queries_missing_length"] == 0


def test_assortativity_counts_missing_lengths():
    glen = np.array([100.0, 200.0])
    qlen = np.array([101.0, np.nan])
    S = np.array([[0.9, 0.1], [0.1, 0.9]])
    out = ro.compute_assortativity(S, qlen, glen, np.random.default_rng(0))
    assert out["n_queries_missing_length"] == 1


def test_stratified_rank1_separates_terciles():
    g_units = [("a", "L"), ("b", "L"), ("c", "L")]
    q_units = [("a", "L")] * 3 + [("b", "L")] * 3 + [("c", "L")] * 3
    known = np.ones(9, dtype=bool)
    # terciles by |len_q - len_mate|: small gaps hit, large gaps miss
    qlen = np.array([100, 101, 102, 110, 111, 112, 140, 141, 142], dtype=float)
    mate_len = np.array([100] * 9, dtype=float)
    S = np.full((9, 3), -1.0)
    for i in range(9):
        S[i, i // 3] = 1.0 if qlen[i] - 100 < 20 else -2.0
        if qlen[i] - 100 >= 20:
            S[i, (i // 3 + 1) % 3] = 1.0  # top tercile matches the WRONG unit
    out = ro.compute_stratified_rank1(S, q_units, g_units, known, qlen, mate_len)
    r = [b["rank1"] for b in out["terciles"]]
    assert r[0] == 1.0 and r[1] == 1.0 and r[2] == 0.0


def test_band_auroc_restricts_gallery_to_length_band():
    glen = np.array([100.0, 300.0])
    qlen = np.array([100.0, 100.0])
    known = np.array([True, False])
    # known's best OVERALL match is the far-length gallery entry; in-band
    # restriction removes it, leaving the mate's (higher-for-known) column.
    S = np.array([[0.6, 0.9], [0.2, 0.9]])
    out = ro.compute_band_auroc(S, known, qlen, glen, band=0.10)
    assert out["auroc"] == 1.0
    assert out["n_excluded_no_inband_gallery_or_no_length"] == 0


def test_band_auroc_counts_queries_with_no_inband_gallery():
    glen = np.array([300.0])
    qlen = np.array([100.0, 100.0])
    known = np.array([True, False])
    S = np.array([[0.5], [0.5]])
    out = ro.compute_band_auroc(S, known, qlen, glen, band=0.10)
    assert out["auroc"] is None
    assert out["n_excluded_no_inband_gallery_or_no_length"] == 2


def test_load_train_identities_prefers_full_record():
    torch = pytest.importorskip("torch")
    path_dir = pytest.importorskip("tempfile").mkdtemp()
    import os
    p = os.path.join(path_dir, "ckpt.pt")
    torch.save(
        {"classes": [["a", "L"], ["b", "L"]],
         "train_identities": ["a", "b", "c_dropped_singleton"]},
        p,
    )
    assert finetune.load_train_identities(p) == {"a", "b", "c_dropped_singleton"}
    torch.save({"classes": [["a", "L"], ["b", "L"]]}, p)
    assert finetune.load_train_identities(p) == {"a", "b"}
