"""Tests for run-3 tooling: dense-subset catalogue filter and the
early-stopping probe builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import finetune
import protocol
import run_ablation


def _frame(units_with_counts, start_day=1):
    rows = []
    i = 0
    for (identity, side), count in units_with_counts.items():
        for j in range(count):
            rows.append(
                {"image_id": "%s%s%d" % (identity, side, j), "identity": identity,
                 "side": side, "path": "p%d.jpg" % i,
                 "date": "2020-01-%02d" % (start_day + j)}
            )
            i += 1
    return pd.DataFrame(rows)


def test_dense_filter_keeps_only_dense_units():
    df = _frame({("a", "L"): 5, ("b", "L"): 4, ("c", "L"): 3, ("d", "R"): 1})
    kept, n_kept, n_before = run_ablation._filter_dense_units(df, 4)
    assert n_before == 4 and n_kept == 2
    assert set(zip(kept["identity"], kept["side"])) == {("a", "L"), ("b", "L")}
    assert len(kept) == 9


def test_dense_filter_is_identical_across_arms():
    # the filter depends only on identity/side counts, so two "arms" (same
    # catalogue rows, different crops) retain exactly the same image ids
    df = _frame({("a", "L"): 4, ("b", "R"): 2})
    kept1, _, _ = run_ablation._filter_dense_units(df.copy(), 3)
    kept2, _, _ = run_ablation._filter_dense_units(df.copy(), 3)
    assert kept1["image_id"].tolist() == kept2["image_id"].tolist()


def test_dense_filter_raises_on_empty():
    df = _frame({("a", "L"): 2})
    with pytest.raises(protocol.ProtocolViolation):
        run_ablation._filter_dense_units(df, 10)


def test_probe_builder_gallery_earliest_query_latest():
    df = _frame({("a", "L"): 3, ("b", "L"): 2, ("c", "L"): 1})
    gallery, query = finetune._build_probe(df, probe_units=10)
    assert len(gallery) == 2  # singleton unit c excluded
    for g, q in zip(gallery, query):
        assert (g["identity"], g["side"]) == (q["identity"], q["side"])
        assert g["date"] < q["date"]
        assert g["image_id"] != q["image_id"]


def test_probe_builder_caps_units_and_is_deterministic():
    df = _frame({("u%02d" % i, "L"): 2 for i in range(20)})
    g1, q1 = finetune._build_probe(df, probe_units=5)
    g2, q2 = finetune._build_probe(df, probe_units=5)
    assert len(g1) == 5
    assert [r["image_id"] for r in g1] == [r["image_id"] for r in g2]
    assert [r["image_id"] for r in q1] == [r["image_id"] for r in q2]


def test_probe_builder_raises_without_multi_image_units():
    df = _frame({("a", "L"): 1, ("b", "R"): 1})
    with pytest.raises(finetune.ProtocolViolation):
        finetune._build_probe(df, probe_units=5)
