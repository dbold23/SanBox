"""Regression tests for melops_data normalization against real-metadata quirks."""

from __future__ import annotations

import pandas as pd
import pytest

import melops_data


def _frame(side_values, orientation=None):
    n = len(side_values)
    df = pd.DataFrame(
        {
            "image_id": ["img%d" % i for i in range(n)],
            "identity": ["id%d" % i for i in range(n)],
            "path": ["p%d.jpg" % i for i in range(n)],
            "date": ["2020-01-0%d" % (i + 1) for i in range(n)],
            "side": side_values,
            "bbox_body": [(0, 0, 10, 10)] * n,
            "bbox_head": [(0, 0, 4, 10)] * n,
            "bbox_headless": [(4, 0, 6, 10)] * n,
        }
    )
    if orientation is not None:
        df["orientation"] = orientation
    return df


def test_real_metadata_left_right_side_values_normalize_to_L_R():
    # The Zenodo Melops metadata spells sides "left"/"right"; the catalogue
    # contract uses "L"/"R" (matching synthetic mode).
    df = melops_data._normalize(_frame(["left", "right"]), "body")
    assert sorted(df["side"].tolist()) == ["L", "R"]
    assert (df["orientation"] == df["side"]).all()
    melops_data._check_catalogue(df)  # must not raise


def test_orientation_column_normalized_alongside_side():
    df = melops_data._normalize(
        _frame(["left", "right"], orientation=["left", "right"]), "body"
    )
    assert sorted(df["orientation"].tolist()) == ["L", "R"]
    melops_data._check_catalogue(df)  # must not raise


def test_unknown_side_values_still_rejected():
    df = melops_data._normalize(_frame(["dorsal", "left"]), "body")
    with pytest.raises(ValueError, match="side must be 'L' or 'R'"):
        melops_data._check_catalogue(df)
