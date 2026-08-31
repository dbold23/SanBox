"""Behavioral tests for diagnose.py (all synthetic / constructed, fast)."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import diagnose
import embedders
import melops_data
import protocol

PROTO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _row(image_id, identity, date, is_known=None):
    row = {"image_id": image_id, "identity": identity, "side": "L", "date": date}
    if is_known is not None:
        row["is_known"] = is_known
    return row


def _ang(deg):
    """2-d unit vector at ``deg`` degrees from the x axis."""
    rad = np.deg2rad(deg)
    return [np.cos(rad), np.sin(rad)]


# ---------------------------------------------------------------------------
# a. Recapture-gap buckets on a constructed frame with known dates
# ---------------------------------------------------------------------------


def test_gap_buckets_on_constructed_frame():
    base = pd.Timestamp("2018-01-01")

    def day(gap):
        return (base + pd.Timedelta(days=gap)).strftime("%Y-%m-%d")

    gallery_df = pd.DataFrame([
        _row("gA", "A", day(0)),
        _row("gB", "B", day(0)),
    ])
    gallery_emb = np.asarray([_ang(0), _ang(90)])  # A on x, B on y
    query_df = pd.DataFrame([
        _row("q10", "A", day(10), True),    # 0-30
        _row("q30", "A", day(30), True),    # 0-30 (upper edge inclusive)
        _row("q31", "A", day(31), True),    # 31-180 (lower edge), nearer B -> rank 2
        _row("q400", "A", day(400), True),  # 366-730
        _row("q800", "B", day(800), True),  # 731+
        _row("qnov", "C", day(900), False), # novel: must not enter any bucket
    ])
    query_emb = np.asarray([
        _ang(25),   # sim to A = cos 25 ~ 0.9063, rank 1
        _ang(25),
        _ang(70),   # sim to A = 0.3420 < sim to B = 0.9397 -> rank 2
        _ang(30),   # 0.8660, rank 1
        _ang(80),   # sim to B = cos 10 ~ 0.9848, rank 1
        _ang(45),
    ])

    details = diagnose.per_query_details(gallery_emb, gallery_df, query_emb, query_df)
    section = diagnose.recapture_gap_section(details, gallery_df, query_df)

    assert section["n_known"] == 5
    by_label = {b["bucket_days"]: b for b in section["buckets"]}
    assert set(by_label) == {"0-30", "31-180", "181-365", "366-730", "731+"}

    assert by_label["0-30"]["n"] == 2
    assert by_label["0-30"]["mean_true_mate_sim"] == pytest.approx(np.cos(np.deg2rad(25)), abs=1e-9)
    assert by_label["0-30"]["rank1"] == 1.0

    assert by_label["31-180"]["n"] == 1
    assert by_label["31-180"]["mean_true_mate_sim"] == pytest.approx(np.cos(np.deg2rad(70)), abs=1e-9)
    assert by_label["31-180"]["rank1"] == 0.0

    assert by_label["181-365"]["n"] == 0
    assert by_label["181-365"]["mean_true_mate_sim"] is None
    assert by_label["181-365"]["rank1"] is None

    assert by_label["366-730"]["n"] == 1
    assert by_label["366-730"]["mean_true_mate_sim"] == pytest.approx(np.cos(np.deg2rad(30)), abs=1e-9)
    assert by_label["366-730"]["rank1"] == 1.0

    assert by_label["731+"]["n"] == 1
    assert by_label["731+"]["mean_true_mate_sim"] == pytest.approx(np.cos(np.deg2rad(10)), abs=1e-9)
    assert by_label["731+"]["rank1"] == 1.0


def test_gap_negative_raises_protocol_violation():
    gallery_df = pd.DataFrame([_row("gA", "A", "2018-06-01")])
    query_df = pd.DataFrame([_row("q", "A", "2018-01-01", True)])
    gallery_emb = np.asarray([_ang(0)])
    query_emb = np.asarray([_ang(10)])
    details = diagnose.per_query_details(gallery_emb, gallery_df, query_emb, query_df)
    with pytest.raises(protocol.ProtocolViolation):
        diagnose.recapture_gap_section(details, gallery_df, query_df)


# ---------------------------------------------------------------------------
# b. Year strata + automated temporal-confound reading
# ---------------------------------------------------------------------------


def _strata_corpus():
    """Two enrolled units; year-1 (2016) known-only stratum, 2020 mixed.

    Embeddings are 3-d unit vectors: a known query sits at cosine 0.95 to its
    mate, a novel query at cosine 0.85 to its nearest impostor, remainder on z.
    """
    gallery_df = pd.DataFrame([
        _row("gA", "A", "2015-01-01"),
        _row("gB", "B", "2015-01-01"),
    ])
    eA, eB, ez = np.eye(3)
    gallery_emb = np.stack([eA, eB])

    def near(mate, sim):
        return sim * mate + np.sqrt(1.0 - sim * sim) * ez

    rows, embs = [], []
    for i in range(3):  # year-1 known queries, A and B
        rows.append(_row("k16a%d" % i, "A", "2016-0%d-01" % (i + 1), True))
        embs.append(near(eA, 0.95))
        rows.append(_row("k16b%d" % i, "B", "2016-0%d-15" % (i + 1), True))
        embs.append(near(eB, 0.95))
    rows.append(_row("k20a", "A", "2020-05-01", True)); embs.append(near(eA, 0.95))
    rows.append(_row("k20b", "B", "2020-06-01", True)); embs.append(near(eB, 0.95))
    for i in range(4):  # late-years-only novel queries
        rows.append(_row("n20_%d" % i, "N%d" % i, "2020-0%d-01" % (i + 3), False))
        embs.append(near(eA, 0.85))
    query_df = pd.DataFrame(rows)
    query_emb = np.stack(embs)
    return gallery_df, gallery_emb, query_df, query_emb


def _renorm(mat):
    return mat / np.linalg.norm(mat, axis=1, keepdims=True)


def test_strata_healthy_corpus_reads_no_inversion():
    g_df, g_emb, q_df, q_emb = _strata_corpus()
    details = diagnose.per_query_details(g_emb, g_df, q_emb, q_df)
    section = diagnose.year_strata_section(details, q_df)
    assert section["pooled_auroc"] == pytest.approx(1.0)
    assert section["reading"].startswith(diagnose.READING_NO_INVERSION)
    by_year = {s["year"]: s for s in section["strata"]}
    assert by_year[2016]["n_novel"] == 0 and by_year[2016]["auroc"] is None
    assert by_year[2020]["auroc"] == pytest.approx(1.0)


def test_strata_early_year_drift_flags_temporal_confound():
    """Simulated acquisition drift: a constant vector added to the year-1
    stratum's query embeddings sinks its known max-sims below the (late-years
    only) novel max-sims, inverting the pooled AUROC while every within-year
    AUROC stays sane -> the reading must call the confound temporal."""
    g_df, g_emb, q_df, q_emb = _strata_corpus()
    years = pd.to_datetime(q_df["date"]).dt.year.to_numpy()
    early = years == 2016
    assert early.sum() == 6
    drifted = q_emb.copy()
    drifted[early] += np.asarray([0.0, 0.0, 6.0])  # constant drift vector
    drifted = _renorm(drifted)
    details = diagnose.per_query_details(g_emb, g_df, drifted, q_df)
    section = diagnose.year_strata_section(details, q_df)
    # preconditions the reading relies on
    assert section["pooled_auroc"] < 0.5
    assert all(s["auroc"] >= 0.5 for s in section["strata"] if s["auroc"] is not None)
    assert section["reading"].startswith(diagnose.READING_CONFOUND)


def test_strata_within_year_inversion_reads_not_temporal():
    """Drift ALL known queries: the inversion persists inside each year, so
    the reading must NOT blame the temporal confound."""
    g_df, g_emb, q_df, q_emb = _strata_corpus()
    is_known = q_df["is_known"].to_numpy().astype(bool)
    drifted = q_emb.copy()
    drifted[is_known] += np.asarray([0.0, 0.0, 6.0])
    drifted = _renorm(drifted)
    details = diagnose.per_query_details(g_emb, g_df, drifted, q_df)
    section = diagnose.year_strata_section(details, q_df)
    assert section["pooled_auroc"] < 0.5
    assert section["reading"].startswith(diagnose.READING_NOT_TEMPORAL)


# ---------------------------------------------------------------------------
# c. Small-gallery calibration
# ---------------------------------------------------------------------------


def test_small_gallery_calibration_no_violation_and_easier(head_corpus):
    # headless arm on the head-concentrated corpus: a genuinely hard task,
    # so shrinking the gallery must not make it harder
    df = melops_data.load_melops(head_corpus, bbox="headless")
    gallery_df, query_df = protocol.one_shot_open_set_split(df, cutoff_fraction=0.5, seed=11)
    embedder = embedders.get_embedder("hist")
    g_emb = embedder.embed([melops_data.load_crop(head_corpus, r) for _, r in gallery_df.iterrows()])
    q_emb = embedder.embed([melops_data.load_crop(head_corpus, r) for _, r in query_df.iterrows()])
    full = protocol.evaluate(g_emb, gallery_df, q_emb, query_df)

    cal = diagnose.small_gallery_calibration(
        g_emb, gallery_df, q_emb, query_df, k=8, n_seeds=3, base_seed=0
    )  # ProtocolViolation would propagate: the subsampled frames re-run the checks
    assert cal["k_effective"] == 8
    assert len(cal["runs"]) == 3
    for run in cal["runs"]:
        assert run["n_gallery"] == 8
        assert run["n_novel"] > 0
    # fewer distractors => the subsampled task is easier than the full gallery
    assert cal["rank1"]["min"] >= full["rank1"]
    assert cal["rank1"]["mean"] >= full["rank1"]
    assert cal["rank1"]["min"] <= cal["rank1"]["mean"] <= cal["rank1"]["max"]


def test_small_gallery_calibration_clamps_k(distributed_corpus):
    df = melops_data.load_melops(distributed_corpus, bbox="body")
    gallery_df, query_df = protocol.one_shot_open_set_split(df, cutoff_fraction=0.5, seed=11)
    embedder = embedders.get_embedder("hist")
    g_emb = embedder.embed([melops_data.load_crop(distributed_corpus, r) for _, r in gallery_df.iterrows()])
    q_emb = embedder.embed([melops_data.load_crop(distributed_corpus, r) for _, r in query_df.iterrows()])
    cal = diagnose.small_gallery_calibration(
        g_emb, gallery_df, q_emb, query_df, k=500, n_seeds=3, base_seed=0
    )
    assert cal["k_requested"] == 500
    assert cal["k_effective"] < 500  # clamped to what the corpus offers
    assert cal["rank1"]["mean"] is not None


# ---------------------------------------------------------------------------
# d. Contact sheet
# ---------------------------------------------------------------------------


def test_contact_sheet_exists_with_expected_grid_size(distributed_corpus, tmp_path):
    out = str(tmp_path / "contact_sheet_head.png")
    path = diagnose.contact_sheet(distributed_corpus, "head", out, grid=4, seed=0)
    assert os.path.exists(path)
    with Image.open(path) as img:
        assert img.size == (4 * diagnose.SHEET_CELL_W, 4 * diagnose.SHEET_CELL_H)
    # deterministic under (seed, arm)
    out2 = str(tmp_path / "again.png")
    diagnose.contact_sheet(distributed_corpus, "head", out2, grid=4, seed=0)
    with open(path, "rb") as f1, open(out2, "rb") as f2:
        assert f1.read() == f2.read()


def test_contact_sheet_rejects_unknown_arm(distributed_corpus, tmp_path):
    with pytest.raises(ValueError):
        diagnose.contact_sheet(distributed_corpus, "cross_orientation",
                               str(tmp_path / "x.png"), grid=2, seed=0)


# ---------------------------------------------------------------------------
# CLI end-to-end
# ---------------------------------------------------------------------------


def test_diagnose_cli_end_to_end(tmp_path):
    root = str(tmp_path / "corpus")
    out = str(tmp_path / "diag")
    cache = str(tmp_path / "cache")
    cmd = [
        sys.executable, os.path.join(PROTO_DIR, "diagnose.py"),
        "--data", "synthetic", "--root", root, "--backbone", "hist",
        "--arm", "body", "--out", out, "--emb-cache", cache,
        "--seed", "3", "--n-individuals", "20", "--grid", "3",
        "--calibration-k", "6",
    ]
    proc = subprocess.run(cmd, cwd=PROTO_DIR, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr

    with open(os.path.join(out, "diagnostics.json")) as f:
        diag = json.load(f)
    assert diag["arm"] == "body" and diag["backbone"] == "hist"
    assert len(diag["recapture_gap"]["buckets"]) == 5
    assert diag["auroc_year_strata"]["reading"]
    assert diag["small_gallery_calibration"]["k_effective"] <= 6
    for arm in ("head", "body", "headless"):
        sheet = os.path.join(out, "contact_sheet_%s.png" % arm)
        assert os.path.exists(sheet)
        with Image.open(sheet) as img:
            assert img.size == (3 * diagnose.SHEET_CELL_W, 3 * diagnose.SHEET_CELL_H)

    with open(os.path.join(out, "diagnostics.md")) as f:
        md = f.read()
    assert "PATTERN-STABILITY" in md
    assert "Recapture-gap curve" in md
    assert "AUROC strata by query year" in md
    assert "Small-gallery calibration" in md
    assert "contact sheets" in md.lower()
    assert "READING:" in proc.stdout
    # the cache was populated (gallery + query for the body arm)
    assert len([f for f in os.listdir(cache) if f.endswith(".npz")]) == 2
