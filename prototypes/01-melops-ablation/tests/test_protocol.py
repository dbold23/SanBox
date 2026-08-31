"""Protocol invariants: leakage, singletons, side partition, determinism."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import protocol
from protocol import ProtocolViolation


def frame(rows):
    return pd.DataFrame(rows, columns=["image_id", "identity", "date", "side", "path"])


def row(image_id, identity, date, side):
    return (image_id, identity, date, side, image_id + ".png")


BASIC = frame(
    [
        row("a1", "A", "2018-01-01", "L"),
        row("a2", "A", "2019-06-01", "L"),
        row("a3", "A", "2020-06-01", "R"),
        row("b1", "B", "2018-02-01", "L"),
        row("c1", "C", "2020-01-01", "L"),  # first sighting after cutoff -> novel
        row("c2", "C", "2020-02-01", "L"),
        row("d1", "D", "2018-03-01", "R"),  # singleton before cutoff
        row("e1", "E", "2020-05-01", "R"),  # singleton after cutoff -> novel
    ]
)


def test_no_image_in_both_roles():
    gallery, queries = protocol.one_shot_open_set_split(BASIC, cutoff_date="2019-01-01")
    assert not set(gallery["image_id"]) & set(queries["image_id"])
    assert set(gallery["image_id"]) | set(queries["image_id"]) == set(BASIC["image_id"])


def test_gallery_is_one_shot_earliest():
    gallery, _ = protocol.one_shot_open_set_split(BASIC, cutoff_date="2019-01-01")
    units = list(zip(gallery["identity"], gallery["side"]))
    assert len(units) == len(set(units))
    # A's L gallery image is its earliest L sighting
    a_l = gallery[(gallery["identity"] == "A") & (gallery["side"] == "L")]
    assert a_l["image_id"].tolist() == ["a1"]


def test_known_novel_partition():
    gallery, queries = protocol.one_shot_open_set_split(BASIC, cutoff_date="2019-01-01")
    enrolled = set(zip(gallery["identity"], gallery["side"]))
    for _, q in queries.iterrows():
        assert ((q["identity"], q["side"]) in enrolled) == bool(q["is_known"])
    # C and E first appear after the cutoff: all their images are novel queries
    novel_ids = set(queries.loc[~queries["is_known"], "identity"])
    assert {"C", "E"} <= novel_ids


def test_singletons_survive():
    gallery, queries = protocol.one_shot_open_set_split(BASIC, cutoff_date="2019-01-01")
    # D: singleton before cutoff -> enrolled, contributes zero known queries
    assert "d1" in set(gallery["image_id"])
    assert "D" not in set(queries.loc[queries["is_known"], "identity"])
    # E: singleton after cutoff -> a novel query, never enrolled
    assert "e1" in set(queries.loc[~queries["is_known"], "image_id"])
    assert "E" not in set(gallery["identity"])


def test_same_individual_other_side_is_separate_unit():
    # A's first R sighting (2020) is after the cutoff: the R unit is novel
    # even though A's L unit is enrolled -- flanks are separate identities.
    gallery, queries = protocol.one_shot_open_set_split(BASIC, cutoff_date="2019-01-01")
    assert ("A", "R") not in set(zip(gallery["identity"], gallery["side"]))
    a3 = queries[queries["image_id"] == "a3"]
    assert not bool(a3["is_known"].iloc[0])


def test_duplicate_image_id_raises():
    bad = frame([row("x1", "A", "2018-01-01", "L"), row("x1", "B", "2018-02-01", "L")])
    with pytest.raises(ProtocolViolation):
        protocol.one_shot_open_set_split(bad, cutoff_date="2019-01-01")


def test_duplicate_path_raises():
    bad = pd.DataFrame(
        [
            ("x1", "A", "2018-01-01", "L", "same.png"),
            ("x2", "B", "2018-02-01", "L", "same.png"),
        ],
        columns=["image_id", "identity", "date", "side", "path"],
    )
    with pytest.raises(ProtocolViolation):
        protocol.one_shot_open_set_split(bad, cutoff_date="2019-01-01")


def test_deterministic_under_seed_with_date_ties():
    tied = frame(
        [
            row("t1", "A", "2018-01-01", "L"),
            row("t2", "A", "2018-01-01", "L"),  # exact tie for earliest
            row("t3", "A", "2019-06-01", "L"),
            row("u1", "B", "2018-01-01", "L"),
        ]
    )
    first = protocol.one_shot_open_set_split(tied, cutoff_date="2019-01-01", seed=3)
    second = protocol.one_shot_open_set_split(tied, cutoff_date="2019-01-01", seed=3)
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])


def _unit(seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(8)
    return v / np.linalg.norm(v)


def test_side_partition_no_cross_side_comparison():
    # Known L query whose embedding is IDENTICAL to a same-identity R gallery
    # decoy and orthogonal-ish to its own L gallery entry. If evaluation ever
    # compared across sides the decoy would win; the partition makes the true
    # L entry the only candidate, so rank1 == 1.0.
    e_true, e_decoy = _unit(1), _unit(2)
    gallery = frame([row("g1", "A", "2018-01-01", "L"), row("g2", "A", "2018-01-01", "R")])
    queries = frame([row("q1", "A", "2019-01-01", "L")])
    queries["is_known"] = [True]
    res = protocol.evaluate(
        np.stack([e_true, e_decoy]), gallery, np.stack([e_decoy]), queries
    )
    assert res["rank1"] == 1.0


def test_adversarial_leak_flags_raise():
    gallery = frame([row("g1", "A", "2018-01-01", "L")])
    emb = np.stack([_unit(1)])
    # novel flag on an enrolled unit -> leakage
    q_leak = frame([row("q1", "A", "2019-01-01", "L")])
    q_leak["is_known"] = [False]
    with pytest.raises(ProtocolViolation):
        protocol.evaluate(emb, gallery, emb, q_leak)
    # known flag on an un-enrolled unit -> no gallery match
    q_orphan = frame([row("q2", "Z", "2019-01-01", "L")])
    q_orphan["is_known"] = [True]
    with pytest.raises(ProtocolViolation):
        protocol.evaluate(emb, gallery, emb, q_orphan)


def test_image_reuse_between_roles_raises():
    gallery = frame([row("g1", "A", "2018-01-01", "L")])
    queries = frame([row("g1", "A", "2019-01-01", "L")])
    queries["is_known"] = [True]
    emb = np.stack([_unit(1)])
    with pytest.raises(ProtocolViolation):
        protocol.evaluate(emb, gallery, emb, queries)


def test_unnormalized_embeddings_rejected():
    gallery = frame([row("g1", "A", "2018-01-01", "L")])
    queries = frame([row("q1", "A", "2019-01-01", "L")])
    queries["is_known"] = [True]
    with pytest.raises(ProtocolViolation):
        protocol.evaluate(np.ones((1, 8)), gallery, np.stack([_unit(1)]), queries)


def test_cross_orientation_split_is_the_only_cross_side_arm():
    gallery, queries = protocol.cross_orientation_split(
        BASIC, enroll_side="L", query_side="R", cutoff_date="2019-01-01"
    )
    assert set(gallery["side"]) == {"L"}
    assert set(queries["side"]) == {"R"}
    # A enrolled on L, so its R images are known queries here (by design)
    assert bool(queries.loc[queries["image_id"] == "a3", "is_known"].iloc[0])
    emb_g = np.stack([_unit(i) for i in range(len(gallery))])
    emb_q = np.stack([_unit(100 + i) for i in range(len(queries))])
    res = protocol.evaluate(emb_g, gallery, emb_q, queries, cross_side=True)
    assert res["n_gallery"] == len(gallery)
    # and same-side evaluation refuses a mixed/cross frame pair
    with pytest.raises(ProtocolViolation):
        protocol.evaluate(emb_g, gallery, emb_q, queries, cross_side=False)


def test_cross_side_eval_requires_disjoint_single_sides():
    gallery = frame([row("g1", "A", "2018-01-01", "L"), row("g2", "B", "2018-01-01", "R")])
    queries = frame([row("q1", "A", "2019-01-01", "R")])
    queries["is_known"] = [True]
    emb_g = np.stack([_unit(1), _unit(2)])
    emb_q = np.stack([_unit(3)])
    with pytest.raises(ProtocolViolation):
        protocol.evaluate(emb_g, gallery, emb_q, queries, cross_side=True)


def test_cutoff_arguments_exclusive():
    with pytest.raises(ValueError):
        protocol.one_shot_open_set_split(BASIC)
    with pytest.raises(ValueError):
        protocol.one_shot_open_set_split(BASIC, cutoff_fraction=0.5, cutoff_date="2019-01-01")


def test_same_date_known_queries_excluded_by_default():
    # a2 shares the gallery date of a1: a same-handling-session near-duplicate.
    df = frame(
        [
            row("a1", "A", "2018-01-01", "L"),
            row("a2", "A", "2018-01-01", "L"),
            row("a3", "A", "2019-06-01", "L"),
        ]
    )
    gallery, queries = protocol.one_shot_open_set_split(df, cutoff_date="2019-01-01", seed=0)
    assert len(gallery) == 1
    assert set(queries["image_id"]) == {"a3"}
    assert queries.attrs["n_same_date_excluded"] == 1

    gallery_inc, queries_inc = protocol.one_shot_open_set_split(
        df, cutoff_date="2019-01-01", seed=0, same_date_policy="include"
    )
    assert len(queries_inc) == 2
    assert queries_inc.attrs["n_same_date_excluded"] == 0

    with pytest.raises(ProtocolViolation):
        protocol.one_shot_open_set_split(df, cutoff_date="2019-01-01", same_date_policy="drop")


def test_nat_dates_raise():
    df = frame([row("a1", "A", "2018-01-01", "L"), row("a2", "A", None, "L")])
    with pytest.raises(ProtocolViolation):
        protocol.one_shot_open_set_split(df, cutoff_date="2019-01-01")
