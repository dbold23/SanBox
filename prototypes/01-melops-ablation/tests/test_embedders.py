"""Metric sanity: random ~ chance floor, hist well above chance; guards."""

from __future__ import annotations

import numpy as np
import pytest

import embedders
import melops_data
import protocol


def _run(root, backbone, bbox="body", seed=0):
    df = melops_data.load_melops(root, bbox=bbox)
    gallery_df, query_df = protocol.one_shot_open_set_split(df, cutoff_fraction=0.5, seed=seed)
    emb = embedders.get_embedder(backbone, seed=seed)
    crops_g = [melops_data.load_crop(root, r) for _, r in gallery_df.iterrows()]
    crops_q = [melops_data.load_crop(root, r) for _, r in query_df.iterrows()]
    res = protocol.evaluate(emb.embed(crops_g), gallery_df, emb.embed(crops_q), query_df)
    # chance floor: known queries pick uniformly within their side's gallery
    side_sizes = gallery_df["side"].value_counts()
    known = query_df[query_df["is_known"]]
    chance = float(np.mean([1.0 / side_sizes[s] for s in known["side"]]))
    return res, chance


def test_random_embedder_near_chance(distributed_corpus):
    res, chance = _run(distributed_corpus, "random")
    assert res["n_known"] >= 10
    assert res["rank1"] <= max(3.0 * chance, 0.25)
    aur = res["open_set_auroc"]
    assert aur is not None and 0.25 <= aur <= 0.75


def test_hist_embedder_far_above_chance(distributed_corpus):
    res, chance = _run(distributed_corpus, "hist")
    assert res["rank1"] >= 0.5
    assert res["rank1"] >= 5.0 * chance
    assert res["mAP"] >= res["rank1"]  # AP = 1/rank >= rank1 indicator mean


def test_embeddings_are_l2_normalized(distributed_corpus):
    df = melops_data.load_melops(distributed_corpus)
    crops = [melops_data.load_crop(distributed_corpus, r) for _, r in df.head(4).iterrows()]
    for name in ("hist", "random"):
        vecs = embedders.get_embedder(name, seed=1).embed(crops)
        assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-9)


def test_random_embedder_deterministic(distributed_corpus):
    df = melops_data.load_melops(distributed_corpus)
    crops = [melops_data.load_crop(distributed_corpus, r) for _, r in df.head(3).iterrows()]
    a = embedders.RandomEmbedder(seed=5).embed(crops)
    b = embedders.RandomEmbedder(seed=5).embed(crops)
    c = embedders.RandomEmbedder(seed=6).embed(crops)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("name", ["megadescriptor", "dinov2", "miewid"])
def test_deep_embedders_guarded_with_pip_hint(name):
    try:
        import timm  # noqa: F401
        import torch  # noqa: F401
        pytest.skip("torch/timm installed; guard path not exercisable")
    except ImportError:
        pass
    with pytest.raises(RuntimeError) as excinfo:
        embedders.get_embedder(name)
    assert "pip install" in str(excinfo.value)


def test_unknown_backbone_rejected():
    with pytest.raises(ValueError):
        embedders.get_embedder("resnet-9000")
