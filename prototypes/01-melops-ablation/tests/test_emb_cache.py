"""Behavioral tests for the npz embedding cache and its run_ablation wiring."""

from __future__ import annotations

import glob
import os

import numpy as np
import pytest

import emb_cache
import embedders
import run_ablation


def _unit_rows(n, d, seed):
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((n, d))
    return mat / np.linalg.norm(mat, axis=1, keepdims=True)


IDS = ["img_b", "img_a", "img_c"]


def test_round_trip_preserves_order_and_normalization(tmp_path):
    cache = str(tmp_path)
    mat = _unit_rows(3, 16, 0)
    path = emb_cache.save(cache, "hist", "head", IDS, mat)
    assert os.path.exists(path)
    out = emb_cache.load(cache, "hist", "head", IDS)
    assert out is not None
    # float32 round trip + re-normalization: close, unit-norm, same order
    assert out.shape == mat.shape
    assert np.allclose(out, mat, atol=1e-6)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-9)


def test_miss_on_empty_cache(tmp_path):
    assert emb_cache.load(str(tmp_path), "hist", "head", IDS) is None


def test_different_arm_never_collides(tmp_path):
    cache = str(tmp_path)
    paths = set()
    for arm in ("head", "body", "headless"):
        paths.add(emb_cache.save(cache, "hist", arm, IDS, _unit_rows(3, 8, 1)))
    assert len(paths) == 3  # three distinct files
    # saving only head must not satisfy body/headless requests
    fresh = str(tmp_path / "fresh")
    emb_cache.save(fresh, "hist", "head", IDS, _unit_rows(3, 8, 2))
    assert emb_cache.load(fresh, "hist", "body", IDS) is None
    assert emb_cache.load(fresh, "hist", "headless", IDS) is None
    assert emb_cache.load(fresh, "hist", "head", IDS) is not None


def test_different_backbone_misses(tmp_path):
    cache = str(tmp_path)
    emb_cache.save(cache, "hist", "head", IDS, _unit_rows(3, 8, 3))
    assert emb_cache.load(cache, "megadescriptor", "head", IDS) is None


def test_reordered_ids_miss(tmp_path):
    cache = str(tmp_path)
    emb_cache.save(cache, "hist", "head", IDS, _unit_rows(3, 8, 4))
    reordered = sorted(IDS)
    assert reordered != IDS
    # ordered filename hash: the reordering maps to a DIFFERENT file, so this
    # is a clean miss without thrashing the original entry
    assert not os.path.exists(emb_cache.cache_path(cache, "hist", "head", reordered))
    assert emb_cache.load(cache, "hist", "head", reordered) is None
    # and even a forged file at the reordered path with the original payload
    # is rejected by the ordered-id check inside load
    import shutil
    shutil.copy(emb_cache.cache_path(cache, "hist", "head", IDS),
                emb_cache.cache_path(cache, "hist", "head", reordered))
    assert emb_cache.load(cache, "hist", "head", reordered) is None


def test_subset_and_superset_ids_miss(tmp_path):
    cache = str(tmp_path)
    emb_cache.save(cache, "hist", "head", IDS, _unit_rows(3, 8, 5))
    assert emb_cache.load(cache, "hist", "head", IDS[:2]) is None
    assert emb_cache.load(cache, "hist", "head", IDS + ["img_d"]) is None


def test_corrupt_file_is_a_miss(tmp_path):
    cache = str(tmp_path)
    path = emb_cache.cache_path(cache, "hist", "head", IDS)
    os.makedirs(cache, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"not an npz")
    assert emb_cache.load(cache, "hist", "head", IDS) is None


def test_run_experiment_cache_hit_skips_embedding(distributed_corpus, tmp_path, monkeypatch):
    cache = str(tmp_path / "cache")
    res1 = run_ablation.run_experiment(
        distributed_corpus, backbone="hist", arms=("head", "body"), seed=0,
        emb_cache_dir=cache,
    )
    files = glob.glob(os.path.join(cache, "*.npz"))
    assert len(files) == 4  # (gallery + query) x (head, body)
    assert any("__head__" in os.path.basename(f) for f in files)
    assert any("__body__" in os.path.basename(f) for f in files)

    def _boom(self, items):
        raise AssertionError("embed() called despite a warm cache")

    monkeypatch.setattr(embedders.HistEmbedder, "embed", _boom)
    res2 = run_ablation.run_experiment(
        distributed_corpus, backbone="hist", arms=("head", "body"), seed=0,
        emb_cache_dir=cache,
    )
    for arm in ("head", "body"):
        assert res2["arms"][arm]["rank1"] == pytest.approx(res1["arms"][arm]["rank1"], abs=1e-9)
        assert res2["arms"][arm]["open_set_auroc"] == pytest.approx(
            res1["arms"][arm]["open_set_auroc"], abs=1e-4
        )


def test_run_experiment_without_cache_unchanged(distributed_corpus, tmp_path):
    res_plain = run_ablation.run_experiment(
        distributed_corpus, backbone="hist", arms=("head",), seed=0
    )
    res_cached = run_ablation.run_experiment(
        distributed_corpus, backbone="hist", arms=("head",), seed=0,
        emb_cache_dir=str(tmp_path / "c"),
    )
    assert res_plain["arms"]["head"]["rank1"] == res_cached["arms"]["head"]["rank1"]


def test_uncached_equals_cached_bitwise(tmp_path, distributed_corpus):
    """Uncached, cold-cached and warm-cached must return bit-identical matrices."""
    import numpy as np
    import emb_cache
    import melops_data
    from embedders import get_embedder

    df = melops_data.load_melops(distributed_corpus, bbox="body").head(6)
    emb_none = emb_cache.embed_frame(get_embedder("hist"), distributed_corpus, df, "hist", "body",
                                     cache_dir=None)
    cache = str(tmp_path / "c")
    emb_cold = emb_cache.embed_frame(get_embedder("hist"), distributed_corpus, df, "hist", "body",
                                     cache_dir=cache)
    emb_warm = emb_cache.embed_frame(get_embedder("hist"), distributed_corpus, df, "hist", "body",
                                     cache_dir=cache)
    assert np.array_equal(emb_none, emb_cold)
    assert np.array_equal(emb_cold, emb_warm)


def test_reordered_ids_get_their_own_file(tmp_path):
    """Ordered filename hash: two orders coexist instead of thrashing one file."""
    import numpy as np
    import emb_cache

    ids = ["a", "b", "c"]
    emb = np.eye(3)
    cache = str(tmp_path)
    emb_cache.save(cache, "hist", "body", ids, emb)
    emb_cache.save(cache, "hist", "body", list(reversed(ids)), emb)
    assert emb_cache.load(cache, "hist", "body", ids) is not None
    assert emb_cache.load(cache, "hist", "body", list(reversed(ids))) is not None
