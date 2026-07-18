import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spotid.ml.dataset import PATCH, extract_patch, make_batch, \
    render_training_view
from spotid.ml.model import SpotEncoder, supcon_loss


def test_extract_patch_shape_and_padding():
    img = (np.random.default_rng(0).uniform(0, 255, (256, 256))
           .astype(np.uint8))
    p = extract_patch(img, np.array([128.0, 128.0]), 40.0)
    assert p.shape == (PATCH, PATCH) and p.dtype == np.float32
    assert 0.0 <= p.min() and p.max() <= 1.0
    # Near-corner center forces border replication and must not crash.
    p2 = extract_patch(img, np.array([2.0, 250.0]), 60.0)
    assert p2.shape == (PATCH, PATCH)


def test_training_view_deterministic_identity():
    rng1 = np.random.default_rng(3)
    rng2 = np.random.default_rng(3)
    a = render_training_view(5, rng1)
    b = render_training_view(5, rng2)
    assert np.allclose(a, b)


def test_make_batch_labels():
    x, y = make_batch(np.random.default_rng(1), n_ids=4, k_views=3,
                      id_pool=50)
    assert x.shape == (12, 1, PATCH, PATCH)
    assert list(np.bincount(y)) == [3, 3, 3, 3]


def test_encoder_normalized_and_trainable():
    m = SpotEncoder(embed_dim=32, width=8)
    x = torch.randn(6, 1, PATCH, PATCH)
    emb = m(x)
    assert emb.shape == (6, 32)
    assert torch.allclose(emb.norm(dim=1), torch.ones(6), atol=1e-5)
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    loss = supcon_loss(emb, y)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_canonicalization_consistent_between_train_and_inference():
    """The canonical frame must not depend on contour point density: the
    training path uses the analytic polygon (uniform in generator angle),
    inference uses segmentation pixels (uniform in arc length)."""
    from spotid.features import segment_spot
    from spotid.ml.dataset import canonical_patch
    from spotid.render import ViewConfig, render_view
    from spotid.shapes import generate_identity

    rng = np.random.default_rng(11)
    cfg = ViewConfig(tilt_max_deg=30, noise_sigma_range=(0.0, 0.005),
                     blur_sigma_range=(0.0, 0.2))
    cors = []
    for seed in range(8):
        img, info = render_view(generate_identity(seed), rng, cfg)
        seg = segment_spot(img)
        assert seg is not None
        p_train = canonical_patch(img, info["polygon_px"])
        p_infer = canonical_patch(img, seg)
        a = p_train - p_train.mean()
        b = p_infer - p_infer.mean()
        cors.append(float((a * b).sum()
                          / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9)))
    # Residual difference is genuine segmentation noise (the contour
    # itself moves under blur/threshold), covered by training jitter.
    # The density bug this guards against produced ~0.6 here.
    assert np.mean(cors) > 0.85, f"train/infer frames diverge: {cors}"


def test_supcon_loss_prefers_clustered_embeddings():
    y = torch.tensor([0, 0, 1, 1])
    good = torch.nn.functional.normalize(torch.tensor(
        [[1.0, 0.0], [0.99, 0.1], [-1.0, 0.0], [-0.99, 0.1]]), dim=1)
    bad = torch.nn.functional.normalize(torch.tensor(
        [[1.0, 0.0], [-1.0, 0.1], [-0.9, 0.2], [0.95, 0.3]]), dim=1)
    assert supcon_loss(good, y) < supcon_loss(bad, y)
