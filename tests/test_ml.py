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


def test_supcon_loss_prefers_clustered_embeddings():
    y = torch.tensor([0, 0, 1, 1])
    good = torch.nn.functional.normalize(torch.tensor(
        [[1.0, 0.0], [0.99, 0.1], [-1.0, 0.0], [-0.99, 0.1]]), dim=1)
    bad = torch.nn.functional.normalize(torch.tensor(
        [[1.0, 0.0], [-1.0, 0.1], [-0.9, 0.2], [0.95, 0.3]]), dim=1)
    assert supcon_loss(good, y) < supcon_loss(bad, y)
