import numpy as np
import pytest

from spotid.features import describe_contour, describe_image, segment_spot
from spotid.matcher import SpotMatcher
from spotid.render import ViewConfig, render_view
from spotid.shapes import generate_identity
from spotid.surface import generate_surface, render_surface_view
from spotid.surface_matcher import SurfaceMatcher, _local_signatures


def test_identity_deterministic():
    a = generate_identity(123)
    b = generate_identity(123)
    assert np.allclose(a, b)
    assert not np.allclose(a, generate_identity(124))


def test_identity_normalized():
    pts = generate_identity(5)
    assert np.allclose(pts.mean(axis=0), 0.0, atol=1e-9)
    assert np.isclose(np.sqrt((pts ** 2).sum(axis=1).mean()), 1.0)


def test_render_and_segment():
    rng = np.random.default_rng(0)
    img, info = render_view(generate_identity(1), rng)
    assert img.shape == (256, 256) and img.dtype == np.uint8
    contour = segment_spot(img)
    assert contour is not None and len(contour) > 30
    # Segmented centroid should sit near the ground-truth polygon centroid.
    gt = info["polygon_px"].mean(axis=0)
    got = contour.mean(axis=0)
    assert np.linalg.norm(gt - got) < 12.0


def test_descriptor_affine_invariance():
    """Descriptors of affinely transformed copies of a contour agree."""
    base = generate_identity(9) * 60.0 + 128.0
    d0 = describe_contour(base)
    rng = np.random.default_rng(2)
    for _ in range(5):
        a = rng.uniform(-1.0, 1.0, (2, 2)) * 0.4 + np.eye(2)
        if abs(np.linalg.det(a)) < 0.3:
            continue
        warped = base @ a.T + rng.uniform(-20, 20, 2)
        d1 = describe_contour(warped)
        cos = d0 @ d1 / (np.linalg.norm(d0) * np.linalg.norm(d1))
        assert cos > 0.995, f"affine invariance broken: cos={cos}"


def test_descriptor_separates_identities():
    rng = np.random.default_rng(3)
    same, diff = [], []
    d_ref = None
    for view in range(4):
        img, _ = render_view(generate_identity(50), rng)
        d = describe_image(img)
        assert d is not None
        d = d / np.linalg.norm(d)
        if d_ref is None:
            d_ref = d
        else:
            same.append(1 - d_ref @ d)
    for s in range(51, 71):
        img, _ = render_view(generate_identity(s), rng)
        d = describe_image(img)
        d = d / np.linalg.norm(d)
        diff.append(1 - d_ref @ d)
    # Raw descriptors (no Fisher weighting) separate on average; perfect
    # separation is the matcher's job (see test_spot_matcher_end_to_end).
    assert np.mean(same) * 3.0 < np.mean(diff), (same, diff)


def test_spot_matcher_end_to_end():
    from spotid.evaluate import enroll_identity
    matcher = SpotMatcher()
    for s in range(20):
        enroll_identity(matcher, s)
    rng = np.random.default_rng(4)
    correct = total = 0
    for s in range(20):
        ident = generate_identity(s)
        for _ in range(5):
            img, _ = render_view(ident, rng, ViewConfig(tilt_max_deg=45))
            res = matcher.identify(img)
            correct += bool(res) and res[0][0] == s
            total += 1
    assert correct / total >= 0.95, f"accuracy {correct}/{total}"


def test_surface_deterministic_and_spaced():
    s1 = generate_surface(7, n_spots=200)
    s2 = generate_surface(7, n_spots=200)
    assert np.allclose(s1.positions, s2.positions)
    assert s1.spots[3].shape_seed == s2.spots[3].shape_seed
    # No two spots closer than the sum of their radii.
    from scipy.spatial import cKDTree
    d, j = cKDTree(s1.positions).query(s1.positions, k=2)
    radii = np.array([sp.radius for sp in s1.spots])
    assert np.all(d[:, 1] > radii + radii[j[:, 1]])


def test_local_signatures_similarity_invariant():
    """Rotation + uniform scale preserve k-NN sets, so signatures are
    exactly invariant."""
    rng = np.random.default_rng(5)
    pts = rng.uniform(-1, 1, (60, 2))
    c, s = np.cos(0.7), np.sin(0.7)
    a = 2.5 * np.array([[c, -s], [s, c]])
    sig0, own0 = _local_signatures(pts)
    sig1, own1 = _local_signatures(pts @ a.T + [3.0, -2.0])
    assert np.array_equal(own0, own1)
    assert np.allclose(sig0, sig1, atol=1e-9)


def test_local_signatures_mild_affine_mostly_stable():
    """Anisotropy can change some k-NN sets, but under a mild affine (like
    a moderate viewing tilt) most signatures must survive — the matcher's
    voting + RANSAC tolerates the rest."""
    rng = np.random.default_rng(5)
    pts = rng.uniform(-1, 1, (200, 2))
    tilt = np.deg2rad(35.0)
    a = np.array([[1.0, 0.0], [0.0, np.cos(tilt)]])
    sig0, _ = _local_signatures(pts)
    sig1, _ = _local_signatures(pts @ a.T)
    err = np.linalg.norm(sig0 - sig1, axis=1) / np.linalg.norm(sig0, axis=1)
    # What matters for voting is that a point keeps at least one of its
    # leave-one-out rows intact, not that every row survives.
    point_ok = (err < 1e-6).reshape(len(pts), -1).any(axis=1)
    assert point_ok.mean() > 0.7, f"only {point_ok.mean():.2f} stable"


@pytest.mark.slow
def test_surface_matcher_full_and_partial():
    surfaces = [generate_surface(i, n_spots=300) for i in range(3)]
    matcher = SurfaceMatcher()
    for s in surfaces:
        matcher.enroll_surface(s)
    rng = np.random.default_rng(6)
    img, info = render_surface_view(surfaces[2], rng)
    res = matcher.identify(img)
    assert res and res[0].surface_id == 2
    # Defaults include mild dropout/clutter/fade, so demand a solid
    # majority rather than near-completeness.
    assert res[0].n_matched > 0.7 * res[0].n_query_spots
    # Partial view: crop to ~30% of the area.
    h, w = img.shape
    crop = img[h // 4: h // 4 + int(h * 0.55),
               w // 4: w // 4 + int(w * 0.55)]
    res_p = matcher.identify(crop, mode="auto")
    assert res_p and res_p[0].surface_id == 2
    assert res_p[0].n_matched >= 8
