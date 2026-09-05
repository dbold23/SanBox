"""
End-to-end check of the angular-spectrum propagator and autofocus.

We synthesize an object plane of opaque disks, forward-propagate it by a
known distance to fake a sensor hologram, then confirm that
reconstruct_hologram_stack + find_best_focus recover that distance.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import Dataprep  # noqa: E402
from config import CONFIG  # noqa: E402


def make_object_plane(size=256, n_disks=6, radius=6, seed=0):
    """Unit-amplitude plane wave with a few opaque disks punched out."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    amplitude = np.ones((size, size), dtype=np.float64)
    for _ in range(n_disks):
        cy, cx = rng.integers(radius * 4, size - radius * 4, size=2)
        amplitude[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = 0.0
    return amplitude


def synthesize_hologram(object_plane, distance):
    """Intensity a sensor would record `distance` microns downstream."""
    h, w = object_plane.shape
    transfer = Dataprep.angular_spectrum_transfer(h, w, distance)
    field = np.fft.ifft2(np.fft.fft2(object_plane) * transfer)
    return np.abs(field) ** 2


@pytest.fixture
def scan_config(monkeypatch):
    monkeypatch.setitem(CONFIG, "WAVELENGTH", 0.650)
    monkeypatch.setitem(CONFIG, "PIXEL_SIZE", 1.55)
    monkeypatch.setitem(CONFIG, "DEPTH_MIN", -1200.0)
    monkeypatch.setitem(CONFIG, "DEPTH_MAX", -200.0)
    monkeypatch.setitem(CONFIG, "DEPTH_STEP", 50.0)
    monkeypatch.setitem(CONFIG, "NORMALIZE_HOLOGRAM", True)


def test_transfer_function_has_no_nan_or_evanescent_energy():
    transfer = Dataprep.angular_spectrum_transfer(128, 128, 5000.0,
                                                  wavelength=0.65,
                                                  pixel_size=0.3)
    # pixel_size < wavelength/2 guarantees evanescent frequencies exist
    assert np.isfinite(transfer).all()
    assert np.count_nonzero(transfer == 0) > 0
    magnitudes = np.abs(transfer[transfer != 0])
    assert np.allclose(magnitudes, 1.0)


def test_reconstruction_is_finite_and_round_trips(scan_config):
    obj = make_object_plane()
    distance = 700.0
    hologram = synthesize_hologram(obj, distance)
    assert np.isfinite(hologram).all()

    field = Dataprep.reconstruct_hologram(hologram, -distance, return_complex=True)
    assert np.isfinite(field).all()
    # Back-propagating the *amplitude* would be exact; we back-propagate the
    # intensity, so require the disks to reappear as dark spots rather than
    # exact equality.
    recon = np.abs(field) ** 2
    disk_mask = obj == 0
    assert recon[disk_mask].mean() < 0.5 * recon[~disk_mask].mean()


@pytest.mark.parametrize("distance", [400.0, 700.0, 1000.0])
@pytest.mark.parametrize("radius", [6, 15, 30])
def test_autofocus_recovers_propagation_distance(scan_config, distance, radius):
    obj = make_object_plane(radius=radius, seed=int(distance) + radius)
    hologram = synthesize_hologram(obj, distance)

    stack, depths = Dataprep.reconstruct_hologram_stack(hologram, verbose=False)
    assert len(stack) == len(depths) == 21
    assert all(np.isfinite(img).all() for img in stack)

    best_idx, scores = Dataprep.find_best_focus(stack)
    assert len(scores) == len(stack)
    assert depths[best_idx] == pytest.approx(-distance, abs=CONFIG["DEPTH_STEP"])


def test_autofocus_survives_8bit_quantization_and_noise(scan_config):
    rng = np.random.default_rng(42)
    obj = make_object_plane(size=512, radius=15, seed=3)
    distance = 700.0
    hologram = synthesize_hologram(obj, distance)
    hologram = hologram + rng.normal(0, 0.02, hologram.shape)
    hologram8 = np.clip(hologram * 120, 0, 255).astype(np.uint8)

    stack, depths = Dataprep.reconstruct_hologram_stack(hologram8, verbose=False)
    best_idx, _ = Dataprep.find_best_focus(stack)
    assert depths[best_idx] == pytest.approx(-distance, abs=CONFIG["DEPTH_STEP"])


def test_focus_score_methods_and_direction(scan_config):
    obj = make_object_plane()
    distance = 700.0
    hologram = synthesize_hologram(obj, distance)
    in_focus = Dataprep.reconstruct_hologram(hologram, -distance)

    # Legacy per-image metrics remain callable and reject unknown names.
    assert Dataprep.focus_score(in_focus, "laplacian") > 0
    assert 0 < Dataprep.focus_score(in_focus, "sparsity") < 1
    with pytest.raises(ValueError):
        Dataprep.focus_score(in_focus, "nope")

    # find_best_focus dispatches to the legacy metrics too.
    stack, _ = Dataprep.reconstruct_hologram_stack(hologram, verbose=False)
    idx, scores = Dataprep.find_best_focus(stack, method="laplacian")
    assert 0 <= idx < len(stack) and len(scores) == len(stack)


def test_depth_scan_supports_float_steps(monkeypatch):
    monkeypatch.setitem(CONFIG, "DEPTH_MIN", -10.5)
    monkeypatch.setitem(CONFIG, "DEPTH_MAX", -8.0)
    monkeypatch.setitem(CONFIG, "DEPTH_STEP", 0.5)
    depths = Dataprep.depth_scan_values()
    assert depths.tolist() == [-10.5, -10.0, -9.5, -9.0, -8.5, -8.0]
