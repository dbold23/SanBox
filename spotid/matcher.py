"""Spot-level identification: nearest neighbor over enrolled descriptors."""

import numpy as np

from .features import describe_image


class SpotMatcher:
    """Gallery of enrolled spot identities.

    Enroll each identity from several sample views. The matcher estimates,
    per descriptor dimension, the within-identity variation (view noise)
    and the between-identity variation (signal), then whitens by the noise:
    dimensions where views of the same spot disagree are shrunk, dimensions
    that separate identities keep their weight (a diagonal Fisher/LDA
    transform). Matching is nearest neighbor by cosine distance in that
    space.
    """

    # Cap on the per-dimension signal-to-noise weight, so a dimension with
    # near-zero measured noise cannot dominate the distance.
    MAX_SNR = 25.0

    def __init__(self):
        self._ids: list = []
        self._samples: list[list[np.ndarray]] = []
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._gallery: np.ndarray | None = None

    def enroll(self, spot_id, descriptors: list[np.ndarray]) -> None:
        """Enroll an identity from one or more descriptor samples. Sample
        views should cover the pose range expected at query time — the
        matcher learns per-dimension view noise from them."""
        descs = [np.asarray(d, float) for d in descriptors if d is not None]
        if not descs:
            raise ValueError(f"no valid descriptors for identity {spot_id!r}")
        self._samples.append(descs)
        self._ids.append(spot_id)
        self._gallery = None  # invalidate cache

    def _prepare(self) -> None:
        means = np.vstack([np.mean(s, axis=0) for s in self._samples])
        self._mean = means.mean(axis=0)
        sig_b = np.maximum(means.std(axis=0), 1e-9)
        resid = [d - np.mean(s, axis=0) for s in self._samples for d in s]
        n_resid = len(resid)
        if n_resid > len(self._samples):  # at least some ids have >1 sample
            sig_w = np.sqrt(np.mean(np.square(np.vstack(resid)), axis=0))
        else:
            sig_w = np.zeros_like(sig_b)
        # Noise floor keeps zero-measured-noise dims bounded (MAX_SNR).
        sig_w = np.maximum(sig_w, sig_b / self.MAX_SNR)
        self._scale = 1.0 / sig_w
        g = (means - self._mean) * self._scale
        self._gallery = g / np.maximum(
            np.linalg.norm(g, axis=1, keepdims=True), 1e-9)

    def __len__(self) -> int:
        return len(self._ids)

    def identify_descriptor(self, desc: np.ndarray, top_k: int = 1):
        """Return [(spot_id, distance), ...] best-first."""
        if desc is None or not self._samples:
            return []
        if self._gallery is None:
            self._prepare()
        q = (np.asarray(desc, float) - self._mean) * self._scale
        q = q / max(np.linalg.norm(q), 1e-9)
        dists = 1.0 - self._gallery @ q
        order = np.argsort(dists)[:top_k]
        return [(self._ids[i], float(dists[i])) for i in order]

    def identify(self, img: np.ndarray, top_k: int = 1):
        """Segment the spot in ``img`` and identify it against the gallery."""
        return self.identify_descriptor(describe_image(img), top_k=top_k)
