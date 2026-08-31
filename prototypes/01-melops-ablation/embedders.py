"""Embedders for the ablation.

Contract (the ``Embedder`` protocol):

* ``fit(paths_or_images)`` -- optional, may be a no-op; must be called before
  ``embed`` for embedders that need it (none of the built-ins do).
* ``embed(items)`` -- ``items`` is a list of PIL images or file paths; returns
  an ``(n, d)`` float64 array whose rows are L2-normalized.

Built-ins runnable with numpy+PIL only:

* ``hist`` -- blockwise color + gradient-orientation histograms. Strong enough
  on the synthetic corpus that Rank-1 is far above chance.
* ``random`` -- seeded chance-floor control. Deterministic per item content,
  independent of it in distribution.

Deep baselines (``megadescriptor``, ``dinov2``, ``miewid``) require torch and
timm and are import-guarded: constructing them without those deps raises a
RuntimeError naming the exact pip line. They are never imported at module
level, so this module works with zero optional dependencies installed.
"""

from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image

_HIST_SIZE = (64, 64)  # (width, height) after resize; square so all arms share geometry
_GRID = 4
_POOL = 4  # pooled intensity map cell size (64/4 -> 16x16 map)
_COLOR_BINS = 4
_ORIENT_BINS = 8


def _as_image(item):
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    return Image.open(item).convert("RGB")


def _l2_normalize(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class HistEmbedder:
    """Blockwise photometric-invariant embedding (numpy + PIL only).

    Three concatenated feature families over a fixed-size resized crop:
    a mean-pooled, per-crop standardized intensity map (spot-constellation
    layout, brightness/contrast invariant), per-cell color histograms on the
    brightness-normalized image, and per-cell magnitude-weighted gradient
    orientation histograms. The whole vector is L2-normalized.
    """

    def __init__(self):
        pass

    def fit(self, items):
        pass

    def embed(self, items):
        rows = [self._embed_one(_as_image(item)) for item in items]
        return _l2_normalize(np.stack(rows).astype(np.float64))

    def _embed_one(self, img):
        img = img.resize(_HIST_SIZE, Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float64) / 255.0
        arr = arr / max(arr.mean(), 1e-6)  # brightness normalization
        gray = arr.mean(axis=2)
        std = gray.std()
        norm_gray = (gray - gray.mean()) / (std if std > 1e-6 else 1.0)
        gy, gx = np.gradient(norm_gray)
        mag = np.hypot(gx, gy)
        orient = np.arctan2(gy, gx)  # [-pi, pi]

        h, w = gray.shape
        feats = []
        # pooled standardized intensity map: h//_POOL x w//_POOL block means
        pooled = norm_gray.reshape(h // _POOL, _POOL, w // _POOL, _POOL).mean(axis=(1, 3))
        feats.append(pooled.ravel())

        ch, cw = h // _GRID, w // _GRID
        for gy_i in range(_GRID):
            for gx_i in range(_GRID):
                sl = (slice(gy_i * ch, (gy_i + 1) * ch), slice(gx_i * cw, (gx_i + 1) * cw))
                cell = arr[sl[0], sl[1], :]
                for c in range(3):
                    hist, _ = np.histogram(cell[:, :, c], bins=_COLOR_BINS, range=(0.0, 2.0))
                    feats.append(hist.astype(np.float64) / cell[:, :, c].size)
                ohist, _ = np.histogram(
                    orient[sl], bins=_ORIENT_BINS, range=(-np.pi, np.pi), weights=mag[sl]
                )
                total = ohist.sum()
                feats.append(ohist / total if total > 0 else ohist)
        return np.concatenate(feats)


class RandomEmbedder:
    """Chance-floor control: content-independent, deterministic under seed.

    Each item's vector is derived from a hash of (seed, item content), so the
    same corpus + seed always gives the same embeddings, but the vectors carry
    no identity information.
    """

    def __init__(self, seed=0, dim=64):
        self.seed = int(seed)
        self.dim = int(dim)

    def fit(self, items):
        pass

    def embed(self, items):
        rows = []
        for item in items:
            if isinstance(item, Image.Image):
                key = hashlib.sha256(item.tobytes()).digest()
            else:
                key = hashlib.sha256(str(item).encode("utf-8")).digest()
            child = np.random.default_rng(
                [self.seed] + [int.from_bytes(key[k : k + 4], "little") for k in range(0, 16, 4)]
            )
            rows.append(child.standard_normal(self.dim))
        return _l2_normalize(np.stack(rows).astype(np.float64))


_TORCH_HINT = (
    "requires torch + timm. On the lab machine (Python 3.9, open egress) run:\n"
    "  pip install torch torchvision timm\n"
    "This session cannot install them (download.pytorch.org / huggingface.co egress-blocked)."
)


class _TimmEmbedder:
    """Shared timm-backed embedder; subclasses pin the exact model string."""

    model_name = None  # override
    input_size = 224

    def __init__(self, device="cpu", batch_size=8):
        try:
            import timm  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            raise RuntimeError("%s %s" % (type(self).__name__, _TORCH_HINT))
        import timm
        import torch

        self._torch = torch
        self.device = device
        self.batch_size = int(batch_size)
        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
        self.model.eval().to(device)

    def fit(self, items):
        pass

    def embed(self, items):
        torch = self._torch
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        out = []
        with torch.no_grad():
            for start in range(0, len(items), self.batch_size):
                batch = []
                for item in items[start : start + self.batch_size]:
                    img = _as_image(item).resize((self.input_size, self.input_size), Image.BILINEAR)
                    arr = (np.asarray(img, dtype=np.float64) / 255.0 - mean) / std
                    batch.append(arr.transpose(2, 0, 1))
                tensor = torch.from_numpy(np.stack(batch)).float().to(self.device)
                feats = self.model(tensor).cpu().numpy()
                out.append(feats)
        return _l2_normalize(np.concatenate(out).astype(np.float64))


class MegaDescriptorEmbedder(_TimmEmbedder):
    """MegaDescriptor-L-384 via timm hf-hub, at 384 px (exact spec string)."""

    model_name = "hf-hub:BVRA/MegaDescriptor-L-384"
    input_size = 384


class DinoV2Embedder(_TimmEmbedder):
    """DINOv2 ViT-S/14 via timm."""

    model_name = "vit_small_patch14_dinov2.lvd142m"
    input_size = 518


class MiewIDEmbedder:
    """MiewID via its repo (wbia-plugin-miew-id / conservationxlabs weights).

    LICENCE UNSETTLED per the feasibility report: the repo carries no OSI
    licence file, only "Copyright Conservation X Labs" -- the lab must settle
    terms with Conservation X Labs before building on these weights.
    """

    def __init__(self, device="cpu", batch_size=8):
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "MiewIDEmbedder requires torch + transformers (weights: "
                "hf.co/conservationxlabs/miewid-msv2, input 440 px). On the lab machine run:\n"
                "  pip install torch transformers\n"
                "This session cannot install them (egress-blocked). Licence is "
                "unsettled: settle terms with Conservation X Labs first."
            )
        import torch
        from transformers import AutoModel

        self._torch = torch
        self.device = device
        self.batch_size = int(batch_size)
        self.model = AutoModel.from_pretrained("conservationxlabs/miewid-msv2", trust_remote_code=True)
        self.model.eval().to(device)

    def fit(self, items):
        pass

    def embed(self, items):
        torch = self._torch
        out = []
        with torch.no_grad():
            for start in range(0, len(items), self.batch_size):
                batch = []
                for item in items[start : start + self.batch_size]:
                    img = _as_image(item).resize((440, 440), Image.BILINEAR)
                    batch.append((np.asarray(img, dtype=np.float64) / 255.0).transpose(2, 0, 1))
                tensor = torch.from_numpy(np.stack(batch)).float().to(self.device)
                feats = self.model(tensor).cpu().numpy()
                out.append(feats)
        return _l2_normalize(np.concatenate(out).astype(np.float64))


_EMBEDDERS = {
    "hist": HistEmbedder,
    "random": RandomEmbedder,
    "megadescriptor": MegaDescriptorEmbedder,
    "dinov2": DinoV2Embedder,
    "miewid": MiewIDEmbedder,
}


def get_embedder(name, seed=0):
    """Factory. ``seed`` is consumed only by embedders that are stochastic."""
    if name not in _EMBEDDERS:
        raise ValueError("unknown backbone %r; choose from %r" % (name, sorted(_EMBEDDERS)))
    if name == "random":
        return RandomEmbedder(seed=seed)
    return _EMBEDDERS[name]()
