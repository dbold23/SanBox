"""Inference: use a trained encoder as a drop-in spot descriptor."""

import numpy as np
import torch

from ..features import segment_spot
from .dataset import canonical_patch
from .model import SpotEncoder


class MLSpotDescriptor:
    """Wraps a trained SpotEncoder as ``describe_image`` compatible with
    SpotMatcher enrollment/identification."""

    def __init__(self, checkpoint: str, device: str = "cpu"):
        ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
        self.model = SpotEncoder(ckpt["embed_dim"], ckpt["width"])
        self.model.load_state_dict(ckpt["model"])
        self.model.eval().to(device)
        self.device = torch.device(device)

    def patch_of(self, img: np.ndarray) -> np.ndarray | None:
        contour = segment_spot(img)
        if contour is None:
            return None
        return canonical_patch(img, contour)

    def describe_image(self, img: np.ndarray) -> np.ndarray | None:
        patch = self.patch_of(img)
        if patch is None:
            return None
        return self.describe_patches(patch[None])[0]

    @torch.no_grad()
    def describe_patches(self, patches: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(patches[:, None, :, :].astype(np.float32))
        return self.model(x.to(self.device)).cpu().numpy()


class EnsembleDescriptor:
    """Classical descriptor + learned embedding, concatenated.

    The two make different mistakes (moment/Fourier signatures vs learned
    texture-shape features), and SpotMatcher's Fisher weighting handles
    heterogeneous blocks natively — measured at +1.2 points overall and
    +5 points at 45-55 degree tilt over the classical descriptor alone.
    """

    def __init__(self, checkpoint: str, device: str = "cpu",
                 ml_weight: float = 1.5):
        from ..features import describe_image as classical
        self._classical = classical
        self._ml = MLSpotDescriptor(checkpoint, device)
        self.ml_weight = ml_weight

    def describe_image(self, img: np.ndarray) -> np.ndarray | None:
        a = self._classical(img)
        b = self._ml.describe_image(img)
        if a is None or b is None:
            return None
        return np.concatenate([a, self.ml_weight * b])
