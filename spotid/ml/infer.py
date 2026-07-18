"""Inference: use a trained encoder as a drop-in spot descriptor."""

import numpy as np
import torch

from ..features import segment_spot
from .dataset import extract_patch
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
        center = contour.mean(axis=0)
        radius = float(np.sqrt(((contour - center) ** 2).sum(axis=1).mean()))
        return extract_patch(img, center, radius)

    def describe_image(self, img: np.ndarray) -> np.ndarray | None:
        patch = self.patch_of(img)
        if patch is None:
            return None
        return self.describe_patches(patch[None])[0]

    @torch.no_grad()
    def describe_patches(self, patches: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(patches[:, None, :, :].astype(np.float32))
        return self.model(x.to(self.device)).cpu().numpy()
