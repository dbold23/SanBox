"""Learned spot embeddings: train a CNN encoder on the synthetic generator
so that views of the same spot map to nearby vectors, viewpoint and
lighting notwithstanding. The trained embedding is a drop-in replacement
for the handcrafted descriptor in SpotMatcher.
"""

from .model import SpotEncoder
from .dataset import canonical_patch, extract_patch, render_training_view

__all__ = ["SpotEncoder", "canonical_patch", "extract_patch",
           "render_training_view"]
