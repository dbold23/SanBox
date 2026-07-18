"""spotid — generate splotch identities and recognize them across viewpoints.

A *spot identity* is a deterministic organic blob shape derived from a seed.
The renderer produces unlimited permutations of an identity (rotation, scale,
perspective tilt, lighting, noise). The matcher recovers which identity a
rendered image shows, invariant to those permutations.
"""

from .shapes import generate_identity
from .render import ViewConfig, render_view
from .features import segment_spot, describe_contour, describe_image
from .matcher import SpotMatcher

__all__ = [
    "generate_identity",
    "ViewConfig",
    "render_view",
    "segment_spot",
    "describe_contour",
    "describe_image",
    "SpotMatcher",
]
