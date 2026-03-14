"""Abstract interface for perceptual hash function wrappers."""

from abc import ABC, abstractmethod
from typing import Any

from PIL import Image


class PHFWrapper(ABC):
    """
    Uniform interface for perceptual hash functions.

    Subclasses wrap a specific PHF library and expose three operations:
    - compute: image → hash value
    - distance: (hash, hash) → scalar distance
    - threshold: the maximum distance considered a collision
    """

    #: Maximum hash distance at which two images are considered a collision.
    threshold: int

    @abstractmethod
    def compute(self, image: Image.Image) -> Any:
        """Compute the perceptual hash of *image*."""

    @abstractmethod
    def distance(self, h1: Any, h2: Any) -> float:
        """Return the distance between two hashes (lower → more similar)."""

    def is_collision(self, h1: Any, h2: Any) -> bool:
        """Return True if the two hashes are within the collision threshold."""
        return self.distance(h1, h2) <= self.threshold

    @property
    def name(self) -> str:
        """Human-readable name used in logs and reports."""
        return self.__class__.__name__
