"""PDQ hash wrapper using the pdqhash library.

PDQ (Photo DNA Quality) is a perceptual hash developed by Facebook/Meta.
It produces a 256-bit binary hash. Two images are considered collisions when
their Hamming distance is ≤ threshold (default: 92 bits out of 256).
"""

import numpy as np
from PIL import Image

from .base import PHFWrapper


class PDQWrapper(PHFWrapper):
    """
    Wrapper around the ``pdqhash`` library.

    Args:
        threshold: Hamming-distance collision threshold (default 92).
    """

    def __init__(self, threshold: int = 92) -> None:
        try:
            import pdqhash  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "pdqhash is required for PDQWrapper. "
                "Install it with: pip install pdqhash"
            ) from e

        self.threshold = threshold

    @property
    def name(self) -> str:
        return "PDQ"

    def compute(self, image: Image.Image) -> np.ndarray:
        """
        Return the PDQ hash as a 1-D binary numpy array of length 256.

        ``pdqhash.compute`` returns ``(hash_vector, quality)``; we keep only
        the hash vector.
        """
        import pdqhash

        img_array = np.array(image.convert("RGB"))
        hash_vector, _quality = pdqhash.compute(img_array)
        return hash_vector  # shape (256,), dtype bool or uint8

    def distance(self, h1: np.ndarray, h2: np.ndarray) -> float:
        """Hamming distance between two PDQ hash vectors."""
        return float(np.sum(h1 != h2))
