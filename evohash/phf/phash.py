"""pHash (perceptual hash) wrapper using the ImageHash library.

pHash computes a 64-bit hash by downsampling the image to 32×32, applying a
DCT, and keeping the top-left 8×8 block. Two images are considered collisions
when their Hamming distance is ≤ threshold (default: 12 bits).
"""

from PIL import Image

from .base import PHFWrapper


class PHashWrapper(PHFWrapper):
    """
    Wrapper around ``imagehash.phash``.

    Args:
        hash_size: Side length of the DCT block (default 8 → 64-bit hash).
        threshold: Hamming-distance collision threshold (default 12).
    """

    def __init__(self, hash_size: int = 8, threshold: int = 12) -> None:
        try:
            import imagehash  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "imagehash is required for PHashWrapper. "
                "Install it with: pip install imagehash"
            ) from e

        self._hash_size = hash_size
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "pHash"

    def compute(self, image: Image.Image):
        """Return an ``imagehash.ImageHash`` for *image*."""
        import imagehash

        return imagehash.phash(image, hash_size=self._hash_size)

    def distance(self, h1, h2) -> float:
        """Hamming distance between two ImageHash values."""
        return float(h1 - h2)
