"""Random Hash Perturbation Defense.

Implements the defense proposed in Madden et al. (NeurIPS 2024 Workshop):
randomly flip a fraction q of hash bits before comparison. This breaks
gradient estimation used by adversarial attacks, since the hash output
becomes stochastic.

For binary hashes (pHash, PDQ, NeuralHash): flip q * hash_length random bits.
For real-valued hashes (PhotoDNA): add Gaussian noise to components.
"""

import numpy as np
from typing import Any
from PIL import Image

from evohash.phf.base import PHFWrapper


class PerturbedHashWrapper(PHFWrapper):
    """Wrapper that adds random perturbation to hash output as a defense.

    Args:
        base: The underlying PHF wrapper to defend.
        q: Fraction of hash bits/components to perturb (0.0 to 1.0).
        noise_std: Standard deviation for real-valued hash noise (PhotoDNA).
    """

    def __init__(self, base: PHFWrapper, q: float = 0.1, noise_std: float = 50.0):
        self._base = base
        self.q = q
        self.noise_std = noise_std
        self.threshold = base.threshold

    def compute(self, image: Image.Image) -> Any:
        h = self._base.compute(image)
        return self._perturb(h)

    def _perturb(self, h: Any) -> Any:
        """Apply random perturbation to hash value."""
        if isinstance(h, np.ndarray):
            h = h.copy()
            n = len(h)
            n_flip = max(1, int(self.q * n))

            if h.dtype in (np.bool_, np.uint8, np.int8, np.int32, np.int64):
                # Binary hash: flip random bits
                indices = np.random.choice(n, size=n_flip, replace=False)
                h[indices] = 1 - h[indices]
            else:
                # Real-valued hash (PhotoDNA): add noise
                indices = np.random.choice(n, size=n_flip, replace=False)
                h[indices] += np.random.normal(0, self.noise_std, size=n_flip)
            return h

        # imagehash.ImageHash objects (pHash)
        if hasattr(h, 'hash'):
            h_copy = h.__class__(h.hash.copy())
            flat = h_copy.hash.ravel()
            n = len(flat)
            n_flip = max(1, int(self.q * n))
            indices = np.random.choice(n, size=n_flip, replace=False)
            flat[indices] = ~flat[indices]
            return h_copy

        return h

    def distance(self, h1: Any, h2: Any) -> float:
        return self._base.distance(h1, h2)

    @property
    def name(self) -> str:
        return f"Perturbed({self._base.name}, q={self.q})"
