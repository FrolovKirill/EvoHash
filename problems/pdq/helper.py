"""Shared attack utilities for the PDQ collision problem.

PDQ produces a 256-bit binary hash.  Distance is Hamming distance over 256 bits;
collision threshold is 92.

This file mirrors problems/phash/helper.py with PDQ-specific documentation.
"""

import numpy as np
from PIL import Image


def to_array(image: Image.Image) -> np.ndarray:
    """Convert a PIL Image to a float32 H×W×3 array in [0, 255]."""
    return np.array(image).astype(np.float32)


def to_image(array: np.ndarray) -> Image.Image:
    """Convert a float32 H×W×3 array (values 0–255) to a PIL Image."""
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def clamp_perturbation(
    original: np.ndarray,
    perturbed: np.ndarray,
    max_l2: float | None = None,
    max_linf: float | None = None,
) -> np.ndarray:
    """
    Clamp *perturbed* so that:
      - pixel values stay in [0, 255]
      - optionally, the L∞ norm of the perturbation ≤ max_linf
      - optionally, the normalised L2 norm of the perturbation ≤ max_l2
    """
    delta = perturbed - original

    if max_linf is not None:
        delta = np.clip(delta, -max_linf, max_linf)

    if max_l2 is not None:
        norm = np.linalg.norm(delta.flatten())
        if norm > max_l2 * np.sqrt(original.size):
            scale = max_l2 * np.sqrt(original.size) / (norm + 1e-8)
            delta = delta * scale

    return np.clip(original + delta, 0, 255)


def normalised_l2(original: np.ndarray, perturbed: np.ndarray) -> float:
    """L2 norm of the perturbation, normalised by sqrt(#pixels)."""
    diff = perturbed.astype(float) - original.astype(float)
    return float(np.linalg.norm(diff.flatten()) / np.sqrt(original.size))


def make_metric(
    success: bool,
    original: np.ndarray,
    attacked: np.ndarray,
    n_queries: int,
    final_dist: float,
) -> dict:
    """Build a per-image metric dict in the format expected by validate.py."""
    return {
        "success": success,
        "l2": normalised_l2(original, attacked),
        "n_queries": n_queries,
        "final_dist": float(final_dist),
    }
