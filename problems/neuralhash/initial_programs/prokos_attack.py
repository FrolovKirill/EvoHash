"""Prokos et al. attack on perceptual hash functions.

Implements the frequency-targeted iterative attack described in:
Prokos et al., "Squint Hard Enough: Attacking Perceptual Hashing with
Adversarial Machine Learning", USENIX Security 2023.

The key insight is that perceptual hash functions primarily depend on
low-frequency image content. The attack concentrates perturbations in
low-to-mid frequency DCT bands that maximally affect the hash output
while preserving perceptual similarity (high frequencies are left alone).

The attack iteratively:
1. Computes a DCT of the image.
2. Estimates gradient of hash distance in DCT coefficient space using
   finite differences over the most impactful frequency bands.
3. Applies a frequency-weighted update favouring low-mid frequencies.
"""

import numpy as np
from PIL import Image


def entrypoint(context: dict) -> dict:
    hash_fn = context["hash_fn"]
    threshold = context["threshold"]
    sources: list[Image.Image] = context["source_images"]
    target_hashes: list = context["target_hashes"]

    attacked_images: list[Image.Image] = []
    metrics: list[dict] = []

    for img, target_hash in zip(sources, target_hashes):
        atk_img, m = _prokos_attack(img, target_hash, hash_fn, threshold)
        attacked_images.append(atk_img)
        metrics.append(m)

    return {"attacked_images": attacked_images, "metrics": metrics}


def _dct2(block: np.ndarray) -> np.ndarray:
    """Type-II 2D DCT via matrix multiplication (no scipy dependency)."""
    from scipy.fft import dctn
    return dctn(block, type=2, norm="ortho")


def _idct2(block: np.ndarray) -> np.ndarray:
    """Inverse 2D DCT."""
    from scipy.fft import idctn
    return idctn(block, type=2, norm="ortho")


def _frequency_weight_mask(h: int, w: int, cutoff_ratio: float = 0.35) -> np.ndarray:
    """
    Create a mask that weights low-to-mid frequencies higher.

    Frequencies below ``cutoff_ratio * max_freq`` get weight 1.0,
    then linearly decays to 0.1 at the Nyquist frequency.
    """
    fy = np.arange(h).reshape(-1, 1) / h
    fx = np.arange(w).reshape(1, -1) / w
    freq = np.sqrt(fy ** 2 + fx ** 2)
    max_freq = np.sqrt(2)
    cutoff = cutoff_ratio * max_freq

    mask = np.where(
        freq <= cutoff,
        1.0,
        np.maximum(0.1, 1.0 - (freq - cutoff) / (max_freq - cutoff) * 0.9),
    )
    return mask.astype(np.float32)


def _prokos_attack(
    img: Image.Image,
    target_hash,
    hash_fn,
    threshold: int,
    n_iter: int = 120,
    n_freq_samples: int = 25,
    lr: float = 5.0,
    sigma: float = 3.0,
) -> tuple[Image.Image, dict]:
    orig = np.array(img).astype(np.float32)
    H, W, C = orig.shape
    current = orig.copy()
    best = orig.copy()
    best_dist = hash_fn.distance(hash_fn.compute(img), target_hash)
    n_queries = 1

    freq_mask = _frequency_weight_mask(H, W)

    for _ in range(n_iter):
        if best_dist <= threshold:
            break

        grad_dct = np.zeros((H, W, C), dtype=np.float32)

        for _ in range(n_freq_samples):
            # Random perturbation in DCT domain, weighted by frequency mask
            noise_dct = np.zeros((H, W, C), dtype=np.float32)
            for c in range(C):
                raw = np.random.randn(H, W).astype(np.float32)
                noise_dct[:, :, c] = raw * freq_mask

            # Convert to spatial domain
            noise_spatial = np.zeros_like(current)
            for c in range(C):
                noise_spatial[:, :, c] = _idct2(noise_dct[:, :, c])

            pos = np.clip(current + sigma * noise_spatial, 0, 255)
            neg = np.clip(current - sigma * noise_spatial, 0, 255)

            d_pos = hash_fn.distance(
                hash_fn.compute(Image.fromarray(pos.astype(np.uint8))), target_hash
            )
            d_neg = hash_fn.distance(
                hash_fn.compute(Image.fromarray(neg.astype(np.uint8))), target_hash
            )
            n_queries += 2

            grad_dct += (d_pos - d_neg) * noise_dct

        grad_dct /= 2.0 * sigma * n_freq_samples

        # Apply frequency-weighted gradient in DCT domain
        for c in range(C):
            grad_dct[:, :, c] *= freq_mask

        # Convert gradient back to spatial domain and update
        grad_spatial = np.zeros_like(current)
        for c in range(C):
            grad_spatial[:, :, c] = _idct2(grad_dct[:, :, c])

        current = np.clip(current - lr * grad_spatial, 0, 255)

        dist = hash_fn.distance(
            hash_fn.compute(Image.fromarray(current.astype(np.uint8))), target_hash
        )
        n_queries += 1

        if dist < best_dist:
            best_dist = dist
            best = current.copy()

    l2 = float(np.linalg.norm((best - orig).flatten()) / np.sqrt(orig.size))
    return Image.fromarray(best.astype(np.uint8)), {
        "success": best_dist <= threshold,
        "l2": l2,
        "n_queries": n_queries,
        "final_dist": float(best_dist),
    }
