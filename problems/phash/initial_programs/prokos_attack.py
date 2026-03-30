"""Prokos et al. frequency-targeted attack on perceptual hash functions.

Concentrates perturbations in low-to-mid frequency DCT bands that maximally
affect hash output while preserving perceptual similarity. Uses momentum
for gradient stabilization.

Reference: Prokos et al., USENIX Security 2023.
"""

import numpy as np
from PIL import Image
from scipy.fft import idctn


def normalised_l2(original, perturbed):
    diff = perturbed.astype(float) - original.astype(float)
    return float(np.linalg.norm(diff.flatten()) / np.sqrt(original.size))


def _frequency_weight_mask(h, w, cutoff_ratio=0.35):
    """Mask weighting low-to-mid frequencies higher."""
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


def _attack_single(img, target_hash, hash_fn, threshold,
                   n_iter=800, n_freq_samples=25, lr=8.0, sigma=80.0,
                   momentum=0.5):
    orig = np.array(img).astype(np.float32)
    H, W, C = orig.shape
    current = orig.copy()
    best = orig.copy()
    best_dist = hash_fn.distance(hash_fn.compute(img), target_hash)
    n_queries = 1

    freq_mask = _frequency_weight_mask(H, W)
    momentum_buf = np.zeros((H, W, C), dtype=np.float32)

    for _ in range(n_iter):
        if best_dist <= threshold:
            break

        grad_dct = np.zeros((H, W, C), dtype=np.float32)

        for _ in range(n_freq_samples):
            noise_dct = np.zeros((H, W, C), dtype=np.float32)
            for c in range(C):
                raw = np.random.randn(H, W).astype(np.float32)
                noise_dct[:, :, c] = raw * freq_mask

            noise_spatial = np.zeros_like(current)
            for c in range(C):
                noise_spatial[:, :, c] = idctn(noise_dct[:, :, c], type=2, norm="ortho")

            pos = np.clip(current + sigma * noise_spatial, 0, 255)
            neg = np.clip(current - sigma * noise_spatial, 0, 255)

            d_pos = hash_fn.distance(
                hash_fn.compute(Image.fromarray(pos.astype(np.uint8))), target_hash)
            d_neg = hash_fn.distance(
                hash_fn.compute(Image.fromarray(neg.astype(np.uint8))), target_hash)
            n_queries += 2
            grad_dct += (d_pos - d_neg) * noise_dct

        grad_dct /= 2.0 * sigma * n_freq_samples

        for c in range(C):
            grad_dct[:, :, c] *= freq_mask

        grad_spatial = np.zeros_like(current)
        for c in range(C):
            grad_spatial[:, :, c] = idctn(grad_dct[:, :, c], type=2, norm="ortho")

        momentum_buf = momentum * momentum_buf + (1.0 - momentum) * grad_spatial
        current = np.clip(current - lr * np.sign(momentum_buf), 0, 255)

        dist = hash_fn.distance(
            hash_fn.compute(Image.fromarray(current.astype(np.uint8))), target_hash)
        n_queries += 1
        if dist < best_dist:
            best_dist = dist
            best = current.copy()

    l2 = normalised_l2(orig, best)
    return Image.fromarray(best.astype(np.uint8)), {
        "success": best_dist <= threshold,
        "l2": l2,
        "n_queries": n_queries,
        "final_dist": float(best_dist),
    }


def entrypoint(context: dict) -> dict:
    hash_fn = context["hash_fn"]
    threshold = context["threshold"]
    sources = context["source_images"]
    target_hashes = context["target_hashes"]

    attacked_images, metrics = [], []
    for img, th in zip(sources, target_hashes):
        atk, m = _attack_single(img, th, hash_fn, threshold,
                                n_iter=800, n_freq_samples=25, lr=8.0, sigma=80.0)
        attacked_images.append(atk)
        metrics.append(m)

    return {"attacked_images": attacked_images, "metrics": metrics}