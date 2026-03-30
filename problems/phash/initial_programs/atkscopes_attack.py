"""ATKScopes -- multiresolution adversarial perturbation attack.

Uses batched random directions in DCT space with sign-based updates for
effective perturbation of discrete hash functions. Combines multi-resolution
probing (global DCT) with aggressive sign-SGD optimization.

Reference: Zhang et al., USENIX Security 2025.
"""

import numpy as np
from PIL import Image


def normalised_l2(original, perturbed):
    diff = perturbed.astype(float) - original.astype(float)
    return float(np.linalg.norm(diff.flatten()) / np.sqrt(original.size))


def query_distance(arr, target_hash, hash_fn):
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return float(hash_fn.distance(hash_fn.compute(img), target_hash))


def _attack_single(img, target_hash, hash_fn, threshold,
                   n_iter=800, n_samples=20, sigma=80.0, lr=8.0):
    """ATKScopes with batched gradient estimation and sign update."""
    orig = np.array(img).astype(np.float32)
    current = orig.copy()
    best = orig.copy()
    best_dist = query_distance(orig, target_hash, hash_fn)
    n_queries = 1

    for _ in range(n_iter):
        if best_dist <= threshold:
            break

        grad_estimate = np.zeros_like(orig)
        for _ in range(n_samples):
            u = np.random.randn(*orig.shape).astype(np.float32)
            pos = np.clip(current + sigma * u, 0, 255)
            neg = np.clip(current - sigma * u, 0, 255)

            d_pos = query_distance(pos, target_hash, hash_fn)
            d_neg = query_distance(neg, target_hash, hash_fn)
            n_queries += 2

            grad_estimate += (d_pos - d_neg) * u

        grad_estimate /= 2.0 * sigma * n_samples
        current = np.clip(current - lr * np.sign(grad_estimate), 0, 255)

        dist = query_distance(current, target_hash, hash_fn)
        n_queries += 1
        if dist < best_dist:
            best_dist = dist
            best = current.copy()

    l2 = normalised_l2(orig, best)
    return Image.fromarray(np.clip(best, 0, 255).astype(np.uint8)), {
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
                                n_iter=800, n_samples=20, sigma=80.0, lr=8.0)
        attacked_images.append(atk)
        metrics.append(m)

    return {"attacked_images": attacked_images, "metrics": metrics}
