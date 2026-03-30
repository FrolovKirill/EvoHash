"""NES (Natural Evolution Strategy) attack.

Estimates a pseudo-gradient of hash distance via antithetic (+/-) random
perturbations, then takes a gradient step to minimise distance.

Reference: Ilyas et al., ICML 2018.
"""

import numpy as np
from PIL import Image


def normalised_l2(original, perturbed):
    diff = perturbed.astype(float) - original.astype(float)
    return float(np.linalg.norm(diff.flatten()) / np.sqrt(original.size))


def _attack_single(img, target_hash, hash_fn, threshold,
                   n_iter=800, n_samples=20, sigma=80.0, lr=8.0):
    orig = np.array(img).astype(np.float32)
    current = orig.copy()
    best = orig.copy()
    best_dist = hash_fn.distance(hash_fn.compute(img), target_hash)
    n_queries = 1

    for _ in range(n_iter):
        if best_dist <= threshold:
            break

        directions = [
            np.random.randn(*orig.shape).astype(np.float32)
            for _ in range(n_samples)
        ]

        grad_estimate = np.zeros_like(orig)
        for v in directions:
            pos = np.clip(current + sigma * v, 0, 255)
            neg = np.clip(current - sigma * v, 0, 255)
            d_pos = hash_fn.distance(
                hash_fn.compute(Image.fromarray(pos.astype(np.uint8))),
                target_hash,
            )
            d_neg = hash_fn.distance(
                hash_fn.compute(Image.fromarray(neg.astype(np.uint8))),
                target_hash,
            )
            n_queries += 2
            grad_estimate += (d_pos - d_neg) * v

        grad_estimate /= 2.0 * sigma * n_samples
        current = np.clip(current - lr * np.sign(grad_estimate), 0, 255)

        dist = hash_fn.distance(
            hash_fn.compute(Image.fromarray(current.astype(np.uint8))),
            target_hash,
        )
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
                                n_iter=800, n_samples=20, sigma=80.0, lr=8.0)
        attacked_images.append(atk)
        metrics.append(m)

    return {"attacked_images": attacked_images, "metrics": metrics}