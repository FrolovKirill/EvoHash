"""ZO-Sign-SGD (Zeroth-Order Sign Stochastic Gradient Descent) attack.

Estimates gradient via random finite differences, then takes the sign of
the estimated gradient for a fixed-magnitude update.

Reference: Liu et al., ICLR 2019.
"""

import numpy as np
from PIL import Image


def normalised_l2(original, perturbed):
    diff = perturbed.astype(float) - original.astype(float)
    return float(np.linalg.norm(diff.flatten()) / np.sqrt(original.size))


def _attack_single(img, target_hash, hash_fn, threshold,
                   n_iter=150, n_samples=20, mu=5.0, lr=1.5):
    orig = np.array(img).astype(np.float32)
    current = orig.copy()
    best = orig.copy()
    best_dist = hash_fn.distance(hash_fn.compute(img), target_hash)
    n_queries = 1

    for _ in range(n_iter):
        if best_dist <= threshold:
            break

        grad_estimate = np.zeros_like(orig)
        for _ in range(n_samples):
            u = np.random.randn(*orig.shape).astype(np.float32)
            fwd = np.clip(current + mu * u, 0, 255)
            d_fwd = hash_fn.distance(
                hash_fn.compute(Image.fromarray(fwd.astype(np.uint8))),
                target_hash,
            )
            d_cur = hash_fn.distance(
                hash_fn.compute(Image.fromarray(current.astype(np.uint8))),
                target_hash,
            )
            n_queries += 2
            grad_estimate += ((d_fwd - d_cur) / mu) * u

        grad_estimate /= n_samples
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
                                n_iter=150, n_samples=20, mu=5.0, lr=1.5)
        attacked_images.append(atk)
        metrics.append(m)

    return {"attacked_images": attacked_images, "metrics": metrics}