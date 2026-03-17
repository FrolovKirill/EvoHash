"""NES (Natural Evolution Strategy) attack.

Estimates a pseudo-gradient of hash distance via antithetic (±) random
perturbations, then takes a gradient step to minimise distance.

Reference: Ilyas et al., ICML 2018.
"""

import numpy as np
from PIL import Image

from .utils import normalised_l2


def _attack_single(img, target_hash, hash_fn, threshold,
                   n_iter=150, n_samples=20, sigma=6.0, lr=3.0):
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
        current = np.clip(current - lr * grad_estimate, 0, 255)

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


def nes_refine_single(source, start, target_hash, hash_fn, threshold,
                      n_iter=80, n_samples=15, sigma=4.0, lr=2.0):
    """NES refinement phase for hybrid attacks. Returns (best, best_dist, n_queries)."""
    current = start.copy()
    best = start.copy()
    best_dist = hash_fn.distance(
        hash_fn.compute(Image.fromarray(current.astype(np.uint8))), target_hash
    )
    q = 1

    for _ in range(n_iter):
        if best_dist <= threshold:
            direction = source - current
            d_norm = np.linalg.norm(direction.flatten())
            if d_norm < 1e-8:
                break
            direction /= d_norm
            for step_mult in [3.0, 1.5, 0.5]:
                cand = np.clip(current + step_mult * direction, 0, 255)
                d = hash_fn.distance(
                    hash_fn.compute(Image.fromarray(cand.astype(np.uint8))),
                    target_hash,
                )
                q += 1
                if d <= threshold:
                    current = cand
                    best = cand.copy()
                    best_dist = d
                    break

        grad = np.zeros_like(current)
        for _ in range(n_samples):
            v = np.random.randn(*current.shape).astype(np.float32)
            pos = np.clip(current + sigma * v, 0, 255)
            neg = np.clip(current - sigma * v, 0, 255)
            d_pos = hash_fn.distance(
                hash_fn.compute(Image.fromarray(pos.astype(np.uint8))), target_hash
            )
            d_neg = hash_fn.distance(
                hash_fn.compute(Image.fromarray(neg.astype(np.uint8))), target_hash
            )
            q += 2
            grad += (d_pos - d_neg) * v

        grad /= 2.0 * sigma * n_samples
        current = np.clip(current - lr * grad, 0, 255)

        dist = hash_fn.distance(
            hash_fn.compute(Image.fromarray(current.astype(np.uint8))), target_hash
        )
        q += 1
        if dist < best_dist:
            best_dist = dist
            best = current.copy()

    return best, best_dist, q


def run(context, **kwargs):
    """Run NES attack on all images in context."""
    hash_fn = context["hash_fn"]
    threshold = context["threshold"]
    sources = context["source_images"]
    target_hashes = context["target_hashes"]

    attacked_images, metrics = [], []
    for img, th in zip(sources, target_hashes):
        atk, m = _attack_single(img, th, hash_fn, threshold, **kwargs)
        attacked_images.append(atk)
        metrics.append(m)

    return {"attacked_images": attacked_images, "metrics": metrics}
