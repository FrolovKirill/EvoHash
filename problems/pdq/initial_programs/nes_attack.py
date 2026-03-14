"""NES (Natural Evolution Strategy) attack against PDQ.

Same NES algorithm as the pHash variant, adapted for the 256-bit PDQ space.
The higher threshold (92/256 ≈ 36%) means more bits must be flipped, so we
use more antithetic samples and a larger learning rate.
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
        atk_img, m = _nes_attack(img, target_hash, hash_fn, threshold)
        attacked_images.append(atk_img)
        metrics.append(m)

    return {"attacked_images": attacked_images, "metrics": metrics}


def _nes_attack(
    img: Image.Image,
    target_hash,
    hash_fn,
    threshold: int,
    n_iter: int = 200,
    n_samples: int = 30,
    sigma: float = 8.0,
    lr: float = 4.0,
) -> tuple[Image.Image, dict]:
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

    l2 = float(
        np.linalg.norm((best - orig).flatten()) / np.sqrt(orig.size)
    )
    return Image.fromarray(best.astype(np.uint8)), {
        "success": best_dist <= threshold,
        "l2": l2,
        "n_queries": n_queries,
        "final_dist": float(best_dist),
    }
