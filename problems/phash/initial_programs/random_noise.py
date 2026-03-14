"""Random Gaussian noise baseline attack against pHash.

This is the simplest possible strategy: add i.i.d. Gaussian noise and keep
the best sample found within a query budget.  It serves as a lower-bound
baseline — any evolved strategy should significantly outperform it.
"""

import numpy as np
from PIL import Image


def entrypoint(context: dict) -> dict:
    """Apply random Gaussian noise to each source image and query pHash."""
    hash_fn = context["hash_fn"]
    threshold = context["threshold"]
    sources: list[Image.Image] = context["source_images"]
    target_hashes: list = context["target_hashes"]

    attacked_images: list[Image.Image] = []
    metrics: list[dict] = []

    for img, target_hash in zip(sources, target_hashes):
        atk_img, m = _random_noise_attack(img, target_hash, hash_fn, threshold)
        attacked_images.append(atk_img)
        metrics.append(m)

    return {"attacked_images": attacked_images, "metrics": metrics}


def _random_noise_attack(
    img: Image.Image,
    target_hash,
    hash_fn,
    threshold: int,
    n_trials: int = 200,
    sigma: float = 8.0,
) -> tuple[Image.Image, dict]:
    """
    Try *n_trials* random Gaussian perturbations; keep the one with the
    smallest Hamming distance to *target_hash*.
    """
    orig = np.array(img).astype(np.float32)
    best_img = orig.copy()
    best_dist = hash_fn.distance(hash_fn.compute(img), target_hash)
    n_queries = 1  # counted the original above

    for _ in range(n_trials):
        noise = np.random.randn(*orig.shape).astype(np.float32) * sigma
        candidate = np.clip(orig + noise, 0, 255)
        cand_pil = Image.fromarray(candidate.astype(np.uint8))
        dist = hash_fn.distance(hash_fn.compute(cand_pil), target_hash)
        n_queries += 1

        if dist < best_dist:
            best_dist = dist
            best_img = candidate
            if best_dist <= threshold:
                break

    l2 = float(
        np.linalg.norm((best_img - orig).flatten()) / np.sqrt(orig.size)
    )
    return Image.fromarray(best_img.astype(np.uint8)), {
        "success": best_dist <= threshold,
        "l2": l2,
        "n_queries": n_queries,
        "final_dist": float(best_dist),
    }
