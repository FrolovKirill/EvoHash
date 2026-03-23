"""Random Gaussian noise baseline attack.

Simplest possible strategy: add i.i.d. Gaussian noise and keep the best
sample. Serves as a lower-bound baseline.
"""

import numpy as np
from PIL import Image


def normalised_l2(original, perturbed):
    diff = perturbed.astype(float) - original.astype(float)
    return float(np.linalg.norm(diff.flatten()) / np.sqrt(original.size))


def _attack_single(img, target_hash, hash_fn, threshold,
                   n_trials=200, sigma=8.0):
    orig = np.array(img).astype(np.float32)
    best_img = orig.copy()
    best_dist = hash_fn.distance(hash_fn.compute(img), target_hash)
    n_queries = 1

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

    l2 = normalised_l2(orig, best_img)
    return Image.fromarray(best_img.astype(np.uint8)), {
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
                                n_trials=200, sigma=8.0)
        attacked_images.append(atk)
        metrics.append(m)

    return {"attacked_images": attacked_images, "metrics": metrics}