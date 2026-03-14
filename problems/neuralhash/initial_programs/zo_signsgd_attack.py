"""ZO-Sign-SGD (Zeroth-Order Sign Stochastic Gradient Descent) attack.

Estimates the gradient via random finite differences, then takes the *sign*
of the estimated gradient (like SignSGD) to produce a fixed-magnitude update.
This keeps L-inf perturbations tightly controlled while still moving in a
useful direction.

Reference: Liu et al., "Signsgd via Zeroth-Order Oracle", ICLR 2019.
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
        atk_img, m = _zo_signsgd_attack(img, target_hash, hash_fn, threshold)
        attacked_images.append(atk_img)
        metrics.append(m)

    return {"attacked_images": attacked_images, "metrics": metrics}


def _zo_signsgd_attack(
    img: Image.Image,
    target_hash,
    hash_fn,
    threshold: int,
    n_iter: int = 150,
    n_samples: int = 20,
    mu: float = 5.0,
    lr: float = 1.5,
) -> tuple[Image.Image, dict]:
    """
    ZO-SignSGD loop.

    At each step, estimate the gradient of hash distance w.r.t. pixel values
    using ``n_samples`` random Gaussian directions with smoothing parameter
    ``mu``, then apply ``sign(grad) * lr`` as the update.
    """
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

        # Sign step — fixed magnitude per pixel
        current = np.clip(current - lr * np.sign(grad_estimate), 0, 255)

        dist = hash_fn.distance(
            hash_fn.compute(Image.fromarray(current.astype(np.uint8))),
            target_hash,
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
