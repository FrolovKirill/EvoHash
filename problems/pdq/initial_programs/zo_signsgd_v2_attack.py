"""ZO-SignSGD v2 — central difference estimator + sign-based momentum.

Improvements over v1:
- Central difference: (f(x+mu*u) - f(x-mu*u)) / (2*mu) — more accurate
- Sign-based momentum: m = beta*m + (1-beta)*sign(g), step = sign(m)

Reference: Liu et al., ICLR 2019.
"""

import numpy as np
from PIL import Image


def normalised_l2(original, perturbed):
    diff = perturbed.astype(float) - original.astype(float)
    return float(np.linalg.norm(diff.flatten()) / np.sqrt(original.size))


def _attack_single(img, target_hash, hash_fn, threshold,
                   n_iter=150, n_samples=20, mu=5.0, lr=1.5,
                   momentum=0.5, grayscale=False):
    orig = np.array(img).astype(np.float32)
    current = orig.copy()
    best = orig.copy()
    best_dist = hash_fn.distance(hash_fn.compute(img), target_hash)
    n_queries = 1

    momentum_buf = np.zeros_like(orig)

    for _ in range(n_iter):
        if best_dist <= threshold:
            break

        grad_estimate = np.zeros_like(orig)
        for _ in range(n_samples):
            u = np.random.randn(*orig.shape).astype(np.float32)
            if grayscale and u.ndim == 3 and u.shape[-1] == 3:
                g = u.mean(axis=2, keepdims=True)
                u = np.repeat(g, 3, axis=2)

            fwd = np.clip(current + mu * u, 0, 255)
            bwd = np.clip(current - mu * u, 0, 255)
            d_fwd = hash_fn.distance(
                hash_fn.compute(Image.fromarray(fwd.astype(np.uint8))),
                target_hash,
            )
            d_bwd = hash_fn.distance(
                hash_fn.compute(Image.fromarray(bwd.astype(np.uint8))),
                target_hash,
            )
            n_queries += 2
            grad_estimate += ((d_fwd - d_bwd) / (2.0 * mu)) * u

        grad_estimate /= n_samples

        s = np.sign(grad_estimate)
        if momentum > 0:
            momentum_buf = momentum * momentum_buf + (1.0 - momentum) * s
            step_dir = np.sign(momentum_buf)
        else:
            step_dir = s

        current = np.clip(current - lr * step_dir, 0, 255)

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
                                n_iter=150, n_samples=20, mu=5.0, lr=1.5,
                                momentum=0.5)
        attacked_images.append(atk)
        metrics.append(m)

    return {"attacked_images": attacked_images, "metrics": metrics}