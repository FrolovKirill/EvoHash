"""HSJA v2 — noise-based initialization, no target_image required.

Finds initial colliding point via Gaussian noise with increasing sigma,
then boundary-walks toward source using sign-based gradient estimation.

Reference: Chen, Jordan & Wainwright, IEEE S&P 2020 (adapted).
"""

import numpy as np
from PIL import Image


def normalised_l2(original, perturbed):
    diff = perturbed.astype(float) - original.astype(float)
    return float(np.linalg.norm(diff.flatten()) / np.sqrt(original.size))


def query_distance(arr, target_hash, hash_fn):
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return float(hash_fn.distance(hash_fn.compute(img), target_hash))


def collides(arr, target_hash, hash_fn, threshold):
    return query_distance(arr, target_hash, hash_fn) <= threshold


def _binary_search(source, colliding, target_hash, hash_fn, threshold, steps=10):
    lo, hi = 0.0, 1.0
    q = 0
    for _ in range(steps):
        mid = (lo + hi) / 2.0
        cand = (1 - mid) * source + mid * colliding
        q += 1
        if collides(cand, target_hash, hash_fn, threshold):
            hi = mid
        else:
            lo = mid
    return (1 - hi) * source + hi * colliding, q


def _find_initial_collision(source, target_hash, hash_fn, threshold,
                            max_tries=200, sigmas=(10, 25, 50, 100, 200)):
    """Find any colliding point by adding Gaussian noise with increasing sigma."""
    n_queries = 0
    shape = source.shape

    for sigma in sigmas:
        tries_per_sigma = max(1, max_tries // len(sigmas))
        for _ in range(tries_per_sigma):
            noise = np.random.randn(*shape).astype(np.float32) * sigma
            candidate = np.clip(source + noise, 0, 255)
            n_queries += 1
            if collides(candidate, target_hash, hash_fn, threshold):
                return candidate, n_queries

    for _ in range(max(20, max_tries // 4)):
        candidate = np.random.randint(0, 256, size=shape).astype(np.float32)
        n_queries += 1
        if collides(candidate, target_hash, hash_fn, threshold):
            return candidate, n_queries

    return None, n_queries


def _estimate_boundary_gradient(current, source, target_hash, hash_fn,
                                threshold, n_samples=40, delta_ratio=0.01):
    """Sign-based boundary gradient estimation."""
    q = 0
    dist_to_source = np.linalg.norm((current - source).flatten())
    delta = max(0.5, delta_ratio * dist_to_source)

    grad = np.zeros_like(current, dtype=np.float32)
    for _ in range(n_samples):
        u = np.random.randn(*current.shape).astype(np.float32)
        u_norm = np.linalg.norm(u.flatten())
        if u_norm < 1e-8:
            continue
        u /= u_norm

        probe = np.clip(current + delta * u, 0, 255)
        q += 1
        sign = 1.0 if collides(probe, target_hash, hash_fn, threshold) else -1.0
        grad += sign * u

    grad /= max(1, n_samples)

    grad_norm = np.linalg.norm(grad.flatten())
    if grad_norm > 1e-8:
        grad /= grad_norm

    direction_to_source = source - current
    if np.dot(grad.flatten(), direction_to_source.flatten()) < 0:
        grad = -grad

    return grad, q


def _attack_single(src_img, target_hash, hash_fn, threshold,
                   n_iter=25, bs_steps=12, grad_samples=40,
                   step_ratio=0.15, delta_ratio=0.01,
                   init_max_tries=200, init_sigmas=(10, 25, 50, 100, 200)):
    source = np.array(src_img).astype(np.float32)

    x_in, q_total = _find_initial_collision(
        source, target_hash, hash_fn, threshold,
        max_tries=init_max_tries, sigmas=init_sigmas,
    )

    if x_in is None:
        l2 = normalised_l2(source, source)
        return Image.fromarray(source.astype(np.uint8)), {
            "success": False, "l2": l2, "n_queries": q_total, "final_dist": 999.0,
        }

    current, q = _binary_search(
        source, x_in, target_hash, hash_fn, threshold, bs_steps
    )
    q_total += q

    best = current.copy()
    best_l2 = float(np.linalg.norm((best - source).flatten()))

    for step in range(n_iter):
        grad, q = _estimate_boundary_gradient(
            current, source, target_hash, hash_fn, threshold,
            n_samples=grad_samples, delta_ratio=delta_ratio,
        )
        q_total += q

        grad_norm = np.linalg.norm(grad.flatten())
        if grad_norm < 1e-8:
            continue

        dist_to_source = np.linalg.norm((current - source).flatten())
        step_size = max(step_ratio * dist_to_source, 0.5)
        candidate = np.clip(current + step_size * grad, 0, 255)

        if collides(candidate, target_hash, hash_fn, threshold):
            q_total += 1
            current, q = _binary_search(
                source, candidate, target_hash, hash_fn, threshold, bs_steps
            )
        else:
            q_total += 1
            current, q = _binary_search(
                source, current, target_hash, hash_fn, threshold, bs_steps
            )
        q_total += q

        l2 = float(np.linalg.norm((current - source).flatten()))
        if l2 < best_l2 and collides(current, target_hash, hash_fn, threshold):
            q_total += 1
            best_l2 = l2
            best = current.copy()

    norm_l2 = normalised_l2(source, best)
    success = collides(best, target_hash, hash_fn, threshold)
    q_total += 1
    final_dist = query_distance(best, target_hash, hash_fn)
    q_total += 1

    return Image.fromarray(np.clip(best, 0, 255).astype(np.uint8)), {
        "success": success,
        "l2": norm_l2,
        "n_queries": q_total,
        "final_dist": final_dist,
    }


def entrypoint(context: dict) -> dict:
    hash_fn = context["hash_fn"]
    threshold = context["threshold"]
    sources = context["source_images"]
    target_hashes = context["target_hashes"]

    attacked_images, metrics = [], []
    for img, th in zip(sources, target_hashes):
        atk, m = _attack_single(img, th, hash_fn, threshold,
                                n_iter=25, bs_steps=12, grad_samples=40,
                                step_ratio=0.15)
        attacked_images.append(atk)
        metrics.append(m)

    return {"attacked_images": attacked_images, "metrics": metrics}