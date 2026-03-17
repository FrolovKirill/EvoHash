"""HopSkipJump Attack (HSJA) for perceptual hash collisions.

Decision-based attack: starts from a known colliding point (the target
image) and iteratively moves toward the source while staying within the
collision boundary, minimising L2 distortion.

Reference: Chen, Jordan & Wainwright, IEEE S&P 2020.
"""

import numpy as np
from PIL import Image

from .utils import collides, normalised_l2


def _binary_search(source, colliding, target_hash, hash_fn, threshold, steps=10):
    """Binary search along source→colliding to find the boundary."""
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


def _estimate_gradient(current, source, target_hash, hash_fn, threshold,
                       n_samples=30, delta=1.0):
    """Estimate the gradient of the decision boundary via MC sampling."""
    q = 0
    direction = current - source
    norm_dir = np.linalg.norm(direction.flatten())
    if norm_dir < 1e-8:
        return np.zeros_like(current), 0

    grad = np.zeros_like(current)
    for _ in range(n_samples):
        rv = np.random.randn(*current.shape).astype(np.float32)
        rv -= np.dot(rv.flatten(), direction.flatten()) / (norm_dir ** 2 + 1e-8) * direction
        rv_norm = np.linalg.norm(rv.flatten())
        if rv_norm < 1e-8:
            continue
        rv /= rv_norm

        candidate = current + delta * rv
        q += 1
        if collides(candidate, target_hash, hash_fn, threshold):
            grad += rv

    return grad, q


def hsja_phase(source, target, target_hash, hash_fn, threshold,
               n_iter=25, bs_steps=10, grad_samples=20):
    """HSJA boundary walk. Returns (best_array, n_queries).

    Shared by standalone HSJA and all hybrid attacks.
    """
    colliding_arr = target.copy()
    current, q_total = _binary_search(
        source, colliding_arr, target_hash, hash_fn, threshold, bs_steps
    )
    best = current.copy()
    best_l2 = float(np.linalg.norm((best - source).flatten()))

    for step in range(n_iter):
        grad, q = _estimate_gradient(
            current, source, target_hash, hash_fn, threshold, grad_samples
        )
        q_total += q

        grad_norm = np.linalg.norm(grad.flatten())
        if grad_norm < 1e-8:
            continue
        grad /= grad_norm

        step_size = max(
            np.linalg.norm((current - source).flatten()) / (step + 1) * 0.5,
            0.5,
        )
        candidate = np.clip(current + step_size * grad, 0, 255)

        if collides(candidate, target_hash, hash_fn, threshold):
            current, q = _binary_search(
                source, candidate, target_hash, hash_fn, threshold, bs_steps
            )
            q_total += q
        else:
            current, q = _binary_search(
                source, current, target_hash, hash_fn, threshold, bs_steps
            )
            q_total += q

        l2 = float(np.linalg.norm((current - source).flatten()))
        if l2 < best_l2 and collides(current, target_hash, hash_fn, threshold):
            q_total += 1
            best_l2 = l2
            best = current.copy()

    return best, q_total


def _attack_single(src_img, tgt_img, target_hash, hash_fn, threshold,
                   n_iter=40, bs_steps=10, grad_samples=30):
    source = np.array(src_img).astype(np.float32)
    target = np.array(tgt_img).astype(np.float32)

    best, n_queries = hsja_phase(
        source, target, target_hash, hash_fn, threshold,
        n_iter=n_iter, bs_steps=bs_steps, grad_samples=grad_samples,
    )

    norm_l2 = normalised_l2(source, best)
    success = collides(best, target_hash, hash_fn, threshold)
    n_queries += 1

    from .utils import query_distance
    final_dist = query_distance(best, target_hash, hash_fn)
    n_queries += 1

    return Image.fromarray(np.clip(best, 0, 255).astype(np.uint8)), {
        "success": success,
        "l2": norm_l2,
        "n_queries": n_queries,
        "final_dist": final_dist,
    }


def run(context, **kwargs):
    """Run HSJA attack on all images in context."""
    hash_fn = context["hash_fn"]
    threshold = context["threshold"]
    sources = context["source_images"]
    target_hashes = context["target_hashes"]
    target_images = context["target_images"]

    attacked_images, metrics = [], []
    for src, tgt, th in zip(sources, target_images, target_hashes):
        atk, m = _attack_single(src, tgt, th, hash_fn, threshold, **kwargs)
        attacked_images.append(atk)
        metrics.append(m)

    return {"attacked_images": attacked_images, "metrics": metrics}
