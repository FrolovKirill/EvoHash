"""HopSkipJump Attack (HSJA) for perceptual hash collisions.

Decision-based attack: starts from a known colliding point (the target
image) and iteratively moves toward the source while staying within the
collision boundary, minimising L2 distortion.

Reference: Chen, Jordan & Wainwright, IEEE S&P 2020.
"""

import numpy as np
from PIL import Image

from .utils import collides, normalised_l2, query_distance


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


# ── HSJA v2: noise-based initialization (no target_image required) ───────────
#
# Based on Den221B's implementation. Key differences from v1:
# - Finds initial colliding point via Gaussian noise with increasing sigma
#   (does NOT require a target_image as starting point)
# - Sign-based boundary gradient estimation: probes around boundary point,
#   +1 if inside collision set, -1 if outside, oriented toward source
# - Step toward source proportional to current distance, then binary search
#   back to boundary


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

    # Fallback: random images
    for _ in range(max(20, max_tries // 4)):
        candidate = np.random.randint(0, 256, size=shape).astype(np.float32)
        n_queries += 1
        if collides(candidate, target_hash, hash_fn, threshold):
            return candidate, n_queries

    return None, n_queries


def _estimate_boundary_gradient(current, source, target_hash, hash_fn,
                                threshold, n_samples=40, delta_ratio=0.01):
    """Sign-based boundary gradient estimation.

    Probes random directions around the boundary point.
    sign = +1 if probe is inside collision set, -1 if outside.
    grad = mean(sign * u), then oriented toward source.
    """
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

    # Normalize
    grad_norm = np.linalg.norm(grad.flatten())
    if grad_norm > 1e-8:
        grad /= grad_norm

    # Orient toward source: dot(grad, source - current) should be > 0
    direction_to_source = source - current
    if np.dot(grad.flatten(), direction_to_source.flatten()) < 0:
        grad = -grad

    return grad, q


def hsja_v2_phase(source, target_hash, hash_fn, threshold,
                  n_iter=25, bs_steps=12, grad_samples=40,
                  step_ratio=0.15, delta_ratio=0.01,
                  init_max_tries=200, init_sigmas=(10, 25, 50, 100, 200)):
    """HSJA v2 boundary walk with noise-based init. Returns (best_array, n_queries).

    Does NOT require a target_image — finds initial collision via noise.
    """
    # Phase 0: find initial colliding point
    x_in, q_total = _find_initial_collision(
        source, target_hash, hash_fn, threshold,
        max_tries=init_max_tries, sigmas=init_sigmas,
    )

    if x_in is None:
        # Could not find collision — return source unchanged
        return source.copy(), q_total

    # Binary search to boundary
    current, q = _binary_search(
        source, x_in, target_hash, hash_fn, threshold, bs_steps
    )
    q_total += q

    best = current.copy()
    best_l2 = float(np.linalg.norm((best - source).flatten()))

    for step in range(n_iter):
        # Estimate boundary gradient (sign-based)
        grad, q = _estimate_boundary_gradient(
            current, source, target_hash, hash_fn, threshold,
            n_samples=grad_samples, delta_ratio=delta_ratio,
        )
        q_total += q

        grad_norm = np.linalg.norm(grad.flatten())
        if grad_norm < 1e-8:
            continue

        # Step toward source along estimated direction
        dist_to_source = np.linalg.norm((current - source).flatten())
        step_size = max(step_ratio * dist_to_source, 0.5)
        candidate = np.clip(current + step_size * grad, 0, 255)

        # Binary search back to boundary
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

    return best, q_total


def _attack_single_v2(src_img, target_hash, hash_fn, threshold, **kwargs):
    """HSJA v2 single-image attack (no target_image needed)."""
    source = np.array(src_img).astype(np.float32)

    best, n_queries = hsja_v2_phase(
        source, target_hash, hash_fn, threshold, **kwargs
    )

    norm_l2 = normalised_l2(source, best)
    success = collides(best, target_hash, hash_fn, threshold)
    n_queries += 1

    final_dist = query_distance(best, target_hash, hash_fn)
    n_queries += 1

    return Image.fromarray(np.clip(best, 0, 255).astype(np.uint8)), {
        "success": success,
        "l2": norm_l2,
        "n_queries": n_queries,
        "final_dist": final_dist,
    }


def run_v2(context, **kwargs):
    """Run HSJA v2 (noise-based init) on all images in context.

    Unlike run(), does NOT require target_images in context.
    """
    hash_fn = context["hash_fn"]
    threshold = context["threshold"]
    sources = context["source_images"]
    target_hashes = context["target_hashes"]

    attacked_images, metrics = [], []
    for img, th in zip(sources, target_hashes):
        atk, m = _attack_single_v2(img, th, hash_fn, threshold, **kwargs)
        attacked_images.append(atk)
        metrics.append(m)

    return {"attacked_images": attacked_images, "metrics": metrics}
