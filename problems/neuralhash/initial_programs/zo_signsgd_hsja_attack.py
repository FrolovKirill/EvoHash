"""ZO-Sign-SGD + HSJA hybrid attack.

Phase 1 — HSJA finds a collision via boundary walking from the target image.
Phase 2 — ZO-Sign-SGD refines the collision point, applying sign-gradient
updates to reduce both hash distance and L2 distortion.
"""

import numpy as np
from PIL import Image


def entrypoint(context: dict) -> dict:
    hash_fn = context["hash_fn"]
    threshold = context["threshold"]
    sources: list[Image.Image] = context["source_images"]
    target_hashes: list = context["target_hashes"]
    target_images: list[Image.Image] = context["target_images"]

    attacked_images: list[Image.Image] = []
    metrics: list[dict] = []

    for src, tgt, target_hash in zip(sources, target_images, target_hashes):
        atk_img, m = _zo_signsgd_hsja(src, tgt, target_hash, hash_fn, threshold)
        attacked_images.append(atk_img)
        metrics.append(m)

    return {"attacked_images": attacked_images, "metrics": metrics}


# ── HSJA phase ───────────────────────────────────────────────────────────────

def _collides(arr, target_hash, hash_fn, threshold):
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return hash_fn.distance(hash_fn.compute(img), target_hash) <= threshold


def _binary_search(source, colliding, target_hash, hash_fn, threshold, steps=10):
    lo, hi = 0.0, 1.0
    q = 0
    for _ in range(steps):
        mid = (lo + hi) / 2.0
        cand = (1 - mid) * source + mid * colliding
        q += 1
        if _collides(cand, target_hash, hash_fn, threshold):
            hi = mid
        else:
            lo = mid
    return (1 - hi) * source + hi * colliding, q


def _hsja_phase(source, target, target_hash, hash_fn, threshold, n_iter=25):
    colliding = target.copy()
    current, q_total = _binary_search(source, colliding, target_hash, hash_fn, threshold)
    best = current.copy()
    best_l2 = float(np.linalg.norm((best - source).flatten()))

    for step in range(n_iter):
        direction = current - source
        norm_dir = np.linalg.norm(direction.flatten())
        if norm_dir < 1e-8:
            break

        grad = np.zeros_like(current)
        for _ in range(20):
            rv = np.random.randn(*current.shape).astype(np.float32)
            rv -= np.dot(rv.flatten(), direction.flatten()) / (norm_dir ** 2 + 1e-8) * direction
            rv_norm = np.linalg.norm(rv.flatten())
            if rv_norm < 1e-8:
                continue
            rv /= rv_norm
            q_total += 1
            if _collides(current + rv, target_hash, hash_fn, threshold):
                grad += rv

        grad_norm = np.linalg.norm(grad.flatten())
        if grad_norm < 1e-8:
            continue
        grad /= grad_norm

        step_size = max(norm_dir / (step + 1) * 0.5, 0.5)
        candidate = np.clip(current + step_size * grad, 0, 255)

        if _collides(candidate, target_hash, hash_fn, threshold):
            current, q = _binary_search(source, candidate, target_hash, hash_fn, threshold)
            q_total += q
        else:
            current, q = _binary_search(source, current, target_hash, hash_fn, threshold)
            q_total += q

        l2 = float(np.linalg.norm((current - source).flatten()))
        if l2 < best_l2 and _collides(current, target_hash, hash_fn, threshold):
            q_total += 1
            best_l2 = l2
            best = current.copy()

    return best, q_total


# ── ZO-Sign-SGD refinement phase ────────────────────────────────────────────

def _zo_signsgd_refine(source, start, target_hash, hash_fn, threshold, n_iter=100):
    current = start.copy()
    best = start.copy()
    best_dist = hash_fn.distance(
        hash_fn.compute(Image.fromarray(current.astype(np.uint8))), target_hash
    )
    q = 1
    n_samples = 15
    mu = 4.0
    lr = 1.0

    for iteration in range(n_iter):
        if best_dist <= threshold:
            # Already colliding — step toward source to reduce L2
            direction = source - current
            d_norm = np.linalg.norm(direction.flatten())
            if d_norm < 1e-8:
                break
            direction /= d_norm
            for mult in [3.0, 1.5, 0.5]:
                cand = np.clip(current + mult * direction, 0, 255)
                d = hash_fn.distance(
                    hash_fn.compute(Image.fromarray(cand.astype(np.uint8))), target_hash
                )
                q += 1
                if d <= threshold:
                    current = cand
                    best = cand.copy()
                    best_dist = d
                    break

        # ZO-Sign-SGD step to minimize hash distance
        grad = np.zeros_like(current)
        for _ in range(n_samples):
            u = np.random.randn(*current.shape).astype(np.float32)
            fwd = np.clip(current + mu * u, 0, 255)
            d_fwd = hash_fn.distance(
                hash_fn.compute(Image.fromarray(fwd.astype(np.uint8))), target_hash
            )
            d_cur = hash_fn.distance(
                hash_fn.compute(Image.fromarray(current.astype(np.uint8))), target_hash
            )
            q += 2
            grad += ((d_fwd - d_cur) / mu) * u

        grad /= n_samples
        current = np.clip(current - lr * np.sign(grad), 0, 255)

        dist = hash_fn.distance(
            hash_fn.compute(Image.fromarray(current.astype(np.uint8))), target_hash
        )
        q += 1
        if dist < best_dist:
            best_dist = dist
            best = current.copy()

    return best, best_dist, q


# ── Combined ─────────────────────────────────────────────────────────────────

def _zo_signsgd_hsja(src_img, tgt_img, target_hash, hash_fn, threshold):
    source = np.array(src_img).astype(np.float32)
    target = np.array(tgt_img).astype(np.float32)

    hsja_result, q1 = _hsja_phase(source, target, target_hash, hash_fn, threshold)
    best, best_dist, q2 = _zo_signsgd_refine(
        source, hsja_result, target_hash, hash_fn, threshold
    )

    n_queries = q1 + q2
    l2 = float(np.linalg.norm((best - source).flatten()) / np.sqrt(source.size))

    return Image.fromarray(np.clip(best, 0, 255).astype(np.uint8)), {
        "success": best_dist <= threshold,
        "l2": l2,
        "n_queries": n_queries,
        "final_dist": float(best_dist),
    }
