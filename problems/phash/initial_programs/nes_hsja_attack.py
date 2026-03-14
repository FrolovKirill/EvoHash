"""NES + HSJA hybrid attack.

Phase 1 — HSJA finds an initial collision with minimal distortion by walking
along the decision boundary from the target image toward the source.
Phase 2 — NES refines the result, using the HSJA output as a warm start and
further reducing L2 distortion while maintaining the collision.

This two-phase approach combines HSJA's ability to reliably *find* collisions
with NES's gradient-estimation refinement for distortion minimisation.
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
        atk_img, m = _nes_hsja(src, tgt, target_hash, hash_fn, threshold)
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
    """Run a short HSJA to find a low-distortion collision starting point."""
    colliding = target.copy()
    current, q_total = _binary_search(
        source, colliding, target_hash, hash_fn, threshold
    )
    best = current.copy()
    best_l2 = float(np.linalg.norm((best - source).flatten()))

    for step in range(n_iter):
        n_samples = 20
        direction = current - source
        norm_dir = np.linalg.norm(direction.flatten())
        if norm_dir < 1e-8:
            break

        grad = np.zeros_like(current)
        for _ in range(n_samples):
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


# ── NES refinement phase ─────────────────────────────────────────────────────

def _nes_refine(source, start, target_hash, hash_fn, threshold, n_iter=80):
    """NES refinement: reduce L2 while maintaining collision."""
    current = start.copy()
    best = start.copy()
    best_dist = hash_fn.distance(
        hash_fn.compute(Image.fromarray(current.astype(np.uint8))), target_hash
    )
    q = 1
    n_samples = 15
    sigma = 4.0
    lr = 2.0

    for _ in range(n_iter):
        if best_dist <= threshold:
            # Already colliding — try to reduce L2 by moving toward source
            direction = source - current
            d_norm = np.linalg.norm(direction.flatten())
            if d_norm < 1e-8:
                break
            direction /= d_norm

            # Small step toward source, check if still colliding
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

        # NES gradient step to minimize hash distance
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


# ── Combined ─────────────────────────────────────────────────────────────────

def _nes_hsja(src_img, tgt_img, target_hash, hash_fn, threshold):
    source = np.array(src_img).astype(np.float32)
    target = np.array(tgt_img).astype(np.float32)

    # Phase 1: HSJA
    hsja_result, q1 = _hsja_phase(source, target, target_hash, hash_fn, threshold)

    # Phase 2: NES refinement
    best, best_dist, q2 = _nes_refine(source, hsja_result, target_hash, hash_fn, threshold)

    n_queries = q1 + q2
    l2 = float(np.linalg.norm((best - source).flatten()) / np.sqrt(source.size))

    return Image.fromarray(np.clip(best, 0, 255).astype(np.uint8)), {
        "success": best_dist <= threshold,
        "l2": l2,
        "n_queries": n_queries,
        "final_dist": float(best_dist),
    }
