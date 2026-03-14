"""SimBa + HSJA hybrid attack.

Phase 1 — HSJA finds an initial collision by boundary-walking from the target
image toward the source.
Phase 2 — SimBa (DCT-basis coordinate descent) refines the collision, reducing
L2 distortion while maintaining or improving the hash collision.
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
        atk_img, m = _simba_hsja(src, tgt, target_hash, hash_fn, threshold)
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


# ── SimBa refinement phase ───────────────────────────────────────────────────

def _dct_basis(height, width, k, c):
    pairs = [(u, v) for u in range(height) for v in range(width)]
    pairs.sort(key=lambda uv: uv[0] ** 2 + uv[1] ** 2)
    u, v = pairs[k % len(pairs)]
    rows = np.arange(height).reshape(-1, 1)
    cols = np.arange(width).reshape(1, -1)
    basis_2d = np.cos(np.pi * u * (2 * rows + 1) / (2 * height)) * \
               np.cos(np.pi * v * (2 * cols + 1) / (2 * width))
    norm = np.linalg.norm(basis_2d)
    if norm > 0:
        basis_2d /= norm
    basis = np.zeros((height, width, 3), dtype=np.float32)
    basis[:, :, c] = basis_2d
    return basis


def _simba_refine(source, start, target_hash, hash_fn, threshold, n_iter=120):
    """SimBa coordinate descent: move toward source in DCT basis while keeping collision."""
    H, W, _ = source.shape
    current = start.copy()
    best_dist = hash_fn.distance(
        hash_fn.compute(Image.fromarray(current.astype(np.uint8))), target_hash
    )
    q = 1
    step_size = 8.0

    max_basis = H * W
    basis_indices = np.random.randint(0, max_basis, size=n_iter)
    channel_indices = np.random.randint(0, 3, size=n_iter)

    for i in range(n_iter):
        if best_dist > threshold:
            # Not colliding — try to get back
            basis = _dct_basis(H, W, int(basis_indices[i]), int(channel_indices[i]))
            pos = np.clip(current + step_size * basis, 0, 255)
            neg = np.clip(current - step_size * basis, 0, 255)
            d_pos = hash_fn.distance(
                hash_fn.compute(Image.fromarray(pos.astype(np.uint8))), target_hash
            )
            d_neg = hash_fn.distance(
                hash_fn.compute(Image.fromarray(neg.astype(np.uint8))), target_hash
            )
            q += 2
            if d_pos < best_dist:
                best_dist = d_pos
                current = pos
            elif d_neg < best_dist:
                best_dist = d_neg
                current = neg
        else:
            # Colliding — step toward source to reduce L2
            direction = source - current
            d_norm = np.linalg.norm(direction.flatten())
            if d_norm < 1e-8:
                break
            direction /= d_norm

            for mult in [4.0, 2.0, 1.0]:
                cand = np.clip(current + mult * direction, 0, 255)
                d = hash_fn.distance(
                    hash_fn.compute(Image.fromarray(cand.astype(np.uint8))), target_hash
                )
                q += 1
                if d <= threshold:
                    current = cand
                    best_dist = d
                    break

    best = current
    norm_l2 = float(np.linalg.norm((best - source).flatten()) / np.sqrt(source.size))
    return best, best_dist, q


# ── Combined ─────────────────────────────────────────────────────────────────

def _simba_hsja(src_img, tgt_img, target_hash, hash_fn, threshold):
    source = np.array(src_img).astype(np.float32)
    target = np.array(tgt_img).astype(np.float32)

    hsja_result, q1 = _hsja_phase(source, target, target_hash, hash_fn, threshold)
    best, best_dist, q2 = _simba_refine(source, hsja_result, target_hash, hash_fn, threshold)

    n_queries = q1 + q2
    l2 = float(np.linalg.norm((best - source).flatten()) / np.sqrt(source.size))

    return Image.fromarray(np.clip(best, 0, 255).astype(np.uint8)), {
        "success": best_dist <= threshold,
        "l2": l2,
        "n_queries": n_queries,
        "final_dist": float(best_dist),
    }
