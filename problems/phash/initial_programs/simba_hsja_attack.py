"""SimBa + HSJA hybrid attack.

Phase 1 — HSJA boundary walk: starts from target image (known collision)
and moves toward source while staying inside the collision set.
Phase 2 — SimBa DCT refinement: reduces L2 distortion using coordinate
descent along DCT basis vectors while maintaining collision.

Reference: Chen et al., IEEE S&P 2020 (HSJA); Guo et al., ICML 2019 (SimBa).
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


def _estimate_gradient(current, source, target_hash, hash_fn, threshold,
                       n_samples=30, delta=1.0):
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
               n_iter=15, bs_steps=10, grad_samples=20):
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


def simba_refine_single(source, start, target_hash, hash_fn, threshold,
                        n_iter=120, step_size=8.0):
    H, W, _ = source.shape
    current = start.copy()
    best_dist = hash_fn.distance(
        hash_fn.compute(Image.fromarray(current.astype(np.uint8))), target_hash
    )
    q = 1

    max_basis = H * W
    rng = np.random.default_rng()
    all_indices = [(k, c) for c in range(3) for k in range(max_basis)]
    order = rng.permutation(len(all_indices))[:n_iter]

    for idx in order:
        k_ref, c_ref = all_indices[idx]
        if best_dist > threshold:
            basis = _dct_basis(H, W, k_ref, c_ref)
            pos = np.clip(current + step_size * basis, 0, 255)
            neg = np.clip(current - step_size * basis, 0, 255)
            d_pos = hash_fn.distance(
                hash_fn.compute(Image.fromarray(pos.astype(np.uint8))), target_hash)
            d_neg = hash_fn.distance(
                hash_fn.compute(Image.fromarray(neg.astype(np.uint8))), target_hash)
            q += 2
            if d_pos < best_dist:
                best_dist = d_pos
                current = pos
            elif d_neg < best_dist:
                best_dist = d_neg
                current = neg
        else:
            direction = source - current
            d_norm = np.linalg.norm(direction.flatten())
            if d_norm < 1e-8:
                break
            direction /= d_norm
            for mult in [4.0, 2.0, 1.0]:
                cand = np.clip(current + mult * direction, 0, 255)
                d = hash_fn.distance(
                    hash_fn.compute(Image.fromarray(cand.astype(np.uint8))), target_hash)
                q += 1
                if d <= threshold:
                    current = cand
                    best_dist = d
                    break

    return current, best_dist, q


def _attack_single(src_img, tgt_img, target_hash, hash_fn, threshold,
                   hsja_n_iter=15, hsja_bs_steps=10, hsja_grad_samples=20,
                   refine_n_iter=120, refine_step_size=8.0):
    source = np.array(src_img).astype(np.float32)
    target = np.array(tgt_img).astype(np.float32)

    # Phase 1: HSJA
    hsja_result, q1 = hsja_phase(
        source, target, target_hash, hash_fn, threshold,
        n_iter=hsja_n_iter, bs_steps=hsja_bs_steps, grad_samples=hsja_grad_samples,
    )

    # Phase 2: SimBa refinement
    best, best_dist, q2 = simba_refine_single(
        source, hsja_result, target_hash, hash_fn, threshold,
        n_iter=refine_n_iter, step_size=refine_step_size,
    )

    n_queries = q1 + q2
    l2 = normalised_l2(source, best)
    return Image.fromarray(np.clip(best, 0, 255).astype(np.uint8)), {
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
    target_images = context["target_images"]

    attacked_images, metrics = [], []
    for src, tgt, th in zip(sources, target_images, target_hashes):
        atk, m = _attack_single(
            src, tgt, th, hash_fn, threshold,
            hsja_n_iter=15, hsja_bs_steps=10, hsja_grad_samples=20,
            refine_n_iter=120, refine_step_size=8.0,
        )
        attacked_images.append(atk)
        metrics.append(m)

    return {"attacked_images": attacked_images, "metrics": metrics}
