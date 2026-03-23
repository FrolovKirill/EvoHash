"""SimBa (Simple Black-box Adversarial) attack — DCT basis variant.

Coordinate descent using DCT basis vectors: tries +/- step along each
basis direction, keeps the one that reduces hash distance.

Reference: Guo et al., ICML 2019.
"""

import numpy as np
from PIL import Image


def normalised_l2(original, perturbed):
    diff = perturbed.astype(float) - original.astype(float)
    return float(np.linalg.norm(diff.flatten()) / np.sqrt(original.size))


def _dct_basis(height, width, k, c):
    """Return the k-th 2D DCT basis vector for shape (H, W, 3), channel c."""
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


def _attack_single(img, target_hash, hash_fn, threshold,
                   n_iter=200, step_size=12.0):
    """SimBa with DCT basis -- deterministic permutation without replacement."""
    orig = np.array(img).astype(np.float32)
    H, W, _ = orig.shape
    current = orig.copy()
    best_dist = hash_fn.distance(hash_fn.compute(img), target_hash)
    n_queries = 1

    rng = np.random.default_rng()
    max_basis = H * W
    all_indices = [(k, c) for c in range(3) for k in range(max_basis)]
    order = rng.permutation(len(all_indices))

    steps_done = 0
    for idx in order:
        if best_dist <= threshold or steps_done >= n_iter:
            break
        k, c = all_indices[idx]
        basis = _dct_basis(H, W, k, c)

        pos = np.clip(current + step_size * basis, 0, 255)
        neg = np.clip(current - step_size * basis, 0, 255)

        d_pos = hash_fn.distance(
            hash_fn.compute(Image.fromarray(pos.astype(np.uint8))), target_hash)
        d_neg = hash_fn.distance(
            hash_fn.compute(Image.fromarray(neg.astype(np.uint8))), target_hash)
        n_queries += 2

        if d_pos < best_dist:
            best_dist = d_pos
            current = pos
        elif d_neg < best_dist:
            best_dist = d_neg
            current = neg

        steps_done += 1

    l2 = normalised_l2(orig, current)
    return Image.fromarray(current.astype(np.uint8)), {
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
                                n_iter=200, step_size=12.0)
        attacked_images.append(atk)
        metrics.append(m)

    return {"attacked_images": attacked_images, "metrics": metrics}