"""SimBa (Simple Black-box Adversarial) attack.

Coordinate descent using either DCT basis vectors or pixel-block basis.
DCT basis is better for pHash/NeuralHash; block basis is better for PDQ.

Reference: Guo et al., ICML 2019.
"""

import numpy as np
from PIL import Image

from .utils import normalised_l2


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


def _block_basis(height, width, block_h, block_w, row, col, channel):
    """Return a normalised block basis vector."""
    basis = np.zeros((height, width, 3), dtype=np.float32)
    r0, r1 = row, min(row + block_h, height)
    c0, c1 = col, min(col + block_w, width)
    basis[r0:r1, c0:c1, channel] = 1.0
    norm = np.linalg.norm(basis)
    if norm > 0:
        basis /= norm
    return basis


def _attack_single_dct(img, target_hash, hash_fn, threshold,
                       n_iter=200, step_size=12.0, n_candidates=10):
    """SimBa with DCT basis."""
    orig = np.array(img).astype(np.float32)
    H, W, _ = orig.shape
    current = orig.copy()
    best_dist = hash_fn.distance(hash_fn.compute(img), target_hash)
    n_queries = 1

    max_basis = H * W
    basis_indices = np.random.randint(0, max_basis, size=(n_iter, n_candidates))
    channel_indices = np.random.randint(0, 3, size=(n_iter, n_candidates))

    for i in range(n_iter):
        if best_dist <= threshold:
            break
        for j in range(n_candidates):
            k = int(basis_indices[i, j])
            c = int(channel_indices[i, j])
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

            if best_dist <= threshold:
                break

    l2 = normalised_l2(orig, current)
    return Image.fromarray(current.astype(np.uint8)), {
        "success": best_dist <= threshold,
        "l2": l2,
        "n_queries": n_queries,
        "final_dist": float(best_dist),
    }


def _attack_single_block(img, target_hash, hash_fn, threshold,
                         n_iter=300, step_size=16.0, block_size=16):
    """SimBa with pixel-block basis (better for PDQ)."""
    orig = np.array(img).astype(np.float32)
    H, W, _ = orig.shape
    current = orig.copy()
    best_dist = hash_fn.distance(hash_fn.compute(img), target_hash)
    n_queries = 1

    row_starts = list(range(0, H, block_size))
    col_starts = list(range(0, W, block_size))

    for _ in range(n_iter):
        if best_dist <= threshold:
            break

        row = row_starts[np.random.randint(len(row_starts))]
        col = col_starts[np.random.randint(len(col_starts))]
        ch = np.random.randint(3)

        basis = _block_basis(H, W, block_size, block_size, row, col, ch)

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

    l2 = normalised_l2(orig, current)
    return Image.fromarray(current.astype(np.uint8)), {
        "success": best_dist <= threshold,
        "l2": l2,
        "n_queries": n_queries,
        "final_dist": float(best_dist),
    }


def simba_refine_single(source, start, target_hash, hash_fn, threshold,
                        n_iter=120, step_size=8.0):
    """SimBa DCT refinement phase for hybrid attacks.
    Returns (best, best_dist, n_queries)."""
    H, W, _ = source.shape
    current = start.copy()
    best_dist = hash_fn.distance(
        hash_fn.compute(Image.fromarray(current.astype(np.uint8))), target_hash
    )
    q = 1

    max_basis = H * W
    basis_indices = np.random.randint(0, max_basis, size=n_iter)
    channel_indices = np.random.randint(0, 3, size=n_iter)

    for i in range(n_iter):
        if best_dist > threshold:
            basis = _dct_basis(H, W, int(basis_indices[i]), int(channel_indices[i]))
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


def run(context, basis="dct", **kwargs):
    """Run SimBa attack. Use basis='dct' or basis='block'."""
    hash_fn = context["hash_fn"]
    threshold = context["threshold"]
    sources = context["source_images"]
    target_hashes = context["target_hashes"]

    attack_fn = _attack_single_dct if basis == "dct" else _attack_single_block

    attacked_images, metrics = [], []
    for img, th in zip(sources, target_hashes):
        atk, m = attack_fn(img, th, hash_fn, threshold, **kwargs)
        attacked_images.append(atk)
        metrics.append(m)

    return {"attacked_images": attacked_images, "metrics": metrics}
