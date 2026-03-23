"""SimBa (block basis) attack tuned for PDQ.

PDQ processes a 64x64 grayscale downsample, so large pixel blocks
are more efficient than DCT basis.

Reference: Guo et al., ICML 2019.
"""

import numpy as np
from PIL import Image


def normalised_l2(original, perturbed):
    diff = perturbed.astype(float) - original.astype(float)
    return float(np.linalg.norm(diff.flatten()) / np.sqrt(original.size))


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


def _attack_single(img, target_hash, hash_fn, threshold,
                   n_iter=300, step_size=16.0, block_size=16):
    """SimBa with pixel-block basis — deterministic permutation without replacement."""
    orig = np.array(img).astype(np.float32)
    H, W, _ = orig.shape
    current = orig.copy()
    best_dist = hash_fn.distance(hash_fn.compute(img), target_hash)
    n_queries = 1

    row_starts = list(range(0, H, block_size))
    col_starts = list(range(0, W, block_size))

    rng = np.random.default_rng()
    all_indices = [(r, c, ch) for r in row_starts for c in col_starts for ch in range(3)]
    order = rng.permutation(len(all_indices))

    steps_done = 0
    for idx in order:
        if best_dist <= threshold or steps_done >= n_iter:
            break
        row, col, ch = all_indices[idx]

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
                                n_iter=300, step_size=16.0, block_size=16)
        attacked_images.append(atk)
        metrics.append(m)

    return {"attacked_images": attacked_images, "metrics": metrics}
