"""SimBa (Simple Black-box Adversarial) attack against PDQ.

PDQ processes a grayscale 64×64 downsample, so low-frequency patterns
(large pixel blocks) have the most impact on the hash.  This SimBa variant
uses pixel-block basis vectors (random block positions) in addition to DCT
basis vectors, giving better coverage of PDQ's internal structure.
"""

import numpy as np
from PIL import Image


def entrypoint(context: dict) -> dict:
    hash_fn = context["hash_fn"]
    threshold = context["threshold"]
    sources: list[Image.Image] = context["source_images"]
    target_hashes: list = context["target_hashes"]

    attacked_images: list[Image.Image] = []
    metrics: list[dict] = []

    for img, target_hash in zip(sources, target_hashes):
        atk_img, m = _simba_attack(img, target_hash, hash_fn, threshold)
        attacked_images.append(atk_img)
        metrics.append(m)

    return {"attacked_images": attacked_images, "metrics": metrics}


def _block_basis(
    height: int,
    width: int,
    block_h: int,
    block_w: int,
    row: int,
    col: int,
    channel: int,
) -> np.ndarray:
    """Return a normalised basis vector that is 1 inside a block and 0 outside."""
    basis = np.zeros((height, width, 3), dtype=np.float32)
    r0, r1 = row, min(row + block_h, height)
    c0, c1 = col, min(col + block_w, width)
    basis[r0:r1, c0:c1, channel] = 1.0
    norm = np.linalg.norm(basis)
    if norm > 0:
        basis /= norm
    return basis


def _simba_attack(
    img: Image.Image,
    target_hash,
    hash_fn,
    threshold: int,
    n_iter: int = 300,
    step_size: float = 16.0,
    block_size: int = 16,
) -> tuple[Image.Image, dict]:
    """
    SimBa using random rectangular pixel blocks aligned to a coarse grid.

    Large blocks perturb many pixels simultaneously, moving the PDQ hash
    (which is computed over a 64×64 downsample) more efficiently than
    single-pixel steps.
    """
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

        # Random block position and channel
        row = row_starts[np.random.randint(len(row_starts))]
        col = col_starts[np.random.randint(len(col_starts))]
        ch = np.random.randint(3)

        basis = _block_basis(H, W, block_size, block_size, row, col, ch)

        pos = np.clip(current + step_size * basis, 0, 255)
        neg = np.clip(current - step_size * basis, 0, 255)

        d_pos = hash_fn.distance(
            hash_fn.compute(Image.fromarray(pos.astype(np.uint8))),
            target_hash,
        )
        d_neg = hash_fn.distance(
            hash_fn.compute(Image.fromarray(neg.astype(np.uint8))),
            target_hash,
        )
        n_queries += 2

        if d_pos < best_dist:
            best_dist = d_pos
            current = pos
        elif d_neg < best_dist:
            best_dist = d_neg
            current = neg

    l2 = float(
        np.linalg.norm((current - orig).flatten()) / np.sqrt(orig.size)
    )
    return Image.fromarray(current.astype(np.uint8)), {
        "success": best_dist <= threshold,
        "l2": l2,
        "n_queries": n_queries,
        "final_dist": float(best_dist),
    }
