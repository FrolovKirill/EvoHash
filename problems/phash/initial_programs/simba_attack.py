"""SimBa (Simple Black-box Adversarial) attack against pHash.

SimBa performs coordinate descent in the DCT basis of the image.  At each
step it randomly draws a DCT basis vector, adds a fixed step in the +/−
direction that most reduces the hash distance, and accumulates the perturbation.

Adapting SimBa to the perceptual hashing setting (rather than classifier
confidence) works especially well for pHash because pHash itself relies on DCT
coefficients, so DCT-basis perturbations are structurally aligned with the
attack objective.

Reference: Guo et al., "Simple Black-box Adversarial Attacks", ICML 2019.
"""

import numpy as np
from PIL import Image


def entrypoint(context: dict) -> dict:
    """Run SimBa (DCT-basis) attack against pHash for every image."""
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


def _dct_basis(height: int, width: int, k: int, c: int) -> np.ndarray:
    """
    Return the k-th 2D DCT basis vector for an image of shape (H, W, 3).

    Only channel *c* is non-zero so that each direction affects one channel.
    The basis is ordered by spatial frequency (low-frequency first).
    """
    # Build a flat list of (u, v) DCT frequency pairs sorted by energy
    pairs = [(u, v) for u in range(height) for v in range(width)]
    pairs.sort(key=lambda uv: uv[0] ** 2 + uv[1] ** 2)

    u, v = pairs[k % len(pairs)]

    # Construct a (H, W) DCT basis image for frequency (u, v)
    rows = np.arange(height).reshape(-1, 1)
    cols = np.arange(width).reshape(1, -1)
    basis_2d = np.cos(np.pi * u * (2 * rows + 1) / (2 * height)) * \
               np.cos(np.pi * v * (2 * cols + 1) / (2 * width))
    norm = np.linalg.norm(basis_2d)
    if norm > 0:
        basis_2d /= norm

    # Embed into a 3-channel image (only channel c is set)
    basis = np.zeros((height, width, 3), dtype=np.float32)
    basis[:, :, c] = basis_2d
    return basis


def _simba_attack(
    img: Image.Image,
    target_hash,
    hash_fn,
    threshold: int,
    n_iter: int = 200,
    step_size: float = 12.0,
    n_candidates: int = 10,
) -> tuple[Image.Image, dict]:
    """
    SimBa attack using random DCT basis vectors.

    At each iteration, randomly samples *n_candidates* DCT basis directions,
    evaluates both +step and -step, and applies whichever reduces hash distance.
    """
    orig = np.array(img).astype(np.float32)
    H, W, _ = orig.shape
    current = orig.copy()
    best_dist = hash_fn.distance(hash_fn.compute(img), target_hash)
    n_queries = 1

    # Pre-sample random basis indices for all iterations
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

            if best_dist <= threshold:
                break

    l2 = float(
        np.linalg.norm((current - orig).flatten()) / np.sqrt(orig.size)
    )
    return Image.fromarray(current.astype(np.uint8)), {
        "success": best_dist <= threshold,
        "l2": l2,
        "n_queries": n_queries,
        "final_dist": float(best_dist),
    }
