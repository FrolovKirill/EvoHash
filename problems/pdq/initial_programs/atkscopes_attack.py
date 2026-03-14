"""AtkScopes — multi-scale patch-based attack on perceptual hash functions.

Implements a multi-resolution attack strategy inspired by the observation that
perceptual hash functions aggregate information at multiple spatial scales.
The attack operates by:

1. Dividing the image into overlapping patches at multiple resolutions.
2. Estimating the sensitivity of the hash output to perturbations in each
   patch (via finite-difference queries).
3. Concentrating perturbation budget on the most hash-sensitive patches.
4. Iteratively refining the perturbation at each scale.

This "scope" approach adaptively focuses attack energy where the PHF is most
vulnerable, rather than distributing noise uniformly.

Reference: Inspired by multi-scale adversarial analysis in content-moderation
attack literature.
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
        atk_img, m = _atkscopes_attack(img, target_hash, hash_fn, threshold)
        attacked_images.append(atk_img)
        metrics.append(m)

    return {"attacked_images": attacked_images, "metrics": metrics}


def _get_patch_slices(H: int, W: int, patch_size: int, stride: int):
    """Generate (row_slice, col_slice) for overlapping patches."""
    patches = []
    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            patches.append((slice(r, r + patch_size), slice(c, c + patch_size)))
    # Include bottom-right corner patches if stride doesn't cover them
    if (H - patch_size) % stride != 0:
        for c in range(0, W - patch_size + 1, stride):
            patches.append((slice(H - patch_size, H), slice(c, c + patch_size)))
    if (W - patch_size) % stride != 0:
        for r in range(0, H - patch_size + 1, stride):
            patches.append((slice(r, r + patch_size), slice(W - patch_size, W)))
    return patches


def _estimate_patch_sensitivity(
    current: np.ndarray,
    patches: list,
    target_hash,
    hash_fn,
    threshold: int,
    sigma: float = 10.0,
    n_probes: int = 2,
) -> tuple[list[float], int]:
    """Estimate how sensitive the hash is to perturbations in each patch."""
    sensitivities = []
    q = 0
    base_dist = hash_fn.distance(
        hash_fn.compute(Image.fromarray(current.astype(np.uint8))), target_hash
    )
    q += 1

    for rs, cs in patches:
        total_change = 0.0
        for _ in range(n_probes):
            probe = current.copy()
            noise = np.random.randn(*probe[rs, cs, :].shape).astype(np.float32) * sigma
            probe[rs, cs, :] = np.clip(probe[rs, cs, :] + noise, 0, 255)
            dist = hash_fn.distance(
                hash_fn.compute(Image.fromarray(probe.astype(np.uint8))), target_hash
            )
            q += 1
            total_change += abs(dist - base_dist)
        sensitivities.append(total_change / n_probes)

    return sensitivities, q


def _attack_patch(
    current: np.ndarray,
    rs,
    cs,
    target_hash,
    hash_fn,
    threshold: int,
    n_samples: int = 8,
    sigma: float = 6.0,
    lr: float = 4.0,
) -> tuple[np.ndarray, int]:
    """NES-style gradient estimate and update restricted to a single patch."""
    q = 0
    grad_patch = np.zeros_like(current[rs, cs, :])

    for _ in range(n_samples):
        v = np.random.randn(*grad_patch.shape).astype(np.float32)
        pos = current.copy()
        neg = current.copy()
        pos[rs, cs, :] = np.clip(pos[rs, cs, :] + sigma * v, 0, 255)
        neg[rs, cs, :] = np.clip(neg[rs, cs, :] - sigma * v, 0, 255)

        d_pos = hash_fn.distance(
            hash_fn.compute(Image.fromarray(pos.astype(np.uint8))), target_hash
        )
        d_neg = hash_fn.distance(
            hash_fn.compute(Image.fromarray(neg.astype(np.uint8))), target_hash
        )
        q += 2
        grad_patch += (d_pos - d_neg) * v

    grad_patch /= 2.0 * sigma * n_samples
    result = current.copy()
    result[rs, cs, :] = np.clip(
        result[rs, cs, :] - lr * grad_patch, 0, 255
    )
    return result, q


def _atkscopes_attack(
    img: Image.Image,
    target_hash,
    hash_fn,
    threshold: int,
    n_outer: int = 8,
    scales: list[int] | None = None,
) -> tuple[Image.Image, dict]:
    if scales is None:
        scales = [56, 28, 14]  # coarse → fine patch sizes

    orig = np.array(img).astype(np.float32)
    H, W, _ = orig.shape
    current = orig.copy()
    best = orig.copy()
    best_dist = hash_fn.distance(hash_fn.compute(img), target_hash)
    n_queries = 1

    for outer in range(n_outer):
        if best_dist <= threshold:
            break

        for patch_size in scales:
            if best_dist <= threshold:
                break

            stride = max(patch_size // 2, 1)
            if patch_size > H or patch_size > W:
                continue

            patches = _get_patch_slices(H, W, patch_size, stride)
            if not patches:
                continue

            # Estimate sensitivity of each patch
            sensitivities, q = _estimate_patch_sensitivity(
                current, patches, target_hash, hash_fn, threshold
            )
            n_queries += q

            # Attack top-k most sensitive patches
            n_attack = max(1, len(patches) // 4)
            top_indices = np.argsort(sensitivities)[-n_attack:][::-1]

            for idx in top_indices:
                if best_dist <= threshold:
                    break
                rs, cs = patches[idx]
                current, q = _attack_patch(
                    current, rs, cs, target_hash, hash_fn, threshold
                )
                n_queries += q

                dist = hash_fn.distance(
                    hash_fn.compute(Image.fromarray(current.astype(np.uint8))),
                    target_hash,
                )
                n_queries += 1
                if dist < best_dist:
                    best_dist = dist
                    best = current.copy()

    l2 = float(np.linalg.norm((best - orig).flatten()) / np.sqrt(orig.size))
    return Image.fromarray(best.astype(np.uint8)), {
        "success": best_dist <= threshold,
        "l2": l2,
        "n_queries": n_queries,
        "final_dist": float(best_dist),
    }
