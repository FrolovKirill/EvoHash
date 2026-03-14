"""HopSkipJump Attack (HSJA) adapted for perceptual hash collisions.

HSJA is a decision-based attack: it only needs a binary oracle ("does the
hash collide?").  It starts from a known colliding point (e.g. the target
image itself — whose hash trivially matches) and iteratively moves toward the
source image while staying within the collision boundary, minimising L2
distortion.

Reference: Chen, Jordan & Wainwright, "HopSkipJumpAttack: A Query-Efficient
Decision-Based Attack", IEEE S&P 2020.
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
        atk_img, m = _hsja_attack(src, tgt, target_hash, hash_fn, threshold)
        attacked_images.append(atk_img)
        metrics.append(m)

    return {"attacked_images": attacked_images, "metrics": metrics}


def _collides(arr: np.ndarray, target_hash, hash_fn, threshold: int) -> bool:
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return hash_fn.distance(hash_fn.compute(img), target_hash) <= threshold


def _binary_search(
    source: np.ndarray,
    colliding: np.ndarray,
    target_hash,
    hash_fn,
    threshold: int,
    steps: int = 10,
) -> tuple[np.ndarray, int]:
    """Binary search along the line source→colliding to find the boundary."""
    lo, hi = 0.0, 1.0
    queries = 0
    for _ in range(steps):
        mid = (lo + hi) / 2.0
        candidate = (1 - mid) * source + mid * colliding
        queries += 1
        if _collides(candidate, target_hash, hash_fn, threshold):
            hi = mid
        else:
            lo = mid
    boundary = (1 - hi) * source + hi * colliding
    return boundary, queries


def _estimate_gradient(
    current: np.ndarray,
    source: np.ndarray,
    target_hash,
    hash_fn,
    threshold: int,
    n_samples: int = 30,
    delta: float = 1.0,
) -> tuple[np.ndarray, int]:
    """Estimate the gradient of the decision boundary via Monte-Carlo sampling."""
    queries = 0
    direction = current - source
    norm_dir = np.linalg.norm(direction.flatten())
    if norm_dir < 1e-8:
        return np.zeros_like(current), 0

    grad = np.zeros_like(current)
    for _ in range(n_samples):
        rv = np.random.randn(*current.shape).astype(np.float32)
        rv -= np.dot(rv.flatten(), direction.flatten()) / (norm_dir ** 2) * direction
        rv_norm = np.linalg.norm(rv.flatten())
        if rv_norm < 1e-8:
            continue
        rv = rv / rv_norm

        candidate = current + delta * rv
        queries += 1
        if _collides(candidate, target_hash, hash_fn, threshold):
            grad += rv

    return grad, queries


def _hsja_attack(
    src_img: Image.Image,
    tgt_img: Image.Image,
    target_hash,
    hash_fn,
    threshold: int,
    n_iter: int = 40,
    bs_steps: int = 10,
    grad_samples: int = 30,
) -> tuple[Image.Image, dict]:
    source = np.array(src_img).astype(np.float32)
    # Start from target image (guaranteed collision with its own hash)
    colliding = np.array(tgt_img).astype(np.float32)
    n_queries = 0

    # Initial binary search to find boundary
    current, q = _binary_search(
        source, colliding, target_hash, hash_fn, threshold, bs_steps
    )
    n_queries += q

    best = current.copy()
    best_l2 = float(np.linalg.norm((best - source).flatten()))

    for step in range(n_iter):
        # Estimate gradient direction at boundary
        grad, q = _estimate_gradient(
            current, source, target_hash, hash_fn, threshold, grad_samples
        )
        n_queries += q

        grad_norm = np.linalg.norm(grad.flatten())
        if grad_norm < 1e-8:
            continue

        grad = grad / grad_norm

        # Step size decay
        step_size = max(
            np.linalg.norm((current - source).flatten()) / (step + 1) * 0.5,
            0.5,
        )

        # Move along gradient (toward source, staying colliding)
        candidate = current + step_size * grad
        candidate = np.clip(candidate, 0, 255)

        # Project back to boundary via binary search
        if _collides(candidate, target_hash, hash_fn, threshold):
            current, q = _binary_search(
                source, candidate, target_hash, hash_fn, threshold, bs_steps
            )
            n_queries += q
        else:
            # If we left the collision region, binary search between current and candidate
            current, q = _binary_search(
                source, current, target_hash, hash_fn, threshold, bs_steps
            )
            n_queries += q

        l2 = float(np.linalg.norm((current - source).flatten()))
        if l2 < best_l2 and _collides(current, target_hash, hash_fn, threshold):
            n_queries += 1
            best_l2 = l2
            best = current.copy()

    norm_l2 = float(np.linalg.norm((best - source).flatten()) / np.sqrt(source.size))
    success = _collides(best, target_hash, hash_fn, threshold)
    n_queries += 1

    return Image.fromarray(np.clip(best, 0, 255).astype(np.uint8)), {
        "success": success,
        "l2": norm_l2,
        "n_queries": n_queries,
        "final_dist": float(
            hash_fn.distance(
                hash_fn.compute(Image.fromarray(np.clip(best, 0, 255).astype(np.uint8))),
                target_hash,
            )
        ),
    }
