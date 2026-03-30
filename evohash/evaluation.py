"""Full benchmark evaluation utilities for EvoHash.

Used by ``scripts/evaluate.py`` to compute all primary and secondary metrics
over the full 100-image-pair test set after evolution is complete.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from evohash.dataset import image_to_array, load_image_pairs
from evohash.phf.base import PHFWrapper


# ── Primary metrics ───────────────────────────────────────────────────────────


def compute_asr(metrics: list[dict]) -> float:
    """Attack Success Rate — fraction of successful collisions."""
    if not metrics:
        return 0.0
    return float(np.mean([bool(m.get("success", False)) for m in metrics]))


def compute_mean_l2(
    originals: list[np.ndarray],
    attacked: list[np.ndarray],
) -> float:
    """
    Mean normalised L2 distance per pixel across all image pairs.

    The normalisation divides by ``sqrt(H * W * C)`` so the value is
    independent of image resolution.
    """
    l2s = []
    for orig, atk in zip(originals, attacked):
        diff = atk.astype(float) - orig.astype(float)
        l2 = np.linalg.norm(diff.flatten()) / np.sqrt(orig.size)
        l2s.append(l2)
    return float(np.mean(l2s)) if l2s else float("nan")


LPIPS_WEIGHT = 50.0  # scales LPIPS (~[0,1]) to be comparable with L2 (~[20,90])


def compute_efficiency(asr: float, mean_l2: float, mean_lpips: float = float("nan")) -> float:
    """Primary fitness metric: ASR / (mean_L2 + λ*LPIPS + ε).

    Falls back to ASR / (mean_L2 + ε) if LPIPS is unavailable (nan).
    """
    if np.isnan(mean_lpips):
        return asr / (mean_l2 + 1e-6)
    return asr / (mean_l2 + LPIPS_WEIGHT * mean_lpips + 1e-6)


# ── Secondary metrics ─────────────────────────────────────────────────────────


def compute_mean_queries(metrics: list[dict]) -> float:
    """Average number of hash queries per attack."""
    if not metrics:
        return float("nan")
    return float(np.mean([m.get("n_queries", 0) for m in metrics]))


def compute_lpips(
    originals: list[np.ndarray],
    attacked: list[np.ndarray],
) -> float:
    """
    Mean LPIPS (AlexNet) perceptual distance.

    Requires ``torch`` and ``lpips`` packages.  Returns ``float('nan')`` if
    either is unavailable.
    """
    try:
        import lpips  # type: ignore
        import torch
    except ImportError:
        return float("nan")

    loss_fn = lpips.LPIPS(net="alex", verbose=False)

    scores = []
    for orig, atk in zip(originals, attacked):
        # LPIPS expects tensors in [-1, 1], shape (1, 3, H, W)
        t_orig = _to_lpips_tensor(orig)
        t_atk = _to_lpips_tensor(atk)
        with torch.no_grad():
            score = loss_fn(t_orig, t_atk).item()
        scores.append(score)

    return float(np.mean(scores)) if scores else float("nan")


def _to_lpips_tensor(arr: np.ndarray):
    """Convert HxWx3 uint8/float array to a normalised LPIPS tensor."""
    import torch

    arr_f = arr.astype(np.float32) / 127.5 - 1.0  # [0,255] → [-1,1]
    tensor = torch.from_numpy(arr_f).permute(2, 0, 1).unsqueeze(0)
    return tensor


# ── Transferability ───────────────────────────────────────────────────────────


def compute_transferability(
    attacked_images: list[np.ndarray],
    originals: list[np.ndarray],
    source_phf: PHFWrapper,
    target_phf: PHFWrapper,
    target_hashes: list[Any],
) -> float:
    """
    Fraction of attacks evolved for *source_phf* that also collide on
    *target_phf* when applied to the same images.
    """
    successes = []
    for atk_arr, orig_arr, tgt_hash in zip(attacked_images, originals, target_hashes):
        atk_img = Image.fromarray(np.clip(atk_arr, 0, 255).astype(np.uint8))
        atk_hash = target_phf.compute(atk_img)
        orig_img = Image.fromarray(np.clip(orig_arr, 0, 255).astype(np.uint8))
        orig_hash = target_phf.compute(orig_img)
        # Success: attacked hash is closer to target than the original was
        dist_atk = target_phf.distance(atk_hash, tgt_hash)
        successes.append(float(dist_atk <= target_phf.threshold))
    return float(np.mean(successes)) if successes else 0.0


# ── Full benchmark runner ─────────────────────────────────────────────────────


def run_benchmark(
    attack_fn,
    phf: PHFWrapper,
    data_dir: Path,
    n_pairs: int = 100,
    image_size: tuple[int, int] = (224, 224),
) -> dict:
    """
    Run *attack_fn* on *n_pairs* image pairs and return a metrics dict.

    Args:
        attack_fn: Callable ``(context: dict) -> dict`` implementing the attack.
        phf: The target PHF wrapper.
        data_dir: Directory containing validation images.
        n_pairs: Number of image pairs to evaluate.
        image_size: Images are resized to this resolution.

    Returns:
        Dict with keys: asr, mean_l2, efficiency, mean_queries, lpips, time_per_attack.
    """
    pairs = load_image_pairs(data_dir, n_pairs=n_pairs, image_size=image_size)
    sources = [p[0] for p in pairs]
    targets = [p[1] for p in pairs]
    target_hashes = [phf.compute(img) for img in targets]

    context = {
        "hash_fn": phf,
        "threshold": phf.threshold,
        "source_images": sources,
        "target_hashes": target_hashes,
        "target_images": targets,
    }

    t0 = time.time()
    result = attack_fn(context)
    elapsed = time.time() - t0

    attacked_pil: list[Image.Image] = result.get("attacked_images", [])
    per_image_metrics: list[dict] = result.get("metrics", [])

    originals = [image_to_array(img) for img in sources]
    attacked_arrs = [image_to_array(img) for img in attacked_pil]

    asr = compute_asr(per_image_metrics)
    mean_l2 = compute_mean_l2(originals, attacked_arrs)
    mean_queries = compute_mean_queries(per_image_metrics)
    lpips_score = compute_lpips(originals, attacked_arrs)
    efficiency = compute_efficiency(asr, mean_l2, lpips_score)
    time_per_attack = elapsed / max(len(sources), 1)

    return {
        "asr": asr,
        "mean_l2": mean_l2,
        "efficiency": efficiency,
        "mean_queries": mean_queries,
        "lpips": lpips_score,
        "time_per_attack": time_per_attack,
    }
