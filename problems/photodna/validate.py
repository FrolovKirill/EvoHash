"""Fitness evaluator for the PhotoDNA collision attack problem."""

from __future__ import annotations

import numpy as np
from PIL import Image


# ── LPIPS model (lazy-loaded once) ───────────────────────────────────────────

_lpips_fn = None
LPIPS_WEIGHT = 50.0


def _get_lpips_fn():
    global _lpips_fn
    if _lpips_fn is None:
        try:
            import lpips as _lpips_mod
            _lpips_fn = _lpips_mod.LPIPS(net="alex", verbose=False)
        except ImportError:
            _lpips_fn = False
    return _lpips_fn if _lpips_fn is not False else None


def _compute_lpips_single(orig_arr: np.ndarray, atk_arr: np.ndarray) -> float:
    fn = _get_lpips_fn()
    if fn is None:
        return float("nan")
    import torch
    def _to_tensor(arr):
        t = torch.from_numpy(arr.astype(np.float32) / 127.5 - 1.0)
        return t.permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        return float(fn(_to_tensor(orig_arr), _to_tensor(atk_arr)).item())


_SENTINEL = {
    "is_valid": 0.0,
    "efficiency": -1000.0,
    "asr": 0.0,
    "l2": 1_000_000.0,
    "lpips": 1.0,
    "n_queries": 0.0,
    "mean_final_dist": 3855.0,
}


def validate(context: dict, data: dict) -> dict:
    """Evaluate PhotoDNA attack results and return fitness metrics."""
    if not isinstance(data, dict):
        return _SENTINEL.copy()

    attacked: list = data.get("attacked_images", [])
    per_image: list[dict] = data.get("metrics", [])
    sources: list[Image.Image] = context["source_images"]
    hash_fn = context["hash_fn"]
    threshold: int = context["threshold"]

    if not attacked or len(attacked) != len(sources):
        return _SENTINEL.copy()
    if len(per_image) != len(sources):
        return _SENTINEL.copy()

    successes: list[float] = []
    l2_values: list[float] = []
    lpips_values: list[float] = []
    query_counts: list[float] = []
    final_dists: list[float] = []

    for orig_img, atk_img, m in zip(sources, attacked, per_image):
        try:
            orig_arr = np.array(orig_img).astype(float)
            atk_arr = np.array(atk_img).astype(float)

            atk_pil = atk_img if isinstance(atk_img, Image.Image) else Image.fromarray(
                np.clip(atk_arr, 0, 255).astype(np.uint8)
            )
            atk_hash = hash_fn.compute(atk_pil)
            target_hash = context["target_hashes"][len(successes)]
            dist = hash_fn.distance(atk_hash, target_hash)
            success = float(dist <= threshold)

            l2 = float(
                np.linalg.norm((atk_arr - orig_arr).flatten())
                / np.sqrt(orig_arr.size)
            )

            lp = _compute_lpips_single(orig_arr, atk_arr)

            successes.append(success)
            l2_values.append(l2)
            lpips_values.append(lp)
            query_counts.append(float(m.get("n_queries", 0)))
            final_dists.append(float(dist))

        except Exception:
            successes.append(0.0)
            l2_values.append(1e6)
            lpips_values.append(1.0)
            query_counts.append(0.0)
            final_dists.append(3855.0)

    asr = float(np.mean(successes))
    mean_l2 = float(np.mean(l2_values))
    mean_lpips = float(np.mean(lpips_values))
    mean_queries = float(np.mean(query_counts))
    mean_final_dist = float(np.mean(final_dists))

    if np.isnan(mean_lpips):
        efficiency = asr / (mean_l2 + 1e-6)
    else:
        efficiency = asr / (mean_l2 + LPIPS_WEIGHT * mean_lpips + 1e-6)

    result = {
        "is_valid": 1.0,
        "efficiency": efficiency,
        "asr": asr,
        "l2": mean_l2,
        "lpips": mean_lpips,
        "n_queries": mean_queries,
        "mean_final_dist": mean_final_dist,
    }

    try:
        from evohash.reporter import log_iteration
        log_iteration(context.get("_report_dir"), context.get("_phf_name", "photodna"), result, context, data)
    except Exception:
        pass

    return result
