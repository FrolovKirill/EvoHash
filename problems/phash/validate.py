"""Fitness evaluator for the pHash collision attack problem.

gigaevo calls ``validate(data, context)`` after running ``entrypoint(context)``
in the evolved program.  The return dict's values are stored as program metrics.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


# ── Sentinel values (returned when a program is invalid) ─────────────────────

_SENTINEL = {
    "is_valid": 0.0,
    "efficiency": -1000.0,
    "asr": 0.0,
    "l2": 1_000_000.0,
    "n_queries": 0.0,
    "mean_final_dist": 64.0,
}


def validate(context: dict, data: dict) -> dict:
    """
    Evaluate pHash attack results and return fitness metrics.

    Args:
        context: Dict produced by ``context.py::build_context()``.
                 Gigaevo passes context as the first argument.
        data: Return value of ``entrypoint(context)`` from the evolved program.
              Expected keys:
              - ``attacked_images``: list of PIL.Image (perturbed sources)
              - ``metrics``: list of per-image dicts with keys
                  ``success`` (bool), ``l2`` (float), ``n_queries`` (int),
                  ``final_dist`` (float, optional)

    Returns:
        Dict with metrics consumed by gigaevo:
        - ``is_valid``   : 1.0 if the program ran without errors, else 0.0
        - ``efficiency`` : ASR / (mean_L2 + ε)   ← PRIMARY metric
        - ``asr``        : Attack Success Rate ∈ [0, 1]
        - ``l2``         : Mean normalised L2 distortion
        - ``n_queries``  : Mean hash queries per image
    """
    if not isinstance(data, dict):
        return _SENTINEL.copy()

    attacked: list = data.get("attacked_images", [])
    per_image: list[dict] = data.get("metrics", [])
    sources: list[Image.Image] = context["source_images"]
    hash_fn = context["hash_fn"]
    threshold: int = context["threshold"]

    # Validate output structure
    if not attacked or len(attacked) != len(sources):
        return _SENTINEL.copy()
    if len(per_image) != len(sources):
        return _SENTINEL.copy()

    successes: list[float] = []
    l2_values: list[float] = []
    query_counts: list[float] = []
    final_dists: list[float] = []

    for orig_img, atk_img, m in zip(sources, attacked, per_image):
        try:
            orig_arr = np.array(orig_img).astype(float)
            atk_arr = np.array(atk_img).astype(float)

            # Re-verify success against the hash function (don't trust the program's
            # self-report alone — guard against trivially faked metrics).
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

            successes.append(success)
            l2_values.append(l2)
            query_counts.append(float(m.get("n_queries", 0)))
            final_dists.append(float(dist))

        except Exception:
            # Any error in a single image → treat as failed attack
            successes.append(0.0)
            l2_values.append(float(np.linalg.norm(np.zeros(3)) + 1e6))
            query_counts.append(0.0)
            final_dists.append(64.0)

    asr = float(np.mean(successes))
    mean_l2 = float(np.mean(l2_values))
    mean_queries = float(np.mean(query_counts))
    mean_final_dist = float(np.mean(final_dists))
    efficiency = asr / (mean_l2 + 1e-6)

    result = {
        "is_valid": 1.0,
        "efficiency": efficiency,
        "asr": asr,
        "l2": mean_l2,
        "n_queries": mean_queries,
        "mean_final_dist": mean_final_dist,
    }

    try:
        from evohash.reporter import log_iteration
        log_iteration(context.get("_report_dir"), context.get("_phf_name", "phash"), result, context, data)
    except Exception:
        pass

    return result
