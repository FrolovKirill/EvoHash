"""Hybrid two-phase attacks: HSJA + score-based refinement.

Phase 1 — HSJA finds a collision by boundary-walking from the target image.
Phase 2 — A score-based refiner (NES, SimBa, or ZO-Sign-SGD) reduces L2
distortion while maintaining or improving the collision.

This module provides three standard hybrids plus a generic runner.
"""

import numpy as np
from PIL import Image

from .hsja import hsja_phase
from .nes import nes_refine_single
from .simba import simba_refine_single
from .zo_signsgd import zo_signsgd_refine_single
from .utils import normalised_l2, query_distance


def _hybrid_single(src_img, tgt_img, target_hash, hash_fn, threshold,
                   refine_fn, hsja_kwargs=None, refine_kwargs=None):
    """Generic hybrid: HSJA phase + refine_fn phase."""
    source = np.array(src_img).astype(np.float32)
    target = np.array(tgt_img).astype(np.float32)

    hsja_kw = hsja_kwargs or {}
    refine_kw = refine_kwargs or {}

    # Phase 1: HSJA
    hsja_result, q1 = hsja_phase(
        source, target, target_hash, hash_fn, threshold, **hsja_kw
    )

    # Phase 2: Refinement
    best, best_dist, q2 = refine_fn(
        source, hsja_result, target_hash, hash_fn, threshold, **refine_kw
    )

    n_queries = q1 + q2
    l2 = normalised_l2(source, best)

    return Image.fromarray(np.clip(best, 0, 255).astype(np.uint8)), {
        "success": best_dist <= threshold,
        "l2": l2,
        "n_queries": n_queries,
        "final_dist": float(best_dist),
    }


def _run_hybrid(context, refine_fn, hsja_kwargs=None, refine_kwargs=None):
    """Generic hybrid runner."""
    hash_fn = context["hash_fn"]
    threshold = context["threshold"]
    sources = context["source_images"]
    target_hashes = context["target_hashes"]
    target_images = context["target_images"]

    attacked_images, metrics = [], []
    for src, tgt, th in zip(sources, target_images, target_hashes):
        atk, m = _hybrid_single(
            src, tgt, th, hash_fn, threshold, refine_fn,
            hsja_kwargs=hsja_kwargs, refine_kwargs=refine_kwargs,
        )
        attacked_images.append(atk)
        metrics.append(m)

    return {"attacked_images": attacked_images, "metrics": metrics}


# ── Specific hybrids ──────────────────────────────────────────────────────────


def run_nes_hsja(context, hsja_kwargs=None, refine_kwargs=None):
    """NES + HSJA hybrid."""
    return _run_hybrid(context, nes_refine_single,
                       hsja_kwargs=hsja_kwargs, refine_kwargs=refine_kwargs)


def run_simba_hsja(context, hsja_kwargs=None, refine_kwargs=None):
    """SimBa + HSJA hybrid."""
    return _run_hybrid(context, simba_refine_single,
                       hsja_kwargs=hsja_kwargs, refine_kwargs=refine_kwargs)


def run_zo_signsgd_hsja(context, hsja_kwargs=None, refine_kwargs=None):
    """ZO-Sign-SGD + HSJA hybrid."""
    return _run_hybrid(context, zo_signsgd_refine_single,
                       hsja_kwargs=hsja_kwargs, refine_kwargs=refine_kwargs)
