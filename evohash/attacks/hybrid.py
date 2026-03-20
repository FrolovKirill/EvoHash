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


# ── JSHA hybrids (Madden et al., NeurIPS 2024 Workshop) ──────────────────────
# Reversed order: soft-label attack FIRST (unconstrained, fast collision),
# then HSJA SECOND (minimize L2 while maintaining collision).


def _jsha_single(src_img, tgt_img, target_hash, hash_fn, threshold,
                 soft_attack_fn, soft_kwargs=None, hsja_kwargs=None):
    """JSHA: soft-label phase (unconstrained) → HSJA phase (L2 minimization)."""
    source = np.array(src_img).astype(np.float32)

    soft_kw = soft_kwargs or {}
    hsja_kw = hsja_kwargs or {}

    # Phase 1: soft-label attack — find collision ignoring L2
    phase1_img, phase1_metrics = soft_attack_fn(
        src_img, target_hash, hash_fn, threshold, **soft_kw
    )
    q1 = phase1_metrics["n_queries"]
    phase1_arr = np.array(phase1_img).astype(np.float32)

    # If Phase 1 didn't find collision, return its result directly
    if not phase1_metrics["success"]:
        l2 = normalised_l2(source, phase1_arr)
        return phase1_img, {
            "success": False,
            "l2": l2,
            "n_queries": q1,
            "final_dist": phase1_metrics["final_dist"],
        }

    # Phase 2: HSJA from phase1 result → minimize L2
    # Start HSJA with phase1_arr as initial point (already colliding)
    hsja_result, q2 = hsja_phase(
        source, phase1_arr, target_hash, hash_fn, threshold, **hsja_kw
    )

    best_dist = query_distance(hsja_result, target_hash, hash_fn)
    n_queries = q1 + q2 + 1
    l2 = normalised_l2(source, hsja_result)

    return Image.fromarray(np.clip(hsja_result, 0, 255).astype(np.uint8)), {
        "success": best_dist <= threshold,
        "l2": l2,
        "n_queries": n_queries,
        "final_dist": float(best_dist),
    }


def _run_jsha(context, soft_attack_fn, soft_kwargs=None, hsja_kwargs=None):
    """Generic JSHA runner."""
    hash_fn = context["hash_fn"]
    threshold = context["threshold"]
    sources = context["source_images"]
    target_hashes = context["target_hashes"]
    target_images = context["target_images"]

    attacked_images, metrics = [], []
    for src, tgt, th in zip(sources, target_images, target_hashes):
        atk, m = _jsha_single(
            src, tgt, th, hash_fn, threshold, soft_attack_fn,
            soft_kwargs=soft_kwargs, hsja_kwargs=hsja_kwargs,
        )
        attacked_images.append(atk)
        metrics.append(m)

    return {"attacked_images": attacked_images, "metrics": metrics}


def run_nes_jsha(context, soft_kwargs=None, hsja_kwargs=None):
    """JSHA: NES (unconstrained) → HSJA (L2 minimization)."""
    from .nes import _attack_single as nes_attack
    return _run_jsha(context, nes_attack,
                     soft_kwargs=soft_kwargs, hsja_kwargs=hsja_kwargs)


def run_simba_jsha(context, soft_kwargs=None, hsja_kwargs=None):
    """JSHA: SimBa DCT (unconstrained) → HSJA (L2 minimization)."""
    from .simba import _attack_single_dct as simba_attack
    return _run_jsha(context, simba_attack,
                     soft_kwargs=soft_kwargs, hsja_kwargs=hsja_kwargs)


def run_zo_signsgd_jsha(context, soft_kwargs=None, hsja_kwargs=None):
    """JSHA: ZO-SignSGD (unconstrained) → HSJA (L2 minimization)."""
    from .zo_signsgd import _attack_single as zo_attack
    return _run_jsha(context, zo_attack,
                     soft_kwargs=soft_kwargs, hsja_kwargs=hsja_kwargs)
