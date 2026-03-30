"""Configurable efficiency metric with multiple distortion combination modes.

All settings are read from environment variables (typically set in ``.env``):

- ``EFFICIENCY_MODE``     — how L2 and LPIPS are combined (default: ``weighted_sum``)
- ``EFFICIENCY_L2_WEIGHT``    — multiplier for L2 (default: ``1.0``)
- ``EFFICIENCY_LPIPS_WEIGHT`` — multiplier for LPIPS (default: ``50.0``)

Supported modes
~~~~~~~~~~~~~~~
- **weighted_sum** : ``d = α*L2 + β*LPIPS``
- **harmonic**     : ``d = 2*a*b / (a + b + ε)``  where ``a = α*L2, b = β*LPIPS``
- **geometric**    : ``d = sqrt(a * b)``
- **l2_only**      : ``d = L2``  (ignores LPIPS entirely)
- **lpips_only**   : ``d = LPIPS``  (ignores L2 entirely)

Efficiency is always: ``ASR / (d + ε)``.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass


# ── LPIPS model (lazy-loaded once per process) ───────────────────────────────

_lpips_fn = None


def _get_lpips_fn():
    global _lpips_fn
    if _lpips_fn is None:
        try:
            import lpips as _lpips_mod
            _lpips_fn = _lpips_mod.LPIPS(net="alex", verbose=False)
        except ImportError:
            _lpips_fn = False  # sentinel: unavailable
    return _lpips_fn if _lpips_fn is not False else None


def compute_lpips_single(orig_arr: np.ndarray, atk_arr: np.ndarray) -> float:
    """Compute LPIPS between two HxWx3 arrays. Returns nan if unavailable."""
    fn = _get_lpips_fn()
    if fn is None:
        return float("nan")
    import torch

    def _to_tensor(arr):
        t = torch.from_numpy(arr.astype(np.float32) / 127.5 - 1.0)
        return t.permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        return float(fn(_to_tensor(orig_arr), _to_tensor(atk_arr)).item())


# ── Read config from environment ─────────────────────────────────────────────

def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def get_config() -> dict:
    """Return current efficiency configuration from environment variables."""
    return {
        "mode": os.environ.get("EFFICIENCY_MODE", "weighted_sum").lower().strip(),
        "l2_weight": _env_float("EFFICIENCY_L2_WEIGHT", 1.0),
        "lpips_weight": _env_float("EFFICIENCY_LPIPS_WEIGHT", 50.0),
    }


# ── Distortion combiners ────────────────────────────────────────────────────

_EPS = 1e-6


def _weighted_sum(l2: float, lpips: float, alpha: float, beta: float) -> float:
    return alpha * l2 + beta * lpips


def _harmonic(l2: float, lpips: float, alpha: float, beta: float) -> float:
    a = alpha * l2
    b = beta * lpips
    if a + b < _EPS:
        return 0.0
    return 2.0 * a * b / (a + b + _EPS)


def _geometric(l2: float, lpips: float, alpha: float, beta: float) -> float:
    a = alpha * l2
    b = beta * lpips
    return math.sqrt(max(a, 0.0) * max(b, 0.0))


def _l2_only(l2: float, lpips: float, alpha: float, beta: float) -> float:
    return l2


def _lpips_only(l2: float, lpips: float, alpha: float, beta: float) -> float:
    return lpips


_MODES = {
    "weighted_sum": _weighted_sum,
    "harmonic": _harmonic,
    "geometric": _geometric,
    "l2_only": _l2_only,
    "lpips_only": _lpips_only,
}


# ── Public API ───────────────────────────────────────────────────────────────

def compute_distortion(mean_l2: float, mean_lpips: float) -> float:
    """Combine L2 and LPIPS into a single distortion value per current config.

    Falls back to L2-only if LPIPS is nan.
    """
    cfg = get_config()
    mode = cfg["mode"]

    if np.isnan(mean_lpips):
        return mean_l2

    combiner = _MODES.get(mode)
    if combiner is None:
        raise ValueError(
            f"Unknown EFFICIENCY_MODE={mode!r}. "
            f"Supported: {', '.join(sorted(_MODES))}"
        )

    return combiner(mean_l2, mean_lpips, cfg["l2_weight"], cfg["lpips_weight"])


def compute_efficiency(asr: float, mean_l2: float, mean_lpips: float = float("nan")) -> float:
    """Compute efficiency = ASR / (distortion + ε)."""
    d = compute_distortion(mean_l2, mean_lpips)
    return asr / (d + _EPS)


def describe_formula() -> str:
    """Return a human-readable description of the current efficiency formula."""
    cfg = get_config()
    mode = cfg["mode"]
    alpha = cfg["l2_weight"]
    beta = cfg["lpips_weight"]

    if mode == "weighted_sum":
        return f"ASR / ({alpha}*L2 + {beta}*LPIPS + eps)"
    elif mode == "harmonic":
        return f"ASR / (harmonic({alpha}*L2, {beta}*LPIPS) + eps)"
    elif mode == "geometric":
        return f"ASR / (geometric({alpha}*L2, {beta}*LPIPS) + eps)"
    elif mode == "l2_only":
        return "ASR / (L2 + eps)"
    elif mode == "lpips_only":
        return "ASR / (LPIPS + eps)"
    else:
        return f"ASR / ({mode}(L2, LPIPS) + eps)"
