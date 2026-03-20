"""HSJA v2 (noise-based init) — thin wrapper.

Unlike HSJA v1, does NOT require a target_image as starting point.
Finds initial collision via Gaussian noise, then boundary-walks toward source.

Reference: Chen, Jordan & Wainwright, IEEE S&P 2020 (adapted).
"""

from evohash.attacks.hsja import run_v2


def entrypoint(context: dict) -> dict:
    return run_v2(context, n_iter=25, bs_steps=12, grad_samples=40,
                  step_ratio=0.15)
