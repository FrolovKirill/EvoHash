"""HopSkipJump Attack (HSJA) — thin wrapper.

Reference: Chen, Jordan & Wainwright, IEEE S&P 2020.
"""

from evohash.attacks.hsja import run


def entrypoint(context: dict) -> dict:
    return run(context, n_iter=40, bs_steps=10, grad_samples=30)
