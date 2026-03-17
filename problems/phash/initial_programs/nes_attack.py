"""NES (Natural Evolution Strategy) attack — thin wrapper.

Reference: Ilyas et al., ICML 2018.
"""

from evohash.attacks.nes import run


def entrypoint(context: dict) -> dict:
    return run(context, n_iter=150, n_samples=20, sigma=6.0, lr=3.0)
