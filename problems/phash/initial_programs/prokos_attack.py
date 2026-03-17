"""Prokos et al. frequency-targeted attack — thin wrapper.

Reference: Prokos et al., USENIX Security 2023.
"""

from evohash.attacks.prokos import run


def entrypoint(context: dict) -> dict:
    return run(context, n_iter=120, n_freq_samples=25, lr=5.0, sigma=3.0)
