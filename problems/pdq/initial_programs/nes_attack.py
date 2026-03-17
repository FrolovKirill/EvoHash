"""NES attack tuned for PDQ — thin wrapper.

PDQ has a larger threshold (92) so we use more samples and higher sigma.
"""

from evohash.attacks.nes import run


def entrypoint(context: dict) -> dict:
    return run(context, n_iter=200, n_samples=30, sigma=8.0, lr=4.0)
