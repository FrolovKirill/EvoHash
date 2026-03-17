"""Analytical DCT-domain attack on pHash — thin wrapper.

pHash-specific: directly manipulates DCT coefficients that determine the hash.
"""

from evohash.attacks.analytical_dct import run


def entrypoint(context: dict) -> dict:
    return run(context, n_iter=20, base_margin=3.0)
