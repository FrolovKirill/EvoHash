"""SimBa (DCT basis) attack — thin wrapper.

Reference: Guo et al., ICML 2019.
"""

from evohash.attacks.simba import run


def entrypoint(context: dict) -> dict:
    return run(context, basis="dct", n_iter=200, step_size=12.0)
