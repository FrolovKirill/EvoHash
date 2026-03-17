"""SimBa (block basis) attack tuned for PDQ — thin wrapper.

PDQ processes a 64x64 grayscale downsample, so large pixel blocks
are more efficient than DCT basis.
"""

from evohash.attacks.simba import run


def entrypoint(context: dict) -> dict:
    return run(context, basis="block", n_iter=300, step_size=16.0, block_size=16)
