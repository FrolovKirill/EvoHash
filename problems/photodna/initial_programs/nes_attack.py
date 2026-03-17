"""NES attack — thin wrapper."""

from evohash.attacks.nes import run


def entrypoint(context: dict) -> dict:
    return run(context, n_iter=200, n_samples=20, sigma=6.0, lr=3.0)
