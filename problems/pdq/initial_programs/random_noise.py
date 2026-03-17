"""Random Gaussian noise baseline — thin wrapper."""

from evohash.attacks.random_noise import run


def entrypoint(context: dict) -> dict:
    return run(context, n_trials=200, sigma=8.0)
