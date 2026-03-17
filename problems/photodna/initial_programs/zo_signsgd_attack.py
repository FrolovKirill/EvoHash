"""ZO-Sign-SGD attack — thin wrapper."""

from evohash.attacks.zo_signsgd import run


def entrypoint(context: dict) -> dict:
    return run(context, n_iter=150, n_samples=20, mu=5.0, lr=1.5)
