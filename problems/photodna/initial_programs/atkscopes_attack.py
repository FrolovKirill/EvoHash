"""AtkScopes multi-scale patch attack — thin wrapper."""

from evohash.attacks.atkscopes import run


def entrypoint(context: dict) -> dict:
    return run(context, n_outer=8, scales=[56, 28, 14])
