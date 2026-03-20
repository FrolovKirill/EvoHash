"""ATKScopes per-coordinate Adam attack — thin wrapper.

Reference: Zhang et al., USENIX Security 2025.
"""

from evohash.attacks.atkscopes import run


def entrypoint(context: dict) -> dict:
    return run(context, scale="global", n_iter=600, lr=0.05, a=0.1)
