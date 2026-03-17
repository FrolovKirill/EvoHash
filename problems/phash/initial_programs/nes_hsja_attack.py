"""NES + HSJA hybrid attack — thin wrapper.

Phase 1: HSJA finds collision. Phase 2: NES refines L2.
"""

from evohash.attacks.hybrid import run_nes_hsja


def entrypoint(context: dict) -> dict:
    return run_nes_hsja(context)
