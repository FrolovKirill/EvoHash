"""SimBa + HSJA hybrid attack — thin wrapper.

Phase 1: HSJA finds collision. Phase 2: SimBa refines L2.
"""

from evohash.attacks.hybrid import run_simba_hsja


def entrypoint(context: dict) -> dict:
    return run_simba_hsja(context)
