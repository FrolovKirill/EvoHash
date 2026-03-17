"""ZO-Sign-SGD + HSJA hybrid attack — thin wrapper.

Phase 1: HSJA finds collision. Phase 2: ZO-Sign-SGD refines L2.
"""

from evohash.attacks.hybrid import run_zo_signsgd_hsja


def entrypoint(context: dict) -> dict:
    return run_zo_signsgd_hsja(context)
