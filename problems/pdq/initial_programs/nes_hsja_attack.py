"""NES + HSJA hybrid attack — thin wrapper."""

from evohash.attacks.hybrid import run_nes_hsja


def entrypoint(context: dict) -> dict:
    return run_nes_hsja(context)
