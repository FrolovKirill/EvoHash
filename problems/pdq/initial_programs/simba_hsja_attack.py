"""SimBa + HSJA hybrid attack — thin wrapper."""

from evohash.attacks.hybrid import run_simba_hsja


def entrypoint(context: dict) -> dict:
    return run_simba_hsja(context)
