"""ZO-Sign-SGD + HSJA hybrid attack — thin wrapper."""

from evohash.attacks.hybrid import run_zo_signsgd_hsja


def entrypoint(context: dict) -> dict:
    return run_zo_signsgd_hsja(context)
