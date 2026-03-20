"""JSHA: SimBa DCT (unconstrained) → HSJA (L2 minimization).

Madden et al., NeurIPS 2024 Workshop.
Phase 1: SimBa finds collision fast (ignoring L2). Phase 2: HSJA minimizes L2.
"""

from evohash.attacks.hybrid import run_simba_jsha


def entrypoint(context: dict) -> dict:
    return run_simba_jsha(context)
