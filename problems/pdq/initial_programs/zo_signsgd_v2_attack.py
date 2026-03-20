"""ZO-SignSGD v2 (central estimator + momentum) — thin wrapper.

Improvements over v1:
- Central difference gradient estimator (more accurate)
- Sign-based momentum for update stabilization

Reference: Liu et al., ICLR 2019.
"""

from evohash.attacks.zo_signsgd import run_v2


def entrypoint(context: dict) -> dict:
    return run_v2(context, n_iter=150, n_samples=20, mu=5.0, lr=1.5,
                  momentum=0.5)
