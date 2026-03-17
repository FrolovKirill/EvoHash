"""Unified attack library for perceptual hash collisions.

All attacks are hash-agnostic: they interact with the hash function through
the standard ``context`` dict containing ``hash_fn``, ``threshold``,
``source_images``, ``target_hashes``, and ``target_images``.

Each module exposes a ``run(context, **kwargs) -> dict`` function that
returns ``{"attacked_images": [...], "metrics": [...]}``.
"""
