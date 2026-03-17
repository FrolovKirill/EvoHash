"""Tests for unified attacks via evohash.attacks.

Uses pHash as the test hash (fast, no external deps beyond imagehash).
Runs each attack on 1 tiny image pair with very small budgets just to
verify the interface contract: run(context) -> {attacked_images, metrics}.
"""

import numpy as np
import pytest
from PIL import Image


def _make_context(phash_wrapper, rng):
    """Build a minimal context dict with 1 image pair."""
    src = Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
    tgt = Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
    return {
        "hash_fn": phash_wrapper,
        "threshold": phash_wrapper.threshold,
        "source_images": [src],
        "target_hashes": [phash_wrapper.compute(tgt)],
        "target_images": [tgt],
    }


def _check_result(result):
    """Validate the attack result contract."""
    assert "attacked_images" in result
    assert "metrics" in result
    assert len(result["attacked_images"]) == 1
    assert len(result["metrics"]) == 1

    m = result["metrics"][0]
    assert "success" in m
    assert "l2" in m
    assert "n_queries" in m
    assert "final_dist" in m
    assert isinstance(m["success"], bool)
    assert m["l2"] >= 0
    assert m["n_queries"] >= 1

    img = result["attacked_images"][0]
    assert isinstance(img, Image.Image)


class TestRandomNoise:
    def test_run(self, phash_wrapper, rng):
        from evohash.attacks.random_noise import run
        ctx = _make_context(phash_wrapper, rng)
        result = run(ctx, n_trials=5, sigma=4.0)
        _check_result(result)


class TestNES:
    def test_run(self, phash_wrapper, rng):
        from evohash.attacks.nes import run
        ctx = _make_context(phash_wrapper, rng)
        result = run(ctx, n_iter=3, n_samples=2, sigma=4.0, lr=2.0)
        _check_result(result)


class TestSimBA:
    def test_run_dct(self, phash_wrapper, rng):
        from evohash.attacks.simba import run
        ctx = _make_context(phash_wrapper, rng)
        result = run(ctx, n_iter=3, basis="dct", step_size=4.0)
        _check_result(result)


class TestZOSignSGD:
    def test_run(self, phash_wrapper, rng):
        from evohash.attacks.zo_signsgd import run
        ctx = _make_context(phash_wrapper, rng)
        result = run(ctx, n_iter=3, n_samples=2, mu=4.0, lr=2.0)
        _check_result(result)


class TestProkos:
    def test_run(self, phash_wrapper, rng):
        from evohash.attacks.prokos import run
        ctx = _make_context(phash_wrapper, rng)
        result = run(ctx, n_iter=3, n_freq_samples=2, sigma=4.0, lr=2.0)
        _check_result(result)


class TestAtkscopes:
    def test_run(self, phash_wrapper, rng):
        from evohash.attacks.atkscopes import run
        ctx = _make_context(phash_wrapper, rng)
        result = run(ctx, scales=[32], n_outer=2)
        _check_result(result)


class TestHSJA:
    def test_run(self, phash_wrapper, rng):
        from evohash.attacks.hsja import run
        ctx = _make_context(phash_wrapper, rng)
        result = run(ctx, n_iter=3)
        _check_result(result)


class TestHybridNESHSJA:
    def test_run(self, phash_wrapper, rng):
        from evohash.attacks.hybrid import run_nes_hsja
        ctx = _make_context(phash_wrapper, rng)
        result = run_nes_hsja(ctx,
                              hsja_kwargs={"n_iter": 2},
                              refine_kwargs={"n_iter": 2})
        _check_result(result)
