"""Tests for evohash.attacks.utils."""

import numpy as np
import pytest
from PIL import Image

from evohash.attacks.utils import (
    clamp_perturbation,
    collides,
    make_metric,
    normalised_l2,
    query_distance,
    to_array,
    to_image,
)


class TestImageConversions:
    def test_to_array_shape(self, sample_image):
        arr = to_array(sample_image)
        assert arr.dtype == np.float32
        assert arr.shape == (64, 64, 3)

    def test_to_image_roundtrip(self, sample_image):
        arr = to_array(sample_image)
        img = to_image(arr)
        assert isinstance(img, Image.Image)
        assert img.size == sample_image.size

    def test_to_image_clips(self):
        arr = np.array([[[300.0, -10.0, 128.0]]], dtype=np.float32)
        img = to_image(arr)
        px = np.array(img)
        assert px[0, 0, 0] == 255
        assert px[0, 0, 1] == 0
        assert px[0, 0, 2] == 128


class TestNormalisedL2:
    def test_identical_is_zero(self):
        arr = np.ones((10, 10, 3), dtype=np.float32) * 128
        assert normalised_l2(arr, arr) == 0.0

    def test_positive_for_diff(self):
        a = np.zeros((10, 10, 3), dtype=np.float32)
        b = np.ones((10, 10, 3), dtype=np.float32) * 10
        assert normalised_l2(a, b) > 0


class TestClampPerturbation:
    def test_output_in_range(self):
        orig = np.full((4, 4, 3), 128.0)
        perturbed = orig + 200  # way out of [0,255]
        out = clamp_perturbation(orig, perturbed)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_linf_constraint(self):
        orig = np.full((4, 4, 3), 128.0)
        perturbed = orig + 50
        out = clamp_perturbation(orig, perturbed, max_linf=10)
        delta = np.abs(out - orig)
        assert delta.max() <= 10 + 1e-6

    def test_l2_constraint(self):
        orig = np.full((4, 4, 3), 128.0)
        perturbed = orig + 100
        out = clamp_perturbation(orig, perturbed, max_l2=1.0)
        l2 = normalised_l2(orig, out)
        assert l2 <= 1.0 + 1e-3


class TestMakeMetric:
    def test_keys(self):
        orig = np.zeros((4, 4, 3), dtype=np.float32)
        atk = np.ones((4, 4, 3), dtype=np.float32)
        m = make_metric(True, orig, atk, 42, 5.0)
        assert set(m.keys()) == {"success", "l2", "n_queries", "final_dist"}
        assert m["success"] is True
        assert m["n_queries"] == 42
        assert m["final_dist"] == 5.0
        assert m["l2"] > 0


class TestHashQueryHelpers:
    def test_query_distance(self, phash_wrapper, sample_image):
        arr = np.array(sample_image).astype(np.float32)
        target_hash = phash_wrapper.compute(sample_image)
        d = query_distance(arr, target_hash, phash_wrapper)
        assert d == 0.0

    def test_collides(self, phash_wrapper, sample_image):
        arr = np.array(sample_image).astype(np.float32)
        target_hash = phash_wrapper.compute(sample_image)
        assert collides(arr, target_hash, phash_wrapper, 12)
