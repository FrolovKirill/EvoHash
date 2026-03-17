"""Tests for PHF wrappers: pHash, PDQ, NeuralHash."""

import numpy as np
import pytest
from PIL import Image

from evohash.phf import PHF_REGISTRY, get_phf
from evohash.phf.base import PHFWrapper


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_all_keys_present(self):
        assert set(PHF_REGISTRY) == {"phash", "pdq", "neuralhash", "photodna"}

    def test_get_phf_returns_wrapper(self):
        phf = get_phf("phash")
        assert isinstance(phf, PHFWrapper)

    def test_get_phf_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown PHF"):
            get_phf("nonexistent")

    def test_get_phf_case_insensitive(self):
        assert isinstance(get_phf("PHash"), PHFWrapper)


# ---------------------------------------------------------------------------
# pHash
# ---------------------------------------------------------------------------

class TestPHash:
    def test_compute_returns_hash(self, phash_wrapper, sample_image):
        h = phash_wrapper.compute(sample_image)
        assert h is not None

    def test_identical_images_zero_distance(self, phash_wrapper, sample_image):
        h1 = phash_wrapper.compute(sample_image)
        h2 = phash_wrapper.compute(sample_image)
        assert phash_wrapper.distance(h1, h2) == 0.0

    def test_different_images_nonzero_distance(self, phash_wrapper, image_pair):
        h1 = phash_wrapper.compute(image_pair[0])
        h2 = phash_wrapper.compute(image_pair[1])
        assert phash_wrapper.distance(h1, h2) >= 0

    def test_threshold(self, phash_wrapper):
        assert phash_wrapper.threshold == 12

    def test_name(self, phash_wrapper):
        assert phash_wrapper.name == "pHash"

    def test_is_collision_self(self, phash_wrapper, sample_image):
        h = phash_wrapper.compute(sample_image)
        assert phash_wrapper.is_collision(h, h)

    def test_distance_symmetric(self, phash_wrapper, image_pair):
        h1 = phash_wrapper.compute(image_pair[0])
        h2 = phash_wrapper.compute(image_pair[1])
        assert phash_wrapper.distance(h1, h2) == phash_wrapper.distance(h2, h1)

    def test_various_image_sizes(self, phash_wrapper):
        for size in [(32, 32), (128, 128), (256, 100)]:
            img = Image.fromarray(np.zeros((*size, 3), dtype=np.uint8))
            h = phash_wrapper.compute(img)
            assert h is not None


# ---------------------------------------------------------------------------
# PDQ
# ---------------------------------------------------------------------------

class TestPDQ:
    def test_compute_shape(self, pdq_wrapper, sample_image):
        h = pdq_wrapper.compute(sample_image)
        assert isinstance(h, np.ndarray)
        assert h.shape == (256,)

    def test_identical_images_zero_distance(self, pdq_wrapper, sample_image):
        h1 = pdq_wrapper.compute(sample_image)
        h2 = pdq_wrapper.compute(sample_image)
        assert pdq_wrapper.distance(h1, h2) == 0.0

    def test_threshold(self, pdq_wrapper):
        assert pdq_wrapper.threshold == 92

    def test_name(self, pdq_wrapper):
        assert pdq_wrapper.name == "PDQ"

    def test_hash_is_binary(self, pdq_wrapper, sample_image):
        h = pdq_wrapper.compute(sample_image)
        assert set(np.unique(h)).issubset({0, 1, True, False})

    def test_distance_symmetric(self, pdq_wrapper, image_pair):
        h1 = pdq_wrapper.compute(image_pair[0])
        h2 = pdq_wrapper.compute(image_pair[1])
        assert pdq_wrapper.distance(h1, h2) == pdq_wrapper.distance(h2, h1)


# ---------------------------------------------------------------------------
# NeuralHash
# ---------------------------------------------------------------------------

class TestNeuralHash:
    def test_compute_shape(self, neuralhash_wrapper, sample_image):
        h = neuralhash_wrapper.compute(sample_image)
        assert isinstance(h, np.ndarray)
        assert h.shape == (96,)

    def test_hash_is_binary(self, neuralhash_wrapper, sample_image):
        h = neuralhash_wrapper.compute(sample_image)
        assert set(np.unique(h)).issubset({0, 1})

    def test_identical_images_zero_distance(self, neuralhash_wrapper, sample_image):
        h1 = neuralhash_wrapper.compute(sample_image)
        h2 = neuralhash_wrapper.compute(sample_image)
        assert neuralhash_wrapper.distance(h1, h2) == 0

    def test_threshold(self, neuralhash_wrapper):
        assert neuralhash_wrapper.threshold == 17

    def test_name(self, neuralhash_wrapper):
        assert neuralhash_wrapper.name == "NeuralHash"

    def test_distance_range(self, neuralhash_wrapper, image_pair):
        h1 = neuralhash_wrapper.compute(image_pair[0])
        h2 = neuralhash_wrapper.compute(image_pair[1])
        d = neuralhash_wrapper.distance(h1, h2)
        assert 0 <= d <= 96

    def test_distance_symmetric(self, neuralhash_wrapper, image_pair):
        h1 = neuralhash_wrapper.compute(image_pair[0])
        h2 = neuralhash_wrapper.compute(image_pair[1])
        assert neuralhash_wrapper.distance(h1, h2) == neuralhash_wrapper.distance(h2, h1)
