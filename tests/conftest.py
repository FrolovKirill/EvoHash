"""Shared fixtures for EvoHash tests."""

import numpy as np
import pytest
from PIL import Image


@pytest.fixture()
def rng():
    """Deterministic random number generator."""
    return np.random.default_rng(42)


@pytest.fixture()
def sample_image(rng):
    """A random 64×64 RGB PIL Image."""
    arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture()
def image_pair(rng):
    """Two distinct random 64×64 RGB PIL Images."""
    a = Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
    b = Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
    return a, b


@pytest.fixture()
def phash_wrapper():
    from evohash.phf.phash import PHashWrapper
    return PHashWrapper()


@pytest.fixture()
def pdq_wrapper():
    from evohash.phf.pdq import PDQWrapper
    return PDQWrapper()


@pytest.fixture()
def neuralhash_wrapper():
    from evohash.phf.neuralhash import NeuralHashWrapper
    return NeuralHashWrapper()


@pytest.fixture()
def photodna_wrapper():
    from evohash.phf.photodna import PhotoDNAWrapper
    return PhotoDNAWrapper()
