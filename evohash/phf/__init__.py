"""Perceptual Hash Function wrappers."""

from .base import PHFWrapper
from .phash import PHashWrapper
from .pdq import PDQWrapper
from .neuralhash import NeuralHashWrapper
from .photodna import PhotoDNAWrapper

#: Map from short name to wrapper class.
PHF_REGISTRY: dict[str, type[PHFWrapper]] = {
    "phash": PHashWrapper,
    "pdq": PDQWrapper,
    "neuralhash": NeuralHashWrapper,
    "photodna": PhotoDNAWrapper,
}


def get_phf(name: str) -> PHFWrapper:
    """Instantiate a PHF wrapper by short name (e.g. ``"phash"``)."""
    key = name.lower()
    if key not in PHF_REGISTRY:
        raise ValueError(
            f"Unknown PHF '{name}'. Available: {list(PHF_REGISTRY)}"
        )
    return PHF_REGISTRY[key]()


__all__ = [
    "PHFWrapper",
    "PHashWrapper",
    "PDQWrapper",
    "NeuralHashWrapper",
    "PhotoDNAWrapper",
    "PHF_REGISTRY",
    "get_phf",
]
