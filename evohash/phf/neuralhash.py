"""NeuralHash wrapper — macOS Vision framework implementation.

Uses Apple's VNCreateNeuralHashprintRequest (Vision framework) via PyObjC
to compute NeuralHash directly on macOS without needing model weights or ONNX.

The hash pipeline:
  1. Run VNCreateNeuralHashprintRequest on the image → 128 float32 values
  2. Project through the 96×128 seed matrix (neuralhash_128x96_seed1.dat)
  3. Apply sign → 96-bit binary hash
  4. Distance = Hamming distance over 96 bits; collision threshold = 17

Requirements:
  - macOS with Vision.framework (macOS 12+)
  - pip install pyobjc-framework-Vision pyobjc-core
  - data/neuralhash_model/seed1.dat (copied from Vision.framework/Resources)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .base import PHFWrapper

# Path to seed matrix (relative to project root)
_SEED_PATH = Path(__file__).parent.parent.parent / "data" / "neuralhash_model" / "seed1.dat"


def _load_seed_matrix() -> np.ndarray:
    """Load the 96×128 seed matrix from neuralhash_128x96_seed1.dat."""
    if not _SEED_PATH.exists():
        raise FileNotFoundError(
            f"NeuralHash seed file not found: {_SEED_PATH}\n"
            "Copy it from /System/Library/Frameworks/Vision.framework/Resources/"
            "neuralhash_128x96_seed1.dat"
        )
    raw = _SEED_PATH.read_bytes()[128:]  # first 128 bytes are a header
    return np.frombuffer(raw, dtype=np.float32).reshape(96, 128)


def _compute_via_vision(image: Image.Image) -> np.ndarray:
    """
    Run VNCreateNeuralHashprintRequest and return the raw 128-float espresso vector.
    """
    import tempfile, os
    try:
        import Vision
        from Foundation import NSURL
    except ImportError as e:
        raise ImportError(
            "PyObjC Vision framework required. Install with:\n"
            "  pip install pyobjc-framework-Vision pyobjc-core"
        ) from e

    # Vision framework works with file URLs — save to a temp file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_path = f.name

    try:
        image.convert("RGB").save(tmp_path, format="JPEG", quality=95)
        url = NSURL.fileURLWithPath_(tmp_path)
        req = Vision.VNCreateNeuralHashprintRequest.alloc().init()
        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, {})
        ok, err = handler.performRequests_error_([req], None)
        if not ok:
            raise RuntimeError(f"Vision request failed: {err}")
        hashprint = req.results()[0].imageNeuralHashprint()
        raw = bytes(hashprint.descriptorData())
        return np.frombuffer(raw, dtype=np.float32).copy()
    finally:
        os.unlink(tmp_path)


class NeuralHashWrapper(PHFWrapper):
    """
    Apple NeuralHash wrapper using the macOS Vision framework.

    Computes a 96-bit perceptual hash via VNCreateNeuralHashprintRequest.
    Requires macOS with Vision.framework and PyObjC.
    """

    threshold: int = 17  # Hamming collision threshold (from Description.md)

    def __init__(self) -> None:
        self._seed_matrix = _load_seed_matrix()

    @property
    def name(self) -> str:
        return "NeuralHash"

    def compute(self, image: Image.Image) -> np.ndarray:
        """
        Compute a 96-bit NeuralHash for *image*.

        Returns a binary numpy array of shape (96,) with values in {0, 1}.
        """
        espresso_vec = _compute_via_vision(image)          # (128,) float32
        projected = self._seed_matrix @ espresso_vec       # (96,) float32
        bits = (projected >= 0).astype(np.uint8)           # (96,) uint8 ∈ {0,1}
        return bits

    def distance(self, h1: np.ndarray, h2: np.ndarray) -> int:
        """Hamming distance between two 96-bit hashes."""
        return int(np.sum(h1 != h2))
