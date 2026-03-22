"""NeuralHash wrapper — cross-platform ONNX implementation.

Uses the ONNX model converted from Apple's NeuralHash:
https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX

Pipeline:
  1. Convert image to RGB, resize to 360×360
  2. Normalise pixel values to [-1, 1]
  3. ONNX model inference → 128-dim embedding
  4. Dot product: seed (96×128) @ embedding (128,) → 96 floats
  5. Binarise via sign → {0, 1}^96
  6. Distance = Hamming; collision threshold = 17

Required files in data/neuralhash_model/:
  - model.onnx   (ONNX model)
  - seed1.dat     (96×128 seed matrix)

Requirements:
  pip install onnxruntime   (CPU)
  pip install onnxruntime-gpu   (GPU, optional)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .base import PHFWrapper

# ---------------------------------------------------------------------------
# Model file paths
# ---------------------------------------------------------------------------

_MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "neuralhash_model"
_SEED_PATH = _MODEL_DIR / "seed1.dat"
_ONNX_PATH = _MODEL_DIR / "model.onnx"


def _load_seed_matrix() -> np.ndarray:
    """Load the 96×128 seed matrix from seed1.dat."""
    if not _SEED_PATH.exists():
        raise FileNotFoundError(
            f"NeuralHash seed file not found: {_SEED_PATH}\n"
            "Download from https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX\n"
            "or copy from /System/Library/Frameworks/Vision.framework/Resources/"
            "neuralhash_128x96_seed1.dat"
        )
    raw = _SEED_PATH.read_bytes()[128:]  # first 128 bytes are a header
    return np.frombuffer(raw, dtype=np.float32).reshape(96, 128)


# ---------------------------------------------------------------------------
# ONNX inference
# ---------------------------------------------------------------------------


def _preprocess(image: Image.Image) -> np.ndarray:
    """RGB → resize 360×360 → normalise to [-1, 1] → (1, 3, 360, 360)."""
    pil = image.convert("RGB").resize((360, 360))
    arr = np.array(pil).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    return arr.transpose(2, 0, 1)[np.newaxis]


class _OnnxSession:
    """Lazy-loaded ONNX inference session (supports CPU and CUDA)."""

    def __init__(self) -> None:
        self._session = None
        self._input_name = None

    # cloudpickle support: InferenceSession is not picklable — drop it and
    # let the lazy-loader recreate it after deserialization.
    def __getstate__(self):
        return {}  # nothing to persist; _load() will recreate on next run()

    def __setstate__(self, state):
        self._session = None
        self._input_name = None

    def _load(self) -> None:
        if self._session is not None:
            return

        if not _ONNX_PATH.exists():
            raise FileNotFoundError(
                f"NeuralHash ONNX model not found: {_ONNX_PATH}\n"
                f"Place model.onnx in {_MODEL_DIR}\n"
                "Download from https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX"
            )

        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError(
                "onnxruntime not installed.\n"
                "  CPU: pip install onnxruntime\n"
                "  GPU: pip install onnxruntime-gpu"
            ) from e

        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        self._session = ort.InferenceSession(str(_ONNX_PATH), providers=providers)
        self._input_name = self._session.get_inputs()[0].name

    def run(self, image: Image.Image) -> np.ndarray:
        """Return 128-dim embedding for the image."""
        self._load()
        arr = _preprocess(image)
        out = self._session.run(None, {self._input_name: arr})
        return out[0].flatten()


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------


class NeuralHashWrapper(PHFWrapper):
    """Apple NeuralHash via ONNX — cross-platform.

    Computes a 96-bit perceptual hash using the converted ONNX model.
    Works on Linux, macOS, and Windows.
    """

    threshold: int = 17

    def __init__(self) -> None:
        self._seed_matrix = _load_seed_matrix()
        self._session = _OnnxSession()

    @property
    def name(self) -> str:
        return "NeuralHash"

    def compute(self, image: Image.Image) -> np.ndarray:
        """Compute 96-bit NeuralHash. Returns uint8 array of shape (96,) with values {0, 1}."""
        embedding = self._session.run(image)
        projected = self._seed_matrix @ embedding
        return (projected >= 0).astype(np.uint8)

    def distance(self, h1: np.ndarray, h2: np.ndarray) -> int:
        """Hamming distance between two 96-bit hashes."""
        return int(np.sum(h1 != h2))
