"""NeuralHash wrapper — stub implementation.

NeuralHash is Apple's proprietary perceptual hashing system used in CSAM
detection (iOS 15+). The model weights are not publicly available through
official channels.

TODO: Integrate an open-source reimplementation if one becomes available under
      a permissive licence. Potential starting points:
      - https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX (model dump,
        unclear licence status)
      Until then, this stub raises NotImplementedError so the rest of the
      framework can be tested with pHash and PDQ.
"""

from PIL import Image

from .base import PHFWrapper


class NeuralHashWrapper(PHFWrapper):
    """
    Stub wrapper for Apple NeuralHash.

    Raises ``NotImplementedError`` on every call.
    Replace the body of ``compute`` and ``distance`` once a usable
    open-source implementation is available.
    """

    threshold: int = 17  # collision threshold from the paper

    @property
    def name(self) -> str:
        return "NeuralHash"

    def compute(self, image: Image.Image):
        # TODO: load ONNX model and run inference
        raise NotImplementedError(
            "NeuralHash is not yet implemented. "
            "Apple's model weights are proprietary. "
            "See evohash/phf/neuralhash.py for details."
        )

    def distance(self, h1, h2) -> float:
        # TODO: implement Hamming distance over NeuralHash bits
        raise NotImplementedError(
            "NeuralHash distance is not yet implemented."
        )
