"""PhotoDNA wrapper — stub implementation.

PhotoDNA is Microsoft's proprietary perceptual hashing technology. It is
available only through the Microsoft PhotoDNA Cloud Service (paid API) or
through partnership agreements with NCMEC-affiliated organisations.

TODO: Integrate once API access is obtained. The REST API returns a hash
      vector of 144 floats. Distance is L1 norm; collision threshold is 3855.
      Reference: https://www.microsoft.com/en-us/photodna
"""

from PIL import Image

from .base import PHFWrapper


class PhotoDNAWrapper(PHFWrapper):
    """
    Stub wrapper for Microsoft PhotoDNA.

    Raises ``NotImplementedError`` on every call.
    Replace with real API calls once access credentials are available.
    """

    threshold: int = 3855  # L1 collision threshold

    @property
    def name(self) -> str:
        return "PhotoDNA"

    def compute(self, image: Image.Image):
        # TODO: call Microsoft PhotoDNA Cloud Service API
        raise NotImplementedError(
            "PhotoDNA is not yet implemented. "
            "Microsoft PhotoDNA requires a paid API licence. "
            "See evohash/phf/photodna.py for details."
        )

    def distance(self, h1, h2) -> float:
        # TODO: L1 distance over 144-element float vectors
        raise NotImplementedError(
            "PhotoDNA distance is not yet implemented."
        )
