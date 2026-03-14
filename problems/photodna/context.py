"""Runtime context for the PhotoDNA collision attack problem.

TODO: PhotoDNA is proprietary (Microsoft).  Access requires a partnership
      agreement or API key from the Microsoft PhotoDNA Cloud Service.
      Once access is obtained:

      1. Implement ``PhotoDNAWrapper.compute`` and ``PhotoDNAWrapper.distance``
         in ``evohash/phf/photodna.py``.
      2. Replace the ``build_context`` body below to instantiate the wrapper.
      3. Remove this TODO comment.
"""


def build_context() -> dict:
    raise NotImplementedError(
        "PhotoDNA is not yet supported.  "
        "See evohash/phf/photodna.py and problems/photodna/context.py."
    )
