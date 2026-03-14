"""Runtime context for the NeuralHash collision attack problem.

TODO: NeuralHash is not yet implemented (Apple's model weights are proprietary).
      This file is a stub.  Once a usable implementation is available:

      1. Replace the ``build_context`` body to instantiate ``NeuralHashWrapper``
         instead of raising ``NotImplementedError``.
      2. Verify that the attack programs in ``initial_programs/`` run correctly.
      3. Remove this TODO comment.
"""


def build_context() -> dict:
    raise NotImplementedError(
        "NeuralHash is not yet supported.  "
        "See evohash/phf/neuralhash.py and problems/neuralhash/context.py "
        "for details on what needs to be implemented."
    )
