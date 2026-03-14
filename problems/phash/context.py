"""Runtime context for the pHash collision attack problem.

gigaevo calls ``build_context()`` once before evaluating any program.
The returned dict is passed as the ``context`` argument to every call of
``entrypoint(context)`` in the evolved programs and to ``validate(data, context)``.
"""

from pathlib import Path

# Project root is two levels above this file: problems/phash/context.py
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DATA_DIR = _PROJECT_ROOT / "data" / "imagenet_val"

#: Number of image pairs used per evaluation during evolution.
#: Keep this small (≤ 20) for fast iteration; the full benchmark uses 100.
N_PAIRS_EVAL = 10


def build_context() -> dict:
    """
    Build and return the shared context for pHash attack evaluation.

    Returns:
        dict with keys:
        - ``hash_fn``       : PHashWrapper instance
        - ``threshold``     : int, collision threshold (12)
        - ``source_images`` : list of PIL Images (sources to perturb)
        - ``target_hashes`` : list of pHash values to collide with
        - ``target_images`` : list of PIL Images (targets, for reference)
    """
    import sys

    sys.path.insert(0, str(_PROJECT_ROOT))

    from evohash.dataset import load_image_pairs
    from evohash.phf.phash import PHashWrapper

    phf = PHashWrapper()

    pairs = load_image_pairs(_DATA_DIR, n_pairs=N_PAIRS_EVAL)
    sources = [p[0] for p in pairs]
    targets = [p[1] for p in pairs]
    target_hashes = [phf.compute(img) for img in targets]

    return {
        "hash_fn": phf,
        "threshold": phf.threshold,
        "source_images": sources,
        "target_hashes": target_hashes,
        "target_images": targets,
    }
