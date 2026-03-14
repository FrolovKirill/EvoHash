"""Runtime context for the PDQ collision attack problem."""

from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DATA_DIR = _PROJECT_ROOT / "data" / "imagenet_val"

N_PAIRS_EVAL = 10


def build_context() -> dict:
    """
    Build and return the shared context for PDQ attack evaluation.

    Returns:
        dict with keys:
        - ``hash_fn``       : PDQWrapper instance
        - ``threshold``     : int, collision threshold (92)
        - ``source_images`` : list of PIL Images (sources to perturb)
        - ``target_hashes`` : list of numpy arrays (256-bit PDQ hashes)
        - ``target_images`` : list of PIL Images (targets, for reference)
    """
    import sys

    sys.path.insert(0, str(_PROJECT_ROOT))

    from evohash.dataset import load_image_pairs
    from evohash.phf.pdq import PDQWrapper

    phf = PDQWrapper()

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
