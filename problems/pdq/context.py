"""Runtime context for the PDQ collision attack problem."""

import os as _os
N_PAIRS_EVAL = int(_os.environ.get("EVOHASH_N_PAIRS", "10"))


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
    import evohash
    from pathlib import Path
    from evohash.dataset import load_image_pairs
    from evohash.phf.pdq import PDQWrapper

    data_dir = Path(evohash.__file__).parent.parent / "data" / "imagenet_val"

    phf = PDQWrapper()

    from evohash.dataset import get_generation_seed
    seed = get_generation_seed("pdq")
    pairs = load_image_pairs(data_dir, n_pairs=N_PAIRS_EVAL, seed=seed)
    sources = [p[0] for p in pairs]
    targets = [p[1] for p in pairs]
    target_hashes = [phf.compute(img) for img in targets]

    from evohash.reporter import init_run
    init_run("pdq", phf.threshold, N_PAIRS_EVAL)

    return {
        "hash_fn": phf,
        "threshold": phf.threshold,
        "source_images": sources,
        "target_hashes": target_hashes,
        "target_images": targets,
        "_report_dir": None,
        "_phf_name": "pdq",
    }
