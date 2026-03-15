"""Runtime context for the NeuralHash collision attack problem.

Uses Apple's Vision framework via PyObjC (macOS only).
Requires: pip install pyobjc-framework-Vision pyobjc-core
"""

N_PAIRS_EVAL = 10


def build_context() -> dict:
    import evohash
    from pathlib import Path
    from evohash.dataset import load_image_pairs
    from evohash.phf.neuralhash import NeuralHashWrapper

    data_dir = Path(evohash.__file__).parent.parent / "data" / "imagenet_val"

    phf = NeuralHashWrapper()

    pairs = load_image_pairs(data_dir, n_pairs=N_PAIRS_EVAL)
    sources = [p[0] for p in pairs]
    targets = [p[1] for p in pairs]
    target_hashes = [phf.compute(img) for img in targets]

    from evohash.reporter import init_run
    init_run("neuralhash", phf.threshold, N_PAIRS_EVAL)

    return {
        "hash_fn": phf,
        "threshold": phf.threshold,
        "source_images": sources,
        "target_hashes": target_hashes,
        "target_images": targets,
        "_report_dir": None,
        "_phf_name": "neuralhash",
    }
