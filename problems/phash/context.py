"""Runtime context for the pHash collision attack problem.

gigaevo calls ``build_context()`` once before evaluating any program.
The returned dict is passed as the ``context`` argument to every call of
``entrypoint(context)`` in the evolved programs and to ``validate(data, context)``.
"""

#: Number of image pairs used per evaluation during evolution.
#: Keep this small (≤ 20) for fast iteration; the full benchmark uses 100.
#: Overridable via EVOHASH_N_PAIRS env var (set by Web UI / runner).
import os as _os
N_PAIRS_EVAL = int(_os.environ.get("EVOHASH_N_PAIRS", "10"))


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
        - ``_report_dir``   : Path where the HTML report is written
        - ``_phf_name``     : str, identifier for the reporter
    """
    # run_evohash.py adds EvoHash/ to PYTHONPATH so gigaevo's exec_runner
    # subprocesses can import the evohash package.  Use evohash.__file__ to
    # locate the data directory rather than relying on __file__ of this module
    # (which is not set when gigaevo exec()s this code in a subprocess).
    import evohash
    from pathlib import Path
    from evohash.dataset import load_image_pairs
    from evohash.phf.phash import PHashWrapper

    data_dir = Path(evohash.__file__).parent.parent / "data" / "imagenet_val"

    phf = PHashWrapper()

    pairs = load_image_pairs(data_dir, n_pairs=N_PAIRS_EVAL)
    sources = [p[0] for p in pairs]
    targets = [p[1] for p in pairs]
    target_hashes = [phf.compute(img) for img in targets]

    from evohash.reporter import init_run
    init_run("phash", phf.threshold, N_PAIRS_EVAL)

    return {
        "hash_fn": phf,
        "threshold": phf.threshold,
        "source_images": sources,
        "target_hashes": target_hashes,
        "target_images": targets,
        "_report_dir": None,
        "_phf_name": "phash",
    }
