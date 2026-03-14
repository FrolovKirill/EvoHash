"""Full benchmark evaluation for EvoHash evolved attack programs.

Runs a given attack strategy over 100 image pairs and reports all primary
and secondary metrics defined in the paper.

Usage::

    # Evaluate the NES baseline against pHash
    python scripts/evaluate.py --phf phash --program problems/phash/initial_programs/nes_attack.py

    # Evaluate a custom evolved program
    python scripts/evaluate.py --phf pdq --program path/to/evolved.py --n-pairs 50

    # Evaluate all initial programs for pHash
    python scripts/evaluate.py --phf phash --all-seeds

Output is printed as a table and optionally saved as CSV.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evohash.dataset import load_image_pairs
from evohash.evaluation import (
    compute_asr,
    compute_efficiency,
    compute_lpips,
    compute_mean_l2,
    compute_mean_queries,
    image_to_array,
)
from evohash.phf import get_phf

DATA_DIR = PROJECT_ROOT / "data" / "imagenet_val"


def _load_program(path: Path):
    """Dynamically load a Python file and return its ``entrypoint`` function."""
    spec = importlib.util.spec_from_file_location("attack_program", path)
    module = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(module)  # type: ignore
    if not hasattr(module, "entrypoint"):
        raise AttributeError(f"No 'entrypoint' function found in {path}")
    return module.entrypoint


def evaluate_program(
    program_path: Path,
    phf_name: str,
    n_pairs: int = 100,
) -> dict:
    """
    Evaluate a single attack program and return metrics.

    Args:
        program_path: Path to a .py file implementing ``entrypoint(context)``.
        phf_name: Short PHF name ('phash', 'pdq', …).
        n_pairs: Number of image pairs to evaluate.

    Returns:
        Dict with: name, phf, asr, mean_l2, efficiency, mean_queries, lpips,
        time_per_attack.
    """
    print(f"Loading program: {program_path.name}")
    attack_fn = _load_program(program_path)

    phf = get_phf(phf_name)

    pairs = load_image_pairs(DATA_DIR, n_pairs=n_pairs)
    sources = [p[0] for p in pairs]
    targets = [p[1] for p in pairs]
    target_hashes = [phf.compute(img) for img in targets]

    context = {
        "hash_fn": phf,
        "threshold": phf.threshold,
        "source_images": sources,
        "target_hashes": target_hashes,
        "target_images": targets,
    }

    print(f"Running attack on {n_pairs} image pairs…")
    t0 = time.time()
    result = attack_fn(context)
    elapsed = time.time() - t0

    attacked_pil = result.get("attacked_images", [])
    per_image = result.get("metrics", [])

    originals = [image_to_array(img) for img in sources]
    attacked_arrs = [image_to_array(img) for img in attacked_pil]

    asr = compute_asr(per_image)
    mean_l2 = compute_mean_l2(originals, attacked_arrs)
    efficiency = compute_efficiency(asr, mean_l2)
    mean_queries = compute_mean_queries(per_image)

    print("Computing LPIPS (may be slow without GPU)…")
    lpips_score = compute_lpips(originals, attacked_arrs)

    time_per = elapsed / max(len(sources), 1)

    return {
        "name": program_path.stem,
        "phf": phf_name,
        "asr": asr,
        "mean_l2": mean_l2,
        "efficiency": efficiency,
        "mean_queries": mean_queries,
        "lpips": lpips_score,
        "time_per_attack": time_per,
    }


def _print_table(rows: list[dict]) -> None:
    """Print evaluation results as a formatted table."""
    try:
        from tabulate import tabulate  # type: ignore

        headers = [
            "Program", "PHF", "ASR", "mean L2", "Efficiency",
            "Queries", "LPIPS", "Time (s)",
        ]
        table = [
            [
                r["name"],
                r["phf"],
                f"{r['asr']:.3f}",
                f"{r['mean_l2']:.2f}",
                f"{r['efficiency']:.4f}",
                f"{r['mean_queries']:.0f}",
                f"{r['lpips']:.4f}" if not _isnan(r["lpips"]) else "N/A",
                f"{r['time_per_attack']:.2f}",
            ]
            for r in rows
        ]
        print("\n" + tabulate(table, headers=headers, tablefmt="github"))
    except ImportError:
        # Fallback: simple print
        print("\n" + "\t".join(["Program", "ASR", "L2", "Efficiency", "Queries"]))
        for r in rows:
            print(
                f"{r['name']}\t{r['asr']:.3f}\t{r['mean_l2']:.2f}\t"
                f"{r['efficiency']:.4f}\t{r['mean_queries']:.0f}"
            )


def _isnan(v) -> bool:
    try:
        return bool(np.isnan(v))
    except Exception:
        return False


def _save_csv(rows: list[dict], out_path: Path) -> None:
    import csv

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate EvoHash attack programs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phf",
        required=True,
        choices=["phash", "pdq"],
        help="Target PHF.",
    )
    parser.add_argument(
        "--program",
        type=Path,
        help="Path to a .py file implementing entrypoint(context).",
    )
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        help="Evaluate all initial programs in problems/<phf>/initial_programs/.",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=100,
        metavar="N",
        help="Number of image pairs (default: 100).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        metavar="FILE",
        help="Save results to a CSV file.",
    )
    args = parser.parse_args()

    if not args.program and not args.all_seeds:
        parser.error("Specify --program or --all-seeds.")

    programs: list[Path] = []
    if args.all_seeds:
        seeds_dir = PROJECT_ROOT / "problems" / args.phf / "initial_programs"
        programs = sorted(seeds_dir.glob("*.py"))
        print(f"Evaluating all seeds in {seeds_dir}:")
        for p in programs:
            print(f"  {p.name}")
        print()
    elif args.program:
        programs = [args.program]

    rows: list[dict] = []
    for prog in programs:
        try:
            metrics = evaluate_program(prog, args.phf, n_pairs=args.n_pairs)
            rows.append(metrics)
            print(
                f"  ✓ {prog.stem}: ASR={metrics['asr']:.3f}, "
                f"L2={metrics['mean_l2']:.2f}, "
                f"Efficiency={metrics['efficiency']:.4f}\n"
            )
        except Exception as e:
            print(f"  ✗ {prog.stem}: {e}\n")

    if rows:
        _print_table(rows)
        if args.output_csv:
            _save_csv(rows, args.output_csv)


if __name__ == "__main__":
    main()
