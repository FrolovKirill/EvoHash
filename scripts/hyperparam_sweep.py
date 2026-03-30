"""Hyperparameter sweep for gradient-based attacks on pHash.

Tests each attack with multiple hyperparameter configurations on a small
number of image pairs. Reports ASR, L2, queries, final_dist, and wall-clock
time per configuration.

Results are saved incrementally to CSV and a detailed log file.

Usage:
    python scripts/hyperparam_sweep.py [--n-pairs 3] [--phf phash]
"""

from __future__ import annotations

import csv
import importlib.util
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evohash.dataset import load_image_pairs
from evohash.phf import get_phf

DATA_DIR = PROJECT_ROOT / "data" / "imagenet_val"
PROGRAMS_DIR = PROJECT_ROOT / "problems" / "phash" / "initial_programs"

# Output paths
RESULTS_CSV = PROJECT_ROOT / "results_hyperparam_sweep.csv"
LOG_FILE = PROJECT_ROOT / "hyperparam_sweep.log"

# ── Logging setup ────────────────────────────────────────────────────────────

def setup_logging():
    logger = logging.getLogger("sweep")
    logger.setLevel(logging.DEBUG)

    # File handler — detailed
    fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))

    # Console handler — concise
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


log = setup_logging()


def load_attack_module(name: str):
    path = PROGRAMS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── CSV writer (incremental) ────────────────────────────────────────────────

CSV_FIELDS = [
    "attack", "phf", "config", "asr", "mean_l2", "mean_queries",
    "mean_final_dist", "min_final_dist", "max_final_dist",
    "time_s", "time_per_pair_s", "n_pairs",
    "per_pair_success", "per_pair_dist", "per_pair_l2", "per_pair_queries",
]

_csv_file = None
_csv_writer = None


def init_csv():
    global _csv_file, _csv_writer
    write_header = not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0
    _csv_file = open(RESULTS_CSV, "a", newline="", encoding="utf-8")
    _csv_writer = csv.DictWriter(_csv_file, fieldnames=CSV_FIELDS)
    if write_header:
        _csv_writer.writeheader()
        _csv_file.flush()


def write_csv_row(row: dict):
    _csv_writer.writerow(row)
    _csv_file.flush()


def close_csv():
    if _csv_file:
        _csv_file.close()


# ── Hyperparameter grids ─────────────────────────────────────────────────────

SWEEPS = {
    "nes_attack": [
        # baseline
        {"n_iter": 150, "n_samples": 20, "sigma": 6.0,  "lr": 3.0},
        # much more iterations
        {"n_iter": 500, "n_samples": 20, "sigma": 6.0,  "lr": 3.0},
        # large sigma to jump plateaus
        {"n_iter": 300, "n_samples": 20, "sigma": 30.0, "lr": 5.0},
        {"n_iter": 300, "n_samples": 20, "sigma": 60.0, "lr": 10.0},
        # many samples for better gradient
        {"n_iter": 200, "n_samples": 50, "sigma": 15.0, "lr": 5.0},
        # extreme: huge perturbations
        {"n_iter": 300, "n_samples": 30, "sigma": 100.0, "lr": 20.0},
    ],
    "simba_attack": [
        # baseline
        {"n_iter": 200, "step_size": 12.0},
        # bigger steps
        {"n_iter": 400, "step_size": 30.0},
        {"n_iter": 400, "step_size": 60.0},
        {"n_iter": 600, "step_size": 100.0},
        # extreme
        {"n_iter": 800, "step_size": 200.0},
    ],
    "zo_signsgd_attack": [
        # baseline
        {"n_iter": 150, "n_samples": 20, "mu": 5.0,   "lr": 1.5},
        # larger mu (bigger finite diff step)
        {"n_iter": 300, "n_samples": 20, "mu": 30.0,  "lr": 3.0},
        {"n_iter": 300, "n_samples": 20, "mu": 60.0,  "lr": 5.0},
        # more samples + large mu
        {"n_iter": 200, "n_samples": 50, "mu": 30.0,  "lr": 5.0},
        # extreme
        {"n_iter": 500, "n_samples": 30, "mu": 100.0, "lr": 10.0},
    ],
    "prokos_attack": [
        # baseline
        {"n_iter": 120, "n_freq_samples": 25, "lr": 5.0,  "sigma": 3.0},
        # larger sigma + more iters
        {"n_iter": 300, "n_freq_samples": 25, "lr": 10.0, "sigma": 15.0},
        {"n_iter": 300, "n_freq_samples": 25, "lr": 20.0, "sigma": 30.0},
        # many frequency samples
        {"n_iter": 200, "n_freq_samples": 50, "lr": 10.0, "sigma": 20.0},
        # extreme
        {"n_iter": 500, "n_freq_samples": 30, "lr": 30.0, "sigma": 60.0},
    ],
    "atkscopes_attack": [
        # baseline (global scale, defaults from scale)
        {"scale": "global", "n_iter": 2000, "a": 50.0,  "lr": 0.05},
        # higher LR
        {"scale": "global", "n_iter": 2000, "a": 50.0,  "lr": 0.2},
        {"scale": "global", "n_iter": 2000, "a": 50.0,  "lr": 1.0},
        # much larger a
        {"scale": "global", "n_iter": 2000, "a": 200.0, "lr": 0.1},
        # pixel scale
        {"scale": "pixel",  "n_iter": 2000, "a": 5.0,   "lr": 0.1},
        # extreme global
        {"scale": "global", "n_iter": 5000, "a": 100.0, "lr": 0.5},
    ],
}


def run_sweep(phf_name: str, n_pairs: int):
    phf = get_phf(phf_name)
    pairs = load_image_pairs(DATA_DIR, n_pairs=n_pairs)
    sources = [p[0] for p in pairs]
    targets = [p[1] for p in pairs]
    target_hashes = [phf.compute(img) for img in targets]

    log.info(f"PHF: {phf_name}, threshold: {phf.threshold}, pairs: {n_pairs}")
    log.info(f"Results CSV: {RESULTS_CSV}")
    log.info(f"Log file: {LOG_FILE}")
    log.info("=" * 110)
    log.debug(f"Source images loaded: {len(sources)}")
    log.debug(f"Target hashes computed: {len(target_hashes)}")

    init_csv()

    total_configs = sum(len(v) for v in SWEEPS.values())
    config_idx = 0

    for attack_name, configs in SWEEPS.items():
        mod = load_attack_module(attack_name)
        attack_fn = mod._attack_single
        log.info(f"\n{'─' * 110}")
        log.info(f"  {attack_name}")
        log.info(f"{'─' * 110}")
        log.info(f"  {'Config':<55} {'ASR':>5} {'L2':>8} {'Queries':>8} "
                 f"{'BestDist':>9} {'Time':>8}")
        log.info(f"  {'-'*55} {'-'*5} {'-'*8} {'-'*8} {'-'*9} {'-'*8}")

        for cfg in configs:
            config_idx += 1
            cfg_str = ", ".join(f"{k}={v}" for k, v in cfg.items())
            log.debug(f"[{config_idx}/{total_configs}] {attack_name} | {cfg_str}")

            successes = 0
            total_l2 = 0.0
            total_queries = 0
            best_dists = []
            per_pair_success = []
            per_pair_dist = []
            per_pair_l2 = []
            per_pair_queries = []

            t0 = time.time()
            for pair_idx, (img, th) in enumerate(zip(sources, target_hashes)):
                tp0 = time.time()
                _, m = attack_fn(img, th, phf, phf.threshold, **cfg)
                tp1 = time.time()

                per_pair_success.append(int(m["success"]))
                per_pair_dist.append(m["final_dist"])
                per_pair_l2.append(m["l2"])
                per_pair_queries.append(m["n_queries"])

                if m["success"]:
                    successes += 1
                total_l2 += m["l2"]
                total_queries += m["n_queries"]
                best_dists.append(m["final_dist"])

                log.debug(
                    f"  pair {pair_idx}: success={m['success']}, "
                    f"dist={m['final_dist']:.1f}, l2={m['l2']:.2f}, "
                    f"queries={m['n_queries']}, time={tp1-tp0:.1f}s"
                )

            elapsed = time.time() - t0

            asr = successes / n_pairs
            avg_l2 = total_l2 / n_pairs
            avg_q = total_queries / n_pairs
            avg_dist = float(np.mean(best_dists))
            min_dist = float(np.min(best_dists))
            max_dist = float(np.max(best_dists))

            marker = " ***" if asr > 0 else ""
            log.info(
                f"  {cfg_str:<55} {asr:>5.1%} {avg_l2:>8.2f} {avg_q:>8.0f} "
                f"{avg_dist:>9.1f} {elapsed:>7.1f}s{marker}"
            )

            # Write to CSV immediately
            write_csv_row({
                "attack": attack_name,
                "phf": phf_name,
                "config": cfg_str,
                "asr": asr,
                "mean_l2": avg_l2,
                "mean_queries": avg_q,
                "mean_final_dist": avg_dist,
                "min_final_dist": min_dist,
                "max_final_dist": max_dist,
                "time_s": round(elapsed, 2),
                "time_per_pair_s": round(elapsed / n_pairs, 2),
                "n_pairs": n_pairs,
                "per_pair_success": str(per_pair_success),
                "per_pair_dist": str([round(d, 1) for d in per_pair_dist]),
                "per_pair_l2": str([round(l, 3) for l in per_pair_l2]),
                "per_pair_queries": str(per_pair_queries),
            })

    close_csv()
    log.info(f"\n{'=' * 110}")
    log.info(f"Done. Results saved to {RESULTS_CSV}")
    log.info(f"Detailed log at {LOG_FILE}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phf", default="phash")
    parser.add_argument("--n-pairs", type=int, default=3)
    args = parser.parse_args()

    log.info(f"\n{'#' * 110}")
    log.info(f"Sweep started at {datetime.now().isoformat()}")
    log.info(f"Args: phf={args.phf}, n_pairs={args.n_pairs}")

    run_sweep(args.phf, args.n_pairs)
