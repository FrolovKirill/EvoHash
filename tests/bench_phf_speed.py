"""Benchmark: compare compute() time across all 4 PHF wrappers.

Measures single-image compute latency and pair distance+collision check,
averaged over N images. Prints a summary table at the end.

Usage:
    python tests/bench_phf_speed.py [--n-images 50]
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from evohash.dataset import DEFAULT_DATA_DIR, _list_images
from evohash.phf import PHF_REGISTRY, get_phf


def make_synthetic_images(n: int, size: int = 224) -> list[Image.Image]:
    """Generate random RGB images for benchmarking."""
    import numpy as np

    rng = np.random.default_rng(42)
    return [
        Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))
        for _ in range(n)
    ]


def load_real_images(n: int) -> list[Image.Image]:
    """Load real images from the dataset directory."""
    paths = _list_images(DEFAULT_DATA_DIR)[:n]
    images = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((224, 224))
        images.append(img)
    return images


def benchmark_phf(name: str, images: list[Image.Image]) -> dict:
    """Benchmark a single PHF: compute, distance, is_collision."""
    try:
        phf = get_phf(name)
    except Exception as e:
        return {"name": name, "error": str(e)}

    n = len(images)

    # --- Warmup (1 call) ---
    try:
        _ = phf.compute(images[0])
    except Exception as e:
        return {"name": name, "error": f"compute failed: {e}"}

    # --- Benchmark compute() ---
    compute_times = []
    hashes = []
    for img in images:
        t0 = time.perf_counter()
        h = phf.compute(img)
        t1 = time.perf_counter()
        compute_times.append(t1 - t0)
        hashes.append(h)

    # --- Benchmark distance() + is_collision() on consecutive pairs ---
    distance_times = []
    n_pairs = n - 1
    for i in range(n_pairs):
        t0 = time.perf_counter()
        phf.distance(hashes[i], hashes[i + 1])
        phf.is_collision(hashes[i], hashes[i + 1])
        t1 = time.perf_counter()
        distance_times.append(t1 - t0)

    # --- Full pipeline: compute pair + distance (simulates attack query) ---
    query_times = []
    for i in range(0, n - 1, 2):
        t0 = time.perf_counter()
        h1 = phf.compute(images[i])
        h2 = phf.compute(images[i + 1])
        phf.distance(h1, h2)
        t1 = time.perf_counter()
        query_times.append(t1 - t0)

    return {
        "name": name,
        "n_images": n,
        "compute_mean_ms": statistics.mean(compute_times) * 1000,
        "compute_median_ms": statistics.median(compute_times) * 1000,
        "compute_std_ms": statistics.stdev(compute_times) * 1000 if n > 1 else 0,
        "compute_total_s": sum(compute_times),
        "distance_mean_us": statistics.mean(distance_times) * 1e6 if distance_times else 0,
        "query_mean_ms": statistics.mean(query_times) * 1000 if query_times else 0,
        "query_total_s": sum(query_times),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark PHF compute speed")
    parser.add_argument("--n-images", type=int, default=50, help="Number of images")
    args = parser.parse_args()

    n = args.n_images

    # Load images
    if DEFAULT_DATA_DIR.exists() and len(_list_images(DEFAULT_DATA_DIR)) >= n:
        print(f"Loading {n} real images from {DEFAULT_DATA_DIR} ...")
        images = load_real_images(n)
    else:
        print(f"Generating {n} synthetic 224x224 images ...")
        images = make_synthetic_images(n)

    print(f"Benchmarking {len(PHF_REGISTRY)} PHFs on {n} images ...\n")

    results = []
    for name in PHF_REGISTRY:
        print(f"  {name} ... ", end="", flush=True)
        r = benchmark_phf(name, images)
        if "error" in r:
            print(f"SKIPPED ({r['error']})")
        else:
            print(f"{r['compute_mean_ms']:.2f} ms/image")
        results.append(r)

    # --- Summary table ---
    print()
    print("=" * 90)
    print(f"{'PHF':<14} {'compute (ms)':<16} {'median (ms)':<14} "
          f"{'std (ms)':<12} {'dist (us)':<12} {'query (ms)':<12} {'total (s)':<10}")
    print("-" * 90)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<14} {'SKIPPED: ' + r['error']}")
            continue
        print(
            f"{r['name']:<14} "
            f"{r['compute_mean_ms']:>10.3f}     "
            f"{r['compute_median_ms']:>9.3f}    "
            f"{r['compute_std_ms']:>8.3f}   "
            f"{r['distance_mean_us']:>8.2f}   "
            f"{r['query_mean_ms']:>8.3f}    "
            f"{r['compute_total_s']:>7.3f}"
        )
    print("=" * 90)

    # --- Relative comparison ---
    valid = [r for r in results if "error" not in r]
    if valid:
        fastest = min(valid, key=lambda r: r["compute_mean_ms"])
        print(f"\nRelative speed (1.0x = {fastest['name']}, "
              f"{fastest['compute_mean_ms']:.3f} ms):")
        for r in sorted(valid, key=lambda r: r["compute_mean_ms"]):
            ratio = r["compute_mean_ms"] / fastest["compute_mean_ms"]
            bar = "#" * int(ratio * 20)
            print(f"  {r['name']:<14} {ratio:>6.1f}x  {bar}")

    # --- Evolution time estimate ---
    print("\n--- Evolution time estimate (per generation, 10 pairs, ~500 queries/pair) ---")
    for r in sorted(valid, key=lambda r: r["compute_mean_ms"]):
        # Each query = 1 compute call; 10 pairs * 500 queries = 5000 computes
        est_s = r["compute_mean_ms"] * 5000 / 1000
        if est_s < 60:
            est_str = f"{est_s:.1f}s"
        else:
            est_str = f"{est_s / 60:.1f}min"
        print(f"  {r['name']:<14} ~{est_str} (PHF compute only, excludes LLM time)")


if __name__ == "__main__":
    main()
