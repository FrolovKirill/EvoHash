"""Download a subset of ImageNet validation images for EvoHash experiments.

This script downloads images into ``data/imagenet_val/`` at the project root.
It tries several strategies in order:

  1. HuggingFace ``datasets`` library (imagenet-1k) — requires HF account +
     ImageNet licence agreement at https://huggingface.co/datasets/imagenet-1k
  2. HuggingFace ``datasets`` library (imagenet-mini) — smaller, no auth needed
  3. Tiny-ImageNet (downloadable from a public URL, 64×64 images)
  4. Synthetic random images (fallback — for smoke-testing only)

Usage::

    python scripts/download_dataset.py              # 200 images (default)
    python scripts/download_dataset.py --n 50       # fewer images
    python scripts/download_dataset.py --synthetic  # always use synthetic

"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "imagenet_val"

# ── Tiny-ImageNet public download ─────────────────────────────────────────────
TINY_IMAGENET_URL = (
    "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
)


def _try_hf_imagenet(n: int, out_dir: Path) -> int:
    """Try to download from HuggingFace imagenet-1k.  Returns #images saved."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("  datasets not installed; skipping HuggingFace strategy.")
        return 0

    print("  Trying HuggingFace imagenet-1k (requires auth)…")
    try:
        ds = load_dataset(
            "imagenet-1k",
            split="validation",
            streaming=True,
            trust_remote_code=True,
        )
        saved = 0
        for i, example in enumerate(ds):
            if saved >= n:
                break
            img = example["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(out_dir / f"imagenet_{i:05d}.JPEG")
            saved += 1
            if saved % 20 == 0:
                print(f"    {saved}/{n} saved…")
        print(f"  Saved {saved} images from imagenet-1k.")
        return saved
    except Exception as e:
        print(f"  imagenet-1k failed: {e}")
        return 0


def _try_hf_imagenet_mini(n: int, out_dir: Path) -> int:
    """Try HuggingFace imagenet-mini (no auth required).  Returns #images saved."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        return 0

    print("  Trying HuggingFace timm/mini-imagenet…")
    try:
        ds = load_dataset(
            "timm/mini-imagenet",
            split="validation",
            streaming=True,
            trust_remote_code=True,
        )
        saved = 0
        for i, example in enumerate(ds):
            if saved >= n:
                break
            img = example.get("image") or example.get("img")
            if img is None:
                continue
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(out_dir / f"mini_imagenet_{i:05d}.JPEG")
            saved += 1
            if saved % 20 == 0:
                print(f"    {saved}/{n} saved…")
        print(f"  Saved {saved} images from mini-imagenet.")
        return saved
    except Exception as e:
        print(f"  mini-imagenet failed: {e}")
        return 0


def _try_tiny_imagenet(n: int, out_dir: Path) -> int:
    """Download Tiny-ImageNet (200 classes, 64×64).  Returns #images saved."""
    print("  Trying Tiny-ImageNet download (~237 MB)…")
    zip_path = DATA_DIR.parent / "tiny-imagenet-200.zip"

    try:
        print(f"    Downloading from {TINY_IMAGENET_URL} …")
        urllib.request.urlretrieve(
            TINY_IMAGENET_URL,
            zip_path,
            reporthook=lambda b, bs, tot: print(
                f"\r    {min(b*bs, tot) / 1e6:.1f}/{tot/1e6:.1f} MB",
                end="",
                flush=True,
            ) if tot > 0 else None,
        )
        print()

        saved = 0
        with zipfile.ZipFile(zip_path) as zf:
            val_files = [
                name for name in zf.namelist()
                if "/val/images/" in name and name.endswith(".JPEG")
            ]
            val_files = val_files[:n]
            for i, name in enumerate(val_files):
                data = zf.read(name)
                img = Image.open(io.BytesIO(data)).convert("RGB")
                img.save(out_dir / f"tiny_imagenet_{i:05d}.JPEG")
                saved += 1

        zip_path.unlink(missing_ok=True)
        print(f"  Saved {saved} images from Tiny-ImageNet.")
        return saved

    except Exception as e:
        print(f"  Tiny-ImageNet download failed: {e}")
        zip_path.unlink(missing_ok=True)
        return 0


def _generate_synthetic(n: int, out_dir: Path, image_size: int = 224) -> int:
    """Generate random synthetic images as a last-resort fallback."""
    print(f"  Generating {n} synthetic random images…")
    rng = np.random.default_rng(42)
    for i in range(n):
        arr = rng.integers(0, 256, (image_size, image_size, 3), dtype=np.uint8)
        # Add some structure so hashes are not all the same
        # (low-frequency sinusoidal pattern per image)
        freq_x = rng.uniform(0.01, 0.1)
        freq_y = rng.uniform(0.01, 0.1)
        x = np.arange(image_size).reshape(1, -1)
        y = np.arange(image_size).reshape(-1, 1)
        pattern = (
            30 * np.sin(2 * np.pi * freq_x * x) * np.sin(2 * np.pi * freq_y * y)
        ).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + pattern[:, :, np.newaxis], 0, 255).astype(
            np.uint8
        )
        img = Image.fromarray(arr)
        img.save(out_dir / f"synthetic_{i:05d}.JPEG")
    print(f"  Saved {n} synthetic images.")
    return n


def download(n: int = 200, force_synthetic: bool = False) -> None:
    """Download *n* images into ``data/imagenet_val/``."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    existing = list(DATA_DIR.glob("*.JPEG")) + list(DATA_DIR.glob("*.jpg"))
    if len(existing) >= n and not force_synthetic:
        print(f"Already have {len(existing)} images in {DATA_DIR}. Nothing to do.")
        return

    print(f"Downloading {n} images to {DATA_DIR}…\n")

    if force_synthetic:
        _generate_synthetic(n, DATA_DIR)
        return

    saved = _try_hf_imagenet(n, DATA_DIR)
    if saved >= n:
        return

    remaining = n - saved
    saved += _try_hf_imagenet_mini(remaining, DATA_DIR)
    if saved >= n:
        return

    remaining = n - saved
    saved += _try_tiny_imagenet(remaining, DATA_DIR)
    if saved >= n:
        return

    remaining = n - saved
    print(
        f"\nAll network strategies failed or partial ({saved}/{n} images so far).\n"
        f"Generating {remaining} synthetic images as fallback…"
    )
    _generate_synthetic(remaining, DATA_DIR)

    total = len(list(DATA_DIR.glob("*.JPEG")) + list(DATA_DIR.glob("*.jpg")))
    print(f"\nDone.  {total} images in {DATA_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download dataset for EvoHash.")
    parser.add_argument(
        "--n-images",
        type=int,
        default=200,
        metavar="N",
        help="Number of images to download (default: 200).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Skip network downloads and generate synthetic images.",
    )
    args = parser.parse_args()

    download(n=args.n_images, force_synthetic=args.synthetic)


if __name__ == "__main__":
    main()
