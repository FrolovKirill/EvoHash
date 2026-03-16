"""ImageNet validation set loader for EvoHash experiments.

Images are expected in ``data/imagenet_val/`` (relative to the project root).
Run ``scripts/download_dataset.py`` to populate this directory.

Usage::

    from evohash.dataset import load_image_pairs

    pairs = load_image_pairs(data_dir, n_pairs=10)
    for src, tgt in pairs:
        # src, tgt are PIL.Image objects (RGB, 224×224)
        ...
"""

import random
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG"}

#: Default directory where ImageNet Val images are stored.
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "imagenet_val"


def _list_images(data_dir: Path) -> list[Path]:
    """Return all image files in *data_dir* sorted by name."""
    files = [
        p for p in sorted(data_dir.iterdir())
        if p.suffix in _IMAGE_EXTENSIONS
    ]
    return files


def load_image_pairs(
    data_dir: Path | str | None = None,
    n_pairs: int = 10,
    image_size: tuple[int, int] = (224, 224),
    seed: int | None = None,
) -> list[tuple[Image.Image, Image.Image]]:
    """
    Return a list of (source, target) PIL Image pairs sampled from *data_dir*.

    Each pair contains two *distinct* images.  The source image will be
    perturbed by an attack strategy; the target image provides the hash to
    collide with.

    Args:
        data_dir: Directory containing JPEG/PNG images.  Defaults to
            ``data/imagenet_val/`` relative to the project root.
        n_pairs: Number of (source, target) pairs to return.
        image_size: Images are resized to this (width, height).
        seed: Optional random seed for reproducibility.

    Returns:
        List of ``(source_image, target_image)`` tuples.

    Raises:
        FileNotFoundError: If *data_dir* does not exist or contains no images.
        ValueError: If there are fewer than ``2 * n_pairs`` images available.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}\n"
            "Run 'python scripts/download_dataset.py' to download images."
        )

    files = _list_images(data_dir)

    if not files:
        raise FileNotFoundError(
            f"No images found in {data_dir}.\n"
            "Run 'python scripts/download_dataset.py' to download images."
        )

    if len(files) < n_pairs * 2:
        raise ValueError(
            f"Need at least {n_pairs * 2} images for {n_pairs} pairs, "
            f"but only {len(files)} found in {data_dir}."
        )

    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)

    pairs: list[tuple[Image.Image, Image.Image]] = []
    for i in range(0, n_pairs * 2, 2):
        src = _load_image(shuffled[i], image_size)
        tgt = _load_image(shuffled[i + 1], image_size)
        pairs.append((src, tgt))

    return pairs


def _load_image(path: Path, size: tuple[int, int]) -> Image.Image:
    """Load and resize a single image to RGB."""
    return Image.open(path).convert("RGB").resize(size, Image.LANCZOS)


def get_generation_seed(phf_name: str, redis_port: int = 6379, redis_db: int = 0) -> int:
    """Read the max completed generation from Redis to use as pair seed.

    All programs within the same generation see the same max_gen (because the
    previous generation is fully complete before new mutants are created),
    so they get identical pairs.  Between generations max_gen increments,
    producing a different shuffle.
    """
    try:
        import redis as _redis
        import json as _json
        r = _redis.Redis(port=redis_port, db=redis_db, socket_connect_timeout=1)
        keys = r.keys(f"{phf_name}:program:*")
        max_gen = -1
        for key in keys:
            raw = r.get(key)
            if raw:
                data = _json.loads(raw)
                g = data.get("lineage", {}).get("generation", -1)
                if isinstance(g, (int, float)) and g > max_gen:
                    max_gen = int(g)
        return max(0, max_gen + 1)
    except Exception:
        return 0


def image_to_array(image: Image.Image) -> np.ndarray:
    """Convert a PIL Image to a float32 numpy array in [0, 255]."""
    return np.array(image).astype(np.float32)


def array_to_image(array: np.ndarray) -> Image.Image:
    """Convert a float32 numpy array (H×W×3, values 0–255) to a PIL Image."""
    clipped = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(clipped)
