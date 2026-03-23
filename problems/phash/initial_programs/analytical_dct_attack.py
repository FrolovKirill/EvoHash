"""Analytical DCT-domain attack on pHash.

Works directly in the 32x32 grayscale DCT space where pHash operates.
pHash-specific: exploits the fact that pHash is entirely determined by
64 DCT coefficients in the top-left 8x8 block of a 32x32 grayscale image.
"""

import numpy as np
from PIL import Image
from scipy.fft import dct, idct
from scipy.ndimage import zoom


def normalised_l2(original, perturbed):
    diff = perturbed.astype(float) - original.astype(float)
    return float(np.linalg.norm(diff.flatten()) / np.sqrt(original.size))


def _dct2(a):
    """2D DCT matching imagehash.phash."""
    return dct(dct(a.astype(float), axis=0), axis=1)


def _idct2(a):
    """Inverse of _dct2."""
    return idct(idct(a, axis=1), axis=0)


def _attack_single(img, target_hash, hash_fn, threshold,
                   n_iter=20, base_margin=3.0):
    orig_rgb = np.array(img).astype(np.float32)
    H, W = orig_rgb.shape[:2]

    w_rgb = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    w_sq = float((w_rgb ** 2).sum())

    target_bits = target_hash.hash  # (8, 8) bool array

    gray32 = np.array(
        img.convert("L").resize((32, 32), Image.LANCZOS)
    ).astype(np.float64)

    current32 = gray32.copy()
    best_rgb = orig_rgb.copy()
    best_dist = float(hash_fn.distance(hash_fn.compute(img), target_hash))
    n_queries = 1
    margin = base_margin

    for _ in range(n_iter):
        if best_dist <= threshold:
            break

        d = _dct2(current32)
        block = d[:8, :8].copy()
        mean = block.mean()

        current_bits = block > mean
        wrong_mask = current_bits != target_bits
        if not wrong_mask.any():
            break

        new_block = block.copy()
        for i in range(8):
            for j in range(8):
                if not wrong_mask[i, j]:
                    continue
                c = block[i, j]
                if target_bits[i, j]:
                    new_block[i, j] = mean + (mean - c) / 63.0 + margin
                else:
                    new_block[i, j] = mean - (c - mean) / 63.0 - margin

        for _ in range(8):
            new_mean = new_block.mean()
            still_wrong = (new_block > new_mean) != target_bits
            if not still_wrong.any():
                break
            for i in range(8):
                for j in range(8):
                    if not still_wrong[i, j]:
                        continue
                    c = new_block[i, j]
                    if target_bits[i, j]:
                        new_block[i, j] = new_mean + (new_mean - c) / 63.0 + margin
                    else:
                        new_block[i, j] = new_mean - (c - new_mean) / 63.0 - margin

        d_new = d.copy()
        d_new[:8, :8] = new_block
        new_gray32 = np.clip(_idct2(d_new), 0, 255)

        delta32 = new_gray32 - gray32
        delta_up = zoom(delta32, (H / 32.0, W / 32.0), order=1)

        delta_rgb = delta_up[:, :, None] * (w_rgb / w_sq)[None, None, :]
        result_rgb = np.clip(orig_rgb + delta_rgb, 0, 255)
        result_pil = Image.fromarray(result_rgb.astype(np.uint8))

        dist = float(hash_fn.distance(hash_fn.compute(result_pil), target_hash))
        n_queries += 1

        if dist < best_dist:
            best_dist = dist
            best_rgb = result_rgb
            current32 = new_gray32

        margin *= 2.0

    final_pil = Image.fromarray(np.clip(best_rgb, 0, 255).astype(np.uint8))
    final_dist = float(hash_fn.distance(hash_fn.compute(final_pil), target_hash))
    n_queries += 1

    l2 = normalised_l2(orig_rgb, best_rgb)
    return final_pil, {
        "success": final_dist <= threshold,
        "l2": l2,
        "n_queries": n_queries,
        "final_dist": final_dist,
    }


def entrypoint(context: dict) -> dict:
    hash_fn = context["hash_fn"]
    threshold = context["threshold"]
    sources = context["source_images"]
    target_hashes = context["target_hashes"]

    attacked_images, metrics = [], []
    for img, th in zip(sources, target_hashes):
        atk, m = _attack_single(img, th, hash_fn, threshold,
                                n_iter=20, base_margin=3.0)
        attacked_images.append(atk)
        metrics.append(m)

    return {"attacked_images": attacked_images, "metrics": metrics}