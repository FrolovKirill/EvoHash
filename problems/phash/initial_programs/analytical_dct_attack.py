"""Analytical DCT-domain attack on pHash.

Works directly in the 32×32 grayscale DCT space where pHash operates,
instead of the full 224×224×3 RGB space used by all other baselines.

Key insight: pHash is entirely determined by 64 DCT coefficients in the
top-left 8×8 block of a 32×32 grayscale image. We can read exactly which
bits need to flip from target_hash.hash (XOR with source), then push the
corresponding coefficients across the mean boundary with a calibrated shift.

The perturbation is computed in 32×32 grayscale space and upsampled back
to 224×224. To minimize RGB L2, the grayscale delta is projected onto the
RGB grayscale direction (0.299R + 0.587G + 0.114B).
"""

import numpy as np
from PIL import Image
from scipy.fft import dct, idct
from scipy.ndimage import zoom


def entrypoint(context: dict) -> dict:
    hash_fn = context["hash_fn"]
    threshold = context["threshold"]
    sources = context["source_images"]
    target_hashes = context["target_hashes"]

    attacked_images, metrics = [], []
    for img, target_hash in zip(sources, target_hashes):
        atk_img, m = _attack(img, target_hash, hash_fn, threshold)
        attacked_images.append(atk_img)
        metrics.append(m)

    return {"attacked_images": attacked_images, "metrics": metrics}


def _dct2(a: np.ndarray) -> np.ndarray:
    """2D DCT matching imagehash.phash: dct along axis=0 then axis=1."""
    return dct(dct(a.astype(float), axis=0), axis=1)


def _idct2(a: np.ndarray) -> np.ndarray:
    """Inverse of _dct2."""
    return idct(idct(a, axis=1), axis=0)


def _attack(
    img: Image.Image,
    target_hash,
    hash_fn,
    threshold: int,
    n_iter: int = 20,
    base_margin: float = 3.0,
) -> tuple:
    orig_rgb = np.array(img).astype(np.float32)
    H, W = orig_rgb.shape[:2]

    # Grayscale weights used by PIL.Image.convert("L")
    w_rgb = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    w_sq = float((w_rgb ** 2).sum())  # ≈ 0.447

    target_bits = target_hash.hash  # (8, 8) bool array

    # Work in 32×32 grayscale — same pipeline as imagehash.phash
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

        # Push each wrong coefficient across the mean boundary.
        # For a single-coefficient shift, the minimum to flip 0→1 is:
        #   c' = mean + (mean - c) / 63 + margin  (derivation: c' > new_mean)
        # For 1→0: c' = mean - (c - mean) / 63 - margin
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

        # Inner loop: correct for mean drift caused by simultaneous changes
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

        # Reconstruct 32×32 grayscale from modified DCT block
        d_new = d.copy()
        d_new[:8, :8] = new_block
        new_gray32 = np.clip(_idct2(d_new), 0, 255)

        # Upsample perturbation from 32×32 to original (H, W)
        delta32 = new_gray32 - gray32
        delta_up = zoom(delta32, (H / 32.0, W / 32.0), order=1)  # (H, W) float

        # Apply grayscale delta to RGB with minimum L2:
        # δ_RGB = δ_gray * w_rgb / ||w_rgb||²  (projects onto grayscale direction)
        delta_rgb = delta_up[:, :, None] * (w_rgb / w_sq)[None, None, :]
        result_rgb = np.clip(orig_rgb + delta_rgb, 0, 255)
        result_pil = Image.fromarray(result_rgb.astype(np.uint8))

        dist = float(hash_fn.distance(hash_fn.compute(result_pil), target_hash))
        n_queries += 1

        if dist < best_dist:
            best_dist = dist
            best_rgb = result_rgb
            current32 = new_gray32

        # Increase margin each iteration to overcome interpolation loss
        margin *= 2.0

    final_pil = Image.fromarray(np.clip(best_rgb, 0, 255).astype(np.uint8))
    final_dist = float(hash_fn.distance(hash_fn.compute(final_pil), target_hash))
    n_queries += 1

    l2 = float(np.linalg.norm((best_rgb - orig_rgb).flatten()) / np.sqrt(orig_rgb.size))
    return final_pil, {
        "success": final_dist <= threshold,
        "l2": l2,
        "n_queries": n_queries,
        "final_dist": final_dist,
    }
